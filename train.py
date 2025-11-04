"""Vanilla predictive coding model trained on MNIST in an unsupervised manner.

Examples:
    $ python train.py
"""
import os

from matplotlib import pyplot as plt
import numpy as np
from sklearn.decomposition import PCA
import torch
from torch import nn, optim
import torch.nn.functional as F
from torchvision import datasets, transforms
from tqdm import tqdm
import wandb

import predictive_coding as pc
from utils import KNNBuffer

plt.rcParams['figure.dpi'] = 300

def load_data(batch_size=500, shuffle_train=True):
    transform = transforms.Compose(
        [transforms.ToTensor(),
         transforms.Lambda(lambda x: torch.flatten(x))])
    data_dir = '../data'
    train_dataset = datasets.MNIST(data_dir,
                                   download=True,
                                   train=True,
                                   transform=transform)
    test_dataset = datasets.MNIST(data_dir,
                                  download=True,
                                  train=False,
                                  transform=transform)

    train_loader = torch.utils.data.DataLoader(train_dataset,
                                               batch_size=batch_size,
                                               shuffle=shuffle_train)
    test_loader = torch.utils.data.DataLoader(test_dataset,
                                              batch_size=batch_size,
                                              shuffle=False)
    print(
        f'# train images: {len(train_dataset)} and # test images: {len(test_dataset)}'
    )
    return train_loader, test_loader



def test(model, test_loader, device):
    model.eval()
    correct = 0
    total = 0
    total_loss = 0
    for data, label in test_loader:
        data, label = data.to(device), label.to(device)
        pred = model(data)
        _, predicted = torch.max(pred, -1)
        total += label.size(0)
        correct += (predicted == label).sum().item()
    model.train()
    accuracy = round(correct / total, 4)
    return accuracy


def create_model(input_size, hidden_size, output_size, activation_fn, device, **kwargs):
    layers = [pc.PCLayer(), nn.Linear(input_size, hidden_size),
                            activation_fn(), pc.PCLayer(),
                            nn.Linear(hidden_size, hidden_size),
                            activation_fn(), pc.PCLayer(),
                            nn.Linear(hidden_size, output_size)]
    model = nn.Sequential(*layers)
    model.train()
    model.to(device)
    return model


def loss_fn(output, _target):
    """This loss function holds to the error of the output layer of the model."""
    return 0.5 * (output - _target).pow(2).sum()

def train_linear_probe_batch(optimizer, classifier, criterion, batch, labels_batch):
    optimizer.zero_grad()
    output = classifier(batch)
    loss = criterion(output, labels_batch)
    loss.backward()
    optimizer.step()
    # compute accuracy
    predicted = torch.argmax(output, -1)
    correct = (predicted == labels_batch).sum().item()
    acc = correct / labels_batch.size(0)
    return {'train_linear_probe_loss': loss.item(), 'train_linear_probe_accuracy': acc}

def knn_classification(knn_buffer, representations, labels, k):
    """Perform kNN classification using the representations in the knn_buffer."""
    buffer_reps, buffer_labels = knn_buffer.get_all()
    # compute distances between representations and buffer_reps
    dists = torch.cdist(representations, buffer_reps)
    # get the indices of the k nearest neighbors
    knn_indices = torch.topk(dists, k, largest=False).indices
    # get the labels of the k nearest neighbors
    knn_labels = buffer_labels[knn_indices]
    # perform majority vote
    preds = []
    for i in range(knn_labels.size(0)):
        counts = torch.bincount(knn_labels[i])
        preds.append(torch.argmax(counts).item())
    preds = torch.tensor(preds).to(representations.device)
    # compute accuracy
    correct = (preds == labels).sum().item()
    acc = correct / labels.size(0)
    return {'train_knn_accuracy': acc}

def run_experimental_setting(config):
    """This is the core function that runs the whole pipeline for a given configuration."""
    # log the configuration to wandb
    wandb.init(project=config['project'],
               group=config['group'],
               config=config,
               reinit=True,
               mode=config['mode'])

    # reset seed to ensure reproducibility
    torch.manual_seed(42)
    train_loader, test_loader = load_data()
    device = config['device']
    if config['workers'] != 0:
        # 0 means no parallelism, while -1 means use all available cores and > 0 means use that many cores
        # so if using parallelism, restrict number of threads to avoid CPU oversubscription
        # otherwise, just let it rip
        torch.set_num_threads(1)
    model = create_model(**config)
    # unpack the optimizer kwargs, because the trainer expects them as dictionaries
    optimizer_x_kwargs = {'lr': config['x_lr']}
    optimizer_p_kwargs = {'lr': config['p_lr']}
    trainer = pc.PCTrainer(model=model,
                           optimizer_x_kwargs=optimizer_x_kwargs,
                           optimizer_p_kwargs=optimizer_p_kwargs,
                           **config)
    
    # train the model
    step = 0
    for epoch in range(config['epochs']):
        losses = []
        energies = []
        overalls = []
        example_count = 0
        for data, label in tqdm(train_loader, desc=f'Epoch {epoch}'):
            data, label = data.to(device), label.to(device)
            # note that the input is a placeholder for the latent code and that the target is the data
            inputs = torch.zeros(data.size(0), config['input_size']).to(device)
            target = data

            results = trainer.train_on_batch(
                inputs=inputs,
                loss_fn=loss_fn,
                loss_fn_kwargs={'_target': target},
                is_optimize_inputs=config['is_optimize_inputs'],
                is_log_progress=config['is_log_progress'],
            )
            losses.append(results['loss'][0])
            energies.append(results['energy'][0])
            overalls.append(results['overall'][0])
            example_count += data.size(0)

            # get model xs and weights to check that everything isn't collapsing to 0
            xs = trainer.get_model_xs_copy()
            xs_abs = [layer.abs().mean().item() for layer in xs]
            xs_abs = {f'layer_{i}_x_abs': xs_abs[i] for i in range(len(xs_abs))}
            weight_abs, _ = trainer.get_weights_norms()
            weight_abs = [layer.item() for layer in weight_abs]
            weight_abs = {f'layer_{i}_weight_abs': weight_abs[i] for i in range(len(weight_abs))}
            metric_dict = dict(loss=results['loss'][0],
                                energy=results['energy'][0],
                                overall=results['overall'][0])
            metric_dict.update(xs_abs)
            metric_dict.update(weight_abs)
                        
            # train the linear probe every linear_probe_steps
            if ((step+1) % config['linear_probe_steps'] == 0) and (epoch >= config['linear_prob_hold_epochs']):
                # get the latent representations from the trainer (use the input attribute)
                latent_representations = trainer.inputs.detach()
                labels_batch = label
                linear_probe_metrics = train_linear_probe_batch(
                    config['linear_probe_optimizer'],
                    config['classifier'],
                    config['linear_probe_criterion'],
                    latent_representations,
                    labels_batch
                )
                tqdm.write(f"Step {step}: Linear Probe Loss: {linear_probe_metrics['train_linear_probe_loss']:.4f}, Accuracy: {linear_probe_metrics['train_linear_probe_accuracy']*100:.2f}%")
                metric_dict.update(linear_probe_metrics)

            # evaluate kNN classification every knn_steps
            if (step+1) % config['knn_steps'] == 0:
                latent_representations = trainer.inputs.detach()
                knn_metrics = knn_classification(
                    config['knn_buffer'],
                    latent_representations,
                    label,
                    config['knn_k']
                )
                tqdm.write(f"Step {step}: kNN Accuracy: {knn_metrics['train_knn_accuracy']*100:.2f}%")
                metric_dict.update(knn_metrics)

            # update the kNN buffer
            latent_representations = trainer.inputs.detach()
            knn_buffer = config['knn_buffer']
            knn_buffer.add_batch(latent_representations, label)

            # log the loss and energy to wandb
            wandb.log(metric_dict, step=step)
            step += 1
    
    return model


def record_latent_representations(model, config, use_train=False, loader=None):
    model.train()
    # iterate over the dataset, generate predictions, save a few images
    generation_trainer = pc.PCTrainer(
        model,
        T=config['T'],
        optimizer_x_fn=config['optimizer_x_fn'],
        optimizer_x_kwargs={'lr': config['x_lr']},
        update_p_at="never",  # the model is already trained
    )

    if loader is None:
        train_loader, test_loader = load_data(shuffle_train=False)
        if use_train:
            data_loader = train_loader
        else:
            data_loader = test_loader
    else:
        data_loader = loader
    representations = []
    ordered_labels = []
    for data, label in tqdm(data_loader, desc='Recording Latent Representations'):
        data, label = data.to(config['device']), label.to(config['device'])
        inputs = torch.zeros(data.size(0), config['input_size']).to(config['device'])
        results = generation_trainer.train_on_batch(
            inputs=inputs,
            loss_fn=loss_fn,
            loss_fn_kwargs={'_target': data},
            is_optimize_inputs=True,
            is_return_outputs=True
        )
        representations.append(generation_trainer.inputs.detach())
        ordered_labels.append(label)

    ordered_labels = torch.cat(ordered_labels, dim=0)
    representations = torch.cat(representations, dim=0).detach().to(config['device'])
    model.eval()
    return representations, ordered_labels

def main():
    config = {
        'device': 'cpu', # change to 'cuda' if GPU is available
        'workers': -1,
        'output_size': 784,
        'hidden_size': 256,
        'input_size': 128,
        'activation_fn': torch.nn.ReLU,
        'x_lr': 0.01,
        'p_lr': 0.001,
        'epochs': 10,
        'T': 150,
        'linear_prob_hold_epochs': 1,
        'linear_probe_steps': 1,
        'linear_probe_lr': 0.01,
        'optimizer_x_fn': optim.SGD,
        'optimizer_p_fn': optim.Adam,
        'update_p_at': 'last',
        'model_type': 'unsupervised',
        'is_optimize_inputs': True,
        'project': 'predictive-coding-tutorial',
        'group': 'unsupervised_learning',
        # 'mode': 'disabled',
        'mode': 'online',
        'is_log_progress': False
    }

    # initialize linear probe objects
    classifier = nn.Sequential(nn.Linear(config['input_size'], 10)).to(config['device'])
    optimizer = optim.Adam(classifier.parameters(), lr=config['linear_probe_lr'])
    criterion = nn.CrossEntropyLoss()
    config['classifier'] = classifier
    config['linear_probe_optimizer'] = optimizer
    config['linear_probe_criterion'] = criterion

    # initialize kNN buffer
    config['knn_buffer_size'] = 1_000 # keep a buffer of representations for kNN classification
    config['knn_k'] = 5
    config['knn_steps'] = 100
    knn_buffer = KNNBuffer(buffer_size=config['knn_buffer_size'],
                           representation_size=config['input_size'],
                           device=config['device'])
    config['knn_buffer'] = knn_buffer

    # run the experimental setting
    model = run_experimental_setting(config)

    # get the latent representations for the test set
    latent_representations, ordered_labels = record_latent_representations(model, config)

    # put the model in eval mode
    model.eval()

        # use the existing classifier and knn buffer to evaluate on the test set
    # chunk into batches of 32
    batch_size = 500
    linear_probe_correct = 0
    linear_probe_losses = 0
    knn_correct = 0
    total = latent_representations.size(0)
    step = wandb.run.step
    for i in tqdm(range(0, len(latent_representations), batch_size), desc='Evaluating Linear Probe and kNN on Test Set'):
        batch = latent_representations[i:i+batch_size]
        labels_batch = ordered_labels[i:i+batch_size]
        # linear probe accuracy
        output = classifier(batch)
        loss = criterion(output, labels_batch)
        linear_probe_losses += loss.item() * labels_batch.size(0)
        predicted = torch.argmax(output, -1)
        linear_probe_correct += (predicted == labels_batch).sum().item()
        # kNN accuracy
        knn_metrics = knn_classification(
            knn_buffer,
            batch,
            labels_batch,
            config['knn_k']
        )
        knn_correct += knn_metrics['train_knn_accuracy'] * labels_batch.size(0)
    linear_probe_acc = linear_probe_correct / total
    linear_probe_loss = linear_probe_losses / total
    knn_acc = knn_correct / total
    print(f'Final Linear Probe Accuracy: {linear_probe_acc * 100:.2f}%')
    print(f'Final kNN Accuracy: {knn_acc * 100:.2f}%')
    wandb.log({'test_linear_probe_accuracy': linear_probe_acc, 'test_linear_probe_loss': linear_probe_loss}, step=step)
    wandb.log({'test_knn_accuracy': knn_acc}, step=step)
    # get one sample per class
    train_loader, test_loader = load_data()
    images = []
    samples = []
    labels = []
    for i in range(10):
        for idx, label_name in enumerate(ordered_labels):
            if label_name == i:
                images.append(test_loader.dataset[idx][0])
                samples.append(latent_representations[idx])
                labels.append(label_name.item())
                break
    
    # generate images from the samples
    generated_images = model(torch.stack(samples).to(config['device'])).detach()

    # plot the inputs with the generated samples underneath
    fig, axs = plt.subplots(2, 10, figsize=(10, 2))
    for i in range(10):
        axs[0, i].imshow(images[i].view(28, 28).cpu().detach().numpy(), cmap='gray')
        axs[0, i].axis('off')
        axs[1, i].imshow(generated_images[i].view(28, 28).cpu().detach().numpy(), cmap='gray')
        axs[1, i].axis('off')
        # axs[0, i].set_title(f'Label: {labels[i]}')
        # axs[1, i].set_title('Generated')
    plt.tight_layout()
    os.makedirs('./analysis', exist_ok=True)
    plt.savefig('./analysis/generated_images.png')
    wandb.log({'generated_images': wandb.Image('./analysis/generated_images.png')})

    # run PCA on the representations to see if we can visually see class separation
    pca = PCA(n_components=2)
    embeddings = pca.fit_transform(latent_representations.cpu().detach().numpy())
    plt.figure(figsize=(6, 6))
    # sample 1000 points and color them by their label
    samples = np.random.choice(embeddings.shape[0], 1000, replace=False)
    sample_embeddings = embeddings[samples]
    sample_labels = ordered_labels[samples]
    colors = ['red', 'blue', 'green', 'orange', 'purple', 'brown', 'pink', 'gray', 'olive', 'cyan']
    for i in range(len(sample_labels)):
        plt.scatter(sample_embeddings[i, 0], sample_embeddings[i, 1], c=colors[sample_labels[i].item()], alpha=0.5)
    # plot the class colors in the legend
    for i in range(10):
        plt.scatter([], [], c=colors[i], label=f'Label: {i}')
    plt.legend()
    plt.savefig('./analysis/pca_embeddings.png')
    wandb.log({'pca_embeddings': wandb.Image('./analysis/pca_embeddings.png')})

if __name__ == '__main__':
    main()
