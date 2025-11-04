import torch

class KNNBuffer:
    """Circular buffer to store representations and labels for kNN classification."""
    def __init__(self, buffer_size, representation_size, device):
        self.buffer_size = buffer_size
        self.representation_size = representation_size
        self.device = device
        self.representations = torch.zeros(buffer_size, representation_size).to(device)
        self.labels = torch.zeros(buffer_size, dtype=torch.long).to(device)
        self.current_index = 0

    def add_batch(self, reps_batch, labels_batch):
        batch_size = reps_batch.size(0)
        end_index = self.current_index + batch_size
        if end_index <= self.buffer_size:
            self.representations[self.current_index:end_index] = reps_batch
            self.labels[self.current_index:end_index] = labels_batch
        else:
            first_part_size = self.buffer_size - self.current_index
            self.representations[self.current_index:] = reps_batch[:first_part_size]
            self.labels[self.current_index:] = labels_batch[:first_part_size]
            remaining_size = batch_size - first_part_size
            self.representations[:remaining_size] = reps_batch[first_part_size:]
            self.labels[:remaining_size] = labels_batch[first_part_size:]
        self.current_index = (self.current_index + batch_size) % self.buffer_size
    
    def get_all(self):
        return self.representations, self.labels
