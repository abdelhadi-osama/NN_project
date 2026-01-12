import numpy as np
import math

class DataGenerator:
    def __init__(self, x, y, batch_size=32, shuffle=True):
        """
        The Waiter.
        Takes the full dataset and serves it in small batches.
        """
        self.x = x
        self.y = y
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.n_samples = x.shape[0]
        # Calculate how many batches we have (e.g. 100 samples / 32 = 4 batches)
        self.num_batches = math.ceil(self.n_samples / self.batch_size)

    def __len__(self):
        """Allows us to say: len(generator) to know total batches."""
        return self.num_batches

    def __iter__(self):
        """
        The Loop Logic.
        This runs every time you say: 'for batch in generator:'
        """
        # 1. Create Indices
        indices = np.arange(self.n_samples)
        
        # 2. Shuffle (if requested)
        if self.shuffle:
            np.random.shuffle(indices)
            
        # 3. Batch Loop
        for start_idx in range(0, self.n_samples, self.batch_size):
            end_idx = min(start_idx + self.batch_size, self.n_samples)
            
            # Get the indices for this specific batch
            batch_indices = indices[start_idx:end_idx]
            
            # Slice the data
            # NOTE: We slice here on demand. We do NOT copy the whole dataset.
            batch_x = self.x[batch_indices]
            batch_y = self.y[batch_indices]
            
            yield batch_x, batch_y