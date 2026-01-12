import numpy as np

class Preprocessor:
    def __init__(self):
        """
        Step 1: The Blueprint.
        We initialize containers to hold the 'Stats' we will learn later.
        We do NOT store the raw data here to save RAM.
        """
        # 1. Normalization Stats (For x_train/x_test)
        self.min_val = None
        self.max_val = None
        
        # 2. Label Encoding Stats (For y_train/y_test)
        self.label_map = {}      # Example: {'M': 0, 'B': 1}
        self.reverse_map = {}    # Example: {0: 'M', 1: 'B'}
        self.num_classes = 0     # Example: 2

    def fit(self, data):
        """
        Step 2: The Learner.
        Calculates statistics from Training Data.
        """
        # 1. Safety Casting: Ensure data is numbers, not strings
        # If we skip this, np.min() on strings gives weird alphabetical results!
        data = data.astype(np.float32)
        
        # 2. Learn Stats
        # axis=0 means "down the columns". 
        # For images (N, 28, 28), it finds min for every pixel position.
        self.min_val = np.min(data, axis=0)
        self.max_val = np.max(data, axis=0)
        
        print(f"    [Preprocessor] Fit Complete. Min shape: {self.min_val.shape}")

    def transform(self, data, flatten=False):
        """
        Step 3: The Transformer.
        Applies normalization and optional flattening.
        """
        # 1. Integrity Check
        if self.min_val is None:
            raise ValueError("Preprocessor must be fit() before transform()!")
        
        # 2. Casting (The Gatekeeper)
        # We force float32 immediately so math works on strings.
        data = data.astype(np.float32)
        
        # 3. Normalization (Min-Max Scaling)
        # We add 1e-8 (epsilon) to avoid dividing by zero if max == min
        range_val = self.max_val - self.min_val
        
        # If the range is 0 (all values are the same), result is 0.
        # Otherwise, apply formula.
        if np.all(range_val == 0):
             data_norm = data - self.min_val
        else:
             data_norm = (data - self.min_val) / (range_val + 1e-8)
            
        # 4. Flattening (Optional - for MLPs)
        # Input: (N, 28, 28) -> Output: (N, 784)
        if flatten and data_norm.ndim > 2:
            num_samples = data_norm.shape[0]
            # Calculate total features (e.g., 28*28 = 784)
            flat_dim = np.prod(data_norm.shape[1:]) 
            data_norm = data_norm.reshape(num_samples, flat_dim)
            
        return data_norm
    

    def fit_encode_labels(self, labels):
        """
        Step 4a: Learn unique labels AND encode them.
        Use this for Training Labels (y_train).
        """
        # 1. If we didn't load a map from Config, learn it now
        if not self.label_map:
            unique_labels = np.unique(labels)
            self.num_classes = len(unique_labels)
            
            for i, label in enumerate(unique_labels):
                self.label_map[label] = i
                self.reverse_map[i] = label
            
            print(f"    [Preprocessor] Labels learned: {self.label_map}")
        
        # 2. Encode
        return self.encode_labels(labels)

    def encode_labels(self, labels):
        """
        Step 4b: Encode labels using the existing map.
        Use this for Validation (y_val) and Test (y_test).
        """
        if not self.label_map:
            raise ValueError("Label encoder not fit yet!")
            
        # 1. Convert Strings/Ints to Indices (0, 1, 2...)
        # We use a list comprehension for speed and safety
        try:
            idxs = np.array([self.label_map[l] for l in labels])
        except KeyError as e:
            raise ValueError(f"Encountered unknown label in test set: {e}")
            
        # 2. Create One-Hot Matrix
        # Shape: (Num_Samples, Num_Classes)
        one_hot = np.zeros((len(idxs), self.num_classes), dtype=np.float32)
        
        # 3. Set the correct index to 1.0
        # Advanced NumPy: specific row, specific column = 1
        one_hot[np.arange(len(idxs)), idxs] = 1.0
        
        return one_hot