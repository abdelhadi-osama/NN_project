import os
import yaml
import gzip
import struct
import urllib.request
import numpy as np
from pathlib import Path
from tqdm import tqdm

class DataLoader:
    def __init__(self, config_path=None, data_dir='./data'):
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)

        # --- DYNAMIC PATH RESOLUTION ---
        if config_path is None:
            current_file_path = Path(__file__).resolve()
            project_root = current_file_path.parent.parent
            self.config_path = project_root / 'config' / 'config.yml'
        else:
            self.config_path = Path(config_path)

        if not self.config_path.exists():
             if Path('config/config.yml').exists():
                 self.config_path = Path('config/config.yml')
             else:
                 raise FileNotFoundError(f"Config file not found at {self.config_path}")

        print(f"[\u2713] Loading config from {self.config_path}...")
        with open(self.config_path, 'r') as f:
            self.config = yaml.safe_load(f)

    def load_dataset(self, dataset_name):
        """
        Returns: (x_train, y_train), (x_val, y_val), (x_test, y_test)
        """
        name = dataset_name.lower().strip()
        all_datasets = self.config.get('data_pipeline', {}).get('datasets', {})

        if name not in all_datasets:
             raise ValueError(f"Dataset '{name}' not found.")
             
        dataset_cfg = all_datasets[name] 
        data_type = dataset_cfg.get('type', 'unknown')
        
        print(f"[\u2713] Dispatching '{name}' -> Type: {data_type}")
        
        if data_type == 'tabular':
            return self._load_tabular_pipeline(name, dataset_cfg)
        elif data_type in ['image', 'image_flattened']:
            return self._load_image_pipeline(name, dataset_cfg)
        else:
            raise ValueError(f"Unsupported data type: {data_type}")

    # ------------------------------------------------------------------
    #  WORKER 1: Infrastructure
    # ------------------------------------------------------------------
    def _get_dataset_dir(self, dataset_name):
        path = self.data_dir / dataset_name
        path.mkdir(parents=True, exist_ok=True)
        return path

    def _download_file(self, url, dest_dir, filename):
        file_path = dest_dir / filename
        if file_path.exists():
            print(f"    -> [Cache] File found: {filename}")
            return file_path

        print(f"    -> [Download] Fetching {filename}...")
        try:
            with urllib.request.urlopen(url) as response:
                total_size = int(response.info().get('Content-Length', 0))
                with tqdm(total=total_size, unit='B', unit_scale=True, desc=filename) as pbar:
                    with open(file_path, 'wb') as f:
                        while True:
                            chunk = response.read(1024)
                            if not chunk: break
                            f.write(chunk)
                            pbar.update(len(chunk))
            return file_path
        except Exception as e:
            if file_path.exists(): os.remove(file_path)
            raise RuntimeError(f"Download failed: {e}")

    # ------------------------------------------------------------------
    #  WORKER 2: Specific Pipelines
    # ------------------------------------------------------------------
    def _load_tabular_pipeline(self, name, config):
        base_dir = self._get_dataset_dir(name)
        downloaded_paths = {}
        
        for file_key, filename in config['files'].items():
            url = config['urls'][file_key]
            downloaded_paths[file_key] = self._download_file(url, base_dir, filename)

        source_type = config.get('source_type', 'unknown')
        if source_type == 'csv':
            target_file = downloaded_paths.get('data')
            
            # FORCE STRING DTYPE to avoid "Too many indices" error
            print(f"    -> [Loading] Parsing CSV: {target_file.name}")
            raw_data = np.genfromtxt(target_file, delimiter=',', dtype=str, encoding='utf-8')
            
            # Verify
            verified_data = self._assemble_and_verify({'data': raw_data}, 'tabular', config)
            
            # Extract
            print(f"    -> [Processing] Extracting features...")
            label_idx = config.get('label_column', -1)
            drop_cols = config.get('drop_columns', [])
            
            y = verified_data[:, label_idx]
            cols_to_remove = [label_idx] + drop_cols
            x = np.delete(verified_data, cols_to_remove, axis=1)

            # Split (Train / Val / Test)
            return self._apply_split_strategy(x, y, config)
        else:
            raise ValueError(f"Unsupported source_type: {source_type}")

    def _load_image_pipeline(self, name, config):
        base_dir = self._get_dataset_dir(name)
        downloaded_paths = {}
        
        for file_key, filename in config['files'].items():
            url = config['urls'][file_key]
            downloaded_paths[file_key] = self._download_file(url, base_dir, filename)

        source_type = config.get('source_type', 'unknown')
        if source_type == 'gzip':
            print(f"    -> [Processing] Parsing Gzip+IDX files...")
            data_dict = {}
            for key, path in downloaded_paths.items():
                data_dict[key] = self._parse_idx_file(path)
            
            # Assemble & Split Inside Here
            return self._assemble_and_verify(data_dict, 'image', config)
        else:
            raise ValueError(f"Unsupported source_type: {source_type}")

    # ------------------------------------------------------------------
    #  WORKER 3: Low-Level Parsers & Verification
    # ------------------------------------------------------------------
    def _parse_idx_file(self, file_path):
        with gzip.open(file_path, 'rb') as f:
            magic_number = struct.unpack('>I', f.read(4))[0]
            if magic_number == 2049: # Labels
                num_items = struct.unpack('>I', f.read(4))[0]
                return np.frombuffer(f.read(), dtype=np.uint8)
            elif magic_number == 2051: # Images
                num_images, rows, cols = struct.unpack('>III', f.read(12))
                data = np.frombuffer(f.read(), dtype=np.uint8)
                return data.reshape(num_images, rows, cols)
            else:
                raise ValueError(f"Invalid Magic Number: {magic_number}")

    def _assemble_and_verify(self, data_dict, data_type, config):
        processed_data = {}
        for key, value in data_dict.items():
            if isinstance(value, list) and len(value) > 0:
                processed_data[key] = np.concatenate(value, axis=0)
            else:
                processed_data[key] = value

        if data_type == 'image':
            x_train_full = processed_data.get('train_images')
            y_train_full = processed_data.get('train_labels')
            x_test  = processed_data.get('test_images')
            y_test  = processed_data.get('test_labels')
            
            # Integrity Check
            if x_train_full.shape[0] != y_train_full.shape[0]:
                raise ValueError("Train shape mismatch")
            if x_test.shape[0] != y_test.shape[0]:
                raise ValueError("Test shape mismatch")
            
            # --- SPLIT LOGIC FOR IMAGES ---
            # We keep x_test as the official Test Set.
            # We split x_train_full into Train and Val.
            val_ratio = config.get('val_split', 0.0)
            
            if val_ratio > 0.0:
                print(f"       [Config] Strategy: Splitting {val_ratio} of Train for Validation")
                (x_train, y_train), (x_val, y_val) = self.train_test_split(
                    x_train_full, y_train_full, test_size=val_ratio
                )
                return (x_train, y_train), (x_val, y_val), (x_test, y_test)
            else:
                return (x_train_full, y_train_full), (None, None), (x_test, y_test)

        elif data_type == 'tabular':
            return processed_data.get('data')

    # ------------------------------------------------------------------
    #  WORKER 4: Splitting Logic
    # ------------------------------------------------------------------
    def _apply_split_strategy(self, x, y, config):
        """
        Splits data into (Train, Val, Test).
        """
        test_ratio = config.get('test_split', 0.2)
        val_ratio = config.get('val_split', 0.0)

        # 1. Cut Test Set
        (x_train_full, y_train_full), (x_test, y_test) = self.train_test_split(
            x, y, test_size=test_ratio
        )
        
        # 2. Cut Validation Set (from the remaining Train Full)
        if val_ratio > 0.0:
            # Note: val_split in config usually means "% of TOTAL". 
            # If we want accurate splits, we should adjust the math, 
            # but for simplicity, we'll take % of the TRAINING portion.
            print(f"       [Config] Strategy: Splitting {val_ratio} of Train for Validation")
            (x_train, y_train), (x_val, y_val) = self.train_test_split(
                x_train_full, y_train_full, test_size=val_ratio
            )
            return (x_train, y_train), (x_val, y_val), (x_test, y_test)
        
        return (x_train_full, y_train_full), (None, None), (x_test, y_test)

    def train_test_split(self, x, y, test_size=0.2, shuffle=True, seed=42):
        if x.shape[0] != y.shape[0]: raise ValueError("Shape mismatch")
        n_samples = x.shape[0]
        indices = np.arange(n_samples)
        
        if shuffle:
            np.random.seed(seed)
            np.random.shuffle(indices)
            
        split_idx = int(n_samples * (1 - test_size))
        train_idx, test_idx = indices[:split_idx], indices[split_idx:]
        
        return (x[train_idx], y[train_idx]), (x[test_idx], y[test_idx])