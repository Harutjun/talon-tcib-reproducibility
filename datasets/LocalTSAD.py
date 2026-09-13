import os
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
import logging

logger = logging.getLogger(__name__)

class LocalTSADDataset(Dataset):
    """
    Sliding-window dataset for local TSAD benchmarks (SMAP, MSL, SWaT, WADI).
    """
    def __init__(self, data_array, labels_array, window_size, stride,
                 x_indices, y_indices, scaler=None, mode='train', metadata=None,
                 discrete_mask=None, scaler_type='standard'):
        """
        Args:
            data_array (np.ndarray): Shape [time_steps, features]
            labels_array (np.ndarray): Shape [time_steps]
            window_size (int): Length of each sliding window
            stride (int): Step size for sliding window
            x_indices (list[int]): Indices of actuator features (X)
            y_indices (list[int]): Indices of sensor features (Y)
            scaler (dict, optional): Scaler parameters (mean/std or min/max). Fit if None.
            mode (str): 'train' or 'test'
            metadata (dict, optional): Extra metadata for this entity/dataset.
            discrete_mask (np.ndarray, optional): Boolean mask indicating discrete columns.
            scaler_type (str): 'standard' (z-score) or 'minmax' ([0, 1] scaling).
        """
        self.window_size = window_size
        self.stride = stride
        self.x_indices = x_indices
        self.y_indices = y_indices
        self.mode = mode
        self.metadata = metadata or {}
        self.scaler_type = scaler_type

        # Store discrete_mask or default to all continuous (False)
        if discrete_mask is not None:
            self.discrete_mask = np.array(discrete_mask, dtype=bool)
        else:
            self.discrete_mask = np.zeros(data_array.shape[1], dtype=bool)

        self.x_discrete_mask = self.discrete_mask[self.x_indices]
        self.y_discrete_mask = self.discrete_mask[self.y_indices]
        self.data_raw = np.array(data_array, copy=True)

        # Fit or apply normalization
        if scaler is None:
            if self.scaler_type == 'minmax':
                d_min = np.min(data_array, axis=0)
                d_max = np.max(data_array, axis=0)
                d_range = d_max - d_min
                d_range[d_range == 0] = 1.0 # avoid div by zero
                self.scaler = {'min': d_min, 'max': d_max, 'range': d_range, 'type': 'minmax'}
            else:
                # Standard scaling
                mean = np.mean(data_array, axis=0)
                std = np.std(data_array, axis=0) + 1e-8
                
                # For discrete channels, custom scaling to keep them in [0, 1]
                if discrete_mask is not None:
                    mean[self.discrete_mask] = 0.0
                    for idx in range(len(self.discrete_mask)):
                        if self.discrete_mask[idx]:
                            col_name = ""
                            if self.metadata and 'feature_names' in self.metadata:
                                col_name = self.metadata['feature_names'][idx].upper()
                            
                            if 'STATUS' in col_name or 'STATE' in col_name:
                                std[idx] = 2.0
                            else:
                                train_max = np.max(data_array[:, idx])
                                if train_max > 1.0:
                                    std[idx] = float(train_max)
                                else:
                                    std[idx] = 1.0
                
                self.scaler = {'mean': mean, 'std': std, 'type': 'standard'}
        else:
            self.scaler = scaler

        if self.scaler.get('type', 'standard') == 'minmax':
            self.data_normalized = (data_array - self.scaler['min']) / self.scaler['range']
        else:
            self.data_normalized = (data_array - self.scaler['mean']) / self.scaler['std']

        self.labels = labels_array

        self.num_samples = max(0, (len(self.data_normalized) - window_size) // stride + 1)
        
    def __len__(self):
        return self.num_samples
    
    def __getitem__(self, idx):
        start_idx = idx * self.stride
        end_idx = start_idx + self.window_size
        
        window_data = self.data_normalized[start_idx:end_idx] # [window_size, features]
        window_labels = self.labels[start_idx:end_idx] # [window_size]
        
        # Split into X and Y
        # Expected outputs usually have shape [channels, sequence_length], so transpose
        x_cond = torch.tensor(window_data[:, self.x_indices].T, dtype=torch.float32) # [x_channels, window_size]
        y_targ = torch.tensor(window_data[:, self.y_indices].T, dtype=torch.float32) # [y_channels, window_size]
        labels = torch.tensor(window_labels, dtype=torch.float32) # [window_size]
        
        meta = {
            'start_idx': start_idx,
            'end_idx': end_idx,
            **self.metadata
        }
        
        # To match the expected format in pipeline unpacking:
        # X, _, anomaly_idxs, datalength = batch[:4]
        # Y = batch[6]
        # In the trainer we can just use named tuple or dictionary, or mimic BallisticDataset
        # BallisticDataset returns: (measurement, state, anomaly_idxs, traj_len, time, params, estimated_states, source_filename)
        # Here we will return a generic tuple, but we should make sure the benchmark runner unpacks it correctly.
        # Let's return (x_cond, y_targ, labels, window_size, meta)
        
        # Wait, the training pipelines unpack like this:
        # X = batch[0]
        # Y = batch[6] (if len(batch)>=7) else batch[1]
        # So we can return:
        # (X, Y, anomaly_idxs, length)
        # X is batch[0], Y is batch[1], anomaly is batch[2], length is batch[3]
        return x_cond, y_targ, labels, torch.tensor(self.window_size, dtype=torch.long), meta


def load_npy_dataset(data_root, entity, x_indices, y_indices, window_size, stride):
    """
    Load SMAP/MSL entity from .npy files.
    """
    train_path = os.path.join(data_root, f"{entity}_train.npy")
    test_path = os.path.join(data_root, f"{entity}_test.npy")
    labels_path = os.path.join(data_root, f"{entity}_labels.npy")
    
    if not all(os.path.exists(p) for p in [train_path, test_path, labels_path]):
        logger.warning(f"Data files missing for entity {entity} in {data_root}")
        return None, None
        
    train_data = np.load(train_path)
    test_data = np.load(test_path)
    test_labels = np.load(labels_path)
    
    # Train labels are typically all 0 for MSL/SMAP
    train_labels = np.zeros(train_data.shape[0])
    
    # Construct discrete mask
    discrete_mask = np.zeros(train_data.shape[1], dtype=bool)
    for idx in range(train_data.shape[1]):
        unique_vals = np.unique(train_data[:, idx])
        unique_vals = unique_vals[~np.isnan(unique_vals)]
        all_int = all(float(v).is_integer() for v in unique_vals)
        if all_int and len(unique_vals) <= 5:
            discrete_mask[idx] = True

    metadata = {'entity': entity, 'dataset': os.path.basename(data_root)}
    
    train_dataset = LocalTSADDataset(
        train_data, train_labels, window_size, stride, x_indices, y_indices,
        mode='train', metadata=metadata, discrete_mask=discrete_mask
    )
    
    test_dataset = LocalTSADDataset(
        test_data, test_labels, window_size, stride, x_indices, y_indices,
        scaler=train_dataset.scaler, mode='test', metadata=metadata,
        discrete_mask=discrete_mask
    )
    
    return train_dataset, test_dataset


def classify_swat_wadi_columns(column_names, x_prefixes, y_prefixes):
    """Classify CSV columns into actuator (X) vs sensor (Y) by tag prefix."""
    import re
    x_cols = []
    y_cols = []
    ambiguous = []
    
    for idx, col in enumerate(column_names):
        col_upper = col.upper().strip()
        # Strip stage prefixes like "1_", "2_", "2A_", "2B_", "3_"
        clean_col = re.sub(r'^\d+[A-Z]?_', '', col_upper)
        
        # Actuators
        is_x = False
        for prefix in x_prefixes:
            # Avoid matching just 'P' to everything, usually P followed by digits
            if prefix == 'P':
                if re.match(r'^P_?\d+', clean_col):
                    is_x = True
                    break
            elif clean_col.startswith(prefix):
                is_x = True
                break
                
        if is_x:
            x_cols.append(idx)
            continue
            
        # Sensors
        is_y = False
        for prefix in y_prefixes:
            if clean_col.startswith(prefix):
                is_y = True
                break
                
        if is_y:
            y_cols.append(idx)
        else:
            ambiguous.append(idx)
            
    return x_cols, y_cols, ambiguous



def load_csv_dataset(data_root, normal_csv, attack_csv, label_column, timestamp_columns,
                     x_prefixes, y_prefixes, window_size, stride, x_override=None, y_override=None,
                     max_rows=None, downsample_rate=1, scaler_type="standard", downsample_mode="median",
                     drop_columns=None):
    """
    Load SWaT/WADI dataset from CSV files.
    
    Args:
        drop_columns (list[str], optional): Column names to explicitly drop (e.g., TranAD's 4
            zero-information WADI columns). Applied after column alignment, before downsampling.
    
    downsample_mode options:
      - 'median': Block median aggregation over downsample_rate rows
      - 'mean': Block mean aggregation over downsample_rate rows
      - 'decimate' / 'slice' / 'none': Takes every Nth row (or no downsampling if downsample_rate=1)
    """
    normal_path = os.path.join(data_root, normal_csv)
    attack_path = os.path.join(data_root, attack_csv)
    dropped_cols = []
    
    if not os.path.exists(normal_path) or not os.path.exists(attack_path):
        logger.warning(f"CSV files missing in {data_root}")
        return None, None
        
    df_train = pd.read_csv(normal_path, low_memory=False, nrows=max_rows)
    df_test = pd.read_csv(attack_path, low_memory=False, nrows=max_rows)

    # Auto-detect if test CSV has a numeric pseudo-header row (e.g., WADI attack file)
    # If all column names are numeric strings, re-read with header on row 1
    if all(str(c).strip().isdigit() for c in df_test.columns):
        logger.info("Detected numeric pseudo-header in attack CSV, re-reading with header=1")
        df_test = pd.read_csv(attack_path, low_memory=False, nrows=max_rows, header=1)

    # Strip whitespace from column names
    df_train.columns = df_train.columns.str.strip()
    df_test.columns = df_test.columns.str.strip()

    # Drop timestamp columns
    if timestamp_columns:
        df_train = df_train.drop(df_train.columns[timestamp_columns], axis=1)
        df_test = df_test.drop(df_test.columns[timestamp_columns], axis=1)
        
    # Get labels
    def parse_labels(labels_series):
        s_str = labels_series.astype(str).str.lower().str.strip()
        is_minus_one = (s_str == '-1')
        if is_minus_one.any():
            return is_minus_one.astype(int).values
        is_attack_str = s_str.str.contains('ttack')
        if is_attack_str.any():
            return is_attack_str.astype(int).values
        return (s_str == '1').astype(int).values

    # Find label column in test
    label_col_name_test = df_test.columns[label_column]
    test_labels = parse_labels(df_test[label_col_name_test])
    df_test = df_test.drop(columns=[label_col_name_test])

    # Check if the exact label column name exists in train
    # If not (e.g., WADI train set lacks the label column), assume all normal (0)
    # and DO NOT drop the last feature column!
    if label_col_name_test in df_train.columns:
        train_labels = parse_labels(df_train[label_col_name_test])
        df_train = df_train.drop(columns=[label_col_name_test])
    else:
        train_labels = np.zeros(len(df_train), dtype=int)
        logger.info(f"Train set lacks label column '{label_col_name_test}', assuming all normal.")

    # Align columns: keep only columns present in both train and test
    common_cols = [c for c in df_train.columns if c in df_test.columns]
    if len(common_cols) < len(df_train.columns):
        train_only = [c for c in df_train.columns if c not in df_test.columns]
        test_only = [c for c in df_test.columns if c not in df_train.columns]
        logger.info(f"Column alignment: {len(common_cols)} common, "
                     f"train-only={train_only}, test-only={test_only}")
    df_train = df_train[common_cols]
    df_test = df_test[common_cols]

    # Drop explicitly specified columns (e.g., TranAD's 4 zero-information WADI columns)
    if drop_columns:
        actual_drops = [c for c in drop_columns if c in df_train.columns]
        if actual_drops:
            dropped_cols.extend(actual_drops)
            logger.info(f"Explicitly dropped columns: {actual_drops}")
            df_train = df_train.drop(columns=actual_drops)
            df_test = df_test.drop(columns=actual_drops)
    
    feature_names = df_train.columns.tolist()
    
    # Coerce all feature columns to numeric before downsampling/aggregation
    df_train = df_train.apply(pd.to_numeric, errors='coerce')
    df_test = df_test.apply(pd.to_numeric, errors='coerce')

    mode_clean = str(downsample_mode).lower().strip()
    if downsample_rate > 1 and mode_clean != 'none':
        logger.info(f"Resampling datasets with step {downsample_rate} (mode='{mode_clean}')")
        
        if mode_clean == 'median':
            train_group = np.arange(len(df_train)) // downsample_rate
            test_group = np.arange(len(df_test)) // downsample_rate
            df_train = df_train.groupby(train_group).median().reset_index(drop=True)
            df_test = df_test.groupby(test_group).median().reset_index(drop=True)
        elif mode_clean == 'mean':
            train_group = np.arange(len(df_train)) // downsample_rate
            test_group = np.arange(len(df_test)) // downsample_rate
            df_train = df_train.groupby(train_group).mean().reset_index(drop=True)
            df_test = df_test.groupby(test_group).mean().reset_index(drop=True)
        else: # 'decimate', 'slice', or fallback
            df_train = df_train.iloc[::downsample_rate].reset_index(drop=True)
            df_test = df_test.iloc[::downsample_rate].reset_index(drop=True)
            
        # For labels, take the max over each chunk to preserve short attacks
        train_labels = np.maximum.reduceat(train_labels, np.arange(0, len(train_labels), downsample_rate))
        test_labels = np.maximum.reduceat(test_labels, np.arange(0, len(test_labels), downsample_rate))
    
    # Drop columns that are entirely non-numeric (all NaN after coercion)
    valid_cols = df_train.columns[df_train.notna().any(axis=0)].tolist()
    if len(valid_cols) < len(feature_names):
        dropped = [c for c in feature_names if c not in valid_cols]
        dropped_cols.extend(dropped)
        logger.info(f"Dropped entirely non-numeric columns: {dropped}")
        df_train = df_train[valid_cols]
        df_test = df_test[valid_cols]
        feature_names = valid_cols
    
    # Handle NaNs (fill with forward fill then 0)
    df_train = df_train.ffill().fillna(0)
    df_test = df_test.ffill().fillna(0)

    # NOTE: Zero-variance columns are NOT dropped. Standard benchmarks (TranAD, TSB-AD)
    # keep all channels. The scaler handles constant columns safely (div-by-zero guard).
    # Log them for reference only.
    train_stds = df_train.std(axis=0)
    zero_var_cols = train_stds[train_stds == 0].index.tolist()
    if zero_var_cols:
        logger.info(f"Zero-variance columns in training data (kept, not dropped): {zero_var_cols}")

    train_data = df_train.values.astype(np.float64)
    test_data = df_test.values.astype(np.float64)
    
    # Construct discrete mask:
    #   - Binary channels (<=2 unique integer values): always discrete, scaled to [0,1]
    #   - Channels with >2 unique integer values: discrete ONLY if they are discrete by
    #     definition (ISA standard: _STATUS, _AL, _AH suffixes). Otherwise treat as continuous.
    discrete_mask = np.zeros(len(feature_names), dtype=bool)
    for idx, col in enumerate(feature_names):
        unique_vals = df_train[col].unique()
        unique_vals = unique_vals[~pd.isna(unique_vals)]
        all_int = all(float(v).is_integer() for v in unique_vals)
        if all_int:
            col_upper = col.upper()
            is_discrete_by_definition = (
                '_STATUS' in col_upper or
                col_upper.endswith('_AL') or col_upper.endswith('_AH') or
                col_upper == 'PLANT_START_STOP_LOG'
            )
            if len(unique_vals) <= 2:
                # Binary channels are always discrete
                discrete_mask[idx] = True
            elif is_discrete_by_definition:
                # >2 values but discrete by ISA definition (e.g., valve STATUS with 0,1,2)
                discrete_mask[idx] = True
            # else: >2 integer values in a non-status channel → treat as continuous
    
    if x_override and y_override:
        # Ensure they are indices; filter to only columns present after drops
        if isinstance(x_override[0], str):
            x_present = [c for c in x_override if c in feature_names]
            y_present = [c for c in y_override if c in feature_names]
            x_missing = [c for c in x_override if c not in feature_names]
            y_missing = [c for c in y_override if c not in feature_names]
            if x_missing:
                logger.info(f"x_override columns not found (dropped/missing): {x_missing}")
            if y_missing:
                logger.info(f"y_override columns not found (dropped/missing): {y_missing}")
            x_indices = [feature_names.index(c) for c in x_present]
            y_indices = [feature_names.index(c) for c in y_present]
        else:
            x_indices = x_override
            y_indices = y_override
    else:
        x_indices, y_indices, ambiguous = classify_swat_wadi_columns(feature_names, x_prefixes, y_prefixes)
        if ambiguous:
            logger.info(f"Ignored ambiguous columns: {[feature_names[i] for i in ambiguous]}")
            
    dataset_name = os.path.basename(data_root)
    metadata = {
        'entity': dataset_name,
        'dataset': dataset_name,
        'feature_names': feature_names,
        'dropped_columns': dropped_cols
    }
    
    train_dataset = LocalTSADDataset(
        train_data, train_labels, window_size, stride, x_indices, y_indices,
        mode='train', metadata=metadata, discrete_mask=discrete_mask, scaler_type=scaler_type
    )
    
    test_dataset = LocalTSADDataset(
        test_data, test_labels, window_size, stride, x_indices, y_indices,
        scaler=train_dataset.scaler, mode='test', metadata=metadata,
        discrete_mask=discrete_mask, scaler_type=scaler_type
    )
    
    return train_dataset, test_dataset
