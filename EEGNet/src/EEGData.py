import numpy as np
import logging
import mne

import torch
from torch.utils.data import TensorDataset, DataLoader, random_split

from pathlib import Path

logger = logging.getLogger(__name__)

class EEGData():
    def __init__(self, data_dir, batch_size = 32, num_workers = 0, pin_memory = True):
        
        self.data_dir = Path(data_dir)
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.pin_memory = pin_memory

        self.raw_data = None
        self.processed_data = None

        self.train_loader = None
        self.val_loader = None
        self.test_loader = None

        self.train_dataset = None
        self.val_dataset = None
        self.test_dataset = None

    def prepare_data(self):
        logger.info("is preparing data...")
        try:
            self._acquire_data()
            self._process_data()
            self._validate_data()
            logger.info("has prepared.")
        except Exception as e:
            raise RuntimeError(f"Error in preparing data: {e}")

    def _acquire_data(self):
        logger.info("[preparing] is acquiring data...")
        self.raw_data = self._acquire_data_by_gdf()

    def _process_data(self):
        if self.raw_data is None:
            raise ValueError("Raw data is None. Cannot process data.")
        logger.info("[preparing] is processing data...")
        X_raw, y_raw = self.raw_data
        mean = np.mean(X_raw, axis=2, keepdims=True)
        std = np.std(X_raw, axis=2, keepdims=True)
        X = (X_raw - mean) / (std + 1e-6)
        X = X[:,np.newaxis,:,:]

        X_tensor = torch.FloatTensor(X)
        y_tensor = torch.LongTensor(y_raw)
        self.processed_data = (X_tensor, y_tensor)

        dataset = TensorDataset(X_tensor, y_tensor)
        ratio = .8
        n_total = len(dataset)
        n_train = int(n_total * ratio)
        n_val = n_total - n_train

        self.train_dataset,self.val_dataset = random_split(dataset,[n_train,n_val])
        self.train_loader = DataLoader(dataset=self.train_dataset,batch_size=self.batch_size,shuffle=True,num_workers = self.num_workers,pin_memory = self.pin_memory)
        self.val_loader = DataLoader(dataset=self.val_dataset,batch_size=self.batch_size,shuffle=False,num_workers = self.num_workers,pin_memory=self.pin_memory,drop_last=False)

    def _validate_data(self):
        pass

    def train_dataloader(self):
        
        if self.train_dataset is None:
            raise ValueError("train_dataset is None")
        
        return self.train_loader

    def val_dataloader(self):

        if self.val_dataset is None:
            raise ValueError("val_dataset is None")
        
        return self.val_loader

    def test_dataloader(self):
        if self.test_dataset is None:
            raise ValueError("test_dataset is None")
        
        return DataLoader(
            dataset = self.test_dataset,
            batch_size = self.batch_size,
            shuffle = False,
            num_workers = self.num_workers,
            pin_memory = self.pin_memory,
            drop_last = False
        )
    
    def _acquire_data_by_gdf(self):
        raw = mne.io.read_raw_gdf(self.data_dir, eog=['EOG-left', 'EOG-central', 'EOG-right'],
                        preload=True, verbose=False)
        raw.pick_types(eeg=True, eog=False, stim=False, exclude='bads')
        raw.filter(4., 38., fir_design='firwin', skip_by_annotation='edge', verbose=False)
        raw.resample(128)
        events, _ = mne.events_from_annotations(raw, verbose=False)
        desired_events = {'769': 0, '770': 1, '771': 2, '772': 3}
        annot_map = {k: v for k, v in _.items() if k in desired_events}
        tmin, tmax = 0, 4.0
        epochs = mne.Epochs(raw, events, event_id=annot_map, tmin=tmin, tmax=tmax,
                        proj=False,
                        baseline=None,
                        preload=True,
                        verbose=False,
                        reject_by_annotation=False
                        )
        X = epochs.get_data() * 1000 
        y = epochs.events[:, -1]
        reverse_map = {v: k for k, v in annot_map.items()}
        y_mapped = np.array([desired_events[reverse_map[original_id]] for original_id in y])
        return (X, y_mapped)