from abc import ABC, abstractmethod
import numpy as np

from torch.utils.data import DataLoader, Dataset
from pathlib import Path

class DataBaseClass():
    def __init__(self, data_dir, batch_size = 32, num_workers = 4, pin_memory = True):
        
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
        
        pass

    def prepare_data(self):
        self.on_preparation_start()

        try:
            self._acquire_data()
            self._process_data()
            self._validate_data()
            self.on_preparation_complete(success=True)
        
        except Exception as e:
            self.on_preparation_complete(success=False,e=e)
            raise

    @abstractmethod
    def _acquire_data(self):
        pass

    @abstractmethod
    def _process_data(self):
        pass

    @abstractmethod
    def _validate_data(self):
        pass

    def train_dataloader(self):
        
        if self.train_dataset is None:
            raise ValueError("train_dataset is None")
        
        return DataLoader(
            dataset = self.train_dataset,
            batch_size = self.batch_size,
            shuffle = True,
            num_workers = self.num_workers,
            pin_memory = self.pin_memory 
        )

    def val_dataloader(self):

        if self.val_dataset is None:
            raise ValueError("val_dataset is None")
        
        return DataLoader(
            dataset = self.val_dataset,
            batch_size = self.batch_size,
            shuffle = False,
            num_workers = self.num_workers,
            pin_memory  = self.pin_memory,
            drop_last = False
        )

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
    
    def on_preparation_start(self):
        pass

    def on_preparation_complete(self,success: bool, e = None):
        pass
