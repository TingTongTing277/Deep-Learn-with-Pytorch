from src.EEGTrainer import EEGTrainer
from src.EEGData import EEGData
from src.EEGModel import EEGModel

import torch
import torch.nn as nn
import torch.optim as optim

import numpy as np
import logging

if __name__ == "__main__":

    logger = logging.getLogger(__name__)
    logging.basicConfig(level=logging.INFO)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.manual_seed(42)
    np.random.seed(42)

    eeg_data = EEGData(data_dir = r"D:\Github\Deep-Learn-with-Pytorch\EEGNet\data\A01T.gdf")
    eeg_data.prepare_data()
    eeg_train_loader = eeg_data.train_dataloader()
    eeg_val_loader = eeg_data.val_dataloader()

    eeg_model = EEGModel()
    eeg_loss_fn = nn.CrossEntropyLoss()
    eeg_optimizer = optim.Adam(eeg_model.parameters(), lr=0.001)

    eeg_trainer = EEGTrainer(eeg_model,eeg_loss_fn,eeg_optimizer)
    eeg_trainer.add_dataloaders(eeg_train_loader,eeg_val_loader)
    eeg_trainer.train(10)
    eeg_trainer.save_checkpoint(r"D:\Github\Deep-Learn-with-Pytorch\EEGNet\model\eeg.pth")
