import logging
logger = logging.getLogger(__name__)

import numpy as np
import torch

class TrainerBase():

    def __init__(self, model, loss_fn, optimizer):
        super().__init__()

        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.model = model.to(self.device)
        self.loss_fn = loss_fn
        self.optimizer = optimizer

        self.train_loader = None
        self.val_loader = None
        self.writer = None

        self.total_epochs = 0
        self.train_losses = []
        self.val_losses = []

        self.train_step = self._make_train_step()
        self.val_step = self._make_val_step()

    def set_seed(self, seed):
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        torch.manual_seed(seed)
        np.random.seed(seed)

    def add_dataloaders(self, train_loader, val_loader = None, test_loader = None):
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.test_loader = test_loader

    def _make_train_step(self):    
        def perform_train_step(x, y):
            self.model.train()
            x, y = x.to(self.device), y.to(self.device)
            y_pred = self.model(x)
            loss = self.loss_fn(y_pred, y)
            loss.backward()
            self.optimizer.step()
            self.optimizer.zero_grad()
            return loss.item()
        return perform_train_step
    
    def _make_val_step(self):
        def perform_val_step(x, y):
            self.model.eval()
            with torch.no_grad():
                x, y = x.to(self.device), y.to(self.device)
                y_pred = self.model(x)
                loss = self.loss_fn(y_pred, y)
            return loss.item()
        return perform_val_step

    def _make_epoch_step(self, validation=False):
        if(validation):
            if self.val_loader is None:
                raise ValueError("val_loader not loaded")
            else:
                data_loader = self.val_loader
                step = self.val_step
        else:
            if self.train_loader is None:
                raise ValueError("train_loader not loaded")
            else:
                data_loader = self.train_loader
                step = self.train_step
        mini_batch_losses = []

        for x_batch, y_batch in data_loader:
            x_batch = x_batch.to(self.device)
            y_batch = y_batch.to(self.device)
            mini_batch_loss = step(x_batch, y_batch)
            mini_batch_losses.append(mini_batch_loss)
        loss = np.mean(mini_batch_losses)
        return loss
    
    def train(self, n_epochs, seed=42):
        if self.train_loader is None:
            raise ValueError("train_loader not loaded")
        
        logger.info("Starting training...")
        for epoch in range(n_epochs):
            self.model.train()
            loss = self._make_epoch_step(validation=False)
            self.train_losses.append(loss)
            self.total_epochs += 1

    def evaluate(self,n_epochs):
        if self.val_loader is None:
            raise ValueError("val_loader not loaded")

        logger.info("Starting evaluation...")
        with torch.no_grad():
            for epoch in range(n_epochs):
                val_loss = self._make_epoch_step(validation=True)
                self.val_losses.append(val_loss)

    def predict(self, x):
        self.model.eval()
        x_tensor = torch.as_tensor(x).float()
        y_hat_tensor = self.model(x_tensor.to(self.device))
        self.model.train()
        return y_hat_tensor.detach().cpu().numpy()

    def load_checkpoint(self, path):
        checkpoint = torch.load(path)
        self.total_epochs = checkpoint['epoch']
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.val_losses = checkpoint['val_losses']
        self.losses = checkpoint['losses']
        self.model.train()

    def save_checkpoint(self, path):
        checkpoint = {
            'epoch': self.total_epochs,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'val_losses': self.val_losses,
            'losses': self.losses
        }
        torch.save(checkpoint, path)
