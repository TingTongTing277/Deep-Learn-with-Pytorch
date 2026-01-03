import torch.nn as nn

class ModelBaseClass(nn.Module):
    def __init__(self):
        super(ModelBaseClass, self).__init__()
        pass

    def forward(self, x):
        raise NotImplementedError("Forward method not implemented.")
    