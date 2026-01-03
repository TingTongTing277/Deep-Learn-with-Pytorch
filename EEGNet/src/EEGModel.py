import torch.nn as nn

class EEGModel(nn.Module):
    def __init__(self, nb_classes=4, Chans=22, Samples=512, dropoutRate=0.5, 
                 kernLength=64, F1=8, D=2, F2=16, norm_rate=0.25):
        super(EEGModel, self).__init__()

        self.conv1 = nn.Sequential(
            nn.Conv2d(1, F1, (1, kernLength), padding=(0, kernLength // 2), bias=False),
            nn.BatchNorm2d(F1, momentum=0.1, affine=True, track_running_stats=True)
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(F1, F1 * D, (Chans, 1), groups=F1, bias=False), 
            nn.BatchNorm2d(F1 * D, momentum=0.1, affine=True, track_running_stats=True),
            nn.ELU(),
            nn.AvgPool2d((1, 4)),
            nn.Dropout(dropoutRate)
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(F1 * D, F1 * D, (1, 16), padding=(0, 8), groups=F1 * D, bias=False),
            nn.Conv2d(F1 * D, F2, (1, 1), bias=False),
            nn.BatchNorm2d(F2, momentum=0.1, affine=True, track_running_stats=True),
            nn.ELU(),
            nn.AvgPool2d((1, 8)),
            nn.Dropout(dropoutRate)
        )
        self.classifier = nn.Linear(F2 * (Samples // 32), nb_classes)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x