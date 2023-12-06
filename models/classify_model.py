from torch import nn


class ClassifyNet(nn.Module):
    def __init__(self, in_channels=3, num_classes=2):
        super(ClassifyNet, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=3),  # 62 x 62 x 32
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2),  # 31 x 31 x 32

            nn.Conv2d(32, 32, kernel_size=3),  # 29 x 29 x 32
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2),  # 14 x 14 x 32

            nn.Conv2d(32, 64, kernel_size=3),  # 12 x 12 x 64
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2),  # 6 x 6 x 64
        )
        self.classfier = nn.Sequential(
            nn.Linear(6 * 6 * 64, 64),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5),
            nn.Linear(64, num_classes),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classfier(x)
        return x
