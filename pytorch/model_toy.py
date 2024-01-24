from torch import nn


class ToyNet(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.conv = nn.Conv2d(in_channels=1, out_channels=1, kernel_size=3, padding=1)
        self.fc = nn.Linear(in_features=28 * 28, out_features=2)
        self.flatten = nn.Flatten()
    
    def forward(self, x):
        x = self.conv(x)
        x = self.flatten(x)
        return self.fc(x)
