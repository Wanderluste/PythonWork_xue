import torch
from torch import nn

#神经网络搭建CIFAR 10 model
class Xue(nn.Module):
    def __init__(self):
        super(Xue, self).__init__()
        self.module = nn.Sequential(
            nn.Conv2d(3, 32, 5, 1,2),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 32, 5, 1,2),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 5, 1,2),
            nn.MaxPool2d(2),
            nn.Flatten(),
            nn.Linear(64 * 4 * 4, 64),
            nn.Linear(64, 10)
        )

    def forward(self, x):
        x = self.module(x)
        return x

if __name__ == '__main__':
    xue = Xue()
    input = torch.ones(64, 3, 32, 32)
    output = xue(input)
    print(output.shape)
