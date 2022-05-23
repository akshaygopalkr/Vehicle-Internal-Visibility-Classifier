import torch
import torch.nn as nn
import torch.nn.functional as F
import pdb


class ResBlock(nn.Module):
    def __init__(self, in_channels, out_channels, downsample):
        super().__init__()
        # resblock with a stride of 2

        if downsample:

            self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=2, padding=1)
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=2),
                nn.BatchNorm2d(out_channels)
            )
        else:
            self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1)

            # Since we are just taking the input in the Sequential layer can be empty
            self.shortcut = nn.Sequential()

        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.bn2 = nn.BatchNorm2d(out_channels)

    def forward(self, input):

        # pdb.set_trace()

        # Apply shortcut to input to use for
        shortcut = self.shortcut(input)

        # convolution for 1st and 2nd layer
        input = self.conv1(input)
        input = self.bn1(input)
        input = nn.ReLU()(input)
        input = self.conv2(input)
        input = self.bn2(input)
        input = nn.ReLU()(input)

        # perform shortcut for last layer
        return nn.ReLU()(input + shortcut)


class ResNet18(nn.Module):
    def __init__(self, in_channels, resblock, outputs):
        super().__init__()
        self.layer0 = nn.Sequential(
            nn.Conv2d(in_channels, 64, kernel_size=7, stride=2, padding=3),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),

        )

        self.layer1 = nn.Sequential(
            resblock(64, 64, downsample=False),
            resblock(64, 64, downsample=False)
        )

        self.layer2 = nn.Sequential(
            resblock(64, 128, downsample=True),
            resblock(128, 128, downsample=False)
        )

        self.layer3 = nn.Sequential(
            resblock(128, 256, downsample=True),
            resblock(256, 256, downsample=False)
        )

        self.layer4 = nn.Sequential(
            resblock(256, 512, downsample=True),
            resblock(512, 512, downsample=False)
        )

        self.gap = torch.nn.AdaptiveAvgPool2d(1)
        self.fc1 = torch.nn.Linear(512, 100)
        self.fc2 = torch.nn.Linear(100, 1)
        self.sigmoid = torch.nn.Sigmoid()

    def forward(self, input):

        # pdb.set_trace()

        input = self.layer0(input)
        input = self.layer1(input)
        input = self.layer2(input)
        input = self.layer3(input)
        input = self.layer4(input)
        input = self.gap(input)
        input = torch.reshape(input, (input.size()[0], input.size()[2], input.size()[1]))
        input = self.fc1(input)
        input = self.fc2(input)
        input = self.sigmoid(input)

        return input


if __name__ == '__main__':
    net = ResNet18(3, ResBlock, outputs=10)
    print(net)