import torch
import torch.nn as nn

class ResNet(nn.Module):
    def __init__(self, input_shape, output_shape, num_blocks):
        super(ResNet, self).__init__()
        self.input_shape = input_shape
        self.output_shape = output_shape
        self.num_blocks = num_blocks

        self.conv1 = nn.Conv2d(input_shape[2], 64, kernel_size=7, stride=2, padding=3)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.layers = self._make_layers()

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * self.num_blocks[-1], output_shape[2])

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layers(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)

        return x

    def _make_layers(self):
        layers = []
        in_channels = 64
        for i, num_blocks in enumerate(self.num_blocks):
            layers.append(self._make_block(in_channels, num_blocks))
            in_channels *= 2
        return nn.Sequential(*layers)

    def _make_block(self, in_channels, num_blocks):
        layers = []
        layers.append(BasicBlock(in_channels, in_channels * 2))
        for _ in range(1, num_blocks):
            layers.append(BasicBlock(in_channels * 2, in_channels * 2))
        return nn.Sequential(*layers)

class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_channels, out_channels, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride),
                nn.BatchNorm2d(out_channels)
            )

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        out += self.shortcut(residual)
        out = self.relu(out)

        return out

if __name__ == "__main__":
    model = ResNet((1, 1, 150, 16), (1, 1, 150, 128), [2, 2, 2, 2])
    print(model)
    print(sum(p.numel() for p in model.parameters() if p.requires_grad))