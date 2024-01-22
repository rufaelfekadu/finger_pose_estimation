import torch
import torch.nn as nn
import torchvision.models as models


class BackHandPose(nn.Module):
    def __init__(self, input_shape1, input_shape2, output_shape, batch_size):
        super(BackHandPose, self).__init__()

        resnet18 = models.resnet18(pretrained=False)
        resnet18_gray = models.resnet18(pretrained=False)
        resnet18_gray.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)

        self.resnet18 = nn.Sequential(resnet18, nn.Flatten())
        self.resnet18_gray = nn.Sequential(resnet18_gray, nn.Flatten())

        self.input_shape1 = input_shape1
        self.input_shape2 = input_shape2
        self.output_shape = output_shape
        self.batch_size = batch_size

        self.lstm = nn.LSTM(256, 128, batch_first=True)
        self.fc1 = nn.Linear(128, 64)
        self.fc2 = nn.Linear(64, output_shape)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x1, x2):
        x1 = x1.view(self.batch_size, -1, *self.input_shape1)
        x2 = x2.view(self.batch_size, -1, *self.input_shape2)

        x1 = self.resnet18(x1)
        x2 = self.resnet18_gray(x2)

        x = torch.cat((x1, x2), dim=2)
        x = x.view(self.batch_size, -1, 256)
        lstm_out, _ = self.lstm(x)
        x = lstm_out[:, -1, :]  # Taking the last time step output

        x = self.fc1(x)
        x = self.fc2(x)
        x = self.sigmoid(x)

        return x

if __name__ == '__main__':

    # Example usage:
    input_shape1 = (3, 224, 224)
    input_shape2 = (1, 224, 224)
    output_shape = 10
    batch_size = 32

    model = BackHandPose(input_shape1, input_shape2, output_shape, batch_size)
    print(model)

    x1 = torch.randn(batch_size, *input_shape1)
    x2 = torch.randn(batch_size, *input_shape2)

    print(model(x1, x2).shape)
