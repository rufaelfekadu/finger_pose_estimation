import torch.nn as nn
import torch.nn.functional as F

class EmgNet(nn.Module):

    def __init__(self, backbone, loss,):
        super(EmgNet, self).__init__()
        self.backbone = backbone
        self.loss = loss

    def forward(self, x, target=None):
        x = self.backbone(x)
        if target is not None:
            loss = self.loss(x, target)
            return x, loss
        return x, None

