from torch.nn.modules.loss import _Loss

class NeuroLoss(_Loss):
    def __init__(self):
        super(NeuroLoss, self).__init__()

        pass

    def forward(self, input, target):
        pass