from torch.nn.modules.loss import _Loss
from torch import nn

metric_dict = {
    'MSE': nn.functional.mse_loss,
    'MAE': nn.functional.l1_loss,
    'SmoothL1': nn.functional.smooth_l1_loss,
}

class NeuroLoss(_Loss):
    def __init__(self, metric='MSE', keypoints=None, weights=None):
        super(NeuroLoss, self).__init__()
        self.metric = metric_dict[metric]
        self.keypoints = keypoints
        self.weights = weights  # Add weights for different variables
        
    def forward(self, input, target):
        loss = self.metric(input, target, reduction='none')
        
        # Apply different weights for different variables if weights are provided
        if self.weights is not None:
            weighted_loss = loss * self.weights.view(1, -1)  # Assuming weights is a tensor
            loss = weighted_loss.mean(dim=0)
        else:
            loss = loss.mean(dim=0)
        
        return loss

def make_loss(cfg, weights=None):
    return NeuroLoss(cfg.SOLVER.METRIC, weights=weights)
