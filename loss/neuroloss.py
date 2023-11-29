from torch.nn.modules.loss import _Loss
from torch import nn

metric_dict = {
    'MSE': nn.functional.mse_loss,
    'MAE': nn.functional.l1_loss
}
class NeuroLoss(_Loss):
    def __init__(self, metric='MSE', keypoints=None):
        super(NeuroLoss, self).__init__()
        self.metric = metric_dict[metric]
        self.keypoints = keypoints
        
    def forward(self, input, target):
        return self.metric(input, target, reduction='none').mean(dim=0)
    
    
def make_loss(cfg):
    return NeuroLoss(cfg.SOLVER.METRIC)