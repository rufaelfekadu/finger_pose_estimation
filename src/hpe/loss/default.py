from .neuroloss import NeuroLoss

def make_loss(cfg, weights=None):
    return NeuroLoss(cfg.SOLVER.METRIC, weights=weights, keypoints=cfg.DATA.LABEL_COLUMNS)