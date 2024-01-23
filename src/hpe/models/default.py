from .neuropose import NeuroPose, make_neuropose_model
from .transformer import TransformerModel, make_transformer_model
from .vit import ViT, make_TS_transformer
from .EmgNet import EmgNet
from hpe.loss import make_loss

MODEL_DICT = {
    'neuropose': make_neuropose_model,
    'transformer': make_transformer_model,
    'ts_transformer': make_TS_transformer,
}
def build_backbone(cfg):
    print(f"Building {cfg.MODEL.NAME.lower()} model ")
    if cfg.MODEL.NAME.lower() not in MODEL_DICT:
        raise NotImplementedError(f"Model {cfg.MODEL.NAME.lower()} not implemented")
    model = MODEL_DICT[cfg.MODEL.NAME.lower()](cfg)
    return model

def build_model(cfg):
    backbone = build_backbone(cfg)
    loss = make_loss(cfg)
    model = EmgNet(backbone, loss)
    return model