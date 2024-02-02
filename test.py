from hpe.trainer import EmgNet
from hpe.config import cfg
import torch



def main():
    cfg_path = 'outputs/transformer/exp0/tb_logs/tb_logs/version_0/hparams.yaml'
    checkpoint_path = 'outputs/transformer/exp0/checkpoints/epoch=189-step=115710.ckpt'
    cfg.merge_from_file("config.yaml")

    trainer = EmgNet.load_from_checkpoint(checkpoint_path=checkpoint_path, map_location='cpu')
    trainer.backbone.eval()
    with torch.no_grad():
        out = trainer.forward(torch.randn(1, 200, 16))
        print(out.shape)
if __name__ == '__main__':
    main()

    