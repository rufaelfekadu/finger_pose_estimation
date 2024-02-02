from hpe.trainer import EmgNet
from hpe.config import cfg



def main():
    cfg_path = 'outputs/transformer/exp0/tb_logs/tb_logs/version_0/hparams.yaml'
    checkpoint_path = 'outputs/transformer/exp0/checkpoints/epoch=189-step=115710.ckpt'
    cfg.merge_from_file("config.yaml")

    trainer = EmgNet.load_from_checkpoint(hparams_file=cfg_path, checkpoint_path=checkpoint_path, map_location='cpu')
    trainer.test()

if __name__ == '__main__':
    main()

    