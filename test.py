from hpe.trainer import EmgNet
from hpe.config import cfg



def main():
    trainer = EmgNet(cfg)
    trainer.load_from_checkpoint(hparams_file=cfg.SOLVER.CHECKPOINT_PATH, checkpoint_path=cfg.SOLVER.CHECKPOINT_PATH, map_location='cpu')

    trainer.test()

    