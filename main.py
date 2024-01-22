
import torch
from pytorch_lightning import loggers as pl_loggers
import pytorch_lightning as pl
from hpe.trainer import EmgNet
from hpe.data import build_dataloader
from hpe.models import build_model
from hpe.config import cfg, to_dict
import argparse
import os

def setup_seed(seed):
    import random
    import numpy as np
    import torch
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # if you are using multi-GPU.
    np.random.seed(seed)  # Numpy module.
    random.seed(seed)  # Python random module.
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.deterministic = True


def main(cfg):

    # Build logger
    csv_logger = pl_loggers.CSVLogger(cfg.SOLVER.LOG_DIR, name=cfg.NAME)
    early_stop_callback = pl.callbacks.EarlyStopping(monitor='val_loss', patience=cfg.SOLVER.PATIENCE)
    checkpoint_callback = pl.callbacks.ModelCheckpoint(dirpath=cfg.SOLVER.LOG_DIR, monitor='val_loss', save_top_k=3, mode='min')
    
    # Build trainer
    model = EmgNet(
        cfg=cfg,
    )

    trainer = pl.Trainer(

        max_epochs=cfg.SOLVER.NUM_EPOCHS,
        check_val_every_n_epoch=1,
        callbacks=[early_stop_callback, checkpoint_callback],
        logger=csv_logger,
        # resume_from_checkpoint=cfg.SOLVER.PRETRAINED_PATH,
        # limit_train_batches=0.1,
        # limit_val_batches=0.1,
        # limit_test_batches=0.1,
        # fast_dev_run=True,
    )

    # Train
    trainer.fit(model)


if __name__ == '__main__':

    # parse arguments
    parser = argparse.ArgumentParser(description="Hand pose estimation")
    parser.add_argument('--config', type=str, default='config.yaml', help='Path to the config file')
    parser.add_argument('--opts', nargs='*', default=[], help='Modify config options using the command-line')
    args = parser.parse_args()

    # merge config file
    cfg.merge_from_file(args.config)
    cfg.merge_from_list(args.opts)
    cfg.SOLVER.LOG_DIR = os.path.join(cfg.SOLVER.LOG_DIR, cfg.DATA.EXP_SETUP, cfg.MODEL.NAME)

    # create log dir
    os.makedirs(cfg.SOLVER.LOG_DIR, exist_ok=True)

    # setup seed
    setup_seed(cfg.SEED)


    main(cfg)