
import torch
from pytorch_lightning import loggers as pl_loggers
import pytorch_lightning as pl
from hpe.trainer import EmgNet, EmgNetClassifier, EmgNetPretrain

from hpe.config import cfg
from hpe.util import prepare_data
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


    # prepare data
    print('Preparing data')
    prepare_data(cfg)

    # Build logger
    csv_logger = pl_loggers.CSVLogger(cfg.SOLVER.LOG_DIR, name='csv_logs')
    tb_logger = pl_loggers.TensorBoardLogger(cfg.SOLVER.LOG_DIR, name='tb_logs')
    early_stop_callback = pl.callbacks.EarlyStopping(monitor='val_loss', patience=cfg.SOLVER.PATIENCE)
    checkpoint_callback = pl.callbacks.ModelCheckpoint(dirpath=os.path.join(cfg.SOLVER.LOG_DIR, 'checkpoints'), monitor='val_loss', save_top_k=1, mode='min')

    # setup model
    cfg.STAGE = 'pretrain'
    model = EmgNetPretrain(cfg=cfg)

    trainer_pretrain = pl.Trainer(
        default_root_dir=os.path.join(cfg.SOLVER.LOG_DIR, 'checkpoints_pretrain'),
        max_epochs=300,
        logger=tb_logger,
        log_every_n_steps=len(model.train_loader),
        limit_val_batches=0,
        limit_test_batches=0,
        # limit_train_batches=0.005,
    )

    print('Pretraining model')
    # pretrain model
    trainer_pretrain.fit(model)
    trainer_pretrain.save_checkpoint(os.path.join(cfg.SOLVER.LOG_DIR, 'pretrained.ckpt'))

    print('Finetuning model')
    model.stage = 'finetune'
    trainer_finetune = pl.Trainer(
        max_epochs=cfg.SOLVER.NUM_EPOCHS,
        logger=tb_logger,
        check_val_every_n_epoch=1,
        log_every_n_steps=len(model.train_loader),
        # limit_val_batches=1,
        # limit_test_batches=1,
        callbacks=[early_stop_callback, checkpoint_callback]
    )
    # finetune model
    trainer_finetune.fit(model)

    print('Testing model')
    # test model
    trainer_finetune.test(model)

    

if __name__ == '__main__':

    # parse arguments
    parser = argparse.ArgumentParser(description="Hand pose estimation")
    parser.add_argument('--config', type=str, default='config.yaml', help='Path to the config file')
    parser.add_argument('--opts', nargs='*', default=[], help='Modify config options using the command-line')
    args = parser.parse_args()

    # merge config file
    cfg.merge_from_file(args.config)
    cfg.merge_from_list(args.opts)
    cfg.SOLVER.LOG_DIR = os.path.join(cfg.SOLVER.LOG_DIR, cfg.DATA.EXP_SETUP)

    # create log dir
    os.makedirs(cfg.SOLVER.LOG_DIR, exist_ok=True)

    # setup seed
    setup_seed(cfg.SEED)


    main(cfg)