#!/bin/bash

# Define variables
exp_setup=$1
data_path="./dataset/emgleap/003/S1/P3"
model="neuropose"
stage="train"
config="config.yaml"
log_dir='outputs/neuropose'


python main.py --config $config \
    --opts \
    STAGE $stage \
    DATA.EXP_SETUP $exp_setup \
    MODEL.NAME $model \
    SOLVER.BATCH_SIZE 32 \
    SOLVER.NUM_EPOCHS 300 \
    SOLVER.LR 0.001 \
    SOLVER.LOG_DIR $log_dir \