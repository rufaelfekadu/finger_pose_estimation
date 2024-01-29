#!/bin/bash

# Define variables
data_path="./dataset/FPE/003/S1/P3"
model="neuropose"
stage="train"
config="config.yaml"
log_dir='outputs/neuropose'


python main.py --config $config \
    --opts \
    DATA.PATH $data_path \
    STAGE $stage \
    MODEL.NAME $model \
    SOLVER.BATCH_SIZE 32 \
    SOLVER.NUM_EPOCHS 100 \
    SOLVER.LOG_DIR $log_dir \