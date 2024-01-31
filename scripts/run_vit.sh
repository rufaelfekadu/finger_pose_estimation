#!/bin/bash

# Define variables
exp_setup="exp0"
data_path="./dataset/emgleap/003/S1/P3"
model="vit"
stage="train"
config="config.yaml"
log_dir='outputs/vit'


python main.py --config $config \
    --opts \
    STAGE $stage \
    DATA.PATH $data_path \
    DATA.EXP_SETUP $exp_setup \
    MODEL.NAME $model \
    SOLVER.BATCH_SIZE 32 \
    SOLVER.NUM_EPOCHS 300 \
    SOLVER.LOG_DIR $log_dir \