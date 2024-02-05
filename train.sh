#!/bin/bash

# Define variables
subject=$1
pose=$2
model=$3
exp_setup="exp0"
data_path="./dataset/ds/"$subject"/S1/"$pose
stage="train"
config="configs/"$model".yaml"
log_dir='outputs/'$exp_setup'/'$model'/'$subject'/'$pose


python main.py --config $config \
    --opts \
    STAGE $stage \
    DATA.PATH $data_path \
    DATA.EXP_SETUP $exp_setup \
    SOLVER.LOG_DIR $log_dir \