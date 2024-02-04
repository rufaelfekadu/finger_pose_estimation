#!/bin/bash

# Define variables
subject=$1
pose=$2
model=$3
exp_setup="exp0"
data_path="./dataset/emgleap/"$subject"/S1/"$pose
stage="train"
config="configs/"$model".yaml"
log_dir='outputs/'$exp_setup'/'$model'/'$subject'/'$pose


python main.py --config $config \
    --opts \
    STAGE $stage \
    DATA.EXP_SETUP $exp_setup \
    SOLVER.LOG_DIR $log_dir \