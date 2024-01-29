#!/bin/bash

# Define variables
model="neuropose"
setup="train"
config="../config.yaml"


python ../main.py --config $config \
    SETUP $setup \
    MODEL.NAME $model \
    SOLVER.BATCH_SIZE 32 \
    SOLVER.EPOCHS 100 \