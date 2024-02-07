#!/bin/bash

# specify the directory
dir="./dataset"

# find and delete all .npz files
find "$dir" -type f -name "*.npz" -exec rm -f {} \;