#!/bin/bash

# specify the directory containing the scripts
dir="./scripts"
data_dir="./dataset/emgleap"

for subject in "001" "002" "003"
do
  for pose in "P1" "P2" "P3"
  do
    for model in "vvit" "ts" "vit" "neuropose"
    do
      # execute the script
      echo "Executing $model $subject $pose"
      "$dir/run_${model}.sh" $subject $pose $model
    done
  done
done 
