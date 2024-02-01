#!/bin/bash

# specify the directory containing the scripts
dir="./scripts"

# iterate over each .sh file in the directory
for script in "$dir"/*.sh
do
  # if the file is a regular file and is executable
  if [ -f "$script" ]
  then
    # execute the script
    chmod +x "$script"
    echo "Executing $script $1"
    "$script" $1
  fi
done