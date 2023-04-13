#!/bin/bash
# use the bash shell
date
ds=$(date +'%Y-%m-%d')
# run the Unix 'date' command
echo "Hello world, from Bridges-2!"
module load anaconda3
conda activate py310

python3 test/test_demo.py > test-mlm-${ds}-$1.log