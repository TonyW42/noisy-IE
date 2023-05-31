#!/bin/bash
# use the bash shell
date
ds=$(date +'%Y-%m-%d')
# run the Unix 'date' command
echo "Hello world, from Bridges-2!"
module load anaconda3
conda activate py310
# run the Unix 'echo' command
mkdir -p /ocean/projects/cis230002p/xhu1/noisy-IE/mlm_b
python3 run.py --expr eval_wnut --n_epochs 50 --num_att_layers $1 --test True --output_dir "/ocean/projects/cis230002p/xhu1/noisy-IE/mlm_b" > test-eval-${ds}-$1.log
echo "done"