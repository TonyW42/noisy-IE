#!/bin/bash
# use the bash shell
date
ds=$(date +'%Y-%m-%d')
ckpt_name=$1
# run the Unix 'date' command
echo "Hello world, from Bridges-2!"
module load anaconda3
conda activate py310
# run the Unix 'echo' command
mkdir -p /ocean/projects/cis230002p/xhu1/noisy-IE/mlm_b
python3 run.py --expr eval_wnut --mlm_epochs 1 --n_epochs 50 --num_att_layers $2 --test True --ckpt_name $ckpt_name --output_dir "/ocean/projects/cis230002p/xhu1/noisy-IE/mlm_b" > test-mlm-${ds}-$1.log
echo "done"