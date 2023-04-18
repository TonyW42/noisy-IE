#!/bin/bash
# use the bash shell
date
# run the Unix 'date' command
echo "Hello world, from Bridges-2!"
module load anaconda3
conda activate py310
# run the Unix 'echo' command
mkdir -p /ocean/projects/cis230002p/xhu1/noisy-IE/mlm_b
python3 run.py --expr mlm_b --mlm_epochs 50 --n_epochs 100 --test True --n_workers 1 --output_dir "/ocean/projects/cis230002p/xhu1/noisy-IE/mlm_b"
echo "done"