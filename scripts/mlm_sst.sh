#!/bin/bash
# use the bash shell
date
# run the Unix 'date' command
echo "Hello world, from Bridges-2!"
module load anaconda3
conda activate py38
# run the Unix 'echo' command
python3 run.py --expr mlm_b --mlm_epochs 50 --n_epochs 100 --output_dir "/ocean/projects/cis230002p/xhu1/noisy-IE/"
