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
CUDA_LAUNCH_BLOCKING=1 find_unused_parameters=True TORCH_DISTRIBUTED_DEBUG=DETAIL accelerate launch run.py --expr mlm_b --mlm_epochs 10 --n_epochs 50 --num_att_layers $2 --test True --output_dir "/ocean/projects/cis230002p/xhu1/noisy-IE/mlm_b" > test-mlm-${ds}-$1.log
echo "done"