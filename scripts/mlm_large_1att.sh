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
CUDA_LAUNCH_BLOCKING=1 find_unused_parameters=True TORCH_DISTRIBUTED_DEBUG=DETAIL accelerate launch run.py --ckpt_name MLM_16_1layer.pt --expr mlm_b --mlm_epochs 10 --n_epochs 50 --last_layer_integration false --num_att_layers 1 --test True --train_batch_size 32 --eval_batch_size 32 --test_batch_size 32 --output_dir "/ocean/projects/cis230002p/xhu1/noisy-IE/mlm_b" > test-mlm-${ds}-1attention.log
echo "done"