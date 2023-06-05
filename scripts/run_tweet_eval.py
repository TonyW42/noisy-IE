# !/usr/bin/python

import os, sys

# using command mkdir
dataset = [
    ('tweeteval-emoji', 20),
    ('tweeteval-emotion', 4),
    ('tweeteval-hate', 2),
    ('tweeteval-irony', 2),
    ('tweeteval-offensive', 2),
    ('tweeteval-sentiment', 3),
    ('tweeteval-stance_abortion', 3),
    ('tweeteval-stance_atheism', 3),
    ('tweeteval-stance_climate', 3),
    ('tweeteval-stance_feminist', 3),
    ('tweeteval-stance_hillary', 3),
]

command_tweet = 'sbatch -p GPU-shared -t 8:00:00 -N 1 -n 1 -o eval_{dataset}_{num_att_layers}_ --gpus=v100-32:1 scripts/eval_script.sh {expr} {dataset} {num_labels} {num_att_layers}'
command_baseline = 'sbatch -p GPU-shared -t 8:00:00 -N 1 -n 1 -o eval_{dataset}_base_ --gpus=v100-32:1 scripts/eval_script.sh {expr} {dataset} {num_labels} {num_att_layers}'
# for dataset_name, labels in dataset:
#     for att_layer_number in range(1,3):
#         os.popen(command_tweet.format(dataset=dataset_name, 
#                                       num_att_layers=att_layer_number, 
#                                       expr=dataset_name, 
#                                       num_labels=labels))

for dataset_name, labels in dataset:
    os.popen(command_baseline.format(dataset=dataset_name, 
                                    expr=dataset_name, 
                                    num_labels=labels))