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
for i in range(0, 2):
    os.popen(command_tweet.format(dataset=dataset[i][0], 
                                  num_att_layers=i, 
                                  expr=dataset[i][0], 
                                  num_labels=dataset[i][1]))