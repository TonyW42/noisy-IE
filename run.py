import transformers 
from transformers import AutoModel, AutoTokenizer
import torch 
from torch import nn, optim
from sklearn.metrics import roc_auc_score, precision_recall_fscore_support

import numpy as np 
import pandas as pd
import random 

import os 
import argparse
from datetime import datetime
import tqdm 

# from model import my_model
# from utils import classifier, setup_seed, make_if_not_exists
# from data import prepare_data
from torch.utils.data import DataLoader
from utils import setup_seed

## use GPU is available 
if torch.cuda.is_available():
    device = "cuda"
else:
    device = "cpu"

## add arguments 
parser = argparse.ArgumentParser()

parser.add_argument('--dataset', type=str, default='wnut17')
parser.add_argument('--seed', type=int, default=42)
parser.add_argument('--log_dir', type=str, default=None)
parser.add_argument('--output_dir', type=str, default=None)
parser.add_argument('--bs', type=int, default=16)
parser.add_argument('--model_name', type=str, default="xlm-roberta-base")
parser.add_argument('--n_epochs', type=int, default=4)
parser.add_argument('--lr', type=float, default=1e-4)
parser.add_argument('--weight_decay', type=float, default=0)
parser.add_argument('--prefix_space', type=bool, default=False)
parser.add_argument('--num_labels', type=int, default=14)


args = parser.parse_args()
prefix_space = args.prefix_space
model_name = args.model_name

def __main__():
    setup_seed(args.seed)

    pass

