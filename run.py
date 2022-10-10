import transformers 
from transformers import AutoModel, AutoTokenizer, DataCollatorForTokenClassification
import torch 
from torch import nn, optim
from datasets import load_dataset
from sklearn.metrics import roc_auc_score, precision_recall_fscore_support

import numpy as np 
import pandas as pd
import random 

import os 
import argparse
from datetime import datetime
import tqdm 

os.environ['KMP_DUPLICATE_LIB_OK']='True'

# from model import my_model
# from utils import classifier, setup_seed, make_if_not_exists
# from data import prepare_data
from torch.utils.data import DataLoader
from utils.utils import setup_seed
from transformers import AutoModelForTokenClassification, TrainingArguments, Trainer
from evaluate_utils import *
## use GPU is available 

from models.default_model import *


if __name__ == '__main__':
    ## add arguments 
    parser = argparse.ArgumentParser()

    parser.add_argument('--dataset', type=str, default='wnut17')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--log_dir', type=str, default=None)
    parser.add_argument('--output_dir', type=str, default="./results")
    parser.add_argument('--bs', type=int, default=16)
    parser.add_argument('--model_name', type=str, default="google/canine-s")
    parser.add_argument('--n_epochs', type=int, default=1) ## change to 4
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--weight_decay', type=float, default=0)
    parser.add_argument('--prefix_space', type=bool, default=True)
    parser.add_argument('--num_labels', type=int, default=13)
    parser.add_argument('--granularities', type=str, default="character,subword 50k")# add cahracter
    parser.add_argument('--add_space_for_char', type=bool, default=True)
    parser.add_argument('--to_char_method', type=str, default="inherit")
    parser.add_argument('--train', type=str, default="True")
    parser.add_argument('--device', type=str, default=None)

    parser.add_argument('--granularities_model', type=dict, 
                        default= {"character": "google/canine-s",
                                "subword 50k": "xlm-roberta-base",
                                "subword 30k" : "bert-base-cased"})

    args = parser.parse_args()
    args.granularities = args.granularities.split(",")
    args.train = True if args.train == "True" else False
    
    if not args.device:
        if torch.cuda.is_available():
            args.device = "cuda"
        else:
            args.device = "cpu"

    setup_seed(args.seed)

    model = HuggingFaceModel(args)
    model.train()