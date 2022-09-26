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

from datasets import load_dataset

## set seed 
def setup_seed(seed):
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def make_if_not_exists(new_dir): 
    if not os.path.exists(new_dir): 
        os.system('mkdir -p {}'.format(new_dir))

