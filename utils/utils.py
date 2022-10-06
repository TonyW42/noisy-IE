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

import math
import numbers
import random

from fractions import Fraction
from decimal import Decimal
from itertools import groupby, repeat
from bisect import bisect_left, bisect_right
from math import hypot, sqrt, fabs, exp, erf, tau, log, fsum
from operator import itemgetter
from collections import Counter, namedtuple

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

#################################################################
####  find mode
#############################
def mode(data):
    """Return the most common data point from discrete or nominal data.
    ``mode`` assumes discrete data, and returns a single value. This is the
    standard treatment of the mode as commonly taught in schools:
        >>> mode([1, 1, 2, 3, 3, 3, 3, 4])
        3
    This also works with nominal (non-numeric) data:
        >>> mode(["red", "blue", "blue", "red", "green", "red", "red"])
        'red'
    If there are multiple modes with same frequency, return the first one
    encountered:
        >>> mode(['red', 'red', 'green', 'blue', 'blue'])
        'red'
    If *data* is empty, ``mode``, raises StatisticsError.
    """
    pairs = Counter(iter(data)).most_common(1)
    try:
        return pairs[0][0]
    except IndexError:
        raise RuntimeError('no mode for empty data') from None


def multimode(data):
    """Return a list of the most frequently occurring values.
    Will return more than one result if there are multiple modes
    or an empty list if *data* is empty.
    >>> multimode('aabbbbbbbbcc')
    ['b']
    >>> multimode('aabbbbccddddeeffffgg')
    ['b', 'd', 'f']
    >>> multimode('')
    []
    """
    counts = Counter(iter(data)).most_common()
    maxcount, mode_items = next(groupby(counts, key=itemgetter(1)), (0, []))
    return list(map(itemgetter(0), mode_items))
#################################################################


def flatten_2d(L):
    new = []
    for l in L:
        for l_i in l:
            new.append(l_i)
    return new

