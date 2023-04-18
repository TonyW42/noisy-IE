import sys

sys.path.append("..")
sys.path.append("../..")
import torch

import numpy as np
import pandas as pd

import os
import argparse
from classifier import *

os.environ["KMP_DUPLICATE_LIB_OK"] = "True"

from utils.utils import setup_seed
from evaluate_utils import *

## use GPU is available

from models.default_model import *


if __name__ == "__main__":
    ## add arguments
    parser = argparse.ArgumentParser()

    parser.add_argument("--dataset", type=str, default="wnut17")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--log_dir", type=str, default=None)
    parser.add_argument("--output_dir", type=str, default="./results")
    parser.add_argument("--bs", type=int, default=8)
    parser.add_argument("--train_batch_size", type=int, default=8)
    parser.add_argument("--eval_batch_size", type=int, default=8)
    parser.add_argument("--test_batch_size", type=int, default=8)
    parser.add_argument("--model_name", type=str, default="google/canine-s")
    parser.add_argument("--n_epochs", type=int, default=25)  ## change to 4
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--weight_decay", type=float, default=0)
    parser.add_argument("--prefix_space", type=bool, default=True)
    parser.add_argument("--num_labels", type=int, default=13)
    parser.add_argument(
        "--granularities", type=str, default="subword_50k,subword_30k"
    )  # add character
    parser.add_argument("--add_space_for_char", type=bool, default=True)
    parser.add_argument("--to_char_method", type=str, default="inherit")
    parser.add_argument("--mode", type=str, default="train")
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--ensemble_method", type=str, default="default")
    parser.add_argument(
        "--model_list", type=str, default="roberta-base|bert-base-cased"
    )
    parser.add_argument("--word_model", type=str, default="roberta-base")
    parser.add_argument("--num_att_layers", type=int, default=6)
    parser.add_argument("--expr", type=str, default="MTL")  ## change back to MTL
    parser.add_argument("--save", type=str, default="true")
    parser.add_argument("--layer_type", type=str, default="att")
    parser.add_argument("--mlm_epochs", type=int, default=100)

    ## NOTE: newly added
    parser.add_argument("--emb_size", type=int, default=768)
    parser.add_argument("--char_model", type=str, default="google/canine-s")
    parser.add_argument("--n_workers", type=int, default=8)

    parser.add_argument(
        "--granularities_model",
        type=dict,
        default={
            "character": "google/canine-s",
            "subword_50k": "roberta-base",
            "subword_30k": "bert-base-cased",
        },
    )

    parser.add_argument("--test", type=bool, default=False)
    parser.add_argument(
        "--embed_size_dict",
        type=dict,
        default={
            "char": 768,
            "word": 768,
            "google/canine-s": 768,
            "google/canine-c": 768,
            "bert-base-cased": 768,
            "bert-base-uncased": 768,
            "xlm-roberta-base": 768,
            "roberta-base": 768,
            "roberta-large": 1024,
            "vinai/bertweet-base": 768,
            "cardiffnlp/twitter-roberta-base-sentiment": 768,
        },
    )

    args = parser.parse_args()
    args.granularities = args.granularities.split(",")
    args.train = True if args.mode == "True" else False
    # args.model_names = [args.granularities_model[key] for key in args.granularities_model]
    args.model_names = args.model_list.split("|")
    # args.vocab_size = {'char': 1114112}
    args.vocab_size = {"char": 259}

    if not args.device:
        if torch.cuda.is_available():
            args.device = "cuda"
        elif torch.backends.mps.is_available():
            args.device = "mps"
        else:
            args.device = "cpu"

    setup_seed(args.seed)

    # args.k = 2

    # model = HuggingFaceModel(args)
    # model.train()
    if args.expr == "baseline":
        train_baseline(args)
    elif args.expr == "sequential":
        train_sequential_2(args)
    elif args.expr == "mlm":
        train_MLM(args)
    elif args.expr == "mlm_c":
        train_MLM_corpus(args)
    elif args.expr == "mlm_b":
        # train_bimodal_MLM(args, args.test)
        wnut_bimodal_MLM(args)
    else:
        train(args)
