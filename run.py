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

# from model import my_model
# from utils import classifier, setup_seed, make_if_not_exists
# from data import prepare_data
from torch.utils.data import DataLoader
from utils.utils import setup_seed
from transformers import AutoModelForTokenClassification, TrainingArguments, Trainer
from evaluate_utils import *
## use GPU is available 
if torch.cuda.is_available():
    device = "cuda"
else:
    device = "cpu"

def hugging_face_model(args):
    from data import character_level_wnut, tokenize_for_char_manual, tokenize_and_align_labels, tokenize_for_char
    from utils.compute import compute_metrics
    model = AutoModelForTokenClassification.from_pretrained(args.model_name, 
            num_labels=args.num_labels)
    wnut = load_dataset("wnut_17")
    wnut_character_level = character_level_wnut(wnut)

    use_old_tok = ["xlm-roberta-base", "xlm-roberta-large"]
    use_new_tok = ["google/canine-s"]
    if args.model_name in use_old_tok:
        tokenized_wnut = wnut_character_level.map(tokenize_and_align_labels, batched=True)
    if args.model_name in use_new_tok:
        try:
            tokenized_wnut = wnut_character_level.map(tokenize_for_char, batched=True)
        except:
            tokenized_wnut = tokenize_for_char_manual(wnut_character_level)

    tokenizer = AutoTokenizer.from_pretrained(args.model_name,
                                            add_prefix_space=args.prefix_space) ## changed here
    data_collator = DataCollatorForTokenClassification(tokenizer=tokenizer)
                                        

    training_args = TrainingArguments(
        output_dir=args.output_dir, ## come back to fix this!
        evaluation_strategy="epoch",
        learning_rate=args.lr,
        per_device_train_batch_size=args.bs,
        per_device_eval_batch_size=args.bs,
        num_train_epochs=args.n_epochs,
        # weight_decay=0.01,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_wnut["train"],
        eval_dataset=tokenized_wnut["test"],
        tokenizer=tokenizer,
        data_collator=data_collator,
        # optimizers = torch.optim.Adam(model.parameters()),
        compute_metrics = compute_metrics, 
    )

    trainer.train()

    ## evalate trained model 
    wnut_f1 = wnut_evaluate_f1(model = model,  
                               tokenized_wnut = tokenized_wnut, 
                               prefix_space = args.prefix_space, 
                               model_name = args.model_name,
                               device = device,
                               method = "first letter")
    # wnut_f1_1 = wnut_evaluate_f1(model = model,  
    #                              tokenized_wnut = tokenized_wnut, 
    #                              prefix_space = args.prefix_space, 
    #                              model_name = args.model_name,
    #                              device = device,
    #                              method = "rule 1")
    # wnut_f1_2 = wnut_evaluate_f1(model = model,  
    #                              tokenized_wnut = tokenized_wnut, 
    #                              prefix_space = args.prefix_space, 
    #                              model_name = args.model_name,
    #                              device = device,
    #                              method = "rule 2")
    print(f"\n The F1-score of the model is {wnut_f1} \n")
    # print(f"\n The F1-score of the model is {wnut_f1_1} \n")
    # print(f"\n The F1-score of the model is {wnut_f1_2} \n")

    

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

args = parser.parse_args()
prefix_space = args.prefix_space
model_name = args.model_name

if __name__ == '__main__':
    ## add arguments 
    # parser = argparse.ArgumentParser()

    # parser.add_argument('--dataset', type=str, default='wnut17')
    # parser.add_argument('--seed', type=int, default=42)
    # parser.add_argument('--log_dir', type=str, default=None)
    # parser.add_argument('--output_dir', type=str, default=None)
    # parser.add_argument('--bs', type=int, default=16)
    # parser.add_argument('--model_name', type=str, default="xlm-roberta-base")
    # parser.add_argument('--n_epochs', type=int, default=4)
    # parser.add_argument('--lr', type=float, default=1e-4)
    # parser.add_argument('--weight_decay', type=float, default=0)
    # parser.add_argument('--prefix_space', type=bool, default=False)
    # parser.add_argument('--num_labels', type=int, default=14)

    # args = parser.parse_args()
    # prefix_space = args.prefix_space
    # model_name = args.model_name

    setup_seed(args.seed)

    hugging_face_model(args)