import torch
import transformers 
from models.model import *
import numpy as np
from transformers import AutoModel, AutoTokenizer, DataCollatorForTokenClassification, get_scheduler
import torch 


def train(args):
    ## initialize model
    model_names = args.model_list.split("|")
    model_dict = {}
    for model_name in model_names:
        model_dict[model_name] = AutoModel.from_pretrained(model_name)
    model = attention_MTL(model_dict = model_dict, args = args)

    criterion = torch.nn.CrossEntropyLoss().to(args.device) ## weight the loss if you wish
    optimizer = torch.optim.AdamW(model.parameters())
    trainloader, devloader, testloader = None, None, None ## TODO: get data
    num_training_steps = args.n_epochs * len(trainloader)
    scheduler = get_scheduler(
        "linear",
        optimizer=optimizer,
        num_warmup_steps=0,
        num_training_steps=num_training_steps
    )
    logger = None ## TODO: add logger to track progress


    classifier = MTL_classifier(
        model = model, 
        cfg = args,
        criterion = criterion, 
        optimizer = optimizer, 
        scheduler = scheduler, 
        device = args.device,
        logger = logger 
    )

    if args.mode == "train":
        classifier.train(args, trainloader, devloader)

    if args.mode == "test":
        pass 
        ## use functions from evaluate_utils to test model.

    ## do something to evaluate the model  
    ## Note: could evaluate using seqeval in the estimator's _eval() function 
    ## need to re-write the _eval() function for token classification. 

