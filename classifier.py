import torch
import transformers 
from models.model import *
import numpy as np

model = None ##TODO: initialize model
args = None ## TODO: import args 
criterion = torch.nn.CrossEntropyLoss().to(args.device) ## weight the loss if you wish
optimizer = None ## TODO: initialize optimizer and define the params you want to optimize only the classification head. Use torch.nn.optim.AdamW()
scheduler = None ## TODO: initialize scheduler 
logger = None ## TODO: add logger to track progress
trainloader, devloader, testloader = None, None, None ## TODO: get data


classifier = MTL_classifier(
    model = model, 
    criterion = criterion, 
    optimizer = optimizer, 
    scheduler = scheduler, 
    device = args.device 
)


if args.mode == "train":
    classifier.train(args, trainloader, devloader)

if args.mode == "test":
    pass 
    ## use functions from evaluate_utils to test model. 
