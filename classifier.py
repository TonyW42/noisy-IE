import torch
import transformers 
from models.model import *
import numpy as np
from transformers import AutoModel, AutoTokenizer, DataCollatorForTokenClassification, get_scheduler, AutoModelForMaskedLM
import torch 


def fetch_loaders(model_names):
    from datasets import load_dataset
    wnut = load_dataset("wnut_17")
    train_encoding_list, train_label_list = [], []
    valid_encoding_list, valid_label_list = [], []
    test_encoding_list, test_label_list = [], []
    for model_name in model_names:
        tokenizer = AutoTokenizer.from_pretrained(model_name, add_prefix_space=True) ## changed here
        xlm_train_encoding = tokenizer(wnut['train']["tokens"], padding="longest",truncation=True, is_split_into_words=True)
        xlm_valid_encoding = tokenizer(wnut['validation']["tokens"], padding="longest", truncation=True, is_split_into_words=True)
        xlm_test_encoding = tokenizer(wnut['test']["tokens"], padding="longest", truncation=True, is_split_into_words=True)

        xlm_train_labels = encode_tags(wnut['train'], xlm_train_encoding)
        xlm_valid_labels = encode_tags(wnut['validation'], xlm_valid_encoding)
        xlm_test_labels = encode_tags(wnut['test'], xlm_test_encoding)

        train_encoding_list.append(xlm_train_encoding)
        valid_encoding_list.append(xlm_valid_encoding)
        test_encoding_list.append(xlm_test_encoding)

        train_label_list.append(xlm_train_labels)
        valid_label_list.append(xlm_valid_labels)
        test_label_list.append(xlm_test_labels)

    data_train = WNUTDatasetMulti(train_encoding_list, train_label_list, model_names)
    data_valid = WNUTDatasetMulti(valid_encoding_list, valid_label_list, model_names)
    data_test = WNUTDatasetMulti(test_encoding_list, test_encoding_list, model_names)
    loader_train = torch.utils.data.DataLoader(
        data_train, batch_size=32
    )
    loader_valid = torch.utils.data.DataLoader(
        data_valid, batch_size=32
    )
    loader_test = torch.utils.data.DataLoader(
        data_test, batch_size=32
    )
    return loader_train, loader_valid, loader_test



def train(args):
    ## initialize model
    model_names = args.model_list.split("|")
    model_dict = {}
    for model_name in model_names:
        model_dict[model_name] = AutoModel.from_pretrained(model_name, num_labels=args.num_labels)
    model = attention_MTL(model_dict = model_dict, args = args)

    criterion = torch.nn.CrossEntropyLoss().to(args.device) ## weight the loss if you wish
    optimizer = torch.optim.AdamW(model.parameters(),lr=args.lr)
    trainloader, devloader, testloader = fetch_loaders(model_names) ## TODO: get data
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

