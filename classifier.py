import torch
import transformers 
from models.model import *
import numpy as np
from transformers import AutoModel, AutoTokenizer, DataCollatorForTokenClassification, get_scheduler, AutoModelForMaskedLM
import torch 

from torch.nn.utils.rnn import pad_sequence #(1)
from collections import defaultdict
def custom_collate(data, seq_len=512): #(2)
    model_names = list(data[0].keys())
    batch_size = len(data)
    input_ids = []
    labels = []
    attention_mask = []
    for m_name in model_names:
      for i in range(batch_size):
        input_ids.append(data[i][m_name]['input_ids'][:seq_len])
    for m_name in model_names:
      for i in range(batch_size):
        attention_mask.append(data[i][m_name]['attention_mask'][:seq_len])
    for m_name in model_names:
      for i in range(batch_size):
        labels.append(data[i][m_name]['labels'][:seq_len])
    input_ids = pad_sequence(input_ids, batch_first=True, padding_value=1) 
    labels = pad_sequence(labels, batch_first=True, padding_value=-100)  
    attention_mask = pad_sequence(attention_mask, batch_first=True, padding_value=0) 

    return_dict = defaultdict(defaultdict)
    for i, m_name in enumerate(model_names):
        return_dict[m_name]['input_ids'] = input_ids[i * batch_size : (i+1) * batch_size]
        return_dict[m_name]['labels'] = labels[i * batch_size : (i+1) * batch_size]
        return_dict[m_name]['attention_mask'] = attention_mask[i * batch_size : (i+1) * batch_size]
    return return_dict


def custom_collate_SST(data, seq_len=512): #(2)
    model_names = list(data[0].keys())
    batch_size = len(data)
    input_ids = []
    labels = []
    attention_mask = []
    for m_name in model_names:
      for i in range(batch_size):
        input_ids.append(data[i][m_name]['input_ids'][:seq_len])
    for m_name in model_names:
      for i in range(batch_size):
        attention_mask.append(data[i][m_name]['attention_mask'][:seq_len])
    input_ids = pad_sequence(input_ids, batch_first=True, padding_value=1) 
    labels = input_ids.clone()
    attention_mask = pad_sequence(attention_mask, batch_first=True, padding_value=0) 

    rand = torch.rand(input_ids.shape)
    # where the random array is less than 0.15, we set true
    mask_arr = rand < 0.15
    mask_arr = mask_arr * (input_ids != -100)

    selection = []

    for i in range(input_ids.shape[0]):
        selection.append(
            torch.flatten(mask_arr[i].nonzero()).tolist()
        )

    for i in range(input_ids.shape[0]):
        labels[i, selection[i]] = -100

    return_dict = defaultdict(defaultdict)
    for i, m_name in enumerate(model_names):
        return_dict[m_name]['input_ids'] = input_ids[i * batch_size : (i+1) * batch_size]
        return_dict[m_name]['labels'] = labels[i * batch_size : (i+1) * batch_size]
        return_dict[m_name]['attention_mask'] = attention_mask[i * batch_size : (i+1) * batch_size]
    
    return return_dict


def fetch_loaders(model_names, args):
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
    data_test = WNUTDatasetMulti(test_encoding_list, test_label_list, model_names)
    loader_train = torch.utils.data.DataLoader(
        data_train, batch_size=args.train_batch_size, collate_fn=custom_collate
    )
    loader_valid = torch.utils.data.DataLoader(
        data_valid, batch_size=args.eval_batch_size, collate_fn=custom_collate
    )
    loader_test = torch.utils.data.DataLoader(
        data_test, batch_size=args.test_batch_size, collate_fn=custom_collate
    )
    return loader_train, loader_valid, loader_test

def fetch_loaders_SST(model_names, args):
    from datasets import load_dataset
    sst = load_dataset("sst")
    train_encoding_list = []
    valid_encoding_list = []
    test_encoding_list = []
    for model_name in model_names:
        tokenizer = AutoTokenizer.from_pretrained(model_name, add_prefix_space=True) ## changed here
        xlm_train_encoding = tokenizer(sst['train']["sentence"], padding="longest",truncation=True)
        xlm_valid_encoding = tokenizer(sst['validation']["sentence"], padding="longest", truncation=True)
        xlm_test_encoding = tokenizer(sst['test']["sentence"], padding="longest", truncation=True)

        train_encoding_list.append(xlm_train_encoding)
        valid_encoding_list.append(xlm_valid_encoding)
        test_encoding_list.append(xlm_test_encoding)

    data_train = SSTDatasetMulti(train_encoding_list, model_names)
    data_valid = SSTDatasetMulti(valid_encoding_list, model_names)
    data_test = SSTDatasetMulti(test_encoding_list, model_names)
    print(len(data_train))
    loader_train = torch.utils.data.DataLoader(
        data_train, batch_size=args.train_batch_size, collate_fn=custom_collate_SST
    )
    loader_valid = torch.utils.data.DataLoader(
        data_valid, batch_size=args.eval_batch_size, collate_fn=custom_collate_SST
    )
    loader_test = torch.utils.data.DataLoader(
        data_test, batch_size=args.test_batch_size, collate_fn=custom_collate_SST
    )
    return loader_train, loader_valid, loader_test

def fetch_loaders2(model_names, args):
    from datasets import load_dataset
    wnut = load_dataset("wnut_17")
    data_loader = wnut_multiple_granularity(wnut, args)
    # model_main = data_loader.model_dict
    tokenized_wnut_main = data_loader.data_

    train_encoding_list, train_label_list = [], []
    valid_encoding_list, valid_label_list = [], []
    test_encoding_list, test_label_list = [], []
    
    for model_name in model_names:
        train_encoding_list.append({'input_ids': tokenized_wnut_main[model_name]['train']['input_ids'],
                                'attention_mask': tokenized_wnut_main[model_name]['train']['attention_mask'],
                                })
        train_label_list.append(tokenized_wnut_main[model_name]['train']['labels'])
        valid_encoding_list.append({'input_ids': tokenized_wnut_main[model_name]['validation']['input_ids'],
                                'attention_mask': tokenized_wnut_main[model_name]['validation']['attention_mask'],
                                })
        valid_label_list.append(tokenized_wnut_main[model_name]['validation']['labels'])
        test_encoding_list.append({'input_ids': tokenized_wnut_main[model_name]['test']['input_ids'],
                                'attention_mask': tokenized_wnut_main[model_name]['test']['attention_mask'],
                                })
        test_label_list.append(tokenized_wnut_main[model_name]['test']['labels'])

    data_train = WNUTDatasetMulti(train_encoding_list, train_label_list, model_names)
    data_valid = WNUTDatasetMulti(valid_encoding_list, valid_label_list, model_names)
    data_test = WNUTDatasetMulti(test_encoding_list, test_label_list, model_names)
    loader_train = torch.utils.data.DataLoader(
        data_train, batch_size=args.train_batch_size, collate_fn=custom_collate
    )
    loader_valid = torch.utils.data.DataLoader(
        data_valid, batch_size=args.eval_batch_size, collate_fn=custom_collate
    )
    loader_test = torch.utils.data.DataLoader(
        data_test, batch_size=args.test_batch_size, collate_fn=custom_collate
    )
    return loader_train, loader_valid, loader_test


def train(args):
    ## initialize model
    model_names = args.model_list.split("|")
    model_dict = torch.nn.ModuleDict()
    for model_name in model_names:
        model_dict[model_name] = AutoModel.from_pretrained(model_name, num_labels=args.num_labels)
    model = flat_MTL(model_dict = model_dict, args = args).to(args.device)

    criterion = torch.nn.ModuleDict()
    for model_name in model_names:
        criterion[model_name] = torch.nn.CrossEntropyLoss().to(args.device)
    # criterion = torch.nn.CrossEntropyLoss().to(args.device) ## weight the loss if you wish
    print(" ====== parameters? ========")
    for name, p in model.named_parameters():
        print(name)
    # params = [p for p in model.parameters()]
    # for name in model.model_dict:
    #   for p in model.model_dict[name].parameters():
    #     params.append(p)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    # trainloader, devloader, testloader = fetch_loaders(model_names, args) ## TODO: get data
    trainloader, devloader, testloader = fetch_loaders2(model_names, args)
    print("new fetch loaders")
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
        classifier.train(args, trainloader, testloader)

    if args.mode == "test":
        pass 
        # use functions from evaluate_utils to test model.

    ## do something to evaluate the model  
    ## Note: could evaluate using seqeval in the estimator's _eval() function 
    ## need to re-write the _eval() function for token classification. 

def train_MLM(args):
    ## initialize model
    model_names = args.model_list.split("|")
    model_dict = torch.nn.ModuleDict()
    for model_name in model_names:
        model_dict[model_name] = AutoModel.from_pretrained(model_name, num_labels=args.num_labels)
    base = MTL_base(model_dict = model_dict, args = args).to(args.device)
    MLM_model = flat_MLM_w_base(base = base, args = args).to(args.device)

    criterion = torch.nn.ModuleDict()
    for model_name in model_names:
        criterion[model_name] = torch.nn.CrossEntropyLoss().to(args.device)
    # criterion = torch.nn.CrossEntropyLoss().to(args.device) ## weight the loss if you wish
    print(" ====== parameters? ========")
    for name, p in MLM_model.named_parameters():
        print(name)
    # params = [p for p in model.parameters()]
    # for name in model.model_dict:
    #   for p in model.model_dict[name].parameters():
    #     params.append(p)
    optimizer = torch.optim.AdamW(MLM_model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    # trainloader, devloader, testloader = fetch_loaders(model_names, args) ## TODO: get data
    ## TODO: fix fetch_loaders_SST function
    trainloader, devloader, testloader = fetch_loaders_SST(model_names, args)
    num_training_steps = args.n_epochs * len(trainloader)
    scheduler = get_scheduler(
        "linear",
        optimizer=optimizer,
        num_warmup_steps=0,
        num_training_steps=num_training_steps
    )
    logger = None ## TODO: add logger to track progress


    MLM_classifier_ = MLM_classifier(
        model = MLM_model, 
        cfg = args,
        criterion = criterion, 
        optimizer = optimizer, 
        scheduler = scheduler, 
        device = args.device,
        logger = logger 
    )
    MLM_classifier_.train(args, trainloader, testloader)  ## train MLM

    #####################################################################
    model = flat_MTL_w_base(base = base, args = args).to(args.device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    # trainloader, devloader, testloader = fetch_loaders(model_names, args) ## TODO: get data
    trainloader, devloader, testloader = fetch_loaders2(model_names, args)
    num_training_steps = args.n_epochs * len(trainloader)
    scheduler = get_scheduler(
        "linear",
        optimizer=optimizer,
        num_warmup_steps=0,
        num_training_steps=num_training_steps
    )
    logger = None ## TODO: add logger to track progress
    classifier = MLM_classifier_(
        model = model, 
        cfg = args,
        criterion = criterion, 
        optimizer = optimizer, 
        scheduler = scheduler, 
        device = args.device,
        logger = logger 
    )
    if args.mode == "train":
        classifier.train(args, trainloader, testloader)


        # use functions from evaluate_utils to test model.


def train_baseline(args):
    model = baseline_model(args = args).to(args.device)
    criterion = torch.nn.CrossEntropyLoss().to(args.device)
    optimizer = torch.optim.AdamW(model.parameters(),lr=args.lr)
    trainloader, devloader, testloader = fetch_loaders([args.word_model], args)
    num_training_steps = args.n_epochs * len(trainloader)
    scheduler = get_scheduler(
        "linear",
        optimizer=optimizer,
        num_warmup_steps=0,
        num_training_steps=num_training_steps
    )
    logger = None
    classifier = baseline_classifier(
        model = model, 
        cfg = args,
        criterion = criterion, 
        optimizer = optimizer, 
        scheduler = scheduler, 
        device = args.device,
        logger = logger 
    )
    if args.mode == "train":
        classifier.train(args, trainloader, testloader)

    if args.mode == "test":
        pass 

def train_sequential(args):
    ## initialize model
    model_names = args.model_list.split("|")
    model_dict = torch.nn.ModuleDict()
    for model_name in model_names:
        model_dict[model_name] = AutoModel.from_pretrained(model_name, num_labels=args.num_labels)
    model = sequential_MTL(model_dict = model_dict, args = args).to(args.device)

    criterion = torch.nn.ModuleDict()
    for model_name in model_names:
        criterion[model_name] = torch.nn.CrossEntropyLoss().to(args.device)
    # criterion = torch.nn.CrossEntropyLoss().to(args.device) ## weight the loss if you wish
    print(" ====== parameters? ========")
    for name, p in model.named_parameters():
        print(name)
    # params = [p for p in model.parameters()]
    # for name in model.model_dict:
    #   for p in model.model_dict[name].parameters():
    #     params.append(p)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    # trainloader, devloader, testloader = fetch_loaders(model_names, args) ## TODO: get data
    trainloader, devloader, testloader = fetch_loaders2(model_names, args)
    num_training_steps = args.n_epochs * len(trainloader)
    scheduler = get_scheduler(
        "linear",
        optimizer=optimizer,
        num_warmup_steps=0,
        num_training_steps=num_training_steps
    )
    logger = None ## TODO: add logger to track progress


    classifier = sequential_classifier(
        model = model, 
        cfg = args,
        criterion = criterion, 
        optimizer = optimizer, 
        scheduler = scheduler, 
        device = args.device,
        logger = logger 
    )
    classifier.JSD = JSD() ## need to have JSD 

    if args.mode == "train":
        classifier.train(args, trainloader, testloader)

    if args.mode == "test":
        pass 
        # use functions from evaluate_utils to test model.

    ## do something to evaluate the model  
    ## Note: could evaluate using seqeval in the estimator's _eval() function 
    ## need to re-write the _eval() function for token classification. 

def train_sequential_2(args):
    ## initialize model
    model_names = args.model_list.split("|")
    model_dict = torch.nn.ModuleDict()
    for model_name in model_names:
        model_dict[model_name] = base_model(model_name, args = args).to(args.device)

    criterion = torch.nn.CrossEntropyLoss().to(args.device)
    trainloader, devloader, testloader = fetch_loaders2(model_names, args)
    num_training_steps = args.n_epochs * len(trainloader)
    # criterion = torch.nn.CrossEntropyLoss().to(args.device) ## weight the loss if you wish
    optimizer_dict = dict()
    scheduler_dict = dict()
    for model_name in model_names:
        optimizer = torch.optim.AdamW(model_dict[model_name].parameters(),
                                      lr=args.lr, weight_decay=args.weight_decay)
        scheduler = get_scheduler(
            "linear",
            optimizer=optimizer,
            num_warmup_steps=0,
            num_training_steps=num_training_steps
        ) 
        optimizer_dict[model_name] = optimizer
        scheduler_dict[model_name] = scheduler

    # trainloader, devloader, testloader = fetch_loaders(model_names, args) ## TODO: get data
    logger = None ## TODO: add logger to track progress


    classifier = sequential_classifier_2(
        model = model_dict, 
        cfg = args,
        criterion = criterion, 
        optimizer = optimizer_dict, 
        scheduler = scheduler_dict, 
        device = args.device,
        logger = logger 
    )
    classifier.prob_loss = torch.nn.MSELoss() ## need to have JSD 

    if args.mode == "train":
        classifier.train(args, trainloader, testloader)

    if args.mode == "test":
        pass 
        # use functions from evaluate_utils to test model.

    ## do something to evaluate the model  
    ## Note: could evaluate using seqeval in the estimator's _eval() function 
    ## need to re-write the _eval() function for token classification. 
