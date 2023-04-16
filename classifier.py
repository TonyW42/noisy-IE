import sys

sys.path.append("..")
sys.path.append("../..")
import torch
from models.model import *
import numpy as np
from transformers import (
    AutoModel,
    get_scheduler,
)
import wandb
import torch
from models.model import bimodal_trainer
from models.model import bimodal_base
from utils.fetch_loader import (
    fetch_loader_wnut,
    fetch_loaders,
    fetch_loaders2,
    fetch_loaders_SST,
    fetch_loader_book_wiki,
    fetch_loader_book_wiki_bimodal,
)
from torch import nn
from accelerate import Accelerator, DistributedDataParallelKwargs


def train(args):
    ## initialize model
    model_names = args.model_list.split("|")
    model_dict = torch.nn.ModuleDict()
    for model_name in model_names:
        model_dict[model_name] = AutoModel.from_pretrained(
            model_name, num_labels=args.num_labels, cache_dir=args.output_dir
        )
    model = flat_MTL(model_dict=model_dict, args=args).to(args.device)

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
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay
    )
    # trainloader, devloader, testloader = fetch_loaders(model_names, args) ## TODO: get data
    trainloader, devloader, testloader = fetch_loaders2(model_names, args)
    print("new fetch loaders")
    num_training_steps = args.n_epochs * len(trainloader)
    scheduler = get_scheduler(
        "linear",
        optimizer=optimizer,
        num_warmup_steps=0,
        num_training_steps=num_training_steps,
    )
    logger = None  ## TODO: add logger to track progress

    classifier = MTL_classifier(
        model=model,
        cfg=args,
        criterion=criterion,
        optimizer=optimizer,
        scheduler=scheduler,
        device=args.device,
        logger=logger,
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
        model_dict[model_name] = AutoModel.from_pretrained(
            model_name, num_labels=args.num_labels, cache_dir=args.output_dir
        )
    base = MTL_base(model_dict=model_dict, args=args).to(args.device)
    MLM_model = flat_MLM_w_base(base=base, args=args).to(args.device)

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
    optimizer = torch.optim.AdamW(
        MLM_model.parameters(), lr=args.lr, weight_decay=args.weight_decay
    )
    # trainloader, devloader, testloader = fetch_loaders(model_names, args) ## TODO: get data
    ## TODO: fix fetch_loaders_SST function
    trainloader, devloader, testloader = fetch_loaders_SST(model_names, args)
    num_training_steps = args.n_epochs * len(trainloader)
    scheduler = get_scheduler(
        "linear",
        optimizer=optimizer,
        num_warmup_steps=0,
        num_training_steps=num_training_steps,
    )
    logger = None  ## TODO: add logger to track progress

    train_epochs = args.n_epochs
    args.n_epochs = args.mlm_epochs
    MLM_classifier_ = MLM_classifier(
        model=MLM_model,
        cfg=args,
        criterion=criterion,
        optimizer=optimizer,
        scheduler=scheduler,
        device=args.device,
        logger=logger,
    )
    MLM_classifier_.train(args, trainloader, testloader)  ## train MLM

    #####################################################################
    model = flat_MTL_w_base(base=base, args=args).to(args.device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay
    )
    # trainloader, devloader, testloader = fetch_loaders(model_names, args) ## TODO: get data
    trainloader, devloader, testloader = fetch_loaders2(model_names, args)
    num_training_steps = args.n_epochs * len(trainloader)
    scheduler = get_scheduler(
        "linear",
        optimizer=optimizer,
        num_warmup_steps=0,
        num_training_steps=num_training_steps,
    )
    logger = None  ## TODO: add logger to track progress

    args.n_epochs = train_epochs
    classifier = MTL_classifier(
        model=model,
        cfg=args,
        criterion=criterion,
        optimizer=optimizer,
        scheduler=scheduler,
        device=args.device,
        logger=logger,
    )
    if args.mode == "train":
        classifier.train(args, trainloader, testloader)

        # use functions from evaluate_utils to test model.


def train_MLM_corpus(args):
    ## initialize model
    model_names = args.model_list.split("|")
    model_dict = torch.nn.ModuleDict()
    for model_name in model_names:
        model_dict[model_name] = AutoModel.from_pretrained(
            model_name, num_labels=args.num_labels, cache_dir=args.output_dir
        )

    """
    Try to store MLM here, using a metadata table to map?
    TODO: time recorder
    TODO: metadata to store base model
    """

    if args.test:
        base = MTL_base(model_dict=model_dict, args=args).to(args.device)
        MLM_model = flat_MLM_w_base(base=base, args=args).to(args.device)
    else:
        base = MTL_base(
            model_dict=model_dict, args=args
        )  # .to(args.device) ## NOTE: no need to push the base to device
        MLM_model = flat_MLM_w_base(base=base, args=args)  # .to(args.device)
        MLM_model = nn.DataParallel(MLM_model)
        MLM_model.to(args.device)

    # freeze the base model
    for param in MLM_model.parameters():
        param.requires_grad = False

    criterion = torch.nn.ModuleDict()
    for model_name in model_names:
        criterion[model_name] = torch.nn.CrossEntropyLoss().to(args.device)
    optimizer = torch.optim.AdamW(
        MLM_model.parameters(), lr=args.lr, weight_decay=args.weight_decay
    )
    trainloader, devloader, testloader = fetch_loader_book_wiki(model_names, args)
    num_training_steps = args.n_epochs * len(trainloader)
    scheduler = get_scheduler(
        "linear",
        optimizer=optimizer,
        num_warmup_steps=0,
        num_training_steps=num_training_steps,
    )
    logger = None  ## TODO: add logger to track progress

    train_epochs = args.n_epochs
    args.n_epochs = args.mlm_epochs
    MLM_classifier_ = MLM_classifier(
        model=MLM_model,
        cfg=args,
        criterion=criterion,
        optimizer=optimizer,
        scheduler=scheduler,
        device=args.device,
        logger=logger,
    )
    MLM_classifier_.train(args, trainloader, testloader)  ## train MLM

    #####################################################################
    model = flat_MTL_w_base(base=base, args=args).to(args.device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay
    )
    # trainloader, devloader, testloader = fetch_loaders(model_names, args) ## TODO: get data
    trainloader, devloader, testloader = fetch_loaders2(model_names, args)
    num_training_steps = args.n_epochs * len(trainloader)
    scheduler = get_scheduler(
        "linear",
        optimizer=optimizer,
        num_warmup_steps=0,
        num_training_steps=num_training_steps,
    )
    logger = None  ## TODO: add logger to track progress

    args.n_epochs = train_epochs
    classifier = MTL_classifier(
        model=model,
        cfg=args,
        criterion=criterion,
        optimizer=optimizer,
        scheduler=scheduler,
        device=args.device,
        logger=logger,
    )
    if args.mode == "train":
        classifier.train(args, trainloader, testloader)

        # use functions from evaluate_utils to test model.


def train_baseline(args):
    model = baseline_model(args=args).to(args.device)
    criterion = torch.nn.CrossEntropyLoss().to(args.device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    trainloader, devloader, testloader = fetch_loaders([args.word_model], args)
    num_training_steps = args.n_epochs * len(trainloader)
    scheduler = get_scheduler(
        "linear",
        optimizer=optimizer,
        num_warmup_steps=0,
        num_training_steps=num_training_steps,
    )
    logger = None
    classifier = baseline_classifier(
        model=model,
        cfg=args,
        criterion=criterion,
        optimizer=optimizer,
        scheduler=scheduler,
        device=args.device,
        logger=logger,
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
        model_dict[model_name] = AutoModel.from_pretrained(
            model_name, num_labels=args.num_labels, cache_dir=args.output_dir
        )
    model = sequential_MTL(model_dict=model_dict, args=args).to(args.device)

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
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay
    )
    # trainloader, devloader, testloader = fetch_loaders(model_names, args) ## TODO: get data
    trainloader, devloader, testloader = fetch_loaders2(model_names, args)
    num_training_steps = args.n_epochs * len(trainloader)
    scheduler = get_scheduler(
        "linear",
        optimizer=optimizer,
        num_warmup_steps=0,
        num_training_steps=num_training_steps,
    )
    logger = None  ## TODO: add logger to track progress

    classifier = sequential_classifier(
        model=model,
        cfg=args,
        criterion=criterion,
        optimizer=optimizer,
        scheduler=scheduler,
        device=args.device,
        logger=logger,
    )
    classifier.JSD = JSD()  ## need to have JSD

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
        model_dict[model_name] = base_model(model_name, args=args).to(args.device)

    criterion = torch.nn.CrossEntropyLoss().to(args.device)
    trainloader, devloader, testloader = fetch_loaders2(model_names, args)
    num_training_steps = args.n_epochs * len(trainloader)
    # criterion = torch.nn.CrossEntropyLoss().to(args.device) ## weight the loss if you wish
    optimizer_dict = dict()
    scheduler_dict = dict()
    for model_name in model_names:
        optimizer = torch.optim.AdamW(
            model_dict[model_name].parameters(),
            lr=args.lr,
            weight_decay=args.weight_decay,
        )
        scheduler = get_scheduler(
            "linear",
            optimizer=optimizer,
            num_warmup_steps=0,
            num_training_steps=num_training_steps,
        )
        optimizer_dict[model_name] = optimizer
        scheduler_dict[model_name] = scheduler

    # trainloader, devloader, testloader = fetch_loaders(model_names, args) ## TODO: get data
    logger = None  ## TODO: add logger to track progress

    classifier = sequential_classifier_2(
        model=model_dict,
        cfg=args,
        criterion=criterion,
        optimizer=optimizer_dict,
        scheduler=scheduler_dict,
        device=args.device,
        logger=logger,
    )
    classifier.prob_loss = torch.nn.MSELoss()  ## need to have JSD

    if args.mode == "train":
        classifier.train(args, trainloader, testloader)

    if args.mode == "test":
        pass
        # use functions from evaluate_utils to test model.

    ## do something to evaluate the model
    ## Note: could evaluate using seqeval in the estimator's _eval() function
    ## need to re-write the _eval() function for token classification.


def train_bimodal_MLM(args, test=False):
    ## initialize model
    ddp_kwargs = DistributedDataParallelKwargs(find_unused_parameters=True)
    accelerator = Accelerator(kwargs_handlers=[ddp_kwargs])
    device = accelerator.device
    args.device = device

    wandb.init(
        # Set the project where this run will be logged
        project="bimodal-MLM",
        # Track hyperparameters and run metadata
        config={
            "learning_rate": args.lr,
            "MLM_epochs": args.mlm_epochs,
    })

    model_dict = torch.nn.ModuleDict()
    model_dict["char"] = AutoModel.from_pretrained(
        args.char_model, cache_dir=args.output_dir
    )
    model_dict["word"] = AutoModel.from_pretrained(
        args.word_model, cache_dir=args.output_dir
    )

    base = bimodal_base(model_dict=model_dict, args=args)
    MLM_model = bimodal_pretrain(base=base, args=args)

    criterion = torch.nn.CrossEntropyLoss()

    ## NOTE: freeze parameters??
    optimizer = torch.optim.AdamW(
        MLM_model.parameters(), lr=args.lr, weight_decay=args.weight_decay
    )

    ## TODO: get loaders
    model_names = args.model_list.split("|")
    trainloader, devloader, testloader = fetch_loader_book_wiki_bimodal(
        model_names, args
    )

    # MLM_model, optimizer, trainloader = accelerator.prepare(MLM_model, optimizer, trainloader)
    ## NOTE: structure of data
    ## data : {"char":  char_data, "word": word_data}
    ## char_data: what returned by char tokenizer + word_id_for_char
    ## word_data: what returned by word tokenizer
    num_training_steps = args.n_epochs * len(trainloader)
    scheduler = get_scheduler(
        "linear",
        optimizer=optimizer,
        num_warmup_steps=0,
        num_training_steps=num_training_steps,
    )
    logger = None  ## TODO: add logger to track progress

    train_epochs = args.n_epochs
    args.n_epochs = args.mlm_epochs
    # args.accelerator = accelerator
    MLM_model, optimizer, trainloader = accelerator.prepare(MLM_model, optimizer, trainloader)
    args.accelerator = accelerator
    MLM_classifier_ = bimodal_trainer(
        model=MLM_model,
        cfg=args,
        criterion=criterion,
        optimizer=optimizer,
        scheduler=scheduler,
        device=args.device,
        logger=logger,
    )

    # MLM_classifier_, optimizer, trainloader = accelerator.prepare(MLM_classifier_, optimizer, trainloader)
    MLM_classifier_.train(args, trainloader, testloader)  ## train MLM

    MLM_classifier_.save(args.output_dir, "MLM_model", force=True)

    ## TODO: evaluate on WNUT 17 and other task
    #####################################################################
    model = bimodal_ner(base=base, args=args).to(args.device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay
    )
    trainloader, devloader, testloader = fetch_loader_wnut(args)
    num_training_steps = args.n_epochs * len(trainloader)
    scheduler = get_scheduler(
        "linear",
        optimizer=optimizer,
        num_warmup_steps=0,
        num_training_steps=num_training_steps,
    )
    logger = None  ## TODO: add logger to track progress

    args.n_epochs = train_epochs
    args.word_model = "word"
    classifier = bimodal_classifier(
        model=model,
        cfg=args,
        criterion=criterion,
        optimizer=optimizer,
        scheduler=scheduler,
        device=args.device,
        logger=logger,
    )
    if args.mode == "train":
        classifier.train(args, trainloader, testloader)

        # use functions from evaluate_utils to test model.
