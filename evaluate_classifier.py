import torch 
from torch import nn
from utils.model_utils import *
import evaluate 
from transformers import AutoModel
from models.model import *
from transformers import AutoModel, AutoModel, AutoTokenizer, DataCollatorForTokenClassification, get_scheduler
from torch.utils.data import RandomSampler, DataLoader, Dataset
from datasets import load_dataset
import wandb

class classification_dataset(Dataset):
    def __init__(self, data, char_tokenizer, word_tokenizer, args):
        self.data = data
        self.char_tokenizer = char_tokenizer
        self.word_tokenizer = word_tokenizer
        self.args = args

    def __len__(self):
        return len(self.data) ## TODO: check this! 

    def __getitem__(self, idx):
        text = self.data[idx]["text"]
        result = dict()
        char_tokenized = self.char_tokenizer(text, 
                                             padding='max_length')
        word_tokenized = self.word_tokenizer(text, 
                                             padding='max_length')
        for key in char_tokenized:
            result[f"char_{key}"] = torch.tensor(char_tokenized[key])
        for key in word_tokenized:
            result[f"word_{key}"] = torch.tensor(word_tokenized[key])
        result["label"] = self.data[idx]["label"]
        return result

def fetch_classification_loader(dataset_name, char_tokenizer, word_tokenizer, args):
    data = None ## TODO: add data 
    if "tweeteval" in dataset_name:
        data = load_dataset("tweet_eval", dataset_name.split("-")[-1])

    train_split = data["train"]
    val_split = data["validation"]
    test_split = data["test"]

    train_dataset = classification_dataset(train_split, char_tokenizer, word_tokenizer, args)
    val_dataset = classification_dataset(val_split, char_tokenizer, word_tokenizer, args)
    test_dataset = classification_dataset(test_split, char_tokenizer, word_tokenizer, args)

    train_loader = DataLoader(train_dataset, batch_size=args.train_batch_size, shuffle = True)
    val_loader = DataLoader(val_dataset, batch_size=args.train_batch_size, shuffle = True)
    test_loader = DataLoader(test_dataset, batch_size=args.train_batch_size, shuffle = True)

    return train_loader, val_loader, test_loader



class model_for_classificaton(nn.Module):
    def __init__(self, base_model, args):
        super().__init__()
        self.base_model = base_model
        self.args = args
        self.classificaton_head = nn.Linear(args.emb_size, args.num_labels)
        self.pooling_dense = nn.Linear(args.emb_size, args.emb_size)
        self.pooling_act = nn.Tanh()

    def forward(self, data):
        if self.args.model_type == "base":
            return self.forward_base(data)
        if self.args.model_type == "bimodal":
            return self.forward_bimodal(data)
    
    def forward_base(self, data):
        encoded = self.base_model(
            input_ids = data["word_input_ids"].to(self.args.device),
            attention_mask = data["word_attention_mask"].to(self.args.device)
                                  )
        cls_emb = encoded["last_hidden_state"][:, 0, :] ## NOTE: this works for Bimodal w/ Roberta, not roberta-base
        cls_emb = self.pooling(cls_emb)
        ## if train roberta-base, should do sth like 
        ## cls_emb = encoded["pooler_output"]
        logits = self.classificaton_head(cls_emb)
        return logits

    def forward_bimodal(self, data):
        encoded = self.base_model(data)
        cls_emb = encoded["word"][:, 0, :] ## NOTE: this works for Bimodal w/ Roberta, not roberta-base
        cls_emb = self.pooling(cls_emb)
        ## if train roberta-base, should do sth like 
        ## cls_emb = encoded["pooler_output"]
        logits = self.classificaton_head(cls_emb)
        return logits
    
    def pooling(self, hidden_states):
        pooled_output = self.pooling_dense(hidden_states)
        pooled_output = self.pooling_act(pooled_output)
        return pooled_output









class classification_trainer(BaseEstimator):
    def step(self, data):
        self.optimizer.zero_grad()
        logits = self.model(data=data)
        loss = self.criterion(logits, data["label"].to(self.device))
        if self.mode == "train":
            # self.args.accelerator.backward(loss)
            loss.backward()
            self.optimizer.step()
            if self.scheduler is not None:
                self.scheduler.step()
        return {
            "loss": loss.detach().cpu().item(),
            "logits": logits.detach().cpu()
        }
    

    def _eval(self, evalloader):
        self.model.eval()
        tbar = tqdm(evalloader, dynamic_ncols=True)
        labels, preds = [], []

        f1_metric = evaluate.load("f1")
        acc_metric = evaluate.load("accuracy")
        precision_metric = evaluate.load("precision")
        recall_metric = evaluate.load("recall")

        for data in tbar:
            ret_step = self.step(data) 
            loss, logits, label = ret_step["loss"], ret_step["logits"], data["label"]
            pred = torch.argmax(logits, dim = -1)
            labels.extend(label.tolist())
            preds.extend(pred.tolist())

        f1 = f1_metric.compute(predictions = preds, references = labels, average="macro")
        acc = acc_metric.compute(predictions = preds, references = labels)
        precision = precision_metric.compute(predictions = preds, references = labels, average="macro")
        recall = recall_metric.compute(predictions = preds, references = labels, average="macro")

        # TODO: add these to W and B!
        print(f"Loss: {loss}, F1: {f1}, Acc: {acc}, Precision: {precision}, Recall: {recall}")
        wandb.log({"loss": loss, "f1": f1, "acc": acc, "precision": precision, "recall": recall})

def train_classification_model(args):
    wandb.init(
        # Set the project where this run will be logged
        project=args.dataset,
        # Track hyperparameters and run metadata
        config={
            "learning_rate": args.lr,
            "n_epoch": args.n_epochs,
            "batch_size": args.train_batch_size,
    })

    model_dict = nn.ModuleDict()
    model_dict["char"] = AutoModel.from_pretrained(
        args.char_model, cache_dir=args.output_dir
    )
    model_dict["word"] = AutoModel.from_pretrained(
        args.word_model, cache_dir=args.output_dir
    )
    args.model_type = "bimodal"
    ## NOTE: change model type here
    if args.model_type == "bimodal":
        base = bimodal_base(model_dict=model_dict, args=args)
    if args.model_type == "base":
        base = model_dict["word"]

    model = model_for_classificaton(base_model = base, args = args)
    model = model.to(args.device)

    criterion = torch.nn.CrossEntropyLoss()

    ## NOTE: freeze parameters??
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay
    )

    ## TODO: get loaders
    # model_names = args.model_list.split("|")
    # model_names = ["roberta-base", "google/canine-s"]
    word_tokenizer = AutoTokenizer.from_pretrained(args.word_model, cache_dir=args.output_dir, add_prefix_space=True)
    char_tokenizer = AutoTokenizer.from_pretrained(args.char_model, cache_dir=args.output_dir, add_prefix_space=True)
    trainloader, devloader, testloader = fetch_classification_loader(
        args.dataset, char_tokenizer, word_tokenizer, args
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
    # args.n_epochs = args.mlm_epochs
    # args.accelerator = accelerator
    classifier_ = classification_trainer(
        model=model,
        cfg=args,
        criterion=criterion,
        optimizer=optimizer,
        scheduler=scheduler,
        device=args.device,
        logger=logger,
    )
    classifier_.train(args, trainloader, testloader)  ## train MLM

    wandb.finish()

