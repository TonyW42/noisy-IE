import torch 
from torch import nn
from utils.model_utils import *
import evaluate 
from transformers import AutoModel
from models.model import *
from transformers import AutoModel, AutoModel, AutoTokenizer, DataCollatorForTokenClassification, get_scheduler
from torch.utils.data import RandomSampler, DataLoader, Dataset
from datasets import load_dataset


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
        char_tokenized = self.char_tokenizer(text, padding=True, truncation=True)
        word_tokenized = self.word_tokenizer(text, padding=True, truncation=True)
        result["char"] = char_tokenized
        result["word"] = word_tokenized
        result["label"] = self.data[idx]["label"]
        return result

def process_wnut_2020_task2(data_dir):
    data = pd.read_csv(data_dir, header = None, sep = "\t", 
                       names = ["id", "text", "label"])
    result = []
    for i in range(len(a)):
        result_tmp = dict()
        result_tmp["text"] = data["text"][i]
        result["label_str"] = data["label"][i]
        result["label"] = 0 if result["label_str"] == "UNINFORMATIVE" else 1
        result.append(result_tmp)
    return result

def process_semeval18_task1(data):
    emotions = [
        "anger", 
        "anticipation",
        "disgust",
        "fear", 
        "joy",
        "love",
        "optimism",
        "pessimism",
        "sadness",
        "surprise",
        "trust"
        ]
    data = data.rename_column("Tweet", "text")
    label = []
    for i in range(len(data)):
        label_sample = [1 for emo in emotions if data[i][emo] else 0]
        label.append(label_sample)
    data.add_column("label", label)
    return data


def fetch_classification_loader(dataset_name, char_tokenizer, word_tokenizer):
    data = None ## TODO: add data 
    if "tweeteval" in dataset_name:
        data = load_dataset("tweet_eval", dataset_name.split("_")[-1])   
        train_split = data["train"]
        val_split = data["validation"]
        test_split = data["test"]
    if dataset_name == "wnut20_task2":
        ## NOTE: if error here just change the data_dir
        train_split = process_wnut_2020_task2("data/wnut20_task2/train.tsv")
        val_split = process_wnut_2020_task2("data/wnut20_task2/valid.tsv")
        test_split = process_wnut_2020_task2("data/wnut20_task2/test.tsv")
    if dataset_name == "semeval18_task1":
        data = load_dataset("sem_eval_2018_task_1", "subtask5.english")
        train_split = process_semeval18_task1(data["train"])
        val_split = process_semeval18_task1(data["validation"])
        test_split = process_semeval18_task1(data["test"])
        

    train_dataset = classification_dataset(train_split, char_tokenizer, word_tokenizer, args)
    val_dataset = classification_dataset(val_split, char_tokenizer, word_tokenizer, args)
    test_dataset = classification_dataset(test_split, char_tokenizer, word_tokenizer, args)

    train_loader = DataLoader(train_dataset, batch_size=args.train_batch_size)
    val_loader = DataLoader(train_dataset, batch_size=args.train_batch_size)
    test_loader = DataLoader(train_dataset, batch_size=args.train_batch_size)

    return train_loader, val_loader, test_loader
    
    

class model_for_classificaton(nn.Module):
    def __init__(self, base_model, args):
        self.base_model = base_model
        self.args = args
        self.classificaton_head = nn.Linear(args.emb_size, args.num_labels)
    
    def forward(self, data):
        encoded = self.base_model(data)
        cls_emb = encoded["word"][:, 0, :] ## NOTE: this works for Bimodal w/ Roberta, not roberta-base
        ## if train roberta-base, should do sth like 
        ## cls_emb = encoded["pooler_output"]
        logits = self.classificaton_head(cls_emb)
        return logits
        

class classification_trainer(BaseEstimator):
    def setp(self, data):
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
            preds.extend(pred.toist())
        
        f1 = f1_metric(predictions = preds, references = labels)
        acc = acc_metric(predictions = preds, references = labels)
        precision = precision_metric(predictions = preds, references = labels)
        recall = recall_metric(predictions = preds, references = labels)

        # TODO: add these to W and B!

class multi_classification_trainer(BaseEstimator):
    def setp(self, data):
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
            preds.extend(pred.toist())
        
        f1 = f1_metric(predictions = preds, references = labels)
        acc = acc_metric(predictions = preds, references = labels)
        precision = precision_metric(predictions = preds, references = labels)
        recall = recall_metric(predictions = preds, references = labels)

        # TODO: add these to W and B!



def train_classification_model(args):
    model_dict = nn.ModuleDict()
    model_dict["char"] = AutoModel.from_pretrained(
        args.char_model, cache_dir=args.output_dir
    )
    model_dict["word"] = AutoModel.from_pretrained(
        args.word_model, cache_dir=args.output_dir
    )

    base = bimodal_base(model_dict=model_dict, args=args)

    model = model_for_classificaton(base_model = base, args = args).to(args.device)

    criterion = torch.nn.CrossEntropyLoss()

    ## NOTE: freeze parameters??
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay
    )

    ## TODO: get loaders
    # model_names = args.model_list.split("|")
    # model_names = ["roberta-base", "google/canine-s"]
    word_tokenizer = AutoTokenizer.from_pretrained(args.word_model)
    char_tokenizer = AutoTokenizer.from_pretrained(args.char_model)
    trainloader, devloader, testloader = fetch_classification_loader(
        args.dataset_name, char_tokenizer, word_tokenizer, args
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
    classifier_ = bimodal_trainer(
        model=model,
        cfg=args,
        criterion=criterion,
        optimizer=optimizer,
        scheduler=scheduler,
        device=args.device,
        logger=logger,
    )
    classifier_.train(args, trainloader, testloader)  ## train MLM




        










