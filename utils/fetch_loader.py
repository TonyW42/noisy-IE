import sys

sys.path.append("..")
sys.path.append("../..")
from transformers import (
    AutoTokenizer,
)
import torch
from models.model import (
    encode_tags,
    wnut_multiple_granularity,
)
from data.pre_processing.loader_helper import (
    SSTDatasetMulti,
    WNUTDatasetMulti,
    SSTDatasetMulti,
)
from datasets import load_dataset
import os

# from datasets import Dataset
from torch.utils.data import Dataset
from data.pre_processing.collator import (
    custom_collate,
    custom_collate_SST,
    custom_collate_book_wiki_eval,
    custom_collate_book_wiki,
    custom_collate_book_wiki_wrapper,
)
from data.efficient.collator_efficient import (
    clean_text,
    tokenize_bimodal_efficient_eval,
    tokenize_bimodal_efficient,
)
from torch.utils.data import RandomSampler

count = 0


def fetch_loaders(model_names, args):
    wnut = load_dataset("wnut_17", cache_dir=args.output_dir)
    train_encoding_list, train_label_list = [], []
    valid_encoding_list, valid_label_list = [], []
    test_encoding_list, test_label_list = [], []
    for model_name in model_names:
        tokenizer = AutoTokenizer.from_pretrained(
            model_name, add_prefix_space=True, cache_dir=args.output_dir
        )  ## changed here
        xlm_train_encoding = tokenizer(
            wnut["train"]["tokens"],
            padding="longest",
            truncation=True,
            is_split_into_words=True,
        )
        xlm_valid_encoding = tokenizer(
            wnut["validation"]["tokens"],
            padding="longest",
            truncation=True,
            is_split_into_words=True,
        )
        xlm_test_encoding = tokenizer(
            wnut["test"]["tokens"],
            padding="longest",
            truncation=True,
            is_split_into_words=True,
        )

        xlm_train_labels = encode_tags(wnut["train"], xlm_train_encoding)
        xlm_valid_labels = encode_tags(wnut["validation"], xlm_valid_encoding)
        xlm_test_labels = encode_tags(wnut["test"], xlm_test_encoding)

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
    sst = load_dataset("sst", cache_dir=args.output_dir)
    train_encoding_list = []
    valid_encoding_list = []
    test_encoding_list = []
    for model_name in model_names:
        tokenizer = AutoTokenizer.from_pretrained(
            model_name, add_prefix_space=True, cache_dir=args.output_dir
        )  ## changed here
        xlm_train_encoding = tokenizer(
            sst["train"]["sentence"], padding="longest", truncation=True
        )
        xlm_valid_encoding = tokenizer(
            sst["validation"]["sentence"], padding="longest", truncation=True
        )
        xlm_test_encoding = tokenizer(
            sst["test"]["sentence"], padding="longest", truncation=True
        )

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


def fetch_loader_book_wiki(model_names, args):
    """
    To load dataset from bookcorpus and wikitext:
    - Wikitext: ['wikitext-103-v1', 'wikitext-2-v1', 'wikitext-103-raw-v1', 'wikitext-2-raw-v1'];
    - WikiText-2 aims to be of a similar size to the PTB while WikiText-103 contains all articles extracted from Wikipedia.
    """

    """
    Store dataset in local to save time, 
    if detected dataset is already downloaded, load from the disk
    """
    if os.path.isfile("data/train_encoding_book_wiki_c.pickle"):
        with open("data/train_encoding_book_wiki_c.pickle", "rb") as handle:
            train_encoding_list = pickle.load(handle)
        with open("data/valid_encoding_book_wiki_c.pickle", "rb") as handle:
            valid_encoding_list = pickle.load(handle)
        with open("data/test_encoding_book_wiki_c.pickle", "rb") as handle:
            test_encoding_list = pickle.load(handle)
        print(
            "=================== Data Loaded from Local Data Folder ==================="
        )
    else:
        # dataset_bookcorpus = load_dataset("bookcorpus", cache_dir=args.output_dir)
        dataset_wiki = load_dataset(
            "wikitext", "wikitext-2-v1", cache_dir=args.output_dir
        )
        train_encoding_list = []
        valid_encoding_list = []
        test_encoding_list = []
        for model_name in model_names:
            tokenizer = AutoTokenizer.from_pretrained(
                model_name, add_prefix_space=True, cache_dir=args.output_dir
            )  ## changed here
            xlm_train_encoding = tokenizer(
                dataset_wiki["train"]["text"],
                padding="longest",
                truncation=True,
            )
            xlm_valid_encoding = tokenizer(
                dataset_wiki["validation"]["text"],
                padding="longest",
                truncation=True,
            )
            xlm_test_encoding = tokenizer(
                dataset_wiki["validation"]["text"],
                padding="longest",
                truncation=True,
            )

            train_encoding_list.append(xlm_train_encoding)
            valid_encoding_list.append(xlm_valid_encoding)
            test_encoding_list.append(xlm_test_encoding)
        # store dataset
        with open("data/train_encoding_book_wiki_c.pickle", "wb") as handle:
            pickle.dump(xlm_train_encoding, handle, protocol=pickle.HIGHEST_PROTOCOL)
        with open("data/valid_encoding_book_wiki_c.pickle", "wb") as handle:
            pickle.dump(xlm_train_encoding, handle, protocol=pickle.HIGHEST_PROTOCOL)
        with open("data/test_encoding_book_wiki_c.pickle", "wb") as handle:
            pickle.dump(xlm_train_encoding, handle, protocol=pickle.HIGHEST_PROTOCOL)
        print("=================== Data Loaded from HuggingFace ===================")

    data_train = SSTDatasetMulti(train_encoding_list, model_names)
    data_valid = SSTDatasetMulti(valid_encoding_list, model_names)
    data_test = SSTDatasetMulti(test_encoding_list, model_names)
    # print(len(data_train))
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
    wnut = load_dataset("wnut_17", cache_dir=args.output_dir)
    data_loader = wnut_multiple_granularity(wnut, args)
    # model_main = data_loader.model_dict
    tokenized_wnut_main = data_loader.data_

    train_encoding_list, train_label_list = [], []
    valid_encoding_list, valid_label_list = [], []
    test_encoding_list, test_label_list = [], []

    for model_name in model_names:
        train_encoding_list.append(
            {
                "input_ids": tokenized_wnut_main[model_name]["train"]["input_ids"],
                "attention_mask": tokenized_wnut_main[model_name]["train"][
                    "attention_mask"
                ],
            }
        )
        train_label_list.append(tokenized_wnut_main[model_name]["train"]["labels"])
        valid_encoding_list.append(
            {
                "input_ids": tokenized_wnut_main[model_name]["validation"]["input_ids"],
                "attention_mask": tokenized_wnut_main[model_name]["validation"][
                    "attention_mask"
                ],
            }
        )
        valid_label_list.append(tokenized_wnut_main[model_name]["validation"]["labels"])
        test_encoding_list.append(
            {
                "input_ids": tokenized_wnut_main[model_name]["test"]["input_ids"],
                "attention_mask": tokenized_wnut_main[model_name]["test"][
                    "attention_mask"
                ],
            }
        )
        test_label_list.append(tokenized_wnut_main[model_name]["test"]["labels"])

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


def fetch_loader_wnut(args):
    """
    To load dataset from bookcorpus and wikitext:
    - Wikitext: ['wikitext-103-v1', 'wikitext-2-v1', 'wikitext-103-raw-v1', 'wikitext-2-raw-v1'];
    - WikiText-2 aims to be of a similar size to the PTB while WikiText-103 contains all articles extracted from Wikipedia.
    """

    """
    Store dataset in local to save time, 
    if detected dataset is already downloaded, load from the disk
    """
    dataset_wnut = load_dataset("wnut_17", cache_dir=args.output_dir)

    word_tokenizer = AutoTokenizer.from_pretrained(
        args.word_model, cache_dir=args.output_dir, add_prefix_space=True
    )
    char_tokenizer = AutoTokenizer.from_pretrained(
        args.char_model, cache_dir=args.output_dir, add_prefix_space=True
    )
    data_train = BookWikiDatasetMulti_efficient_eval(
        dataset_wnut["train"], char_tokenizer, word_tokenizer, args
    )
    data_valid = BookWikiDatasetMulti_efficient_eval(
        dataset_wnut["validation"], char_tokenizer, word_tokenizer, args
    )
    data_test = BookWikiDatasetMulti_efficient_eval(
        dataset_wnut["test"], char_tokenizer, word_tokenizer, args
    )
    loader_train = torch.utils.data.DataLoader(
        data_train,
        batch_size=args.train_batch_size,
        collate_fn=custom_collate_book_wiki_eval,
    )
    loader_valid = torch.utils.data.DataLoader(
        data_valid,
        batch_size=args.train_batch_size,
        collate_fn=custom_collate_book_wiki_eval,
    )
    loader_test = torch.utils.data.DataLoader(
        data_test,
        batch_size=args.train_batch_size,
        collate_fn=custom_collate_book_wiki_eval,
    )
    return loader_train, loader_valid, loader_test


def fetch_loader_book_wiki_bimodal(model_names, args):
    """
    To load dataset from bookcorpus and wikitext:
    - Wikitext: ['wikitext-103-v1', 'wikitext-2-v1', 'wikitext-103-raw-v1', 'wikitext-2-raw-v1'];
    - WikiText-2 aims to be of a similar size to the PTB while WikiText-103 contains all articles extracted from Wikipedia.
    """

    """
    Store dataset in local to save time, 
    if detected dataset is already downloaded, load from the disk
    """
    dataset_bookcorpus = load_dataset("bookcorpus", cache_dir=args.output_dir)
    dataset_wiki = load_dataset(
        "wikitext", "wikitext-103-v1", cache_dir=args.output_dir
    )

    word_tokenizer = AutoTokenizer.from_pretrained(
        args.word_model, cache_dir=args.output_dir
    )  ## changed here
    char_tokenizer = AutoTokenizer.from_pretrained(
        args.char_model, cache_dir=args.output_dir
    )

    char_tokenizer.unk_token_id = 256
    char_tokenizer.cls_token_id = 257
    char_tokenizer.sep_token_id = 258

    data_full = dataset_wiki["train"]["text"] + dataset_bookcorpus["train"]["text"]

    data_train = BookWikiDatasetMulti_efficient(
        data_full[: len(data_full)],
        char_tokenizer,
        word_tokenizer,
        args,
    )

    loader_train = torch.utils.data.DataLoader(
        data_train,
        batch_size=args.train_batch_size,
        shuffle=True,
        collate_fn=custom_collate_book_wiki_wrapper,
    )
    return loader_train, None, None


## NOTE: efficient version of dataset
## Tokenize "on the fly"
class BookWikiDatasetMulti_efficient(Dataset):
    def __init__(self, text, char_tokenizer, word_tokenizer, args):
        self.text = text
        self.char_tokenizer = char_tokenizer
        self.word_tokenizer = word_tokenizer
        self.args = args

    def __len__(self):
        return len(self.text)

    def __getitem__(self, idx):
        return tokenize_bimodal_efficient(
            self.text[idx], self.char_tokenizer, self.word_tokenizer, self.args
        )


class BookWikiDatasetMulti_efficient_eval(Dataset):
    def __init__(self, text, char_tokenizer, word_tokenizer, args):
        self.text = text
        self.char_tokenizer = char_tokenizer
        self.word_tokenizer = word_tokenizer
        self.args = args

    def __len__(self):
        return len(self.text)

    def __getitem__(self, idx):
        return tokenize_bimodal_efficient_eval(
            self.text[idx], self.char_tokenizer, self.word_tokenizer, self.args, idx
        )


def tokenize_bimodal(text, char_tokenizer, word_tokenizer, args):
    """
    input:
        text: input text type: str
        char_tokenizer: character tokenizer
        word_tokenizer: word tokenizer
    output:
        result : dict {"word" : word_tokenized,
                       "char" : char_tokenized,
                       "char_ids" :  the #word the character belongs to. Same length as character input_ids
                       }
    """
    global count
    text = clean_text(text)
    ## NOTE: change padding type and custom collator
    if len(text):
        char_tokenized = char_tokenizer(text, padding=True, truncation=True)
        word_tokenized = word_tokenizer(text, padding=True, truncation=True)
        char_ids = []

        char_list = char_tokenizer.tokenize(text)
        word_list = word_tokenizer.tokenize(text)

        if "xlm" in args.word_model:
            char_list.insert(0, " ")

        current_word_id = 0
        for word in word_list:
            char_ids.extend([current_word_id for i in range(len(word))])
            current_word_id += 1
        if "xlm" in args.word_model:
            char_ids[0] = -100
        else:
            char_ids.insert(0, -100)
        ## if not truncated, then there is [SEP] token. append -100
        # if char_tokenized["input_ids"][-1] == char_tokenizer.sep_token_id:
        char_ids.append(-100)  ## [CLS] and [SEP] token should not be aligned
        max_len = char_tokenizer.model_max_length
        ## if too long, truncate and set last one to -100
        if len(char_ids) > max_len:
            char_ids = char_ids[:max_len]
            char_ids[-1] = -100

        if len(char_ids) == len(char_tokenized["input_ids"]):
            return {
                "char": char_tokenized,
                "word": word_tokenized,
                "char_word_ids": char_ids,
            }
        else:
            count += 1
            return None
