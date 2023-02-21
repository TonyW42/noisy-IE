from torch.nn.utils.rnn import pad_sequence  # (1)
from collections import defaultdict
from transformers import (
    AutoModel,
    AutoTokenizer,
    DataCollatorForTokenClassification,
    get_scheduler,
    AutoModelForMaskedLM,
)
import torch
from models.model import (
    SSTDatasetMulti,
    BookWikiDatasetMulti,
    encode_tags,
    WNUTDatasetMulti,
    wnut_multiple_granularity,
)
from datasets import load_dataset
import os
import pickle


def custom_collate(data, seq_len=512):  # (2)
    model_names = list(data[0].keys())
    batch_size = len(data)
    input_ids = []
    labels = []
    attention_mask = []
    for m_name in model_names:
        for i in range(batch_size):
            input_ids.append(data[i][m_name]["input_ids"][:seq_len])
    for m_name in model_names:
        for i in range(batch_size):
            attention_mask.append(data[i][m_name]["attention_mask"][:seq_len])
    for m_name in model_names:
        for i in range(batch_size):
            labels.append(data[i][m_name]["labels"][:seq_len])
    input_ids = pad_sequence(input_ids, batch_first=True, padding_value=1)
    labels = pad_sequence(labels, batch_first=True, padding_value=-100)
    attention_mask = pad_sequence(attention_mask, batch_first=True, padding_value=0)

    return_dict = defaultdict(defaultdict)
    for i, m_name in enumerate(model_names):
        return_dict[m_name]["input_ids"] = input_ids[
            i * batch_size : (i + 1) * batch_size
        ]
        return_dict[m_name]["labels"] = labels[i * batch_size : (i + 1) * batch_size]
        return_dict[m_name]["attention_mask"] = attention_mask[
            i * batch_size : (i + 1) * batch_size
        ]
    return return_dict


def custom_collate_SST(data, seq_len=512, probability=0.15):  # (2)
    model_names = list(data[0].keys())
    batch_size = len(data)
    input_ids = []
    labels = []
    attention_mask = []
    for m_name in model_names:
        for i in range(batch_size):
            input_ids.append(data[i][m_name]["input_ids"][:seq_len])
    for m_name in model_names:
        for i in range(batch_size):
            attention_mask.append(data[i][m_name]["attention_mask"][:seq_len])
    input_ids = pad_sequence(input_ids, batch_first=True, padding_value=1)
    labels = input_ids.clone()
    attention_mask = pad_sequence(attention_mask, batch_first=True, padding_value=0)

    rand = torch.rand(input_ids.shape)
    # where the random array is less than 0.15, we set true
    mask_arr = rand < probability
    mask_arr = mask_arr * (input_ids != -100)

    selection = []

    for i in range(input_ids.shape[0]):
        selection.append(torch.flatten(mask_arr[i].nonzero()).tolist())

    for i in range(input_ids.shape[0]):
        labels[i, selection[i]] = -100

    return_dict = defaultdict(defaultdict)
    for i, m_name in enumerate(model_names):
        return_dict[m_name]["input_ids"] = input_ids[
            i * batch_size : (i + 1) * batch_size
        ]
        return_dict[m_name]["labels"] = labels[i * batch_size : (i + 1) * batch_size]
        return_dict[m_name]["attention_mask"] = attention_mask[
            i * batch_size : (i + 1) * batch_size
        ]

    return return_dict


def fetch_loaders(model_names, args):
    wnut = load_dataset("wnut_17")
    train_encoding_list, train_label_list = [], []
    valid_encoding_list, valid_label_list = [], []
    test_encoding_list, test_label_list = [], []
    for model_name in model_names:
        tokenizer = AutoTokenizer.from_pretrained(
            model_name, add_prefix_space=True
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
    sst = load_dataset("sst")
    train_encoding_list = []
    valid_encoding_list = []
    test_encoding_list = []
    for model_name in model_names:
        tokenizer = AutoTokenizer.from_pretrained(
            model_name, add_prefix_space=True
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
    if os.path.isfile("data/train_encoding_book_wiki.pickle"):
        with open("train_encoding_book_wiki.pickle", "rb") as handle:
            train_encoding_list = pickle.load(handle)
        with open("valid_encoding_book_wiki.pickle", "rb") as handle:
            valid_encoding_list = pickle.load(handle)
        with open("test_encoding_book_wiki.pickle", "rb") as handle:
            test_encoding_list = pickle.load(handle)
        print(
            "=================== Data Loaded from Local Data Folder ==================="
        )
    else:
        dataset_bookcorpus = load_dataset("bookcorpus")
        dataset_wiki = load_dataset("wikitext", "wikitext-2-v1")
        train_encoding_list = []
        valid_encoding_list = []
        test_encoding_list = []
        for model_name in model_names:
            tokenizer = AutoTokenizer.from_pretrained(
                model_name, add_prefix_space=True
            )  ## changed here
            xlm_train_encoding = tokenizer(
                dataset_bookcorpus["train"]["text"] + dataset_wiki["train"]["text"],
                padding="longest",
                truncation=True,
            )
            xlm_valid_encoding = tokenizer(
                dataset_bookcorpus["validation"]["text"]
                + dataset_wiki["validation"]["text"],
                padding="longest",
                truncation=True,
            )
            xlm_test_encoding = tokenizer(
                dataset_bookcorpus["test"]["text"] + dataset_wiki["validation"]["text"],
                padding="longest",
                truncation=True,
            )

            train_encoding_list.append(xlm_train_encoding)
            valid_encoding_list.append(xlm_valid_encoding)
            test_encoding_list.append(xlm_test_encoding)
        # store dataset
        with open("train_encoding_book_wiki.pickle", "wb") as handle:
            pickle.dump(xlm_train_encoding, handle, protocol=pickle.HIGHEST_PROTOCOL)
        with open("valid_encoding_book_wiki.pickle", "wb") as handle:
            pickle.dump(xlm_train_encoding, handle, protocol=pickle.HIGHEST_PROTOCOL)
        with open("test_encoding_book_wiki.pickle", "wb") as handle:
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
    wnut = load_dataset("wnut_17")
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


def fetch_loader_book_wiki_bimodal(model_names, args, test):
    """
    To load dataset from bookcorpus and wikitext:
    - Wikitext: ['wikitext-103-v1', 'wikitext-2-v1', 'wikitext-103-raw-v1', 'wikitext-2-raw-v1']; 
    - WikiText-2 aims to be of a similar size to the PTB while WikiText-103 contains all articles extracted from Wikipedia. 
    """

    """
    Store dataset in local to save time, 
    if detected dataset is already downloaded, load from the disk
    """

    def convert_char(xlm_train_encoding):
        encoded = defaultdict(list)
        for input_id, att_mask in zip(
            xlm_train_encoding["input_ids"], xlm_train_encoding["attention_mask"]
        ):
            each_i, each_a = [], []
            for each_input_id, each_att_mask in zip(input_id, att_mask):
                if each_input_id not in range(0, 4):
                    original_word = tokenizer.decode([each_input_id]).strip()
                    length = len(original_word)
                else:
                    length = 1
                each_i.append(each_input_id * length)
                each_a.append(each_att_mask * length)

            encoded["input_ids"].append(each_i)
            encoded["attention_mask"].append(each_a)
        return encoded

    if os.path.isfile("data/train_encoding_book_wiki.pickle"):
        with open("data/train_encoding_book_wiki.pickle", "rb") as handle:
            train_encoding_list = pickle.load(handle)
        print(
            "=================== Data Loaded from Local Data Folder ==================="
        )
    else:
        dataset_bookcorpus = load_dataset("bookcorpus")
        dataset_wiki = load_dataset("wikitext", "wikitext-2-v1")

        train_encoding_list = []

        model_name = (
            model_names[0] if "canine" not in model_names[0] else model_names[1]
        )

        tokenizer = AutoTokenizer.from_pretrained(
            model_name, add_prefix_space=True, cache_dir=args.output_dir
        )  ## changed here
        if test:
            xlm_train_encoding = tokenizer(
                dataset_bookcorpus["train"]["text"][:10],
                padding="longest",
                truncation=True,
            )
        else:
            xlm_train_encoding = tokenizer(
                dataset_bookcorpus["train"]["text"] + dataset_wiki["train"]["text"],
                padding="longest",
                truncation=True,
            )

        train_encoding_list.append(
            {"word": xlm_train_encoding, "char": convert_char(xlm_train_encoding)}
        )
        # store dataset
        with open("data/train_encoding_book_wiki.pickle", "wb") as handle:
            pickle.dump(train_encoding_list, handle, protocol=pickle.HIGHEST_PROTOCOL)

        print("=================== Data Loaded from HuggingFace ===================")

    data_train = BookWikiDatasetMulti(train_encoding_list, model_names)
    loader_train = torch.utils.data.DataLoader(
        data_train, batch_size=args.train_batch_size, collate_fn=custom_collate_SST
    )
    return loader_train, None, None
