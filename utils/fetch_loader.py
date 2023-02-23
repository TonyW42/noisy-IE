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


def custom_collate_book_wiki(data, seq_len=512, probability=0.15):
    ### TODO: random mask by probability given
    model_names = ["word", "char"]
    batch_size = len(data)

    return_dict = defaultdict(defaultdict)

    for m_name in model_names:
        input_ids = []
        attention_mask = []
        char_word_id = []
        for i in range(batch_size):
            input_ids.append(torch.tensor(data[i][m_name]["input_ids"][:seq_len]))
            attention_mask.append(
                torch.tensor(data[i][m_name]["attention_mask"][:seq_len])
            )
            char_word_id.append(torch.tensor(data[i]["char_word_ids"][:seq_len]))

        input_ids = pad_sequence(input_ids, batch_first=True, padding_value=0) # why pad_value = -100 doesn't work
        attention_mask = pad_sequence(attention_mask, batch_first=True, padding_value=0)
        char_word_id = pad_sequence(char_word_id, batch_first=True, padding_value=-100)

        rand = torch.rand(input_ids.shape)
        # where the random array is less than 0.15, we set true
        mask_arr = rand < probability
        mask_arr = mask_arr * (input_ids != 0)

        selection = []
        for i in range(input_ids.shape[0]):
            selection.append(torch.flatten(mask_arr[i].nonzero()).tolist())
        for i in range(input_ids.shape[0]):
            input_ids[i, selection[i]] = 0

        return_dict[m_name]["input_ids"] = input_ids
        return_dict[m_name]["attention_mask"] = attention_mask
        return_dict["char_word_ids"] = char_word_id.clone().detach()

    return return_dict


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
        dataset_bookcorpus = load_dataset("bookcorpus", cache_dir=args.output_dir)
        dataset_wiki = load_dataset("wikitext", "wikitext-2-v1", cache_dir=args.output_dir)
        train_encoding_list = []
        valid_encoding_list = []
        test_encoding_list = []
        for model_name in model_names:
            tokenizer = AutoTokenizer.from_pretrained(
                model_name, add_prefix_space=True, cache_dir=args.output_dir
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
    if os.path.isfile("data/train_encoding_book_wiki.pickle"):
        with open("data/train_encoding_book_wiki.pickle", "rb") as handle:
            train_encoding_list = pickle.load(handle)
        print(
            "=================== Data Loaded from Local Data Folder ==================="
        )
    else:
        dataset_bookcorpus = load_dataset("bookcorpus", cache_dir=args.output_dir)
        dataset_wiki = load_dataset("wikitext", "wikitext-2-v1", cache_dir=args.output_dir)

        train_encoding_list = []

        word_tokenizer = AutoTokenizer.from_pretrained(
            args.word_model, cache_dir=args.output_dir
        )  ## changed here
        char_tokenizer = AutoTokenizer.from_pretrained(
            args.char_model, cache_dir=args.output_dir
        )
        if test:
            for each_data in dataset_bookcorpus["train"]["text"][:10]:
                train_encoding_list.append(
                    tokenize_bimodal(each_data, char_tokenizer, word_tokenizer)
                )
                print("====================")
                print(train_encoding_list[-1])
        else:
            for each_data in (
                dataset_bookcorpus["train"]["text"] + dataset_wiki["train"]["text"]
            ):
                train_encoding_list.append(
                    tokenize_bimodal(each_data, char_tokenizer, word_tokenizer)
                )

        # store dataset
        with open("data/train_encoding_book_wiki.pickle", "wb") as handle:
            pickle.dump(train_encoding_list, handle, protocol=pickle.HIGHEST_PROTOCOL)

        print("=================== Data Loaded from HuggingFace ===================")

    data_train = BookWikiDatasetMulti(train_encoding_list, model_names)
    loader_train = torch.utils.data.DataLoader(
        data_train,
        batch_size=args.train_batch_size,
        collate_fn=custom_collate_book_wiki,
    )
    return loader_train, None, None


def tokenize_bimodal(text, char_tokenizer, word_tokenizer):
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
    ## NOTE: change padding type and custom collator
    char_tokenized = char_tokenizer(text, padding=True, truncation=True)
    word_tokenized = word_tokenizer(text, padding=True, truncation=True)
    char_ids = []  ## NOTE: should we match CLS tokens to each?

    char_list = char_tokenizer.tokenize(text)
    word_list = word_tokenizer.tokenize(text)

    current_word_id = 0
    for word in word_list:
        char_ids.extend([current_word_id for i in range(len(word))])
        current_word_id += 1
    char_ids.insert(0, -100)
    ## if not truncated, then there is [SEP] token. append -100
    if char_tokenized["input_ids"][-1] == char_tokenizer.sep_token_id:
        char_ids.append(-100)  ## [CLS] and [SEP] token should not be aligned

    assert len(char_ids) == len(char_tokenized["input_ids"])

    return {"char": char_tokenized, "word": word_tokenized, "char_word_ids": char_ids}
