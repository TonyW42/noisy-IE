import torch
from models.info import encode_tag_each


def clean_text(x):
    if isinstance(x, str):
        x = x.replace("<unk>", "")
        x = " ".join(x.split())
    return x


def clean_tokenized_text(x_list):
    for i in range(len(x_list)):
        if x_list[i] == 57344:
            x_list[i] = 256
        elif x_list[i] == 57345:
            x_list[i] = 257
    return x_list


def tokenize_bimodal_efficient_eval(data, char_tokenizer, word_tokenizer, args, idx):
    text = data["tokens"]
    text = clean_text(text)
    char_tokenized = char_tokenizer(
        text,
        padding=True,
        truncation=True,
        is_split_into_words=True,
    )
    word_tokenized = word_tokenizer(
        text,
        padding=True,
        truncation=True,
        is_split_into_words=True,
    )
    word_labels = encode_tag_each(data, word_tokenized, idx)

    word_tokenized["labels"] = word_labels
    char_tokenized["labels"] = char_tokenized["token_type_ids"]
    return {
        "char": char_tokenized,
        "word": word_tokenized,
    }


def tokenize_bimodal_efficient(text, char_tokenizer, word_tokenizer, args):
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
    text = clean_text(text)
    char_tokenized = char_tokenizer(text, padding=True, truncation=True)
    word_tokenized = word_tokenizer(text, padding=True, truncation=True)
    if len(text):
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
        char_ids = clean_tokenized_text(char_ids)
        char_tokenized["input_ids"] = clean_tokenized_text(char_tokenized["input_ids"])
        if len(char_ids) == len(char_tokenized["input_ids"]):
            result = dict()
            for key in char_tokenized:
                result[f"char_{key}"] = char_tokenized[key]
            for key in word_tokenized:
                result[f"word_{key}"] = word_tokenized[key]
            result["char_word_ids"] = char_ids

            return result
    ## if not match then return empty set. Collator should padd empty set to max len
    result = dict()
    for key in char_tokenized:
        result[f"char_{key}"] = torch.tensor([], dtype=torch.long)
    for key in word_tokenized:
        result[f"word_{key}"] = torch.tensor([], dtype=torch.long)
    result["char_word_ids"] = torch.tensor([], dtype=torch.long)
    return result
