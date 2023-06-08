from torch.nn.utils.rnn import pad_sequence  # (1)
from collections import defaultdict
import torch


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


def custom_collate_book_wiki_wrapper(data, seq_len=512, probability=0.15):
    result = custom_collate_book_wiki(data, seq_len, probability)
    return result


def custom_collate_book_wiki(data, seq_len=512, probability=0.15):
    ### TODO: random mask by probability given
    ### NOTE: character truncation should be 2k 
    model_names = ["word", "char"]
    batch_size = len(data)
    return_dict = dict()

    for m_name in model_names:
        input_ids = []
        attention_mask = []
        char_word_id = []
        for i in range(batch_size):
            attention_mask.append(
                torch.tensor(data[i][f"{m_name}_attention_mask"][:seq_len])
            )
            char_word_id.append(torch.tensor(data[i]["char_word_ids"][:seq_len]))
            input_ids.append(torch.tensor(data[i][f"{m_name}_input_ids"][:seq_len]))

        input_ids = pad_sequence(
            input_ids, batch_first=True, padding_value=0
        )  # why pad_value = -100 doesn't work
        attention_mask = pad_sequence(attention_mask, batch_first=True, padding_value=0)
        char_word_id = pad_sequence(char_word_id, batch_first=True, padding_value=-100) # formerly -100

        rand = torch.rand(input_ids.shape)
        # where the random array is less than 0.15, we set true
        mask_arr = rand < probability
        mask_arr = mask_arr * (input_ids != 0)

        selection = []
        for i in range(input_ids.shape[0]):
            selection.append(torch.flatten(mask_arr[i].nonzero()).tolist())
        for i in range(input_ids.shape[0]):
            input_ids[i, selection[i]] = 0

        return_dict['char_word_ids'] = char_word_id.clone().detach()
        return_dict[f'{m_name}_input_ids'] = input_ids
        return_dict[f'{m_name}_attention_mask'] = attention_mask
    return return_dict


def custom_collate_book_wiki_eval(data, seq_len=512, probability=0.15):
    ### TODO: random mask by probability given
    model_names = ["word", "char"]
    batch_size = len(data)

    return_dict = dict()

    for m_name in model_names:
        input_ids = []
        attention_mask = []
        labels = []
        for i in range(batch_size):
            attention_mask.append(
                torch.tensor(data[i][f"{m_name}_attention_mask"][:seq_len])
            )
            labels.append(torch.tensor(data[i][f"{m_name}_labels"][:seq_len]))
            input_ids.append(torch.tensor(data[i][f"{m_name}_input_ids"][:seq_len]))


        input_ids = pad_sequence(input_ids, batch_first=True, padding_value=0)
        attention_mask = pad_sequence(attention_mask, batch_first=True, padding_value=0)
        labels = pad_sequence(labels, batch_first=True, padding_value=-100)

        return_dict[f'{m_name}_labels'] = labels
        return_dict[f'{m_name}_input_ids'] = input_ids
        return_dict[f'{m_name}_attention_mask'] = attention_mask

    return return_dict