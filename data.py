import transformers 
import torch 
import numpy as np 
import pandas as pd
import datasets 
from datasets import load_dataset
### wnut dataset mapping
from collections import defaultdict
from datasets import DatasetDict, Dataset
from transformers import AutoTokenizer
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler, random_split

from run import *

use_old_tok = ["xlm-roberta-base", "xlm-roberta-large"]
use_new_tok = ["google/canine-s"]

id2tag = {0: 'O',
          1: 'B-corporation',
          2: 'I-corporation',
          3: 'B-creative-work',
          4: 'I-creative-work',
          5: 'B-group',
          6: 'I-group',
          7: 'B-location',
          8: 'I-location',
          9: 'B-person',
          10: 'I-person',
          11: 'B-product',
          12: 'I-product',}
tag2id = {tag: id for id, tag in id2tag.items()}


def encode_tags(examples, tokenized_inputs):
    labels = []
    for i, label in enumerate(examples[f"ner_tags"]):
        word_ids = tokenized_inputs.word_ids(batch_index=i)  # Map tokens to their respective word.
        previous_word_idx = None
        label_ids = []
        for word_idx in word_ids:  # Set the special tokens to -100.
            if word_idx is None:
                label_ids.append(-100)
            elif word_idx != previous_word_idx:  # Only label the first token of a given word.
                label_ids.append(label[word_idx])
            else:
                label_ids.append(-100)
            previous_word_idx = word_idx
        labels.append(label_ids)

    return labels


## get raw data 
def get_unprocessed_data(data_name):
    return load_dataset(data_name)

  
def get_tokenizer(model_name, prefix_space):
    tokenizer = AutoTokenizer.from_pretrained(model_name,
                                              add_prefix_space=prefix_space) ## changed here
    return tokenizer


class CharacterLevelMapping:
  def __init__(self, to_char_method="inherit") -> None:
      self.to_char_method = to_char_method

  ## takes wnut dataset and transform it to character level 
  def character_level_wnut(self, wnut):
    wnut_character_level = DatasetDict()
    mapping_dict = None
    if self.to_char_method == "first_character_inherit":
      mapping_dict = {0:0, 1:2, 2:2, 3:4, 4:4, 5:6, 6:6, 7:8, 8: 8, 9:10, 10:10, 11:12, 12:12}
    if self.to_char_method == "inherit":
      mapping_dict = {0:0, 1:1, 2:2, 3:3, 4:4, 5:5, 6:6, 7:7, 8: 8, 9:9, 10:10, 11:11, 12:12}
    assert mapping_dict is not None
    # reference tag num: https://huggingface.co/datasets/wnut_17
    def iterate_dataset(type_):
      id_list, token_list, ner_tag_list = [], [], []
      index = 0
      for token_arr, ner_tag_arr in zip(wnut[type_]['tokens'], wnut[type_]['ner_tags']):
        token_arr_list, ner_tag_arr_list = [], []
        for token, ner_tag in zip(token_arr, ner_tag_arr):
          for ch_ind, character in enumerate(token):
            token_arr_list.append(character)
            if ch_ind == 0:
              ner_tag_arr_list.append(ner_tag)
            else:
              ner_tag_arr_list.append(mapping_dict[ner_tag])
          token_arr_list.append(' ')
          ner_tag_arr_list.append(mapping_dict[ner_tag])
        token_list.append(token_arr_list)
        ner_tag_list.append(ner_tag_arr_list)
        id_list.append(str(index))
        index += 1
      wnut_character_level[type_] = Dataset.from_dict({
          'id': id_list, 
          'tokens': token_list, 
          'ner_tags' : ner_tag_list,
          })
    iterate_dataset('train')
    iterate_dataset('validation')
    iterate_dataset('test')
    return wnut_character_level



class Tokenization:
  def __init__(self, model_name, prefix_space):
    self.model_name = model_name
    self.prefix_space = prefix_space
    self.tokenizer = get_tokenizer(model_name, prefix_space)

  def tokenize_and_align_labels(self, examples):
    tokenized_inputs = self.tokenizer(examples["tokens"], truncation=True, is_split_into_words=True)

    labels = []
    for i, label in enumerate(examples[f"ner_tags"]):
        word_ids = tokenized_inputs.word_ids(batch_index=i)  # Map tokens to their respective word.
        previous_word_idx = None
        label_ids = []
        for word_idx in word_ids:  # Set the special tokens to -100.
            if word_idx is None:
                label_ids.append(-100)
            elif word_idx != previous_word_idx:  # Only label the first token of a given word.
                label_ids.append(label[word_idx])
            else:
                label_ids.append(-100)
            previous_word_idx = word_idx
        labels.append(label_ids)

    tokenized_inputs["labels"] = labels
    return tokenized_inputs

  def tokenize_for_char(self, examples):
    tokenized_inputs = self.tokenizer(examples["tokens"], truncation=True, is_split_into_words=True)
    tokenized_inputs["labels"] = examples["ner_tags"]

    labels_list = examples["ner_tags"]
    if self.prefix_space:
        labels_list.insert(0, -100)  ## insert 0 or -100? 
        labels_list.append(-100)
        tokenized_inputs["attention_mask"][0] = 0
        tokenized_inputs["attention_mask"][-1] = 0

    tokenized_inputs["labels"] = labels_list
    assert len(tokenized_inputs["labels"]) == len(tokenized_inputs["input_ids"])
    assert len(tokenized_inputs["input_ids"])  ==   len(tokenized_inputs["attention_mask"])
    return tokenized_inputs
  
  def tokenize_for_char_manual(self, wnut):
    new_wnut = DatasetDict()
    def iterate_dataset(type_):
      id_list, mask_list, label_list = [], [], []
      for samp in wnut[type_]:
        tokenized_samp = self.tokenize_for_char(samp)
        id_list.append(tokenized_samp["input_ids"])
        mask_list.append(tokenized_samp["attention_mask"])
        label_list.append(tokenized_samp["labels"])
      new_wnut[type_] = Dataset.from_dict(
          {
              "input_ids": id_list,
              "attention_mask" : mask_list, 
              "labels" : label_list
          }
      )
    iterate_dataset('train')
    iterate_dataset('validation')
    iterate_dataset('test')
    return new_wnut 

  def tokenize_wnut_char(self, model_name):
      wnut = get_unprocessed_data("wnut_17")
      clm = CharacterLevelMapping()
      wnut_character_level = clm.character_level_wnut(wnut)
      tokenized_wnut = None
      if model_name in use_old_tok:
          tokenized_wnut = wnut_character_level.map(self.tokenize_and_align_labels, batched=True)
      if model_name in use_new_tok:
          tokenized_wnut = self.tokenize_for_char_manual(wnut_character_level)
      if (not model_name in use_new_tok) and (not model_name in use_old_tok):
          raise ValueError("Please update your model")
      assert is_aligned(tokenized_wnut)
      return tokenized_wnut


# tokenizer = get_tokenizer(model_name, prefix_space)
# # print(tokenizer(" "))

# def tokenize_and_align_labels(examples):
#     tokenized_inputs = tokenizer(examples["tokens"], truncation=True, is_split_into_words=True)

#     labels = []
#     for i, label in enumerate(examples[f"ner_tags"]):
#         word_ids = tokenized_inputs.word_ids(batch_index=i)  # Map tokens to their respective word.
#         previous_word_idx = None
#         label_ids = []
#         for word_idx in word_ids:  # Set the special tokens to -100.
#             if word_idx is None:
#                 label_ids.append(-100)
#             elif word_idx != previous_word_idx:  # Only label the first token of a given word.
#                 label_ids.append(label[word_idx])
#             else:
#                 label_ids.append(-100)
#             previous_word_idx = word_idx
#         labels.append(label_ids)

#     tokenized_inputs["labels"] = labels
#     return tokenized_inputs


# def tokenize_for_char(examples):
#   tokenized_inputs = tokenizer(examples["tokens"], truncation=True, is_split_into_words=True)
#   tokenized_inputs["labels"] = examples["ner_tags"]

#   labels_list = examples["ner_tags"]
#   if prefix_space:
#       labels_list.insert(0, -100)  ## insert 0 or -100? 
#       labels_list.append(-100)
#       tokenized_inputs["attention_mask"][0] = 0
#       tokenized_inputs["attention_mask"][-1] = 0

#   tokenized_inputs["labels"] = labels_list
#   assert len(tokenized_inputs["labels"]) == len(tokenized_inputs["input_ids"])
#   assert len(tokenized_inputs["input_ids"])  ==   len(tokenized_inputs["attention_mask"])
#   return tokenized_inputs

# def tokenize_for_char_manual(wnut):
#   new_wnut = DatasetDict()
#   def iterate_dataset(type_):
#     id_list, mask_list, label_list = [], [], []
#     for samp in wnut[type_]:
#       tokenized_samp = tokenize_for_char(samp)
#       id_list.append(tokenized_samp["input_ids"])
#       mask_list.append(tokenized_samp["attention_mask"])
#       label_list.append(tokenized_samp["labels"])
#     new_wnut[type_] = Dataset.from_dict(
#         {
#             "input_ids": id_list,
#             "attention_mask" : mask_list, 
#             "labels" : label_list
#         }
#     )
#   iterate_dataset('train')
#   iterate_dataset('validation')
#   iterate_dataset('test')
#   return new_wnut 


def is_aligned(dataset):
    for key in ["train", "validation", "test"]:
        for samp in dataset[key]:
            len_input = len(samp["input_ids"])
            len_attn  = len(samp["attention_mask"])
            len_label = len(samp["labels"])
            if len_input!=len_attn or len_input!=len_label or len_attn!=len_label:
                return False
    return True 

def granularity_aligned(data_length_dict):
    temp = -1
    for _, data_length in data_length_dict.items():
        if temp == -1:
            temp = data_length
        else:
            if data_length != temp:
                return False
    return True

# def tokenize_wnut_char(model_name):
#     wnut = get_unprocessed_data("wnut_17")
#     wnut_character_level = character_level_wnut(wnut)
#     tokenized_wnut = None
#     if model_name in use_old_tok:
#         tokenized_wnut = wnut_character_level.map(tokenize_and_align_labels, batched=True)
#     if model_name in use_new_tok:
#         tokenized_wnut = tokenize_for_char_manual(wnut_character_level)
#     if (not model_name in use_new_tok) and (not model_name in use_old_tok):
#         raise ValueError("Please update your model")
#     assert is_aligned(tokenized_wnut)
#     return tokenized_wnut

if __name__ == '__main__':
  prefix_space = True
  model_name = "google/canine-s"
  tok = Tokenization(model_name, prefix_space)
  a = tok.tokenize_wnut_char()
  print(a["train"][0]["input_ids"])
  print(a["train"][0]["labels"])
  # print(a["train"][0])
  # print(len(a["train"][0]["input_ids"]))
  # print(len(a["train"][0]["attention_mask"]))
  # print(len(a["train"][0]["labels"]))
  # assert len(a["train"][0]["input_ids"]) == len(a["train"][0]["attention_mask"])
  # assert len(a["train"][0]["input_ids"]) == len(a["train"][0]["labels"])



