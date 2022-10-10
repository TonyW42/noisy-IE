import numpy as np 
import torch 
import evaluate 
import statistics
import math
from utils.utils import mode, multimode, flatten_2d
# from run import *  

#####################################################################
########## compute character-level F1 
#####################################################################
def get_space_id(model_name):
  space_id = None
  if model_name == "google/canine-s":
      space_id = 32
  return space_id 



def wnut_get_predictions(model, tokenized_wnut, prefix_space, device):
  model.eval()

  predicted = []
  label = []
  count = 0
  len_test = len(tokenized_wnut["test"])
  input_ids_list = []
  for samp in tokenized_wnut["test"]:
    input_ids = samp["input_ids"]
    attn_mask = samp["attention_mask"]
    labels = samp["labels"]

    input_ids = torch.unsqueeze(torch.tensor(input_ids), dim=0).to(device)
    attn_mask = torch.unsqueeze(torch.tensor(attn_mask), dim = 0).to(device)
    input_ids_squeezed = input_ids.view(-1)

    encoded = model(input_ids = input_ids,
                    attention_mask = attn_mask)
    
    pred = torch.argmax(encoded["logits"], dim = 2)
    pred = pred.view(-1)
    if prefix_space: ## added, remove trailing space
      pred = pred[1:(len(pred) - 1)] 
      input_ids_squeezed = input_ids_squeezed[1: (len(input_ids_squeezed) - 1)]
      labels = labels[1:(len(labels)-1)]

    predicted.append(pred)
    label.append(labels)
    input_ids_list.append(input_ids_squeezed)
    # print(pred)
    # metric.add_batch(predictions=pred, 
    #                  references=torch.tensor(labels))
    assert len(pred) == len(labels) & len(pred) == len(input_ids_squeezed) & len(labels) == len(input_ids_squeezed)

    if count % 200 == 0:
      print(f"finish {count} / {len_test}")
      # break 
    #   return {
    #     "pred" : predicted,
    #     "label": label,
    #     "input_ids": input_ids_list
    # }
    count += 1
  len_test = len(tokenized_wnut["test"])
  print(f"Finished {len_test}/{len_test}")

  return {
      "pred" : predicted,
      "label": label,
      "input_ids": input_ids_list
  }

def wnut_separate_char_prediction(pred_output, model_name):
  pred, label, input_ids = pred_output["pred"], pred_output["label"], pred_output["input_ids"]
  pred_word = []
  label_word = []
  ids_word = []
  space_id = get_space_id(model_name) ## get space id 
  for i in range(0, len(pred)):
    p = pred[i]
    l = label[i]
    tok_id = input_ids[i]
    tmp_pred = []
    tmp_label = []
    tmp_input = []
    for j in range(0, len(p)):
      p_j = int(p[j])
      l_j = int(l[j])
      tok_id_j = int(tok_id[j])
      # print(tok_id_j)
      if tok_id_j == space_id:
        if len(tmp_pred) != 0: 
          pred_word.append(tmp_pred)
          label_word.append(tmp_label)
          ids_word.append(tmp_input)
        tmp_pred = []
        tmp_label = []
        tmp_input = []
      else:
        tmp_pred.append(p_j)
        tmp_label.append(l_j)
        tmp_input.append(tok_id_j)
    # print(ids_word)
  return {
      "pred_word"  :  pred_word, 
      "label_word" : label_word, 
      "input_word" : ids_word,
  }

def wnut_char_to_word_first_letter(char_pred):
  pred_word, label_word, input_word = char_pred["pred_word"], char_pred["label_word"], char_pred["input_word"]

  label = []
  inputs = []
  pred = []
  for i in range(0, len(pred_word)):
    p = pred_word[i]
    input_i = input_word[i]
    l = label_word[i]

    pred.append(p[0])
    inputs.append(input_i[0])
    label.append(l[0])
  
  return {
      "pred" : pred, 
      "word": inputs,
      "label" : label
  }

def wnut_char_rule_1(l):
  return mode(l) ## if equally likely, then take the first one 

## if contains B, then B. Then take
def wnut_char_rule_2(l):
  B_list = [1, 3, 5, 7, 9, 11]
  I_2_B = {0:0, 2:1, 4:3, 6:5, 8:7, 10:9, 12:11, 1:1, 3:3, 5:5, 7:7, 9:9, 11:11}
  if len(l) == 1: return l[0]
  if l[0] in B_list:
    mode_l = mode(l[1:len(l)])
    label = I_2_B[mode_l]
  else:
    label = mode(l)
  return label

def rule_3(l):
  return l[0]

def wnut_char_to_word_rule(char_pred, rule = 3):
  pred_word, label_word, input_word = char_pred["pred_word"], char_pred["label_word"], char_pred["input_word"]
  label = []
  pred = []
  for i in range(0, len(pred_word)):
    p = pred_word[i]
    l = label_word[i]
    pred_rule = None
    if rule == 1:
      pred_rule = wnut_char_rule_1(p)
    if rule == 2: 
      pred_rule = wnut_char_rule_2(p)
    if rule == 3:
      pred_rule = rule_3(p)
      

    pred.append(pred_rule)
    label.append(l[0])
  return {
      "pred" : pred, 
      "label" : label
  }
# c_rule_1 = char_to_word_rule(b, rule = 1)
# c_rule_2 = char_to_word_rule(b, rule = 2)

def wnut_f1(pred, ref, average = "macro"):
  f1_metric = evaluate.load("f1")
  return f1_metric.compute(predictions = pred, references = ref, 
                           average = average)

def wnut_evaluate_f1(model, tokenized_wnut, prefix_space, model_name, device, method = "first letter"):
  raw_predictions = wnut_get_predictions(model, tokenized_wnut, prefix_space, device)
  pred_for_char = wnut_separate_char_prediction(raw_predictions, model_name)
  pred = None
  if method == "first letter":
    pred = wnut_char_to_word_first_letter(pred_for_char)
  if method == "rule 1":
    pred = wnut_char_to_word_rule(pred_for_char, rule = 1)
  if method == "rule 2":
    pred = wnut_char_to_word_rule(pred_for_char, rule = 2)
  wnut_f1_score = wnut_f1(pred = pred["pred"], ref = pred["label"])
  return wnut_f1_score


#####################################################################
####### compute character level logits 
#####################################################################
  
def wnut_get_character_logits_raw(model, tokenized_wnut, prefix_space, device):
  model.to(device)
  model.eval()

  predicted = []
  label = []
  count = 0
  len_test = len(tokenized_wnut["test"])
  input_ids_list = []
  for samp in tokenized_wnut["test"]:
    with torch.no_grad():
      input_ids = samp["input_ids"]
      attn_mask = samp["attention_mask"]
      labels = samp["labels"]

      input_ids = torch.unsqueeze(torch.tensor(input_ids), dim=0).to(device)
      attn_mask = torch.unsqueeze(torch.tensor(attn_mask), dim = 0).to(device)
      input_ids_squeezed = input_ids.view(-1)

      encoded = model(input_ids = input_ids,
                      attention_mask = attn_mask)
      
      pred = encoded["logits"]
      pred = torch.squeeze(pred)
      if prefix_space: ## added, remove trailing space
        pred = pred[1:(len(pred) - 1)] 
        input_ids_squeezed = input_ids_squeezed[1: (len(input_ids_squeezed) - 1)]
        labels = labels[1:(len(labels)-1)]
      
      pred = torch.nn.functional.softmax(pred, dim = 1) ## computes probability
      predicted.append(pred.detach().cpu().numpy())
      label.append(labels)
      input_ids_list.append(input_ids_squeezed.detach().cpu().numpy())
      # print(pred)
      # metric.add_batch(predictions=pred, 
      #                  references=torch.tensor(labels))
      assert len(pred) == len(labels) & len(pred) == len(input_ids_squeezed) & len(labels) == len(input_ids_squeezed)

      if count % 200 == 0:
        print(f"finish {count} / {len_test}")
        # break 
      #   return {
      #     "pred" : predicted,
      #     "label": label,
      #     "input_ids": input_ids_list
      # }
      count += 1
  len_test = len(tokenized_wnut["test"])
  print(f"Finished {len_test}/{len_test}")

  return {
      "pred" : predicted,
      "label": label,
      "input_ids": input_ids_list
  }
# wnut_get_character_logits(model, tokenized_wnut, prefix_space, device)

def wnut_separate_char_logits(pred_output, model_name):
  pred, label, input_ids = pred_output["pred"], pred_output["label"], pred_output["input_ids"]
  pred_word = []
  label_word = []
  ids_word = []
  space_id = get_space_id(model_name) ## get space id 
  for i in range(0, len(pred)):
    p = pred[i]
    l = label[i]
    tok_id = input_ids[i]
    tmp_pred = []
    tmp_label = []
    tmp_input = []
    for j in range(0, len(p)):
      p_j = p[j]
      l_j = int(l[j])
      tok_id_j = int(tok_id[j])
      # print(tok_id_j)
      if tok_id_j == space_id:
        if len(tmp_pred) != 0: 
          pred_word.append(tmp_pred)
          label_word.append(tmp_label)
          ids_word.append(tmp_input)
        tmp_pred = []
        tmp_label = []
        tmp_input = []
      else:
        tmp_pred.append(p_j)
        tmp_label.append(l_j)
        tmp_input.append(tok_id_j)
    # print(ids_word)
  return {
      "pred_word"  :  pred_word, 
      "label_word" : label_word, 
      "input_word" : ids_word,
  }

def wnut_get_char_logits(model, tokenized_wnut, prefix_space, device, model_name, rule=3):
  logits_raw = wnut_get_character_logits_raw(model, tokenized_wnut, prefix_space, device)
  logits_grouped = wnut_separate_char_logits(logits_raw, model_name)
  logits_word = wnut_char_to_word_rule(logits_grouped, rule = rule)
  return logits_word

#####################################################################
####### compute subword level logits 
#####################################################################
def wnut_get_subword_logits(model, tokenized_wnut, prefix_space, device, model_name, rule=3):
  model.eval()

  predicted = []
  label = []
  count = 0
  len_test = len(tokenized_wnut["test"])
  input_ids_list = []
  for samp in tokenized_wnut["test"]:
    with torch.no_grad():
      input_ids = samp["input_ids"]
      attn_mask = samp["attention_mask"]
      labels = samp["labels"]

      input_ids = torch.unsqueeze(torch.tensor(input_ids), dim=0).to(device)
      attn_mask = torch.unsqueeze(torch.tensor(attn_mask), dim = 0).to(device)
      input_ids_squeezed = input_ids.view(-1)

      encoded = model(input_ids = input_ids,
                      attention_mask = attn_mask)
      
      pred = encoded["logits"]
      pred = torch.squeeze(pred)
      if prefix_space: ## added, remove trailing space
        pred = pred[1:(len(pred) - 1)] 
        input_ids_squeezed = input_ids_squeezed[1: (len(input_ids_squeezed) - 1)]
        labels = labels[1:(len(labels)-1)]
      
      pred = torch.nn.functional.softmax(pred, dim = 1) ## computes probability

      predicted.append(pred.detach().cpu().numpy())
      label.append(labels)
      input_ids_list.append(input_ids_squeezed.detach().cpu().numpy())
      # print(pred)
      # metric.add_batch(predictions=pred, 
      #                  references=torch.tensor(labels))
      assert len(pred) == len(labels) & len(pred) == len(input_ids_squeezed) & len(labels) == len(input_ids_squeezed)

      if count % 200 == 0:
        print(f"finish {count} / {len_test}")
        # break 
      #   return {
      #     "pred" : predicted,
      #     "label": label,
      #     "input_ids": input_ids_list
      # }
      count += 1
  len_test = len(tokenized_wnut["test"])
  print(f"Finished {len_test}/{len_test}")

  predicted_all = flatten_2d(predicted)
  label_all = flatten_2d(label)
  input_ids_all = flatten_2d(input_ids_list)
  null_indices = [i for i in range(0, len(label_all)) if label_all[i] == -100]

  predicted = [predicted_all[i] for i in range(0, len(label_all)) if i not in null_indices]
  label = [label_all[i] for i in range(0, len(label_all)) if i not in null_indices]
  input_ids = [input_ids_all[i] for i in range(0, len(label_all)) if i not in null_indices]
  assert len(predicted) == len(label) & len(input_ids) == len(predicted)
  print(len(predicted))
  return {
      "pred" : predicted,
      "label": label,
      "input_ids": input_ids_list
  }
  