import torch 
import transformers
import numpy as np
from utils.utils import *
from data import *
from utils.model_utils import *
from torch import nn

class weighted_ensemble(BaseClassifier):
    def __init__(self, model_dict, args):
        super().__init__()
        self.model_dict = model_dict
        total_embed_size = 0
        for model_name in model_dict:
            total_embed_size += self.args.embed_size_dict[model_name]
        self.lin1 = nn.Linear(total_embed_size, args.num_labels)
        self.args = args
    
    def forward(self, input_info_dict):
        hidden_states_all = [] ## [num_model, bs, seq_len, embed_size]
        for model_name in self.model_dict:
            model = self.model_dict[model_name]
            input_info = input_info_dict[model_name]
            input_ids, attn_mask, token_type_ids = input_info["input_ids"], input_info["attn_mask"], input_info["token_type_ids"]
            encoded = model(input_ids = input_ids, 
                            attention_mask = attn_mask,
                            token_type_ids = token_type_ids)
            hidden_states = encoded[0]  ## [bs, seq_len, embed_size]
            hidden_states_all.append(hidden_states) 
            ##########################
            ## make sure that seq_len is aligned fr subword/character model
            ###########################



        hidden_states_cat = torch.cat(hidden_states_all, dim = 3)
        logits = self.lin1(hidden_states_cat)
        return logits ## [bs, seq_len, num_labels]
        














