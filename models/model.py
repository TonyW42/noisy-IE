import sys 
sys.path.append("..") 
sys.path.append("../..") 
import torch 
import transformers
import numpy as np
from utils.utils import *
from data import *
from utils.model_utils import *
from torch import nn
from collections import defaultdict
from datasets import DatasetDict, Dataset


## need dataset/loader structure such as the following:
## integrate to data.py if possible 
class wnut_multiple_granularity(Dataset):
    def __init__(self, wnut, args):
        """
        preprocess multiple granularity for each model here
        Input: wnut - result of load_dataset("wnut_17")
        """
        self.args = args
        self.data_ = dict()
        self.data_length = defaultdict(int)
        tokenizer_dict = {}
        self.model_dict = dict()
        print("============== loading models ==============")
        for name in args.model_names:
            tokenizer_dict[name] = AutoTokenizer.from_pretrained(name, add_prefix_space=self.args.prefix_space)
            self.model_dict[name] = transformers.AutoModelForTokenClassification.from_pretrained(name, num_labels=self.args.num_labels)
        self.tokenizer_dict = tokenizer_dict

        ## use previous implementation of tokenization for each granularity
        # TODO
        # Should end up something like 
        # self.data = {model_name : DatasetDict }, where 
        # datadict is created using previous tokenization method in data.py 
        for name, granularity in zip(self.args.model_names, self.args.granularities):
            if granularity == "character":
                clm = CharacterLevelMapping(self.args.to_char_method)
                wnut_character_level = clm.character_level_wnut(wnut)
                tok = Tokenization(self.args.granularities_model[granularity], self.args.prefix_space)
                tokenized_wnut = tok.tokenize_for_char_manual(wnut_character_level)
                self.data_[name] = tokenized_wnut
                self.data_length[name] = tokenized_wnut['train'].num_rows
            elif granularity == "subword_50k" or granularity == "subword_30k":
                tok = Tokenization(self.args.granularities_model[granularity], self.args.prefix_space)
                tokenized_wnut = wnut.map(tok.tokenize_and_align_labels, batched=True) ## was previously wnut_character level 
                self.data_[name] = tokenized_wnut
                self.data_length[name] = tokenized_wnut['train'].num_rows
            assert is_aligned(tokenized_wnut)
        assert granularity_aligned(self.data_length)

        # raise NotImplementedError
    
    def __len__(self):
        ## return the length of dataset of any granularity (they must be all equal)
        for model_name, data_length in self.data_length_dict.items():
            print("model_name {model_name}, the length of dataset is : {length} ".format(model_name=model_name, length=data_length))
        return set(self.data_length.values())[0]
    
    def __getitem__(self, idx):
        ## should return {model_name : input_info}, where 
        ## input_info is input_ids, attn_mask, token_type_ids for each granularity. 
        # self.input_info = defaultdict(defaultdict)
        # self.input_info_test = defaultdict(defaultdict)
        # for model_name in self.args.model_names:
        #     self.input_info[model_name]['input_ids'] = self.data[model_name]['train']['input_ids'] 
        #     self.input_info[model_name]['attn_mask'] = self.data[model_name]['train']['attention_mask'] 
        #     self.input_info[model_name]['token_type_ids'] = self.data[model_name]['train']['labels'] 
        #     self.input_info_test[model_name]['input_ids'] = self.data[model_name]['test']['input_ids'] 
        #     self.input_info_test[model_name]['attn_mask'] = self.data[model_name]['test']['attention_mask'] 
        #     self.input_info_test[model_name]['token_type_ids'] = self.data[model_name]['test']['labels'] 
        # return self.input_info
        return self.data
        # raise NotImplementedError


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
            input_ids, attn_mask, token_type_ids = input_info["input_ids"], input_info["attention_mask"], input_info["ner_tags"]
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
        

class weighted_estimater(BaseEstimator):
    def __init__(self, model, tokenizer, criterion=None, optimizer=None, scheduler=None, logger=None, writer=None, pred_thold=None, device='cpu', **kwargs):
        super().__init__(model, tokenizer, criterion, optimizer, scheduler, logger, writer, pred_thold, device, **kwargs)
        self.clip_grad_norm = 1

    def step(self, data):
        logits = self.model(
            data['input_ids'].to(self.device, dtype=torch.long), 
            data['attention_mask'].to(self.device, dtype=torch.long), 
            data['ner_tags'].to(self.device, dtype=torch.long)
        )
        if self.mode in {'train', 'dev'}: 
            # training or developmenting, ground true labels are provided
            if self.mode == 'train': 
                self.optimizer.zero_grad()
            loss = self.criterion(logits, data['labels'].to(self.device, dtype=torch.float))
            if self.mode == 'train': 
                loss.backward()
                nn.utils.clip_grad_norm_(self.model.parameters(), self.clip_grad_norm) # TODO: not equivalent to tf.clip_by_global_norm
                self.optimizer.step()
                if self.scheduler is not None: 
                    self.scheduler.step()
            return (
                loss.detach().cpu().item(), 
                torch.softmax(logits).detach().cpu().numpy(), 
                data['labels'].numpy()
            )
        elif self.mode == 'test': 
            # testing, no ground true label is provided
            return None, torch.sigmoid(logits).detach().cpu().numpy(), None
        else: 
            raise ValueError(self.mode)












