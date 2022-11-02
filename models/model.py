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
import typing

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


class WNUTDatasetMulti(torch.utils.data.Dataset):
    def __init__(self, encodings, labels, model_names):
        # inputs are as Lists of encodings, labels, and models names : []
        self.encodings = encodings
        self.labels = labels
        self.model_names = model_names

    def __getitem__(self, idx):
        result = {}
        for encoding, label, model_name in zip(self.encodings, self.labels, self.model_names):
            item = {key: torch.tensor(val[idx]) for key, val in encoding.items()}
            item['labels'] = torch.tensor(label[idx])
            result[model_name] = item
        return result

    def __len__(self):
        return len(self.labels[0])


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
        self.wnut = wnut

        self.loader()

    def loader(self):
        tokenizer_dict = {}
        self.model_dict = dict()
        print("============== loading models ==============")
        for name in self.args.model_names:
            tokenizer_dict[name] = AutoTokenizer.from_pretrained(name, add_prefix_space=self.args.prefix_space)
            self.model_dict[name] = transformers.AutoModelForTokenClassification.from_pretrained(name, num_labels=self.args.num_labels)
        self.tokenizer_dict = tokenizer_dict

        ## use previous implementation of tokenization for each granularity
        # TODO
        # Should end up something like 
        # self.data = {model_name : DatasetDict }, where 
        # datadict is created using previous tokenization method in data.py 
        for granularity in self.args.granularities:
            name = self.args.granularities_model[granularity]
            if granularity == "character":
                clm = CharacterLevelMapping(self.args.to_char_method)
                wnut_character_level = clm.character_level_wnut(self.wnut)
                tok = Tokenization(self.args.granularities_model[granularity], self.args.prefix_space)
                tokenized_wnut = tok.tokenize_for_char_manual(wnut_character_level)
                self.data_[name] = tokenized_wnut
                self.data_length[name] = tokenized_wnut['train'].num_rows
            elif granularity == "subword_50k" or granularity == "subword_30k":
                tok = Tokenization(self.args.granularities_model[granularity], self.args.prefix_space)
                tokenized_wnut = self.wnut.map(tok.tokenize_and_align_labels, batched=True) ## was previously wnut_character level 
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
        ## return {model_name : input_info}, where 
        ## input_info is input_ids, attn_mask, token_type_ids for each granularity. 
        return_dict_train = dict()
        return_dict_val = dict()
        return_dict_test = dict()
        for granularity in self.args.granularities:
            name = self.args.granularities_model[granularity]
            return_dict_train[name] = self.data_[name]['train'][idx]
            return_dict_val[name] = self.data_[name]['validation'][idx]
            return_dict_test[name] = self.data_[name]['test'][idx]

        return return_dict_train



## multitask learning 
class attention_MTL(nn.module):
    def __init__(self, model_dict, args):
        super().__init__()  ## delete this ??
        self.model_dict = model_dict
        self.args = args
        self.lin_layer_dict = dict()
        self.W_dict = dict()
        self.attention_dict = dict()
        
        ## weight for adding attention scores 
        self.weights = nn.Parameter(torch.empty(len(model_dict)))

        ## or we can have one weight for each model
        self.weight_dict = dict()
    

        for model_name in model_dict:
            # add one linear layer per model
            lin_layer = nn.Linear(self.args.embed_size_dict[model_name], args.num_labels)
            self.lin_layer_dict[model_name] = lin_layer
            ## attention weight matrix for each model 
            self.W_dict = nn.Linear(self.args.embed_size_dict[model_name], self.args.embed_size_dict[model_name])

            ## one set of weights for each model: UNCOMMENT THIS 
            self.weight_dict[model_name] = nn.Parameter(torch.empty(len(model_dict)))
            attention = nn.MultiheadAttention(
                embed_dim = self.args.embed_size_dict[model_name],
                num_heads = 1,
                batch_first=True)
            self.attention_dict[model_name] = attention
            
    
    def forward(self, input_info_dict):
        hidden_states_all = [] ## [num_model, bs, seq_len, embed_size]
        logits_dict = dict()
        hidden_state_dict = dict()
        for model_name in self.model_dict:
            ## get information
            model = self.model_dict[model_name]
            input_info = input_info_dict[model_name]
            # input_ids, attn_mask, token_type_ids = input_info["input_ids"], input_info["attention_mask"], input_info["labels"]
            ## get contexualized representation
            encoded = model(return_dict = True, output_hidden_states=True, **input_info)
            hidden_states = encoded["hidden_states"][-1]  ## [bs, seq_len, embed_size]
            hidden_state_dict[model_name] = hidden_states
            self.logits_dict[model_name] = self.lin_layer_dict[model_name](hidden_states)
        
        ## compute cross attention / self attention
        for model_name_q in hidden_state_dict:
            query = hidden_state_dict[model_name_q]
            hidden_states_sum = torch.zeros_like(query)
            count = 0
            weights = self.weights ## can change to one weight one model
            for model_name_k in hidden_state_dict:
                key = hidden_state_dict[model_name_k]
                attn_output, attn_output_weights = self.attention_dict[model_name](
                    query = query, 
                    key = key, 
                    value = key
                )
                ## write the attention ourselves? 
                hidden_states_sum = torch.add(hidden_states_sum, attn_output * weights[count])
                count += 1
            logit = self.lin_layer_dict[model_name](hidden_states_sum)
            self.logits_dict[model_name] = logit
    
        return self.logits_dict ## {model_name: logit}

        

class MTL_classifier(BaseEstimator):
    def step(self, data):
        logits_dict = self.model(
            input_infor_dict = data
        )
        if self.mode == "train":
            self.optimizer.zero_grad()
            loss = torch.tensor(0)
            for model_name, logit in logits_dict:
                loss += self.criterion(logit, data[model_name]["labels"])
                ## todo: penalize disagreement by adding other loss 
                ## todo: penalize weighted loss instead of simple sum? 
            loss.backward()
            self.optimizer.step()
            if self.scheduler is not None:
                self.scheduler.step()
            return {
                "loss" : loss.detach().cpu().item(), 
                "logits_dict" : logits_dict.detach().cpu().item(), ## softmax this 
                "label" : data[self.args.word_model]["labels"]
                    }
        elif self.mode == "test":
            return {
                "loss" : None, 
                "logits_dict" : logits_dict.detach().cpu().item(),
                "label" : data[self.args.word_model]["labels"]
                }


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
        

# class weighted_estimater(BaseEstimator):
#     def __init__(self, model, tokenizer, criterion=None, optimizer=None, scheduler=None, logger=None, writer=None, pred_thold=None, device='cpu', **kwargs):
#         super().__init__(model, tokenizer, criterion, optimizer, scheduler, logger, writer, pred_thold, device, **kwargs)
#         self.clip_grad_norm = 1

#     def step(self, data):
#         logits = self.model(
#             data['input_ids'].to(self.device, dtype=torch.long), 
#             data['attention_mask'].to(self.device, dtype=torch.long), 
#             data['ner_tags'].to(self.device, dtype=torch.long)
#         )
#         if self.mode in {'train', 'dev'}: 
#             # training or developmenting, ground true labels are provided
#             if self.mode == 'train': 
#                 self.optimizer.zero_grad()
#             loss = self.criterion(logits, data['labels'].to(self.device, dtype=torch.float))
#             if self.mode == 'train': 
#                 loss.backward()
#                 nn.utils.clip_grad_norm_(self.model.parameters(), self.clip_grad_norm) # TODO: not equivalent to tf.clip_by_global_norm
#                 self.optimizer.step()
#                 if self.scheduler is not None: 
#                     self.scheduler.step()
#             return (
#                 loss.detach().cpu().item(), 
#                 torch.softmax(logits).detach().cpu().numpy(), 
#                 data['labels'].numpy()
#             )
#         elif self.mode == 'test': 
#             # testing, no ground true label is provided
#             return None, torch.sigmoid(logits).detach().cpu().numpy(), None
#         else: 
#             raise ValueError(self.mode)




if __name__ == '__main__':
    ## add arguments 
    parser = argparse.ArgumentParser()

    parser.add_argument('--dataset', type=str, default='wnut17')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--log_dir', type=str, default=None)
    parser.add_argument('--output_dir', type=str, default="./results")
    parser.add_argument('--bs', type=int, default=16)
    parser.add_argument('--model_name', type=str, default="google/canine-s")
    parser.add_argument('--n_epochs', type=int, default=1) ## change to 4
    parser.add_argument('--lr', type=float, default=2e-5)
    parser.add_argument('--weight_decay', type=float, default=0)
    parser.add_argument('--prefix_space', type=bool, default=True)
    parser.add_argument('--num_labels', type=int, default=13)
    parser.add_argument('--granularities', type=str, default="character,subword_50k")# add cahracter
    parser.add_argument('--add_space_for_char', type=bool, default=True)
    parser.add_argument('--to_char_method', type=str, default="inherit")
    parser.add_argument('--train', type=str, default="True")
    parser.add_argument('--device', type=str, default=None)
    parser.add_argument('--ensemble_method', type=str, default="soft")

    parser.add_argument('--granularities_model', type=dict, 
                        default= {"character": "google/canine-s",
                                "subword_50k": "xlm-roberta-base",
                                "subword_30k" : "bert-base-cased"})

    parser.add_argument('--embed_size_dict', type=dict, 
                        default= {"google/canine-s": 768,
                                  "google/canine-c": 768,
                                  "bert-base-cased" : 768,
                                  "bert-base-uncased" : 768,
                                  "xlm-roberta-base" : 1024,
                                  "roberta-base": 768,
                                  "roberta-large": 1024,
                                  "vinai/bertweet-base": 768,
                                  "cardiffnlp/twitter-roberta-base-sentiment": 768})

    args = parser.parse_args()
    args.granularities = args.granularities.split(",")
    args.model_names = [args.granularities_model[key] for key in args.granularities_model]

    from datasets import load_dataset

    print(args.model_names)
    print(args.granularities)

    wnut = load_dataset("wnut_17")
    wnut_g = wnut_multiple_granularity(wnut, args)
    print(wnut_g.data_.keys())
    print(wnut_g[0])

