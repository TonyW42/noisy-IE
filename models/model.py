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
            print(f"============== loading {name} ==============")
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
            print(f"============== tokenizing {name} ==============")
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
class attention_MTL(nn.Module):
    def __init__(self, model_dict, args):
        super().__init__()  ## delete this ??
        self.model_dict = model_dict
        self.args = args
        self.lin_layer_dict = nn.ParameterDict()
        # self.W_dict = nn.ParameterDict()
        self.attention_dict = nn.ParameterDict()
        
        ## weight for adding attention scores 
        # self.weights = nn.Parameter(torch.empty(len(model_dict)))
        # self.weights = nn.Parameter(torch.ones(len(model_dict)), requires_grad=True)

        ## or we can have one weight for each model
        self.weight_dict = nn.ParameterDict()
    

        for model_name in model_dict:
            # add one linear layer per model
            lin_layer = nn.Linear(self.args.embed_size_dict[model_name], args.num_labels)
            self.lin_layer_dict[model_name] = lin_layer
            ## attention weight matrix for each model 
            # self.W_dict = nn.Linear(self.args.embed_size_dict[model_name], self.args.embed_size_dict[model_name])

            ## one set of weights for each model: UNCOMMENT THIS 
            self.weight_dict[model_name] = nn.Parameter(torch.ones(len(model_dict)), requires_grad=True)
            attention = nn.MultiheadAttention(
                embed_dim = self.args.embed_size_dict[model_name],
                num_heads = 1,
                batch_first=True)
            self.attention_dict[model_name] = attention
    
    def forward(self, input_info_dict):
        hidden_states_all = [] ## [num_model, bs, seq_len, embed_size]
        self.logits_dict = dict()
        hidden_state_dict = dict()
        for model_name in self.model_dict:
            ## get information
            model = self.model_dict[model_name]
            input_info = input_info_dict[model_name]
            input_ids, attn_mask, token_type_ids = input_info["input_ids"], input_info["attention_mask"], input_info["labels"]
            ## get contexualized representation
            encoded = model(return_dict = True, output_hidden_states=True, input_ids=input_ids.to(self.args.device), attention_mask = attn_mask.to(self.args.device))
            hidden_states = encoded["hidden_states"][-1]  ## [bs, seq_len, embed_size]
            hidden_state_dict[model_name] = hidden_states
            self.logits_dict[model_name] = self.lin_layer_dict[model_name](hidden_states)
        
        ## compute cross attention / self attention
        for model_name_q in hidden_state_dict:
            query = hidden_state_dict[model_name_q]
            hidden_states_sum = torch.zeros_like(query)
            count = 0
            # weights = self.weight_dict[model_name_q] ## can change to one weight one model
            for model_name_k in hidden_state_dict:
                key = hidden_state_dict[model_name_k]
                attn_output, attn_output_weights = self.attention_dict[model_name](
                    query = query, 
                    key = key, 
                    value = key
                )
                ## write the attention ourselves? 
                hidden_states_sum = torch.add(hidden_states_sum, attn_output * self.weight_dict[model_name_q][count])
                count += 1
            logit = nn.functional.softmax(self.lin_layer_dict[model_name](hidden_states_sum), dim=-1)
            self.logits_dict[model_name] = logit
    
        return self.logits_dict ## {model_name: logit}

        

class MTL_classifier(BaseEstimator):

    def step(self, data):
        logits_dict = self.model(
            input_info_dict = data
        )
        if self.mode == "train":
            self.optimizer.zero_grad()
            loss = torch.tensor(0.00, requires_grad = True)
            for model_name, logit in logits_dict.items():
                if model_name == 'xlm-roberta-base':
                    loss_tmp1 = self.criterion[model_name](logit.view(-1, self.cfg.num_labels), data[model_name]["labels"].view(-1))
                else: 
                    loss_tmp2 = self.criterion[model_name](logit.view(-1, self.cfg.num_labels), data[model_name]["labels"].view(-1))
                # loss = torch.cat((loss, torch.unsqueeze(self.criterion[model_name](logit.view(-1, self.cfg.num_labels), data[model_name]["labels"].view(-1)), 0)))
                ## todo: penalize disagreement by adding other loss 
                ## todo: penalize weighted loss instead of simple sum? 
            # loss = torch.sum(loss)
            loss = loss_tmp1 + loss_tmp2
            loss.backward()
            self.optimizer.step()
            # print("=========  step weight ===========")
            # print(list(self.model.parameters())[0].grad)
            # print(self.model.attention_layers[1].in_proj_weight[0][:10])
            # print(self.model.attention_layers[2].in_proj_weight[0][:10])
            # print(self.model.attention_layers[3].in_proj_weight[0][:10])
            # print(self.model.attention_layers[4].in_proj_weight[0][:10])
            # print(self.model.attention_layers[5].in_proj_weight[0][:10])
            # print("==================================")

            # print(self.model.attention_layers[0].out_proj.weight[0][:10])
            # print(self.model.attention_layers[1].out_proj.weight[0][:10])
            # print(self.model.attention_layers[2].out_proj.weight[0][:10])
            # print(self.model.attention_layers[3].out_proj.weight[0][:10])
            # print(self.model.attention_layers[4].out_proj.weight[0][:10])
            # print(self.model.attention_layers[5].out_proj.weight[0][:10])
            if self.scheduler is not None:
                self.scheduler.step()
            for key, val in logits_dict.items():
                logits_dict[key] = val.detach().cpu()
            self.optimizer.zero_grad()
            return {
                "loss" : loss.detach().cpu().item(), 
                "logits_dict" : logits_dict, ## softmax this 
                "label" : data[self.cfg.word_model]["labels"]
                    }
        elif self.mode in ("dev", "test"):
            for key, val in logits_dict.items():
                logits_dict[key] = val.detach().cpu()
            return {
                "loss" : None, 
                "logits_dict" : logits_dict,
                "label" : data[self.cfg.word_model]["labels"]
                }
    
    def _eval(self, evalloader): 
        self.model.eval()
        tbar = tqdm(evalloader, dynamic_ncols=True)
        eval_loss = []
        ys = []
        preds = []

        if self.evaluate_metric is None:
            self.evaluate_metric = dict()
            self.evaluate_metric['f1'] = evaluate.load("f1")

        for data in tbar: 
            ret_step = self.step(data)   ## y: [bs, seq_len]
            loss, logits_dict, y = ret_step['loss'], ret_step['logits_dict'], ret_step['label']
            logit_word = logits_dict[self.cfg.word_model]
            # prob = torch.nn.functional.softmax(logit_word, dim=-1) ## softmax logit_word, [bs, seq_len, num_label] 
            pred = torch.argmax(logit_word, dim = -1) ## predicted, [bs, seq_len]
            if self.mode == 'dev': 
                # tbar.set_description('dev_loss - {:.4f}'.format(loss))
                # eval_loss.append(loss)
                ys.append(y)
            preds.append(pred) ## use pred for F1 and change how you append 
        # loss = np.mean(eval_loss).item() if self.mode == 'dev' else None
        # ys = np.concatenate(ys, axis=0) if self.mode == 'dev' else None
        # probs = np.concatenate(probs, axis=0)
        
        if self.mode == 'dev':
            flatten_ys, flatten_pred = np.array([]), np.array([])
            for y_ in ys:
                flatten_ys = np.append(flatten_ys, np.array(y_).ravel())
            for p_ in preds:
                flatten_pred = np.append(flatten_pred, np.array(p_).ravel())
            
            eval_ys, eval_pred = np.array([]), np.array([])
            for y_, p_ in zip(flatten_ys, flatten_pred):
                if y_ != -100:
                    eval_ys = np.append(eval_ys, y_)
                    eval_pred = np.append(eval_pred, p_)

            results = self.evaluate_metric['f1'].compute(predictions=eval_pred, references=eval_ys, average='macro')
            print(f"====== F1 result: {results}======")

            # if self.writer is not None: 
            #     self.writer.add_scalar('dev/loss', loss, self.dev_step)
            #     self.writer.add_scalar('dev/macro/auc', macro_auc, self.dev_step)
            #     self.writer.add_scalar('dev/micro/auc', micro_auc, self.dev_step)
            #     if self.pred_thold is not None: 
            #         yhats = (probs > self.pred_thold).astype(int)
            #         macros = precision_recall_fscore_support(ys, yhats, average='macro')
            #         self.writer.add_scalar('dev/macro/precision', macros[0], self.dev_step)
            #         self.writer.add_scalar('dev/macro/recall', macros[1], self.dev_step)
            #         self.writer.add_scalar('dev/macro/f1', macros[2], self.dev_step)
            #         micros = precision_recall_fscore_support(ys, yhats, average='micro')
            #         self.writer.add_scalar('dev/micro/precision', micros[0], self.dev_step)
            #         self.writer.add_scalar('dev/micro/recall', micros[1], self.dev_step)
            #         self.writer.add_scalar('dev/micro/f1', micros[2], self.dev_step)
        return eval_pred, eval_ys


# class weighted_ensemble(BaseClassifier):
#     def __init__(self, model_dict, args):
#         super().__init__()
#         self.model_dict = model_dict
#         total_embed_size = 0
#         for model_name in model_dict:
#             total_embed_size += self.args.embed_size_dict[model_name]
#         self.lin1 = nn.Linear(total_embed_size, args.num_labels)
#         self.args = args
    
#     def forward(self, input_info_dict):
#         hidden_states_all = [] ## [num_model, bs, seq_len, embed_size]
#         for model_name in self.model_dict:
#             model = self.model_dict[model_name]
#             input_info = input_info_dict[model_name]
#             input_ids, attn_mask, token_type_ids = input_info["input_ids"], input_info["attention_mask"], input_info["ner_tags"]
#             encoded = model(input_ids = input_ids, 
#                             attention_mask = attn_mask,
#                             token_type_ids = token_type_ids)
#             hidden_states = encoded[0]  ## [bs, seq_len, embed_size]
#             hidden_states_all.append(hidden_states) 
#             ##########################
#             ## make sure that seq_len is aligned fr subword/character model
#             ###########################



#         hidden_states_cat = torch.cat(hidden_states_all, dim = 3)
#         logits = self.lin1(hidden_states_cat)
#         return logits ## [bs, seq_len, num_labels]



class flat_MTL(nn.Module):
    def __init__(self, model_dict, args):
        super().__init__()  ## delete this ??
        self.model_dict = model_dict
        self.args = args
        self.lin_layer_dict = nn.ModuleDict()
        self.num_att_layers = 1
        self.attention_layers = nn.ModuleList(
            [nn.MultiheadAttention(
            embed_dim = self.args.embed_size_dict[self.args.word_model],
            num_heads = 1,
            batch_first=True) for _ in range(self.num_att_layers)]
        )
        # self.attention_layer = nn.MultiheadAttention(
        #     embed_dim = self.args.embed_size_dict[self.args.word_model],
        #     num_heads = 1,
        #     batch_first=True)
        
        for model_name in model_dict:
            # add one linear layer per model
            lin_layer = nn.Linear(self.args.embed_size_dict[model_name], args.num_labels)
            self.lin_layer_dict[model_name] = lin_layer
    
    def forward(self, input_info_dict):
        hidden_states_all = [] ## [num_model, bs, seq_len, embed_size]
        self.logits_dict = dict()
        self.hidden_states_dict = dict()
        for model_name in self.model_dict:
            ## get information
            model = self.model_dict[model_name]
            input_info = input_info_dict[model_name]
            input_ids, attn_mask, token_type_ids = input_info["input_ids"], input_info["attention_mask"], input_info["labels"]
            ## get contexualized representation
            encoded = model(return_dict = True, output_hidden_states=True, input_ids=input_ids, attention_mask = attn_mask)
            hidden_states = encoded["hidden_states"][-1]  ## [bs, seq_len, embed_size]
            ## TODO: add positional embbeding 
            hidden_states_all.append(hidden_states)
        hidden_states_all = torch.cat(hidden_states_all, dim = 1)
        attn_output, attn_output_weights = self.attention_layers[0](
            query = hidden_states_all, 
            key = hidden_states_all, 
            value = hidden_states_all
        )

        for i in range(1, self.num_att_layers):
            attn_output, attn_output_weights = self.attention_layers[i](
            query = attn_output, 
            key = attn_output, 
            value = attn_output
        )

        ## separate hidden states from the global attention output
        count = 0
        for model_name in self.model_dict: 
            input_info = input_info_dict[model_name]
            seq_len_model = input_info["input_ids"].shape[1]
            count_next = count + seq_len_model
            self.hidden_states_dict[model_name] = attn_output[:, count:count_next, :]
            self.logits_dict[model_name] = nn.functional.softmax(self.lin_layer_dict[model_name](self.hidden_states_dict[model_name]), dim=-1)
            count += seq_len_model
    
        return self.logits_dict ## {model_name: logit}
            
            
    
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

