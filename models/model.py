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
from transformers import AutoModel
import torch.nn.functional as F
from datasets import load_metric

id2tag = {
    0: "O",
    1: "B-corporation",
    2: "I-corporation",
    3: "B-creative-work",
    4: "I-creative-work",
    5: "B-group",
    6: "I-group",
    7: "B-location",
    8: "I-location",
    9: "B-person",
    10: "I-person",
    11: "B-product",
    12: "I-product",
}
tag2id = {tag: id for id, tag in id2tag.items()}


def encode_tags(examples, tokenized_inputs):
    labels = []
    for i, label in enumerate(examples[f"ner_tags"]):
        word_ids = tokenized_inputs.word_ids(
            batch_index=i
        )  # Map tokens to their respective word.
        previous_word_idx = None
        label_ids = []
        for word_idx in word_ids:  # Set the special tokens to -100.
            if word_idx is None:
                label_ids.append(-100)
            elif (
                word_idx != previous_word_idx
            ):  # Only label the first token of a given word.
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
        for encoding, label, model_name in zip(
            self.encodings, self.labels, self.model_names
        ):
            item = {key: torch.tensor(val[idx]) for key, val in encoding.items()}
            item["labels"] = torch.tensor(label[idx])
            result[model_name] = item
        return result

    def __len__(self):
        return len(self.labels[0])


## SST data for
class SSTDatasetMulti(torch.utils.data.Dataset):
    def __init__(self, encodings, model_names):
        # inputs are as Lists of encodings, labels, and models names : []
        self.encodings = encodings
        self.model_names = model_names

    def __getitem__(self, idx):
        result = {}
        for encoding, model_name in zip(self.encodings, self.model_names):
            item = {key: torch.tensor(val[idx]) for key, val in encoding.items()}
            item["labels"] = item["input_ids"]
            result[model_name] = item
        return result

    def __len__(self):
        return len(self.encodings[0]["input_ids"])  ## TODO HERE!


class BookWikiDatasetMulti(torch.utils.data.Dataset):
    def __init__(self, encodings, model_names):
        # inputs are as Lists of encodings, labels, and models names : []
        self.encodings = encodings[0]
        self.model_names = model_names

    def __getitem__(self, idx):
        result = {}
        item = {
            key: torch.tensor(val[idx]) for key, val in self.encodings["char"].items()
        }
        result["char"] = item
        item = {
            key: torch.tensor(val[idx]) for key, val in self.encodings["word"].items()
        }
        result["word"] = item
        return result

    def __len__(self):
        return len(self.encodings["word"]["input_ids"])  ## TODO HERE!


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
            tokenizer_dict[name] = AutoTokenizer.from_pretrained(
                name, add_prefix_space=self.args.prefix_space
            )
            self.model_dict[
                name
            ] = transformers.AutoModelForTokenClassification.from_pretrained(
                name, num_labels=self.args.num_labels
            )
        self.tokenizer_dict = tokenizer_dict

        ## use previous implementation of tokenization for each granularity
        # TODO
        # Should end up something like
        # self.data = {model_name : DatasetDict }, where
        # datadict is created using previous tokenization method in data.py
        for name in self.args.model_names:
            # name = self.args.granularities_model[granularity]
            print(f"============== tokenizing {name} ==============")
            # if granularity == "character":
            if "canine" in name:
                clm = CharacterLevelMapping(self.args.to_char_method)
                wnut_character_level = clm.character_level_wnut(self.wnut)
                tok = Tokenization(name, self.args.prefix_space)
                tokenized_wnut = tok.tokenize_for_char_manual(wnut_character_level)
                self.data_[name] = tokenized_wnut
                self.data_length[name] = tokenized_wnut["train"].num_rows
            # elif granularity == "subword_50k" or granularity == "subword_30k":
            else:
                # tok = Tokenization(self.args.granularities_model[granularity], self.args.prefix_space)
                tok = Tokenization(name, self.args.prefix_space)
                tokenized_wnut = self.wnut.map(
                    tok.tokenize_and_align_labels, batched=True
                )  ## was previously wnut_character level
                self.data_[name] = tokenized_wnut
                self.data_length[name] = tokenized_wnut["train"].num_rows
            assert is_aligned(tokenized_wnut)
        assert granularity_aligned(self.data_length)

        # raise NotImplementedError

    def __len__(self):
        ## return the length of dataset of any granularity (they must be all equal)
        for model_name, data_length in self.data_length_dict.items():
            print(
                "model_name {model_name}, the length of dataset is : {length} ".format(
                    model_name=model_name, length=data_length
                )
            )
        return set(self.data_length.values())[0]

    def __getitem__(self, idx):
        ## return {model_name : input_info}, where
        ## input_info is input_ids, attn_mask, token_type_ids for each granularity.
        return_dict_train = dict()
        return_dict_val = dict()
        return_dict_test = dict()
        for granularity in self.args.granularities:
            name = self.args.granularities_model[granularity]
            return_dict_train[name] = self.data_[name]["train"][idx]
            return_dict_val[name] = self.data_[name]["validation"][idx]
            return_dict_test[name] = self.data_[name]["test"][idx]

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
            lin_layer = nn.Linear(
                self.args.embed_size_dict[model_name], args.num_labels
            )
            self.lin_layer_dict[model_name] = lin_layer
            ## attention weight matrix for each model
            # self.W_dict = nn.Linear(self.args.embed_size_dict[model_name], self.args.embed_size_dict[model_name])

            ## one set of weights for each model: UNCOMMENT THIS
            self.weight_dict[model_name] = nn.Parameter(
                torch.ones(len(model_dict)), requires_grad=True
            )
            attention = nn.MultiheadAttention(
                embed_dim=self.args.embed_size_dict[model_name],
                num_heads=1,
                batch_first=True,
            )
            self.attention_dict[model_name] = attention

    def forward(self, input_info_dict):
        hidden_states_all = []  ## [num_model, bs, seq_len, embed_size]
        self.logits_dict = dict()
        hidden_state_dict = dict()
        for model_name in self.model_dict:
            ## get information
            model = self.model_dict[model_name]
            input_info = input_info_dict[model_name]
            input_ids, attn_mask, token_type_ids = (
                input_info["input_ids"],
                input_info["attention_mask"],
                input_info["labels"],
            )
            ## get contexualized representation
            encoded = model(
                return_dict=True,
                output_hidden_states=True,
                input_ids=input_ids.to(self.args.device),
                attention_mask=attn_mask.to(self.args.device),
            )
            hidden_states = encoded["hidden_states"][-1]  ## [bs, seq_len, embed_size]
            hidden_state_dict[model_name] = hidden_states
            self.logits_dict[model_name] = self.lin_layer_dict[model_name](
                hidden_states
            )

        ## compute cross attention / self attention
        for model_name_q in hidden_state_dict:
            query = hidden_state_dict[model_name_q]
            hidden_states_sum = torch.zeros_like(query)
            count = 0
            # weights = self.weight_dict[model_name_q] ## can change to one weight one model
            for model_name_k in hidden_state_dict:
                key = hidden_state_dict[model_name_k]
                attn_output, attn_output_weights = self.attention_dict[model_name](
                    query=query, key=key, value=key
                )
                ## write the attention ourselves?
                hidden_states_sum = torch.add(
                    hidden_states_sum,
                    attn_output * self.weight_dict[model_name_q][count],
                )
                count += 1
            logit = self.lin_layer_dict[model_name](hidden_states_sum)
            self.logits_dict[model_name] = logit

        return self.logits_dict  ## {model_name: logit}


# do not eval
class MLM_classifier(BaseEstimator):
    def step(self, data):
        self.optimizer.zero_grad()
        logits_dict = self.model(input_info_dict=data)
        if self.mode == "train":
            count = 0
            for model_name, logit in logits_dict.items():
                vocab_size = (
                    1114112
                    if model_name == "google/canine-s"
                    else self.model.base.model_dict[model_name].config.vocab_size
                )
                if count == 0:
                    loss = self.criterion[model_name](
                        logit.view(-1, vocab_size),
                        data[model_name]["labels"].view(-1).to(self.cfg.device),
                    )
                else:
                    loss += self.criterion[model_name](
                        logit.view(-1, vocab_size),
                        data[model_name]["labels"].view(-1).to(self.cfg.device),
                    )
                count += 1
            loss.backward()
            self.optimizer.step()
            if self.scheduler is not None:
                self.scheduler.step()
            for key, val in logits_dict.items():
                logits_dict[key] = val.detach().cpu()
            self.optimizer.zero_grad()
            return {
                "loss": loss.detach().cpu().item(),
                "logits_dict": logits_dict,  ## softmax this
                "label": data[self.cfg.word_model]["input_ids"],
            }
        elif self.mode in ("dev", "test"):
            for key, val in logits_dict.items():
                logits_dict[key] = val.detach().cpu()
            return {
                "loss": None,
                "logits_dict": logits_dict,
                "label": data[self.cfg.word_model]["input_ids"],
            }

    def _eval(self, evalloader):
        print("we do not eval")


class MTL_classifier(BaseEstimator):
    def step(self, data):
        self.optimizer.zero_grad()
        logits_dict = self.model(input_info_dict=data)
        if self.mode == "train":
            # loss = torch.tensor(0.00, requires_grad = True)
            count = 0
            for model_name, logit in logits_dict.items():
                if count == 0:
                    loss = self.criterion[model_name](
                        logit.view(-1, self.cfg.num_labels),
                        data[model_name]["labels"].view(-1).to(self.cfg.device),
                    )
                else:
                    loss += self.criterion[model_name](
                        logit.view(-1, self.cfg.num_labels),
                        data[model_name]["labels"].view(-1).to(self.cfg.device),
                    )
                count += 1
                # if model_name == 'xlm-roberta-base':
                #     loss_tmp1 = self.criterion[model_name](logit.view(-1, self.cfg.num_labels), data[model_name]["labels"].view(-1))
                # else:
                #     loss_tmp2 = self.criterion[model_name](logit.view(-1, self.cfg.num_labels), data[model_name]["labels"].view(-1))
                # loss = torch.cat((loss, torch.unsqueeze(self.criterion[model_name](logit.view(-1, self.cfg.num_labels), data[model_name]["labels"].view(-1)), 0)))
                ## todo: penalize disagreement by adding other loss
                ## todo: penalize weighted loss instead of simple sum?
            # loss = torch.sum(loss)
            # loss = loss_tmp1 + loss_tmp2
            loss.backward()
            self.optimizer.step()
            # print("=========  step weight ===========")
            # print(list(self.model.parameters())[0].grad)
            # print(self.model.bert_layers[0].self_attention.query_lin.weight)
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
                "loss": loss.detach().cpu().item(),
                "logits_dict": logits_dict,  ## softmax this
                "label": data[self.cfg.word_model]["labels"],
            }
        elif self.mode in ("dev", "test"):
            for key, val in logits_dict.items():
                logits_dict[key] = val.detach().cpu()
            return {
                "loss": None,
                "logits_dict": logits_dict,
                "label": data[self.cfg.word_model]["labels"],
            }

    def _eval(self, evalloader):
        self.model.eval()
        tbar = tqdm(evalloader, dynamic_ncols=True)
        eval_loss = []
        ys = []
        preds = []

        if self.evaluate_metric is None:
            self.evaluate_metric = dict()
            self.evaluate_metric["f1"] = evaluate.load("f1")
            self.evaluate_metric["all"] = load_metric("seqeval")

        for data in tbar:
            ret_step = self.step(data)  ## y: [bs, seq_len]
            loss, logits_dict, y = (
                ret_step["loss"],
                ret_step["logits_dict"],
                ret_step["label"],
            )
            logit_word = logits_dict[self.cfg.word_model]
            prob = torch.nn.functional.softmax(
                logit_word, dim=-1
            )  ## softmax logit_word, [bs, seq_len, num_label]
            pred = torch.argmax(prob, dim=-1)  ## predicted, [bs, seq_len]
            if self.mode == "dev":
                # tbar.set_description('dev_loss - {:.4f}'.format(loss))
                # eval_loss.append(loss)
                ys.append(y)
            preds.append(pred)  ## use pred for F1 and change how you append
        # loss = np.mean(eval_loss).item() if self.mode == 'dev' else None
        # ys = np.concatenate(ys, axis=0) if self.mode == 'dev' else None
        # probs = np.concatenate(probs, axis=0)

        if self.mode == "dev":
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

            results = self.evaluate_metric["f1"].compute(
                predictions=eval_pred, references=eval_ys, average="macro"
            )
            print(f"====== F1 result: {results}======")

            true_predictions = [
                [
                    id2tag[p]
                    for (p, l) in zip(np.array(p_).ravel(), np.array(y_).ravel())
                    if l != -100
                ]
                for p_, y_ in zip(preds, ys)
            ]
            true_labels = [
                [
                    id2tag[l]
                    for (p, l) in zip(np.array(p_).ravel(), np.array(y_).ravel())
                    if l != -100
                ]
                for p_, y_ in zip(preds, ys)
            ]

            result_ = self.evaluate_metric["all"].compute(
                predictions=true_predictions, references=true_labels,
            )
            # print(f"{result_}")
            print(f"===== *F1 result: {result_['overall_f1']}======")

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
        hidden_states_all = []  ## [num_model, bs, seq_len, embed_size]
        for model_name in self.model_dict:
            model = self.model_dict[model_name]
            input_info = input_info_dict[model_name]
            input_ids, attn_mask, token_type_ids = (
                input_info["input_ids"],
                input_info["attention_mask"],
                input_info["ner_tags"],
            )
            encoded = model(
                input_ids=input_ids,
                attention_mask=attn_mask,
                token_type_ids=token_type_ids,
            )
            hidden_states = encoded[0]  ## [bs, seq_len, embed_size]
            hidden_states_all.append(hidden_states)
            ##########################
            ## make sure that seq_len is aligned fr subword/character model
            ###########################

        hidden_states_cat = torch.cat(hidden_states_all, dim=3)
        logits = self.lin1(hidden_states_cat)
        return logits  ## [bs, seq_len, num_labels]


class flat_MTL(nn.Module):
    def __init__(self, model_dict, args):
        super().__init__()  ## delete this ??
        self.model_dict = model_dict
        self.args = args
        self.lin_layer_dict = nn.ModuleDict()
        self.num_att_layers = args.num_att_layers
        if self.args.layer_type == "att":
            self.is_bert_layers = False
        else:
            self.is_bert_layers = True

        if self.is_bert_layers:
            self.bert_layers = nn.ModuleList(
                [
                    BertLayer(emb_size=self.args.embed_size_dict[self.args.word_model])
                    for _ in range(self.num_att_layers)
                ]
            )
        else:
            self.attention_layers = nn.ModuleList(
                [
                    nn.MultiheadAttention(
                        embed_dim=self.args.embed_size_dict[self.args.word_model],
                        num_heads=1,
                        batch_first=True,
                    )
                    for _ in range(self.num_att_layers)
                ]
            )

        # self.attention_layer = nn.MultiheadAttention(
        #     embed_dim = self.args.embed_size_dict[self.args.word_model],
        #     num_heads = 1,
        #     batch_first=True)

        for model_name in model_dict:
            # add one linear layer per model
            lin_layer = nn.Linear(
                self.args.embed_size_dict[model_name], args.num_labels
            )
            self.lin_layer_dict[model_name] = lin_layer

    def forward(self, input_info_dict):
        hidden_states_all = []  ## [num_model, bs, seq_len, embed_size]
        self.logits_dict = dict()
        self.hidden_states_dict = dict()
        for model_name in self.model_dict:
            ## get information
            model = self.model_dict[model_name]
            input_info = input_info_dict[model_name]
            input_ids, attn_mask, token_type_ids = (
                input_info["input_ids"],
                input_info["attention_mask"],
                input_info["labels"],
            )
            ## get contexualized representation
            # print(input_ids.shape)
            encoded = model(
                return_dict=True,
                output_hidden_states=True,
                input_ids=input_ids.to(self.args.device),
                attention_mask=attn_mask.to(self.args.device),
            )
            hidden_states = encoded["hidden_states"][-1]  ## [bs, seq_len, embed_size]
            ## TODO: add positional embbeding
            hidden_states_all.append(hidden_states)
        hidden_states_all = torch.cat(hidden_states_all, dim=1)

        if self.num_att_layers > 0:
            if self.is_bert_layers:
                attn_output = self.bert_layers[0](hidden_states_all)

                for i in range(1, self.num_att_layers):
                    attn_output = self.bert_layers[i](hidden_states_all)["attn_output"]
            else:
                attn_output, attn_output_weights = self.attention_layers[0](
                    query=hidden_states_all,
                    key=hidden_states_all,
                    value=hidden_states_all,
                )

                for i in range(1, self.num_att_layers):
                    attn_output, attn_output_weights = self.attention_layers[i](
                        query=attn_output, key=attn_output, value=attn_output
                    )
        else:
            attn_output = hidden_states_all

        ## separate hidden states from the global attention output
        ##################################################################
        ######## SOURCE OF A bug
        count = 0
        for model_name in self.model_dict:
            input_info = input_info_dict[model_name]
            seq_len_model = input_info["input_ids"].shape[1]
            count_next = count + seq_len_model
            self.hidden_states_dict[model_name] = attn_output[:, count:count_next, :]
            self.logits_dict[model_name] = self.lin_layer_dict[model_name](
                self.hidden_states_dict[model_name]
            )
            count += seq_len_model
        #####################################################################

        return self.logits_dict  ## {model_name: logit}


class flat_MTL_for_MLM(nn.Module):
    def __init__(self, model_dict, args):
        super().__init__()  ## delete this ??
        self.model_dict = model_dict
        self.args = args
        self.lin_layer_dict = nn.ModuleDict()
        self.num_att_layers = args.num_att_layers
        self.attention_layers = nn.ModuleList(
            [
                nn.MultiheadAttention(
                    embed_dim=self.args.embed_size_dict[self.args.word_model],
                    num_heads=1,
                    batch_first=True,
                )
                for _ in range(self.num_att_layers)
            ]
        )
        # self.attention_layer = nn.MultiheadAttention(
        #     embed_dim = self.args.embed_size_dict[self.args.word_model],
        #     num_heads = 1,
        #     batch_first=True)

        for model_name in model_dict:
            # add one linear layer per model
            lin_layer = nn.Linear(
                model_dict[model_name].config.hidden_size,
                model_dict[model_name].config.vocab_size,
            )
            self.lin_layer_dict[model_name] = lin_layer

    def forward(self, input_info_dict):
        hidden_states_all = []  ## [num_model, bs, seq_len, embed_size]
        self.logits_dict = dict()
        self.hidden_states_dict = dict()
        for model_name in self.model_dict:
            ## get information
            model = self.model_dict[model_name]
            input_info = input_info_dict[model_name]
            input_ids, attn_mask, token_type_ids = (
                input_info["input_ids"],
                input_info["attention_mask"],
                input_info["labels"],
            )
            ## get contexualized representation
            encoded = model(
                return_dict=True,
                output_hidden_states=True,
                input_ids=input_ids.to(self.args.device),
                attention_mask=attn_mask.to(self.args.device),
            )
            hidden_states = encoded["hidden_states"][-1]  ## [bs, seq_len, embed_size]
            ## TODO: add positional embbeding
            hidden_states_all.append(hidden_states)
        hidden_states_all = torch.cat(hidden_states_all, dim=1)
        if self.num_att_layers > 0:
            attn_output, attn_output_weights = self.attention_layers[0](
                query=hidden_states_all, key=hidden_states_all, value=hidden_states_all
            )

            for i in range(1, self.num_att_layers):
                attn_output, attn_output_weights = self.attention_layers[i](
                    query=attn_output, key=attn_output, value=attn_output
                )
        else:
            attn_output = hidden_states_all

        ## separate hidden states from the global attention output
        ##################################################################
        ######## SOURCE OF A bug
        count = 0
        for model_name in self.model_dict:
            input_info = input_info_dict[model_name]
            seq_len_model = input_info["input_ids"].shape[1]
            count_next = count + seq_len_model
            self.hidden_states_dict[model_name] = attn_output[:, count:count_next, :]
            self.logits_dict[model_name] = self.lin_layer_dict[model_name](
                self.hidden_states_dict[model_name]
            )
            count += seq_len_model
        #####################################################################

        return self.logits_dict  ## {model_name: logit}


class MTL_base(nn.Module):
    def __init__(self, model_dict, args):
        super().__init__()  ## delete this ??
        self.model_dict = model_dict
        self.args = args
        self.lin_layer_dict = nn.ModuleDict()
        self.num_align_layers = args.num_att_layers
        if self.args.layer_type == "att":
            self.align_layers = nn.ModuleList(
                [
                    nn.MultiheadAttention(
                        embed_dim=self.args.embed_size_dict[self.args.word_model],
                        num_heads=1,
                        batch_first=True,
                    )
                    for _ in range(self.num_align_layers)
                ]
            )
        if self.args.layer_type == "bert":
            self.align_layers = nn.ModuleList(
                [
                    BertLayer(emb_size=self.args.embed_size_dict[self.args.word_model])
                    for _ in range(self.num_align_layers)
                ]
            )

    def forward(self, input_info_dict):
        hidden_states_all = []  ## [num_model, bs, seq_len, embed_size]
        self.hidden_states_dict = dict()
        for model_name in self.model_dict:
            ## get information
            model = self.model_dict[model_name]
            input_info = input_info_dict[model_name]
            input_ids, attn_mask, token_ids = (
                input_info["input_ids"],
                input_info["attention_mask"],
                input_info["labels"],
            )
            ## get contexualized representation
            # print(input_ids.shape)
            encoded = model(
                return_dict=True,
                output_hidden_states=True,
                input_ids=input_ids.to(self.args.device),
                attention_mask=attn_mask.to(self.args.device),
            )
            hidden_states = encoded["hidden_states"][-1]  ## [bs, seq_len, embed_size]
            ## TODO: add positional embbeding
            hidden_states_all.append(hidden_states)
        hidden_all = torch.cat(hidden_states_all, dim=1)

        for i in range(self.num_align_layers):
            if self.args.layer_type == "bert":
                hidden_all = self.align_layers[i](hidden_all)
            if self.args.layer_type == "att":
                hidden_all, _ = self.align_layers[0](
                    query=hidden_all, key=hidden_all, value=hidden_all
                )

        ## separate hidden states from the global attention output
        ##################################################################
        ######## SOURCE OF A bug
        count = 0
        for model_name in self.model_dict:
            input_info = input_info_dict[model_name]
            seq_len_model = input_info["input_ids"].shape[1]
            count_next = count + seq_len_model
            self.hidden_states_dict[model_name] = hidden_all[:, count:count_next, :]
            count += seq_len_model
        #####################################################################

        return self.hidden_states_dict  ## {model_name: logit}


class flat_MTL_w_base(nn.Module):
    def __init__(self, base, args):
        super().__init__()
        self.args = args
        self.base = base
        self.lin_layer_dict = nn.ModuleDict()
        for model_name in base.model_dict:
            # add one linear layer per model
            lin_layer = nn.Linear(
                self.args.embed_size_dict[model_name], args.num_labels
            )
            self.lin_layer_dict[model_name] = lin_layer

    def forward(self, input_info_dict):
        self.logits_dict = dict()
        hidden_states_dict = self.base(input_info_dict)
        for model_name in self.lin_layer_dict:
            self.logits_dict[model_name] = self.lin_layer_dict[model_name](
                hidden_states_dict[model_name]
            )
        return self.logits_dict


class flat_MLM_w_base(nn.Module):
    def __init__(self, base, args):
        super().__init__()
        self.args = args
        self.base = base
        self.lin_layer_dict = nn.ModuleDict()
        for model_name in base.model_dict:
            # add one linear layer per model
            vocab_size = (
                1114112
                if model_name == "google/canine-s"
                else base.model_dict[model_name].config.vocab_size
            )
            lin_layer = nn.Linear(self.args.embed_size_dict[model_name], vocab_size)
            self.lin_layer_dict[model_name] = lin_layer

    def forward(self, input_info_dict):
        self.logits_dict = dict()
        hidden_states_dict = self.base(input_info_dict)
        for model_name in self.lin_layer_dict:
            self.logits_dict[model_name] = self.lin_layer_dict[model_name](
                hidden_states_dict[model_name]
            )
        return self.logits_dict


class baseline_model(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.backbone = AutoModel.from_pretrained(args.word_model)
        self.lin = nn.Linear(768, args.num_labels)
        self.args = args

    def forward(self, data):
        input_info = data[self.args.word_model]
        input_ids, attn_mask, token_type_ids = (
            input_info["input_ids"],
            input_info["attention_mask"],
            input_info["labels"],
        )
        encoded = self.backbone(
            return_dict=True,
            output_hidden_states=True,
            input_ids=input_ids.to(self.args.device),
            attention_mask=attn_mask.to(self.args.device),
        )
        hidden_states = encoded["hidden_states"][-1]
        logits = self.lin(hidden_states)
        return logits


class baseline_classifier(BaseEstimator):
    def step(self, data):
        self.optimizer.zero_grad()
        logits = self.model(data=data)
        ## softmax the logit before loss!
        loss = self.criterion(
            logits.view(-1, self.cfg.num_labels),
            data[self.cfg.word_model]["labels"].view(-1).to(self.cfg.device),
        )
        if self.mode == "train":
            # loss = torch.tensor(0.00, requires_grad = True)
            loss.backward()
            self.optimizer.step()
            if self.scheduler is not None:
                self.scheduler.step()
            self.optimizer.zero_grad()
            return {
                "loss": loss.detach().cpu().item(),
                "logits": logits.detach().cpu(),  ## softmax this
                "label": data[self.cfg.word_model]["labels"],
            }
        elif self.mode in ("dev", "test"):
            return {
                "loss": loss.detach().cpu().item(),
                "logits": logits.detach().cpu(),  ## softmax this
                "label": data[self.cfg.word_model]["labels"],
            }

    def _eval(self, evalloader):
        self.model.eval()
        tbar = tqdm(evalloader, dynamic_ncols=True)
        eval_loss = []
        ys = []
        preds = []

        if self.evaluate_metric is None:
            self.evaluate_metric = dict()
            self.evaluate_metric["f1"] = evaluate.load("f1")
            self.evaluate_metric["all"] = load_metric("seqeval")

        for data in tbar:
            ret_step = self.step(data)  ## y: [bs, seq_len]
            loss, logits, y = ret_step["loss"], ret_step["logits"], ret_step["label"]
            # logit_word = logits_dict[self.cfg.word_model]
            # prob = torch.nn.functional.softmax(logit_word, dim=-1) ## softmax logit_word, [bs, seq_len, num_label]
            pred = torch.argmax(logits, dim=-1)  ## predicted, [bs, seq_len]
            if self.mode == "dev":
                # tbar.set_description('dev_loss - {:.4f}'.format(loss))
                # eval_loss.append(loss)
                ys.append(y)
            preds.append(pred)  ## use pred for F1 and change how you append
        # loss = np.mean(eval_loss).item() if self.mode == 'dev' else None
        # ys = np.concatenate(ys, axis=0) if self.mode == 'dev' else None
        # probs = np.concatenate(probs, axis=0)

        if self.mode == "dev":
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

            results = self.evaluate_metric["f1"].compute(
                predictions=eval_pred, references=eval_ys, average="macro"
            )
            print(f"====== F1 result: {results}======")

            true_predictions = [
                [
                    id2tag[p]
                    for (p, l) in zip(np.array(p_).ravel(), np.array(y_).ravel())
                    if l != -100
                ]
                for p_, y_ in zip(preds, ys)
            ]
            true_labels = [
                [
                    id2tag[l]
                    for (p, l) in zip(np.array(p_).ravel(), np.array(y_).ravel())
                    if l != -100
                ]
                for p_, y_ in zip(preds, ys)
            ]

            result_ = self.evaluate_metric["all"].compute(
                predictions=true_predictions, references=true_labels,
            )
            # print(f"{result_}")
            print(f"===== *F1 result: {result_['overall_f1']}======")

        return eval_pred, eval_ys


class self_attention(nn.Module):
    def __init__(self, emb_size, dropout_p=0.1, eps=1e-12):
        super().__init__()
        self.emb_size = emb_size
        ## k, v, q projection
        self.query_lin = nn.Linear(emb_size, emb_size)
        self.key_lin = nn.Linear(emb_size, emb_size)
        self.value_lin = nn.Linear(emb_size, emb_size)
        self.dropout = nn.Dropout(p=dropout_p)  ## add dropout if you want
        # ## output projection
        # self.out_proj = nn.Linear(emb_size)
        # self.layerNorm = nn.LayerNorm(emb_size, eps = eps)
        # self.out_dropout = nn.Dropout(p = dropout_p)

    def forward(self, query, key, value):
        ## in-projection, [bs, seq_len, emb_size]
        q = self.query_lin(query)
        k = self.key_lin(key)
        v = self.value_lin(value)
        ## dot-product attention
        attention_weights = F.softmax(
            torch.matmul(q, k.transpose(1, 2)) / (k.shape[2] ** 0.5), dim=-1
        )  ## [bs, seq_len, seq_len]
        attention_output = torch.matmul(attention_weights, v)
        return {"attn_weights": attention_weights, "attn_output": attention_output}


class BertLayer(nn.Module):
    def __init__(self, emb_size, intermeidate_size=None, dropout_p=0.1, eps=1e-12):
        super().__init__()
        ## attention layer
        self.self_attention = self_attention(emb_size=emb_size, dropout_p=dropout_p)
        self.attention_dense = nn.Linear(emb_size, emb_size)
        self.attention_layer_norm = nn.LayerNorm(emb_size, eps=eps)
        self.attention_dropout = nn.Dropout(p=dropout_p)
        ## intermediate layer
        if intermeidate_size is None:
            intermeidate_size = emb_size
        self.interm_dense = nn.Linear(emb_size, intermeidate_size)
        self.interm_af = nn.GELU()
        ## output
        self.out_dense = nn.Linear(intermeidate_size, emb_size)
        self.out_layer_norm = nn.LayerNorm(emb_size, eps=eps)
        self.out_dropout = nn.Dropout(dropout_p)

    def add_norm(self, inputs, output, dense_layer, dropout, ln_layer):
        output_dense = dense_layer(output)
        droped_out = dropout(output_dense)
        result = ln_layer(droped_out + inputs)
        return result

    def forward(self, hidden_states):
        ## attention
        attention_out = self.self_attention(
            hidden_states, hidden_states, hidden_states
        )["attn_output"]
        ## add-norm
        attention_normed = self.add_norm(
            hidden_states,
            attention_out,
            self.attention_dense,
            self.attention_dropout,
            self.attention_layer_norm,
        )
        ## intermediate
        forward_linear = self.interm_dense(attention_normed)
        forward_act = self.interm_af(forward_linear)
        ## output
        out = self.out_dense(forward_act)
        out = self.add_norm(
            attention_normed,
            forward_act,
            self.out_dense,
            self.out_dropout,
            self.out_layer_norm,
        )
        return out


class sequential_MTL(nn.Module):
    def __init__(self, model_dict, args):
        super().__init__()
        self.model_dict = model_dict
        self.args = args
        self.lin_layer_dict = nn.ModuleDict()
        self.JSD = JSD()

        for model_name in model_dict:
            # add one linear layer per model
            lin_layer = nn.Linear(
                model_dict[model_name].config.hidden_size, self.args.num_labels
            )
            self.lin_layer_dict[model_name] = lin_layer

    def forward(self, data):
        prob = None
        prob_dict = dict()
        for model_name in self.model_dict:
            model = self.model_dict[model_name]
            input_info = data[model_name]
            input_ids, attn_mask, token_type_ids = (
                input_info["input_ids"],
                input_info["attention_mask"],
                input_info["labels"],
            )
            ## get contexualized representation
            encoded = model(
                return_dict=True,
                output_hidden_states=True,
                input_ids=input_ids.to(self.args.device),
                attention_mask=attn_mask.to(self.args.device),
            )
            hidden_states = encoded["hidden_states"][-1]  ## [bs, seq_len, embed_size]
            logit = self.lin_layer_dict[model_name](hidden_states)
            logit_prob = F.log_softmax(logit, dim=-1)  ##TODO: Log softmax???
            prob_dict[model_name] = logit_prob
            if prob is None:
                prob = logit_prob
            else:
                prob = torch.add(prob, logit_prob)
        return prob, prob_dict


class sequential_classifier(BaseEstimator):
    def step(self, data):
        # self.optimizer.zero_grad()
        prob, prob_dict = self.model(data=data)
        if self.mode == "train":
            count = 0
            for model_name, probs in prob_dict.items():
                self.optimizer.zero_grad()  ## clear grad at the start of every iteration
                if count == 0:
                    pred = probs.view(-1, self.cfg.num_labels)
                    ref = data[model_name]["labels"].view(-1).to(self.cfg.device)
                    ## rule out padding
                    pred = pred[ref != -100]
                    ref = ref[ref != -100]

                    loss = self.criterion[model_name](pred, ref)
                    loss.backward()
                    self.optimizer.step()
                    if self.scheduler is not None:
                        self.scheduler.step()
                    count += 1

                    ## residuals
                    ref_one_hot = F.one_hot(
                        ref, num_classes=self.cfg.num_labels
                    )  ## [num_tokens_in_batch, num_labels]
                    residuals = torch.subtract(ref_one_hot, pred)
                else:
                    pred = probs.view(-1, self.cfg.num_labels)
                    ref = data[model_name]["labels"].view(-1).to(self.cfg.device)

                    pred = pred[ref != -100]
                    ref = ref[ref != -100]
                    loss = self.JSD(pred, residuals)

                    loss.backward()
                    self.optimizer.step()
                    if self.scheduler is not None:
                        self.scheduler.step()

                    residuals = torch.subtract(residuals, pred)
                    count += 1
            pred = prob.view(-1, self.cfg.num_labels)
            ref = data[self.cfg.word_model]["labels"].view(-1).to(self.cfg.device)
            return {"pred": pred, "label": ref, "loss": loss}

        elif self.mode in ("dev", "test"):
            pred = prob.view(-1, self.cfg.num_labels)
            ref = data[self.cfg.word_model]["labels"].view(-1).to(self.cfg.device)
            loss = nn.CrossEntropyLoss().to(self.cfg.device)(pred, ref)
            return {"pred": pred, "label": ref, "loss": loss}

    def _eval(self, evalloader):
        self.model.eval()
        tbar = tqdm(evalloader, dynamic_ncols=True)
        eval_loss = []
        ys = []
        preds = []

        if self.evaluate_metric is None:
            self.evaluate_metric = dict()
            self.evaluate_metric["f1"] = evaluate.load("f1")
            self.evaluate_metric["all"] = load_metric("seqeval")

        for data in tbar:
            ret_step = self.step(data)  ## y: [bs, seq_len]
            loss, pred, labels = ret_step["loss"], ret_step["pred"], ret_step["label"]
            pred = torch.argmax(pred, dim=-1)  ## predicted, [bs, seq_len]
            ys.append(labels)
            preds.append(pred)
        preds = preds[ys != -100]
        ys = ys[ys != -100]

        results = self.evaluate_metric["f1"].compute(
            predictions=preds, references=ys, average="macro"
        )
        print(f"====== F1 result: {results}======")

        return preds, ys


class JSD(nn.Module):
    def __init__(self):
        super().__init__()
        self.KL = nn.KLDivLoss(reduction="batchmean")

    def forward(self, p, q):
        ## Note: p, q should already been log_softmaxed
        return self.KL(p, q) + self.KL(q, p)


class base_model(nn.Module):
    def __init__(self, name, args):
        super().__init__()
        self.backbone = AutoModel.from_pretrained(name)
        self.lin = nn.Linear(self.backbone.config.hidden_size, args.num_labels)
        self.args = args

    def forward(self, input_ids, attn_mask):
        encoded = self.backbone(
            return_dict=True,
            output_hidden_states=True,
            input_ids=input_ids,
            attention_mask=attn_mask,
        )
        hidden_states = encoded["hidden_states"][-1]
        logits = self.lin(hidden_states)
        prob = F.log_softmax(logits, dim=-1)
        return logits, prob


class sequential_classifier_2(BaseEstimator):
    def step(self, data):
        # self.optimizer.zero_grad()
        count = 0
        num_models = len(self.model)
        for model_name in self.model:
            self.optimizer[model_name].zero_grad()
            model = self.model[model_name]
            input_info = data[model_name]
            input_ids, attn_mask, token_type_ids = (
                input_info["input_ids"],
                input_info["attention_mask"],
                input_info["labels"],
            )

            logits, prob = model(
                input_ids=input_ids.to(self.cfg.device),
                attn_mask=attn_mask.to(self.cfg.device),
            )
            pred = prob.view(-1, self.cfg.num_labels)
            logits = logits.view(-1, self.cfg.num_labels)
            ref = data[model_name]["labels"].view(-1).to(self.cfg.device)

            pred = pred[ref != -100]  ## [#token, num_label]
            logits = logits[ref != -100]
            ref = ref[ref != -100]  ## [#token]

            if count == 0:
                loss = self.criterion(logits, ref)
                ref_one_hot = F.one_hot(ref, num_classes=self.cfg.num_labels)
                residuals = torch.subtract(
                    ref_one_hot, F.softmax(logits, dim=-1)
                ).detach()
                prob_sum = F.softmax(logits, dim=-1)
            else:
                loss = self.prob_loss(logits, residuals)
                prob_sum = torch.add(prob_sum, logits)
                residuals = torch.subtract(residuals, logits).detach()
                ## backprop
            count += 1
            retain_graph = not (count == num_models)
            if self.mode == "train":
                loss.backward(retain_graph=retain_graph)
                self.optimizer[model_name].step()
                if self.scheduler is not None:
                    self.scheduler[model_name].step()
                # self.optimizer[model_name].zero_grad() ## delete this

        # print(ref)
        # print(F.log_softmax(prob_sum))
        loss = nn.NLLLoss()(F.log_softmax(prob_sum, dim=-1), ref)
        return {
            "loss": loss.detach().cpu().item(),
            "pred": torch.argmax(prob_sum, dim=-1).detach().cpu(),  ## [#tokens]
            "prob": prob_sum.detach().cpu(),  ## [#tokens, num_class]
            "label": ref.detach().cpu(),  ## [#tokens]
        }

    def _eval(self, evalloader):
        self.model.eval()
        tbar = tqdm(evalloader, dynamic_ncols=True)
        eval_loss = []
        ys = []
        preds = []

        if self.evaluate_metric is None:
            self.evaluate_metric = dict()
            self.evaluate_metric["f1"] = evaluate.load("f1")
            self.evaluate_metric["all"] = load_metric("seqeval")

        for data in tbar:
            ret_step = self.step(data)  ## y: [bs, seq_len]
            loss, pred, labels = ret_step["loss"], ret_step["pred"], ret_step["label"]
            ys.extend(labels.tolist())
            preds.extend(pred.tolist())

        results = self.evaluate_metric["f1"].compute(
            predictions=preds, references=ys, average="macro"
        )
        print(f"====== F1 result: {results}======")

        return preds, ys


class BertLayer_bimodal(nn.Module):
    def __init__(self, emb_size, intermeidate_size=None, dropout_p=0.1, eps=1e-12):
        super().__init__()
        ## attention layer
        self.self_attention = self_attention(emb_size=emb_size, dropout_p=dropout_p)
        self.attention_dense = nn.Linear(emb_size, emb_size)
        self.attention_layer_norm = nn.LayerNorm(emb_size, eps=eps)
        self.attention_dropout = nn.Dropout(p=dropout_p)
        ## intermediate layer
        if intermeidate_size is None:
            intermeidate_size = emb_size
        self.interm_dense = nn.Linear(emb_size, intermeidate_size)
        self.interm_af = nn.GELU()
        ## output
        self.out_dense = nn.Linear(intermeidate_size, emb_size)
        self.out_layer_norm = nn.LayerNorm(emb_size, eps=eps)
        self.out_dropout = nn.Dropout(dropout_p)

    def add_norm(self, inputs, output, dense_layer, dropout, ln_layer):
        output_dense = dense_layer(output)
        droped_out = dropout(output_dense)
        result = ln_layer(droped_out + inputs)
        return result

    def forward(self, query, key, value):
        ## attention
        attention_out = self.self_attention(query=query, key=key, value=value)[
            "attn_output"
        ]
        ## add-norm
        attention_normed = self.add_norm(
            query,
            attention_out,
            self.attention_dense,
            self.attention_dropout,
            self.attention_layer_norm,
        )
        ## intermediate
        forward_linear = self.interm_dense(attention_normed)
        forward_act = self.interm_af(forward_linear)
        ## output
        out = self.out_dense(forward_act)
        out = self.add_norm(
            attention_normed,
            forward_act,
            self.out_dense,
            self.out_dropout,
            self.out_layer_norm,
        )
        return out


class co_attention(nn.Module):
    def __init__(self, emb_size):
        super().__init__()
        self.emb_size = emb_size
        self.cotrm = BertLayer_bimodal(emb_size=emb_size)
        self.trm = BertLayer_bimodal(emb_size=emb_size)

    def forward(self, mod1, mod2):
        co_trm = self.cotrm(query=mod1, key=mod2, value=mod2)
        trm = self.trm(query=co_trm, key=co_trm, value=co_trm)
        return trm


class bimodal_base(nn.Module):
    def __init__(self, model_dict, args):
        super().__init__()
        self.model_dict = model_dict
        self.args = args
        self.char_co_attention = nn.ModuleList(
            [co_attention(args.emb_size) for i in range(args.num_att_layers)]
        )
        self.word_co_attention = nn.ModuleList(
            [co_attention(args.emb_size) for i in range(args.num_att_layers)]
        )

    def forward(self, data):
        char_data = data["char"]
        word_data = data["word"]
        char_encoded = self.model_dict["char"](
            input_ids=char_data["input_ids"].to(self.args.device),
            attention_mask=char_data["input_ids"].to(self.args.device),
        )
        word_encoded = self.model_dict["word"](
            input_ids=word_data["input_ids"].to(self.args.device),
            attention_mask=word_data["input_ids"].to(self.args.device),
        )
        char_hidden = char_encoded["last_hidden_state"]
        word_hidden = word_encoded["last_hidden_state"]
        for i in range(self.args.k):
            char_new = self.char_co_attention(mod1=char_hidden, mod2=word_hidden)
            word_new = self.word_co_attention(mod1=word_hidden, mod2=char_hidden)
            char_hidden = char_new
            word_hidden = word_new
        return {"char": char_hidden, "word": word_hidden}


class bimodal_pretrain(nn.Module):
    def __init__(self, base, args):
        super().__init__()
        args.char_vocab_size = 1114112
        # args.char_vocab_size = base.model_dict["char"].config.vocab_size
        args.word_vocab_size = base.model_dict["word"].config.vocab_size
        self.base = base
        self.char_mlm_layer = nn.Linear(args.emb_size, args.char_vocab_size)
        self.word_mlm_layer = nn.Linear(args.emb_size, args.word_vocab_size)
        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))

    def forward(self, data):
        encoded = self.base(data=data)
        char_mlm_logits = self.char_mlm_layer(encoded["word"])
        word_mlm_logits = self.word_mlm_layer(encoded["word"])
        ## TODO: check correctness
        ## TODO: check whether word/char is aligned.
        similarity = torch.matmul(encoded["word"], torch.transpose(encoded["char"], 1, 2)) / self.logit_scale
        return {
            "char": char_mlm_logits,
            "word": word_mlm_logits,
            "similarity": similarity,
        }


class bimodal_trainer(BaseEstimator):
    def step(self, data):
        self.optimizer.zero_grad()
        logits_dict = self.model(data=data)
        ## TODO: check data structure
        char_mlm_loss = self.criterion(logits_dict["char"], data["char"]["input_ids"])
        word_mlm_loss = self.criterion(logits_dict["word"], data["word"]["input_ids"])
        ## TODO: check dimension here
        alignment_loss = self.criterion(
            logits_dict["similarity"], data["char_word_ids"]
        )
        ## TODO: weight loss
        loss = char_mlm_loss + word_mlm_loss + alignment_loss
        if self.mode == "train":
            loss.backward()
            self.optimizer.step()
            if self.scheduler is not None:
                self.scheduler.step()

            self.optimizer.zero_grad()
        elif self.mode in ("dev", "test"):
            pass
        for key, val in logits_dict.items():
            logits_dict[key] = val.detach().cpu()
        return {
            "loss": loss.detach().cpu().item(),
            "char_mlm_loss": char_mlm_loss.detach().cpu().item(),
            "word_mlm_loss": word_mlm_loss.detach().cpu().item(),
            "alignment_loss": alignment_loss.detach().cpu().item(),
            "logits_dict": logits_dict,
        }

    def _eval(self, evalloader):
        self.model.eval()
        tbar = tqdm(evalloader, dynamic_ncols=True)
        loss, char_mlm_loss, word_mlm_loss, alignment_loss = [], [], [], []

        if self.evaluate_metric is None:
            self.evaluate_metric = dict()
            self.evaluate_metric["f1"] = evaluate.load("f1")
            self.evaluate_metric["all"] = load_metric("seqeval")

        for data in tbar:
            ret_step = self.step(data)  ## y: [bs, seq_len]
            loss.append(ret_step["loss"])
            char_mlm_loss.append(ret_step["char_mlm_loss"])
            word_mlm_loss.append(ret_step["word_mlm_loss"])
            alignment_loss.append(ret_step["alignment_loss"])
        mean_loss = np.mean(loss)
        mean_char_mlm_loss = np.mean(char_mlm_loss)
        mean_word_mlm_loss = np.mean(word_mlm_loss)
        mean_alignment_loss = np.mean(alignment_loss)
        print(f"mean loss: {mean_loss}")
        print(f"mean_char_mlm_loss: {mean_char_mlm_loss}")
        print(f"mean_word_mlm_loss: {mean_word_mlm_loss}")
        print(f"mean_alignment_loss: {mean_alignment_loss}")
        return {
            "mean loss": mean_loss,
            "mean_char_mlm_loss": mean_char_mlm_loss,
            "mean_word_mlm_loss": mean_word_mlm_loss,
            "mean_alignment_loss": mean_alignment_loss,
        }


if __name__ == "__main__":
    ## add arguments
    parser = argparse.ArgumentParser()

    parser.add_argument("--dataset", type=str, default="wnut17")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--log_dir", type=str, default=None)
    parser.add_argument("--output_dir", type=str, default="./results")
    parser.add_argument("--bs", type=int, default=16)
    parser.add_argument("--model_name", type=str, default="google/canine-s")
    parser.add_argument("--n_epochs", type=int, default=1)  ## change to 4
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--weight_decay", type=float, default=0)
    parser.add_argument("--prefix_space", type=bool, default=True)
    parser.add_argument("--num_labels", type=int, default=13)
    parser.add_argument(
        "--granularities", type=str, default="character,subword_50k"
    )  # add character
    parser.add_argument("--add_space_for_char", type=bool, default=True)
    parser.add_argument("--to_char_method", type=str, default="inherit")
    parser.add_argument("--train", type=str, default="True")
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--ensemble_method", type=str, default="soft")

    parser.add_argument(
        "--granularities_model",
        type=dict,
        default={
            "character": "google/canine-s",
            "subword_50k": "xlm-roberta-base",
            "subword_30k": "bert-base-cased",
        },
    )

    parser.add_argument(
        "--embed_size_dict",
        type=dict,
        default={
            "google/canine-s": 768,
            "google/canine-c": 768,
            "bert-base-cased": 768,
            "bert-base-uncased": 768,
            "xlm-roberta-base": 1024,
            "roberta-base": 768,
            "roberta-large": 1024,
            "vinai/bertweet-base": 768,
            "cardiffnlp/twitter-roberta-base-sentiment": 768,
        },
    )

    args = parser.parse_args()
    args.granularities = args.granularities.split(",")
    args.model_names = [
        args.granularities_model[key] for key in args.granularities_model
    ]

    from datasets import load_dataset

    print(args.model_names)
    print(args.granularities)

    wnut = load_dataset("wnut_17")
    wnut_g = wnut_multiple_granularity(wnut, args)
    print(wnut_g.data_.keys())
    print(wnut_g[0])
