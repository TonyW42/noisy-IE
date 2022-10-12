from cmath import log
import sys 
sys.path.append("..") 
sys.path.append("../..") 
from data import Tokenization, is_aligned, CharacterLevelMapping
from utils.compute import compute_metrics
from transformers import AutoModelForTokenClassification, TrainingArguments, Trainer, AutoTokenizer, DataCollatorForTokenClassification
from datasets import load_dataset
import argparse
import torch 
from evaluate_utils import *
import numpy as np
from utils.log import *


class HuggingFaceModel:
    def __init__(self, args):
        self.args = args
        self.log = Log("Hugging Face Model")

    def train(self):
        # model = AutoModelForTokenClassification.from_pretrained(self.args.model_name, num_labels=self.args.num_labels)
        wnut = load_dataset("wnut_17")
        logits_sum = []
        count = 0
        model_prob = dict()
        for granularity in self.args.granularities:
            if granularity == "character":
                clm = CharacterLevelMapping(self.args.to_char_method)
                wnut_character_level = clm.character_level_wnut(wnut)
                tok = Tokenization(self.args.granularities_model[granularity], self.args.prefix_space)
                tokenized_wnut = tok.tokenize_for_char_manual(wnut_character_level)
            elif granularity == "subword_50k" or granularity == "subword_30k":
                self.log.info(self.args.granularities_model[granularity])
                tok = Tokenization(self.args.granularities_model[granularity], self.args.prefix_space)
                tokenized_wnut = wnut.map(tok.tokenize_and_align_labels, batched=True) ## was previously wnut_character level 
            assert is_aligned(tokenized_wnut)

            model_name = self.args.granularities_model[granularity]
            model = AutoModelForTokenClassification.from_pretrained(model_name, num_labels=self.args.num_labels)
            tokenizer = AutoTokenizer.from_pretrained(model_name, add_prefix_space=self.args.prefix_space) ## changed here
            data_collator = DataCollatorForTokenClassification(tokenizer=tokenizer)
                                        

            training_args = TrainingArguments(
                output_dir=self.args.output_dir, ## come back to fix this!
                evaluation_strategy="epoch",
                learning_rate=self.args.lr,
                per_device_train_batch_size=self.args.bs,
                per_device_eval_batch_size=self.args.bs,
                num_train_epochs=self.args.n_epochs,
                # weight_decay=0.01,
            )

            trainer = Trainer(
                model=model,
                args=training_args,
                train_dataset=tokenized_wnut["train"],
                eval_dataset=tokenized_wnut["test"],
                tokenizer=tokenizer,
                data_collator=data_collator,
                # optimizers = torch.optim.Adam(model.parameters()),
                compute_metrics = compute_metrics, 
            )
            if self.args.train:
                trainer.train()
            
            if granularity == "character":
                pred_tmp = wnut_get_char_logits(model = model,  
                                                tokenized_wnut = tokenized_wnut, 
                                                prefix_space = self.args.prefix_space, 
                                                model_name = self.args.granularities_model[granularity],
                                                device = self.args.device,
                                                rule = 3)
                logits_tmp = pred_tmp["pred"]
                label = pred_tmp["label"]
                self.log.info("-------- granularity == character ---------")
                self.log.info("logits size: (%d, %d)", len(logits_tmp), len(logits_tmp[0]))
            elif granularity == "subword_30k" or granularity  == "subword_50k":
                ## get subword logits 
                pred_tmp = wnut_get_subword_logits(model = model,  
                                                tokenized_wnut = tokenized_wnut, 
                                                prefix_space = self.args.prefix_space, 
                                                model_name = self.args.granularities_model[granularity],
                                                device = self.args.device,
                                                rule = 3)
                logits_tmp = pred_tmp["pred"]
                label = pred_tmp["label"]
                self.log.info("-------- granularity == subword ---------")
                self.log.info("logits size: (%d, %d)", len(logits_tmp), len(logits_tmp[0]))
            # if count == 0:
            #     # logits_sum = np.array([s.detach().numpy() for s in logits_tmp])
            #     logits_sum = torch.tensor(logits_tmp)
            # if count > 0:
            #     # logits_sum += np.array([s.detach().numpy() for s in logits_tmp]) 
            #     logits_sum = torch.add(logits_sum, torch.tensor(logits_tmp))
            model_prob[granularity] = logits_tmp 

            count += 1

        # logits_sum = [s.tolist() for s in logits_sum]
        # logits_sum = torch.tensor(logits_sum)
        pred = None
        if self.args.ensemble_method == "soft":
            logits_prob_matrix = torch.tensor([model_prob[key] for key in model_prob])
            logits_sum = torch.sum(logits_prob_matrix, dim = 0)
            pred = torch.argmax(logits_sum, dim = 1)
        elif self.args.ensemble_method == "most_confident":
            logits_prob_matrix = torch.tensor([model_prob[key] for key in model_prob])
            logits_max = torch.max(logits_prob_matrix, dim = 0).values
            pred = torch.argmax(logits_max, dim = 1)
        elif self.args.ensemble_method == "hard":
            model_pred = torch.tensor([np.argmax(model_prob[key], axis = 1) for key in model_prob])
            pred = torch.mode(model_pred, dim = 0).values

        assert pred is not None 


            
            
        # pred = torch.argmax(logits_sum, dim = 1)
        ensumble_f1 = wnut_f1(pred = pred, ref = label)
        print(f"\n The F1-score of the model is {ensumble_f1} \n")
        self.log.info(f"\n The F1-score of the model is {ensumble_f1} \n")

            
        # ## evalate trained model 
        # wnut_f1 = wnut_evaluate_f1(model = model,  
        #                            tokenized_wnut = tokenized_wnut, 
        #                            prefix_space = args.prefix_space, 
        #                            model_name = args.model_name,
        #                            device = device,
        #                            method = "first letter")

        # print(f"\n The F1-score of the model is {wnut_f1} \n")
        # print(f"\n The F1-score of the model is {wnut_f1_1} \n")
        # print(f"\n The F1-score of the model is {wnut_f1_2} \n")


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

    args = parser.parse_args()
    args.granularities = args.granularities.split(",")

    model = HuggingFaceModel(args)
    model.train()