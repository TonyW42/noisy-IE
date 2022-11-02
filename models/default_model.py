from cmath import log
import sys 
sys.path.append("..") 
sys.path.append("../..") 
from utils.compute import compute_metrics
from transformers import AutoModelForTokenClassification, TrainingArguments, Trainer, AutoTokenizer, DataCollatorForTokenClassification
from datasets import load_dataset
import argparse
import torch 
from evaluate_utils import *
from torch.utils.tensorboard.writer import SummaryWriter
import numpy as np
from utils.log import *
from models.model import *
from utils.save_model import current_time


class HuggingFaceModel:
    def __init__(self, args):
        self.args = args
        self.log = Log("Hugging Face Model")
        self.writer = SummaryWriter(os.path.join(self.args.output_dir, current_time())
    )

    def train(self):
        # model = AutoModelForTokenClassification.from_pretrained(self.args.model_name, num_labels=self.args.num_labels)
        wnut = load_dataset("wnut_17")
        logits_sum = []
        label_t = None
        count = 0
        model_prob = dict()
        data_loader = wnut_multiple_granularity(wnut, self.args)
        model_main = data_loader.model_dict
        tokenized_wnut_main = data_loader.data_
        for granularity in self.args.granularities:
            model_name = self.args.granularities_model[granularity]
            tokenized_wnut = tokenized_wnut_main[model_name]

            model = model_main[model_name]
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
            if count == 0:
                # logits_sum = np.array([s.detach().numpy() for s in logits_tmp])
                logits_sum = torch.tensor(logits_tmp)
                label_t = torch.tensor(torch.argmax(torch.tensor(logits_tmp), dim = 1))
            if count > 0:
                # logits_sum += np.array([s.detach().numpy() for s in logits_tmp]) 
                logits_sum = torch.add(logits_sum, torch.tensor(logits_tmp))
                label_t = torch.stack((label_t, torch.argmax(torch.tensor(logits_tmp), dim = 1)), 0)
            count += 1

        # logits_sum = [s.tolist() for s in logits_sum]
        # logits_sum = torch.tensor(logits_sum)
        pred = torch.argmax(logits_sum, dim = 1)
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

    def _train(self):
        wnut = load_dataset("wnut_17")
        print(self.args.model_names)
        data_loader = wnut_multiple_granularity(wnut, self.args)
        tokenized_wnut_main = data_loader.data_
        model = weighted_ensemble(data_loader.model_dict, self.args)
        for granularity in self.args.granularities:
            model_name = self.args.granularities_model[granularity]
            tokenized_wnut = tokenized_wnut_main[model_name]

            tokenizer = AutoTokenizer.from_pretrained(model_name, add_prefix_space=self.args.prefix_space) ## changed here
            data_collator = DataCollatorForTokenClassification(tokenizer=tokenizer)

            criterion = nn.BCEWithLogitsLoss().to(self.args.device)

            if not self.args.train: 
                optimizer = None
                scheduler = None
            else: 
                no_weight_decay = lambda param_name: any(
                    no_decay_name in param_name for no_decay_name in ['LayerNorm', 'layer_norm', 'bias']
                )
                optimizer = optim.AdamW(
                    [
                        {'params': [param for param_name, param in model.named_parameters() if not no_weight_decay(param_name)]}, 
                        {'params': [param for param_name, param in model.named_parameters() if no_weight_decay(param_name)], 'weight_decay': 0}
                    ], 
                    lr=self.args.lr, 
                    betas=(0.9, 0.999), 
                    eps=1E-6, 
                    weight_decay=0.01
                )
                train_steps = self.args.n_epochs * data_loader.__len__
                warmup_steps = self.args.warmup_proportion * train_steps
                scheduler = optim.lr_scheduler.LambdaLR(
                    optimizer, 
                    lambda global_step: max(
                        0, 
                        min(global_step / warmup_steps, 1 - (global_step - warmup_steps) / train_steps)
                    ) # slanted triangular lr
                )

            estimator = weighted_estimater( # TODO: change to your estimator!
                model, 
                tokenizer, 
                criterion, 
                optimizer=optimizer, 
                scheduler=scheduler, 
                logger=self.log, 
                writer=self.writer, 
                pred_thold=None, 
                device=self.args.device, 
                # add other hyperparameters here
            )
            print('Running the model...')
            if self.args.train: 
                self.logger.info('Training...')
                estimator.train(self.args, tokenized_wnut['train'], tokenized_wnut['validation'])
                probs, _ = estimator.test(tokenized_wnut['test'])
                assert probs.shape[0] == len(tokenized_wnut['test']) 
            


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

    model = HuggingFaceModel(args)
    model._train()