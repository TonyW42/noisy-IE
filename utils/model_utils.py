import os
import sys
import random
import logging
from datetime import datetime

from tqdm import tqdm
import numpy as np
import torch
from torch import nn
from sklearn.metrics import roc_auc_score, precision_recall_fscore_support


def seed_everything(seed): 
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def make_if_not_exists(new_dir): 
    if not os.path.exists(new_dir): 
        os.system('mkdir -p {}'.format(new_dir))


def config_logging(log_dir): 
    make_if_not_exists(log_dir)
    file_handler = logging.FileHandler(
        filename=os.path.join(log_dir, 'experiment.log')
    )
    stdout_handler = logging.StreamHandler(sys.stdout)
    logging.basicConfig(
        level=logging.INFO, 
        format='%(asctime)s\t%(message)s', 
        datefmt='%m/%d/%Y %H:%M:%S', 
        handlers=[file_handler, stdout_handler]
    )
    logger = logging.getLogger('train')
    return logger


def convert_to_unicode(text): 
    """Converts text to Unicode (if it's not already), assuming utf-8 input"""
    if isinstance(text, str): 
        return text
    elif isinstance(text, bytes): 
        return text.decode("utf-8", "ignore")
    else: 
        raise ValueError("Unsupported string type: {}".format(type(text)))


class BaseClassifier(nn.Module): 
    """A generic class that combines a representation learner and a classifier"""
    def __init__(self, backbone, head): 
        """
        Input
        ----------
        backbone: 
            Usually a (pretrained) BertModel, or any sentence embedding extractor
        head: 
            Usually a FFN, or any NN for classification
        """
        super().__init__()
        self.backbone = backbone
        self.head = head

    def forward(self, input_ids, input_mask, seg_ids): 
        """
        Input
        ----------
        input_ids: 
            Raw text => tokens => input_ids, starting with id of [CLS] and ending with id of [SEP] (or padding id 0), shape (B, max_len)
        input_mask: 
            1 for tokens that are not padded (hence not masked) and 0 otherwise, shape (B, max_len)
        set_ids: 
            Segment identifiers, but in this case 0 for all tokens, shape (B, max_len)
        Output
        ----------
        logits: 
            Logit of being true for each label, shape (B, n_labels)
        """
        encoded = self.backbone(
            input_ids=input_ids, 
            attention_mask=input_mask, 
            token_type_ids=seg_ids, 
        )
        sentence_embd = encoded[1] # 'pooler_output' 
        logits = self.head(sentence_embd)
        return logits


class BaseEstimator(object): 
    """A wrapper class to perform training, evluation or testing while accumulating and logging results"""
    def __init__(
        self, 
        model, 
        cfg,
        # tokenizer, 
        criterion=None, 
        optimizer=None, 
        scheduler=None, 
        logger=None, 
        writer=None, 
        pred_thold=None, 
        device='cpu', 
        **kwargs
    ): 
        self.model = model
        # self.tokenizer = tokenizer
        self.criterion = criterion
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.logger = logger
        self.writer = writer
        self.pred_thold = pred_thold
        self.device = device
        self.mode = None # {'train', 'dev', 'test'}

        self.epoch = 0
        self.train_step = 0
        self.dev_step = 0
        self.test_step = 0
        self.kwargs = kwargs
        self.cfg = cfg

        self.evaluate_metric = None

    def step(self, data): 
        """
        This function is responsible for feeding the data into the model and obtain predictions. 
        If `self.mode == 'train'`, perform backpropagation and update optimizer and, if given, scheduler; 
        if `self.mode == 'dev'`, we still compute loss and return ground true label, but no backpropagation nor optimizer update; 
        if `self.mode == 'test'`, no ground true label is presented so loss will not be calculated. 
        Input
        ----------
        data: 
            A dictionary of mini-batch input obtained from Dataset.__getitem__, each with shape (B, max_len), type torch.tensor
            Before fed into the model, inputs should be cast to appropriate the type (torch.long) and converted to self.device
        Output
        ----------
        loss: 
            A scalar for the entire batch, type float; None if no label provided
        prob: 
            Model predictions as the probability for each label, shape (B, n_labels), type np.ndarray
        y: 
            Ground true labels for the batch, shape (B, n_labels), type np.ndarray; None if no label provided
        """
        raise NotImplementedError('Implement it in the child class!')

    def _train_epoch(self, trainloader, devloader=None): 
        self.mode = 'train'
        self.model.train()
        tbar = tqdm(trainloader, dynamic_ncols=True)
        for data in tbar: 
            ret_step = self.step(data)
            loss = ret_step['loss']
            y = ret_step['label']
            # prob = ret_step
            self.train_step += 1
            tbar.set_description('train_loss - {:.4f}'.format(loss))
            if self.writer is not None: 
                self.writer.add_scalar('train/loss', loss, self.train_step)
                self.writer.add_scalar('train/micro/auc', roc_auc_score(y, prob, average='micro'), self.train_step)
                if self.pred_thold is not None: 
                    # yhat = (prob > self.pred_thold).astype(int)
                    micros = precision_recall_fscore_support(y, yhat, average='micro')
                    self.writer.add_scalar('train/micro/precision', micros[0], self.train_step)
                    self.writer.add_scalar('train/micro/recall', micros[1], self.train_step)
                    self.writer.add_scalar('train/micro/f1', micros[2], self.train_step)
        if devloader is not None: 
            self.dev(devloader)

    def train(self, cfg, trainloader, devloader=None): 
        self.mode = 'train'
        assert self.optimizer is not None, 'Optimizer is required'
        assert hasattr(cfg, 'output_dir'), 'Output directory must be specified'
        make_if_not_exists(cfg.output_dir)
        for i in range(cfg.n_epochs): 
            print(f"Training epoch {i}")
            self._train_epoch(trainloader, devloader)
            self.epoch += 1
            checkpoint_path = os.path.join(cfg.output_dir, '{}.pt'.format(datetime.now().strftime('%m-%d_%H-%M')))
            if cfg.save.lower() == "true": 
                self.save(checkpoint_path)
            if self.logger is not None: 
                self.logger.info('[CHECKPOINT]\t{}'.format(checkpoint_path))

    def _eval(self, evalloader): 
        self.model.eval()
        tbar = tqdm(evalloader, dynamic_ncols=True)
        eval_loss = []
        ys = []
        preds = []
        for data in tbar: 
            loss, logits_dict, y = self.step(data)   ## y: [bs, seq_len]
            logit_word = logits_dict[self.args.word_model]
            prob = None ## softmax logit_word, [bs, seq_len, num_label] 
            pred = torch.argmax(prob, dim = -1) ## predicted, [bs, seq_len]
            if self.mode == 'dev': 
                tbar.set_description('dev_loss - {:.4f}'.format(loss))
                eval_loss.append(loss)
                ys.append(y)
            preds.append(pred) ## use pred for F1 and change how you append 
        loss = np.mean(eval_loss).item() if self.mode == 'dev' else None
        ys = np.concatenate(ys, axis=0) if self.mode == 'dev' else None
        probs = np.concatenate(probs, axis=0)
        if self.mode == 'dev': 
            macro_auc = roc_auc_score(ys, probs, average='macro')
            micro_auc = roc_auc_score(ys, probs, average='micro')
            if self.writer is not None: 
                self.writer.add_scalar('dev/loss', loss, self.dev_step)
                self.writer.add_scalar('dev/macro/auc', macro_auc, self.dev_step)
                self.writer.add_scalar('dev/micro/auc', micro_auc, self.dev_step)
                if self.pred_thold is not None: 
                    yhats = (probs > self.pred_thold).astype(int)
                    macros = precision_recall_fscore_support(ys, yhats, average='macro')
                    self.writer.add_scalar('dev/macro/precision', macros[0], self.dev_step)
                    self.writer.add_scalar('dev/macro/recall', macros[1], self.dev_step)
                    self.writer.add_scalar('dev/macro/f1', macros[2], self.dev_step)
                    micros = precision_recall_fscore_support(ys, yhats, average='micro')
                    self.writer.add_scalar('dev/micro/precision', micros[0], self.dev_step)
                    self.writer.add_scalar('dev/micro/recall', micros[1], self.dev_step)
                    self.writer.add_scalar('dev/micro/f1', micros[2], self.dev_step)
        return probs, ys

    def dev(self, devloader): 
        self.mode = 'dev'
        results = self._eval(devloader)
        self.dev_step += 1
        return results

    def test(self, testloader): 
        self.mode = 'test'
        results = self._eval(testloader)
        self.test_step += 1
        return results

    def save(self, checkpoint_path): 
        if self.epoch % 20 == 0:
            checkpoint = {
                'epoch': self.epoch, 
                'train_step': self.train_step, 
                'dev_step': self.dev_step, 
                'test_step': self.test_step, 
                'model': self.model.state_dict(), 
                'optimizer': self.optimizer.state_dict() if self.optimizer is not None else None, 
                'scheduler': self.scheduler.state_dict() if self.scheduler is not None else None
            }
            torch.save(checkpoint, checkpoint_path)

    def load(self, checkpoint_path): 
        print('Loading checkpoint {}'.format(checkpoint_path))
        checkpoint = torch.load(checkpoint_path)
        self.epoch = checkpoint['epoch']
        self.train_step = checkpoint['train_step']
        self.dev_step = checkpoint['dev_step']
        self.test_step = checkpoint['test_step']
        self.model.load_state_dict(checkpoint['model'])
        if self.optimizer is not None: 
            self.optimizer.load_state_dict(checkpoint['optimizer'])
        else: 
            print('Optimizer is not loaded')
        if self.scheduler is not None: 
            self.scheduler.load_state_dict(checkpoint['scheduler'])
        else: 
            print('Scheduler is not loaded')