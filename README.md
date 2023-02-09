[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

# noisy-IE
Noisy-IE is a research project that combines multi-level models for Noisy Information Extraction. 

# Installation
## Requirements
```
transformers 
torch 
datasets
argparse
tqdm 
evaluate 
```

- To install all the requirements
`pip install -r requirements.txt`

- To run the character level model and get combined word level F1:
`python3 run.py`

- To prevent memory leak, whenever you add a tensor to a list, make sure to call `tensor.detach()` first to remove it from the torch computational graph. To be save, start everything with with `torch.no_grad()` in evaluation mode. 

- Added different ensemble methods: Use `--ensemble_method` to specify. `soft` addes softmax probability, `hard` does mojority vote (and resolve conflict based on first apperance), and `most_confident` takes most confident one as prediction (max of max softmax probability across models.) Default is `soft`, which stands for softmax probability. 

It is preferred to use Python >= 3.7 on macOS and Linux. 

## Data 
For this project, we will focus on named entity recognition (NER) on noisy user-generated texts. For evaluation purpose, we will use WNUT-17 (Derczynski et.al., 2017). We will be gathering data using Huggingface's API. The data can be accessed using the link below:

- https://huggingface.co/datasets/wnut_17

## Experiments

We provide a wrapper file that allows you to replicate our experiments using command lines. The general command to run your experiment is `python3 run.py`. To run each experiment, you may want to specify the following flags:

* `--model_list`: the list of model in the entanglement model separated by `|`. In the standard setting, we will use the character/word granularity. Therefore, the default option will be `"roberta-base|google/canine-s"`. Note: Don't forget the `"`

* `--expr`: experiment. With a few specification:
    - `baseline`: baseline experiment, single model + linear classification head 
    - `MTL`: run the standard entanglement model. 
    - `mlm`: run masked-language pretraining before evaluating on WNUT-17 (still under improvement)

* `--word_model`: the model used to evaluate on WNUT-17. This is necessary when you want to include more than 1 subword model. 

* `--layer_type`: The type of alignment layers. Can have two typess:
    - `attn`: simple attention layers 
    - `bert`: transformer layers (Vaswani, 2017) with residual connection and normalization. 

* `--num_att_layers`: number of alignment layers

Here are some flags for standard hyper-parameter setting:

* `--seed`: set random seed. Default uses the current system time.
* `--bs`: batch size. Default is 16.
* `--n_epochs`: number of epochs. Default is 25. 
* `--lr`: (initial) learning rate. Default is 2e-5. 

We provide some example code for running our experiments below. 

### Baseline experiment
Here is one example command line code to run baseline experiments, with roberta-base

```python3 run.py --mode train --model_list "roberta-base" --word_model roberta-base \
                --num_att_layers 0 --mode train --bs 32 --train_batch_size 10 --eval_batch_size 10 --test_batch_size 10 \
                --weight_decay 1e-8 --lr 2e-5 --n_epochs 50 --expr baseline \
                --granularities "subword_50k"```

### Entanglement model
Here is one example command line code to run a basic extanglement model, with roberta-base and canine, using 4 attention layers as alignment layer

``` python3 run.py --mode train --model_list "roberta-base|google/canine-s" --word_model roberta-base \
                --num_att_layers 4 --mode train --bs 32 --train_batch_size 10 --eval_batch_size 10 --test_batch_size 10 \
                --weight_decay 1e-8 --lr 2e-5 --n_epochs 50 --expr MTL \
                --granularities "character,subword_50k"```
