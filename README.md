# noisy-IE

## Requirements
transformers 
torch 
datasets
argparse
tqdm 
evaluate 

- To install all the requirements
`pip install -r requirements.txt`

- To run the character level model and get combined word level F1:
`python3 run.py`

- To prevent memory leak, whenever you add a tensor to a list, make sure to call `tensor.detach()` first to remove it from the torch computational graph. To be save, start everything with with `torch.no_grad()` in evaluation mode. 
