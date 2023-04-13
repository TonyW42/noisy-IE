import os
import sys
import tempfile
import torch
import torch.distributed as dist
import torch.nn as nn
import torch.optim as optim
import torch.multiprocessing as mp
from torch.utils.data import Dataset, DataLoader

from torch.nn.parallel import DistributedDataParallel as DDP

X = 30000
def setup(rank, world_size):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'

    # initialize the process group
    dist.init_process_group("nccl", rank=rank, world_size=world_size)

def cleanup():
    dist.destroy_process_group()


class ToyModel(nn.Module):
    def __init__(self):
        super(ToyModel, self).__init__()
        self.net1 = nn.Linear(X, X)
        self.relu = nn.ReLU()
        self.net2 = nn.Linear(X, 5)

    def forward(self, x):
        return self.net2(self.relu(self.net1(x)))

# class ToyModel(torch.nn.Module):
#   def __init__(self):
#     super(ToyModel, self).__init__()
#     self.lin = nn.Linear(X, 5)

  # def forward(self, a):
  #   a = self.lin(a)
  #   b = self.lin(a)
  #   return a + b

class my_dataset(Dataset):
  def __init__(self):
    self.data = torch.rand(512, X)
  
  def __len__(self):
    return 512
  
  def __getitem__(self, idx):
    return self.data[idx]

from torch.utils.data.distributed import DistributedSampler
def prepare(rank, world_size, batch_size=32, pin_memory=False, num_workers=0):
    dataset = my_dataset()
    sampler = DistributedSampler(dataset, num_replicas=world_size, rank=rank, shuffle=False, drop_last=False)
    
    dataloader = DataLoader(dataset, batch_size=batch_size, pin_memory=pin_memory, num_workers=num_workers, drop_last=False, shuffle=False, sampler=sampler)
    
    return dataloader

def demo_basic(rank, world_size):
    print(f"Running basic DDP example on rank {rank}.")
    setup(rank, world_size)

    # create model and move it to GPU with id rank
    model = ToyModel().to(rank)
    model = DDP(model, device_ids=[rank])

    loss_fn = nn.MSELoss()
    optimizer = optim.SGD(model.parameters(), lr=0.001)
    dataloader = prepare(rank, world_size)

    for i in range(20):
      for i, x in enumerate(dataloader):
        optimizer.zero_grad()
        outputs = model(x.to(rank))
        labels = torch.randn(32, 5).to(rank)
        loss_fn(outputs, labels).backward()
        optimizer.step()
        print(torch.cuda.memory_summary(device = rank))

    cleanup()


def run_demo(demo_fn, world_size):
    mp.spawn(demo_fn,
             args=(world_size,),
             nprocs=world_size)

if __name__ == '__main__':
    run_demo(demo_basic, 4)