import torch
torch.multiprocessing.set_start_method('spawn')

import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel
import os

def dist_init(host_addr, rank, local_rank, world_size, port=23456):
    host_addr_full = 'tcp://' + host_addr + ':' + str(port)
    torch.distributed.init_process_group("nccl", init_method=host_addr_full,
                                         rank=rank, world_size=world_size)
    num_gpus = torch.cuda.device_count()
    torch.cuda.set_device(local_rank)
    assert torch.distributed.is_initialized()

rank = int(os.environ['SLURM_PROCID'])
local_rank = int(os.environ['SLURM_LOCALID'])
world_size = int(os.environ['SLURM_NTASKS'])
# get_ip函数自己写一下 不同服务器这个字符串形式不一样
# 保证所有task拿到的是同一个ip就成
ip = get_ip(os.environ['SLURM_STEP_NODELIST'])

dist_init(ip, rank, local_rank, world_size)


# 接下来是写dataset和dataloader，这个网上有很多教程
# 我这给的也只是个形式，按自己需求写好就ok
dataset = your_dataset()  #主要是把这写好
datasampler = DistributedSampler(dataset, num_replicas=world_size, rank=rank)
dataloader = DataLoader(dataset, batch_size=batch_size_per_gpu, sampler=source_sampler)

model = your_model()     #也是按自己的模型写
model = DistributedDataPrallel(model, device_ids=[local_rank], output_device=local_rank)