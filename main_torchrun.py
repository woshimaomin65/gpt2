import pdb
import sys
import os
import torch
import yaml
from argparse import Namespace
from utils.import_class import import_class
from torch.utils.data.dataloader import DataLoader
from utils.tools import batch_end_callback

import torch.multiprocessing as mp #ddp
from torch.utils.data import DistributedSampler #ddp
from torch.distributed import init_process_group, destroy_process_group #ddp
from torch.nn.parallel import DistributedDataParallel as DDP   #ddp
#ddp

def ddp_setup(**kw):
    init_process_group(backend="nccl")
    
def cleanup():
    destroy_process_group()

def main_ddp(config):
    rank = int(os.environ["LOCAL_RANK"])
    #是否需要ddp
    config = Namespace(**config)
    if config.is_multi:
        print(f'gpu_id: {os.environ["RANK"]}')    
    else:
        print(f'gpu_id: {rank}')    
    ddp_setup(**config.ddp_setup["params"]) #ddp
    #初始化模型 tokenizer， model_class
    model = import_class(**config.model)(**config.model["params"])
    #加载模型
    if config.model["is_from_pretrained"]:
        model.from_pretrained(**config.model["params"])
    model.to(rank) #ddp
    model = DDP(model, device_ids=[rank])  #ddp
    #加载训练数据
    train_dataset = import_class(**config.train_dataset)(**config.train_dataset["params"])
    train_sampler=torch.utils.data.RandomSampler(train_dataset, replacement=True, num_samples=int(1e10))
    train_loader = DataLoader(train_dataset, **config.train_dataloader)
    #加载测试数据
    test_dataset = import_class(**config.test_dataset)(**config.test_dataset["params"])
    test_sampler=torch.utils.data.RandomSampler(test_dataset, replacement=True, num_samples=int(1e10))
    test_loader = DataLoader(test_dataset, **config.test_dataloader)
    #初始化训练框架
    trainer = import_class(**config.trainer)(model, train_loader, test_loader, is_ddp=True, **config.trainer["params"])
    if rank == 0:
        trainer.set_callback('on_batch_end', batch_end_callback)
    trainer.run()
    #训练
    try:
        trainer.run()
    except:
        cleanup()    
    cleanup()

if __name__ == "__main__":
    #加载配置文件
    try:
        is_multi = True
    except:
        is_multi = False
    config = yaml.safe_load(open("conf/conf_mingpt.yml"))
    config['is_multi'] = is_multi
    config = Namespace(**config)
    config=vars(config)
    main_ddp(config)
