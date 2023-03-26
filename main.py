import torch
import yaml
from argparse import Namespace
from torch.utils.data.dataloader import DataLoader
from utils.import_class import import_class


#加载配置文件
config = yaml.safe_load(open("conf_mingpt.yml"))
config = Namespace(**config)

#初始化模型 tokenizer， model_class

#加载模型

#加载数据
#训练数据
train_dataset = import_class(**config.train_dataset)(**config.train_dataset.params)
train_sampler=torch.utils.data.RandomSampler(train_dataset, replacement=True, num_samples=int(1e10))
train_loader = DataLoader(train_dataset, **config.train_dataloader)
#测试数据
test_dataset = import_class(**config.test_dataset)(**config.test_dataset.params)
test_sampler=torch.utils.data.RandomSampler(test_dataset, replacement=True, num_samples=int(1e10))
test_loader = DataLoader(test_dataset, **config.train_dataloader)

#初始化训练框架





