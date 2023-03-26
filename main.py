import pdb
import torch
import yaml
from argparse import Namespace
from torch.utils.data.dataloader import DataLoader
from utils.import_class import import_class
from utils.tools import batch_end_callback
#加载配置文件
config = yaml.safe_load(open("conf/conf_mingpt.yml"))
config = Namespace(**config)
#初始化模型 tokenizer， model_class
model = import_class(**config.model)(**config.model["params"])
#加载模型
if config.model["is_from_pretrained"]:
    model.from_pretrained(**config.model["params"])
#加载训练数据
train_dataset = import_class(**config.train_dataset)(**config.train_dataset["params"])
train_sampler=torch.utils.data.RandomSampler(train_dataset, replacement=True, num_samples=int(1e10))
train_loader = DataLoader(train_dataset, **config.train_dataloader)
#for batch in train_loader:
#    pdb.set_trace()
#    a = 5
#加载测试数据
test_dataset = import_class(**config.test_dataset)(**config.test_dataset["params"])
test_sampler=torch.utils.data.RandomSampler(test_dataset, replacement=True, num_samples=int(1e10))
test_loader = DataLoader(test_dataset, **config.test_dataloader)
#初始化训练框架
trainer = import_class(**config.trainer)(model, train_loader, test_loader, **config.trainer["params"])
trainer.set_callback('on_batch_end', batch_end_callback)
#训练
trainer.run()





