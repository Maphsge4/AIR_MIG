from torch.utils.data.dataloader import DataLoader
from torchvision.datasets import FakeData
from torchvision.transforms import ToTensor
import torch
import time
import logging

print(torch.cuda.device_count())
print(torch.cuda.current_device())
print(torch.cuda.get_device_name(0))
print(torch.cuda.get_device_capability(device=0))