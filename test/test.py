from torch.utils.data.dataloader import DataLoader
from torchvision.datasets import FakeData
from torchvision.transforms import ToTensor
import torch
import time
import logging

# 引入Offload
from lib.offload_selectn import OffloadModel

logging.getLogger().setLevel(logging.INFO)

# 定义训练配置
num_inputs = 8
num_outputs = 8
num_hidden = 4
num_layers = 2
batch_size = 8

# 数据加载
transform = ToTensor()
dataloader = DataLoader(
    FakeData(
        image_size=(1, num_inputs, num_inputs),
        num_classes=num_outputs,
        transform=transform,
    ),
    batch_size=batch_size,
)

# 定义了Sequential模型，注意前面提到的：模型假定为nn.Sequential模型，并根据参数数量（几乎）平均分片到nn.Modules 列表之中。
model = torch.nn.Sequential(
    torch.nn.Linear(num_inputs * num_inputs, num_hidden),
    *([torch.nn.Linear(num_hidden, num_hidden) for _ in range(num_layers)]),
    torch.nn.Linear(num_hidden, num_outputs),
)

offload_model = OffloadModel(  # 使用 OffloadModel 来包装模型
    model=model,  # 原生模型
    device=torch.device("cuda"),  # 用于计算向前和向后传播的设备
    offload_device=torch.device("cpu"),  # 模型将存储在其上的offload 设备
    num_slices=3,  # 模型应分片的片数
    checkpoint_activation=False,
    num_microbatches=1,
    device_list=[0, 0, 1]
)

torch.cuda.set_device(0)
device = torch.device("cuda")

criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(offload_model.parameters(), lr=0.001)  # 使用OffloadModel

# To train 1 epoch.
offload_model.train()  # 使用 OffloadModel
for batch_inputs, batch_outputs in dataloader:
    batch_inputs, batch_outputs = batch_inputs.to("cuda"), batch_outputs.to("cuda")
    start = time.time_ns()
    optimizer.zero_grad()
    inputs = batch_inputs.reshape(-1, num_inputs * num_inputs)
    # print(inputs.device)  # debug
    with torch.cuda.amp.autocast():
        output = offload_model(inputs)  # 前向传播
        loss = criterion(output, target=batch_outputs)
        loss.backward()  # 反向传播
    optimizer.step()
