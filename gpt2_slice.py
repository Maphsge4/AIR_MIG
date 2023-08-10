"""
    gpt2 benchmark 叶博代码+profiling
    能跑通，能打印
    TODO: dropout层的参数量为0
"""

from lib.transformers import GPT2Tokenizer, GPT2Config, GPT2Model
import torch
import os
import pathlib
import datetime
import gc
import torch.nn as nn
from lib.profiler import FlopsProfiler
from lib.my_offload import OffloadModel 
from typing import List, Tuple

model_name = 'gpt2-large'
batch_size = 32

# gpt2_config = {
#     "gpt2-large": {
#         "activation_function": "gelu_new",
#         "architectures": [
#             "GPT2LMHeadModel"
#         ],
#         "attn_pdrop": 0.1,
#         "bos_token_id": 50256,
#         "embd_pdrop": 0.1,
#         "eos_token_id": 50256,
#         "initializer_range": 0.02,
#         "layer_norm_epsilon": 1e-05,
#         "model_type": "gpt2",
#         "n_ctx": 1024,
#         "n_embd": 1280,
#         "n_head": 20,
#         "n_layer": 36,
#         "n_positions": 1024,
#         "resid_pdrop": 0.1,
#         "summary_activation": None,
#         "summary_first_dropout": 0.1,
#         "summary_proj_to_labels": True,
#         "summary_type": "cls_index",
#         "summary_use_proj": True,
#         "task_specific_params": {
#             "text-generation": {
#             "do_sample": True,
#             "max_length": 50
#             }
#         },
#         "vocab_size": 50257
#     }
# }

gpt2_config = {
    "gpt2-large": {
        "activation_function": "gelu_new",
        "architectures": [
            "GPT2LMHeadModel"
        ],
        "attn_pdrop": 0.1,
        "bos_token_id": 50256,
        "embd_pdrop": 0.1,
        "eos_token_id": 50256,
        "initializer_range": 0.02,
        "layer_norm_epsilon": 1e-05,
        "model_type": "gpt2",
        "n_ctx": 1024,
        "n_embd": 768,
        "n_head": 12,
        "n_layer": 12,
        "n_positions": 1024,
        "resid_pdrop": 0.1,
        "summary_activation": None,
        "summary_first_dropout": 0.1,
        "summary_proj_to_labels": True,
        "summary_type": "cls_index",
        "summary_use_proj": True,
        "task_specific_params": {
            "text-generation": {
            "do_sample": True,
            "max_length": 50
            }
        },
        "vocab_size": 50257
    }
}

def seed_all(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(True)  # 这么设置使用确定性算法，如果代码中有算法cuda没有确定性实现，则会报错，可以验证代码中有没有cuda没有确定性实现的代码
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'

seed_all(1)

config = GPT2Config(**gpt2_config[model_name])
model = GPT2Model(config=config)
device_id = 1
torch.cuda.set_device(device_id)
print(f"=> model params: {sum(p.numel() for p in model.parameters())}")

# 获取modulelist
mlist = model.h
# print(mlist)  # debug
mslices : List[nn.Module] = []
for i, layer_module in enumerate(mlist):
    mslices.append(layer_module)

offload_model = OffloadModel( # 使用 OffloadModel 来包装模型
    model=mslices, # 原生模型
    device=torch.device("cuda"), # 用于计算向前和向后传播的设备
    offload_device=torch.device("cpu"), # 模型将存储在其上的offload 设备
    num_slices=10, # 模型应分片的片数
    checkpoint_activation=False,
    num_microbatches=1,
)
model.hh = offload_model

# # 需要改改，怎样只让那几层先进GPU
# print("原来的")
# model = model.cuda()
# model.h = model.h.cpu()
print("现在的")
model.to_cuda()
print("max:", torch.cuda.max_memory_allocated(device=torch.device("cuda")))  # 显存量
print("now", torch.cuda.memory_allocated(device=torch.device("cuda")))  # 显存量

def prepare_dataloader(length, batch_size):
    vocab_size = 50257
    from torch.utils.data import Dataset, DataLoader
    class RandomDataset(Dataset):
        def __init__(self, length, batch_size, seq_len=512):
            self.len = length
            self.batch_size = batch_size
            self.seq_len = seq_len
            self.data = torch.randint(3, vocab_size, (length, seq_len))  # vocab_size: 30522

        def __getitem__(self, index):
            # return self.data[:, :, :, index]
            input_ids = self.data[index]
            mask_labels = torch.randint(3, vocab_size, (int(self.seq_len*0.15),))  # vocab_size: 30522
            return (input_ids, mask_labels)

        def __len__(self):
            return self.len


    ds = RandomDataset(length, batch_size)

    # train_sampler = ElasticDistributedSampler(train_dataset)
    loader = DataLoader(
        ds,
        batch_size=batch_size,
        num_workers=1,
        pin_memory=True,
        # sampler=train_sampler,
    )

    return loader

dataloader = prepare_dataloader(2 * batch_size, batch_size)

class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self, name: str, fmt: str = ":f"):
        self.name = name
        self.fmt = fmt
        self.reset()

    def reset(self) -> None:
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1) -> None:
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self):
        fmtstr = "{name} {val" + self.fmt + "} ({avg" + self.fmt + "})"
        return fmtstr.format(**self.__dict__)

class ProgressMeter(object):
    def __init__(self, num_batches: int, meters, prefix: str = ""):
        self.batch_fmtstr = self._get_batch_fmtstr(num_batches)
        self.meters = meters
        self.prefix = prefix

    def display(self, batch: int) -> None:
        entries = [self.prefix + self.batch_fmtstr.format(batch)]
        entries += [str(meter) for meter in self.meters]
        print("\t".join(entries))

    def _get_batch_fmtstr(self, num_batches: int) -> str:
        num_digits = len(str(num_batches // 1))
        fmt = "{:" + str(num_digits) + "d}"
        return "[" + fmt + "/" + fmt.format(num_batches) + "]"

def validate(data_loader, device_id, print_freq=10):
    trace_dir = pathlib.Path(__file__).parent.joinpath("traces")
    now = datetime.datetime.now().strftime("%Y_%m_%d:%H.%M.%S")
    trace_dir.mkdir(exist_ok=True)
    # print("yes!!!")
    batch_time = AverageMeter("Time", ":6.3f")
    losses = AverageMeter("Loss", ":.4e")
    top1 = AverageMeter("Acc@1", ":6.2f")
    top5 = AverageMeter("Acc@5", ":6.2f")
    progress = ProgressMeter(
        len(data_loader), [batch_time], prefix="Test: "
    )

    # switch to evaluate mode
    model.eval()
    import time

    prof = FlopsProfiler(model)  # add profile
    # prof_step = len(data_loader) // 3  # 整除3，所以会在33%的时候输出profile！
    prof_step = 30  # debug

    with torch.no_grad():
        end = time.time()
        gc.collect()
        with torch.profiler.profile(record_shapes=True, profile_memory=True, with_stack=True) as p:
            for i, (images, target) in enumerate(data_loader):
                # gc.collect()
                if i == prof_step:  # add profile
                    prof.start_profile()

                if device_id is not None:
                    images = images.cuda(device_id, non_blocking=True)
                target = target.cuda(device_id, non_blocking=True)

                # compute output
                output = model(images)
                output = output.last_hidden_state
                # loss = criterion(output, target)

                if i == prof_step:  # add profile
                    prof.print_model_profile(profile_step=i)
                    prof.end_profile()

                # measure accuracy and record loss
                # acc1, acc5 = accuracy(output, target, topk=(1, 5))
                # losses.update(loss.item(), images.size(0))
                # top1.update(acc1[0], images.size(0))
                # top5.update(acc5[0], images.size(0))

                # measure elapsed time
                batch_time.update(time.time() - end)
                end = time.time()

                if i % print_freq == 0:
                    progress.display(i)
                
                print("end_max:", torch.cuda.max_memory_allocated(device=torch.device("cuda")))  # 显存量
                print("end_now", torch.cuda.memory_allocated(device=torch.device("cuda")))  # 显存量

                # if i == prof_step + 30:
                #     return 999

            gc.collect()
        p.export_memory_timeline(str(trace_dir.joinpath(f"linear_stack_{now}.html")), torch.cuda.current_device())
                
validate(dataloader, device_id)
