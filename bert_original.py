"""
    bert benchmark 叶博代码+profiling
    能跑通，能打印
    TODO: dropout层的参数量为0
"""

# python bert_training.py --training-or-inference=inference

import argparse
import io
import os
import shutil
import time
from contextlib import contextmanager
from datetime import timedelta
from typing import List, Tuple

import numpy
import torch
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.nn as nn
import torch.nn.parallel
import torch.optim
import torch.utils.data
import torch.utils.data.distributed
# import torchvision.datasets as datasets
# import torchvision.models as models
# import torchvision.transforms as transforms
from torch.distributed.elastic.utils.data import ElasticDistributedSampler
from torch.nn.parallel import DistributedDataParallel
from torch.optim import SGD
from torch.utils.data import DataLoader

from transformers import BertConfig, BertForMaskedLM, BERT_PRETRAINED_CONFIG_ARCHIVE_MAP

from lib.profiler import FlopsProfiler

BERT_CONFIGS = {
    "bert-base-uncased": {
        "architectures": [
            "BertForMaskedLM"
        ],
        "attention_probs_dropout_prob": 0.1,
        "gradient_checkpointing": False,
        "hidden_act": "gelu",
        "hidden_dropout_prob": 0.1,
        "hidden_size": 768,
        "initializer_range": 0.02,
        "intermediate_size": 3072,
        "layer_norm_eps": 1e-12,
        "max_position_embeddings": 512,
        "model_type": "bert",
        "num_attention_heads": 12,
        "num_hidden_layers": 12,
        "pad_token_id": 0,
        "position_embedding_type": "absolute",
        "transformers_version": "4.6.0.dev0",
        "type_vocab_size": 2,
        "use_cache": True,
        "vocab_size": 30522,
    },
    "bert-large-uncased": {
        "architectures": [
            "BertForMaskedLM"
        ],
        "attention_probs_dropout_prob": 0.1,
        "gradient_checkpointing": False,
        "hidden_act": "gelu",
        "hidden_dropout_prob": 0.1,
        "hidden_size": 1024,
        "initializer_range": 0.02,
        "intermediate_size": 4096,
        "layer_norm_eps": 1e-12,
        "max_position_embeddings": 512,
        "model_type": "bert",
        "num_attention_heads": 16,
        "num_hidden_layers": 24,
        "pad_token_id": 0,
        "position_embedding_type": "absolute",
        "transformers_version": "4.6.0.dev0",
        "type_vocab_size": 2,
        "use_cache": True,
        "vocab_size": 30522
    },
}

# model_names = sorted(
#     name
#     for name in models.__dict__
#     if name.islower() and not name.startswith("__") and callable(models.__dict__[name])
# )

parser = argparse.ArgumentParser(description="PyTorch Elastic ImageNet Training")
# parser.add_argument("data", metavar="DIR", help="path to dataset")
parser.add_argument(
    "--data", default="./"
)
parser.add_argument(
    "-a",
    "--arch",
    metavar="ARCH",
    default="resnet18",
    # choices=model_names,
    # help="model architecture: " + " | ".join(model_names) + " (default: resnet18)",
)
parser.add_argument(
    "-j",
    "--workers",
    default=0,
    type=int,
    metavar="N",
    help="number of data loading workers",
)
parser.add_argument(
    "--epochs", default=90, type=int, metavar="N", help="number of total epochs to run"
)
parser.add_argument(
    "-b",
    "--batch-size",
    default=32,
    type=int,
    metavar="N",
    help="mini-batch size (default: 32), per worker (GPU)",
)
parser.add_argument(
    "--lr",
    "--learning-rate",
    default=0.1,
    type=float,
    metavar="LR",
    help="initial learning rate",
    dest="lr",
)
parser.add_argument("--momentum", default=0.9, type=float, metavar="M", help="momentum")
parser.add_argument(
    "--wd",
    "--weight-decay",
    default=1e-4,
    type=float,
    metavar="W",
    help="weight decay (default: 1e-4)",
    dest="weight_decay",
)
parser.add_argument(
    "-p",
    "--print-freq",
    default=10,
    type=int,
    metavar="N",
    help="print frequency (default: 10)",
)
parser.add_argument(
    "--dist-backend",
    default="nccl",
    choices=["nccl", "gloo"],
    type=str,
    help="distributed backend",
)
parser.add_argument(
    "--checkpoint-file",
    default="/tmp/checkpoint.pth.tar",
    type=str,
    help="checkpoint file path, to load and save to",
)
parser.add_argument(
    "--iterations",
    default=100,
    type=int,
)

parser.add_argument(
    "--training-or-inference",
    default="inference",
    choices=["training", "inference"],
    type=str
)


def seed_all(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(True)  # 这么设置使用确定性算法，如果代码中有算法cuda没有确定性实现，则会报错，可以验证代码中有没有cuda没有确定性实现的代码
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'


def main():
    args = parser.parse_args()
    # device_id = int(os.environ["LOCAL_RANK"])
    device_id = 0
    torch.cuda.set_device(device_id)
    print(f"=> set cuda device = {device_id}")

    # dist.init_process_group(
    #     backend=args.dist_backend, init_method="env://", timeout=timedelta(seconds=10)
    # )

    model, criterion, optimizer = initialize_model(
        args.arch, args.lr, args.momentum, args.weight_decay, device_id
    )

    train_loader, val_loader = initialize_data_loader(
        args.data, args.batch_size, args.workers, args.iterations
    )

    # resume from checkpoint if one exists;
    state = load_checkpoint(
        args.checkpoint_file, device_id, args.arch, model, optimizer
    )
    print_freq = args.print_freq

    if args.training_or_inference == "inference":
        # print("yes!!")  # debug
        validate(val_loader, model, criterion, device_id, print_freq)
        return

    start_epoch = state.epoch + 1
    print(f"=> start_epoch: {start_epoch}, best_acc1: {state.best_acc1}")

    for epoch in range(start_epoch, args.epochs):
        state.epoch = epoch
        # train_loader.batch_sampler.sampler.set_epoch(epoch)
        adjust_learning_rate(optimizer, epoch, args.lr)

        # train for one epoch
        train(train_loader, model, criterion, optimizer, epoch, device_id, print_freq, args.batch_size)

        # evaluate on validation set
        # acc1 = validate(val_loader, model, criterion, device_id, print_freq)

        # remember best acc@1 and save checkpoint
    #    is_best = acc1 > state.best_acc1
    #    state.best_acc1 = max(acc1, state.best_acc1)


class State:
    """
    Container for objects that we want to checkpoint. Represents the
    current "state" of the worker. This object is mutable.
    """

    def __init__(self, arch, model, optimizer):
        self.epoch = -1
        self.best_acc1 = 0
        self.arch = arch
        self.model = model
        self.optimizer = optimizer

    def capture_snapshot(self):
        """
        Essentially a ``serialize()`` function, returns the state as an
        object compatible with ``torch.save()``. The following should work
        ::

        snapshot = state_0.capture_snapshot()
        state_1.apply_snapshot(snapshot)
        assert state_0 == state_1
        """
        return {
            "epoch": self.epoch,
            "best_acc1": self.best_acc1,
            "arch": self.arch,
            "state_dict": self.model.state_dict(),
            "optimizer": self.optimizer.state_dict(),
        }

    def apply_snapshot(self, obj, device_id):
        """
        The complimentary function of ``capture_snapshot()``. Applies the
        snapshot object that was returned by ``capture_snapshot()``.
        This function mutates this state object.
        """

        self.epoch = obj["epoch"]
        self.best_acc1 = obj["best_acc1"]
        self.state_dict = obj["state_dict"]
        self.model.load_state_dict(obj["state_dict"])
        self.optimizer.load_state_dict(obj["optimizer"])

    def save(self, f):
        torch.save(self.capture_snapshot(), f)

    def load(self, f, device_id):
        # Map model to be loaded to specified single gpu.
        snapshot = torch.load(f, map_location=f"cuda:{device_id}")
        self.apply_snapshot(snapshot, device_id)


def load_bert_config(bert_variant, mode="inline"):
    import json
    if mode == "inline":
        data = BERT_CONFIGS[bert_variant]
    elif mode == "local":
        config_file = os.path.join(
            os.path.dirname(os.path.abspath(__file__)),
            f"{bert_variant}.json",
        )
        with open(config_file) as f:
            data = json.load(f)
    elif mode == "remote":
        config_url = BERT_PRETRAINED_CONFIG_ARCHIVE_MAP[bert_variant]
        import urllib.request as urlrequest
        with urlrequest.urlopen(config_url) as url:
            data = json.load(url)
    return data


def initialize_model(
        arch: str, lr: float, momentum: float, weight_decay: float, device_id: int
):
    print(f"=> creating model: bert")
    # get bert model

    # model = BertModel.from_pretrained('bert-base-uncased', local_files_only=True, cache_dir="bert-base-uncased")
    # use BERT_PRETRAINED_CONFIG_ARCHIVE_MAP to get the config file
    bert_variant = "bert-large-uncased"
    config = BertConfig(**load_bert_config(bert_variant))
    model = BertForMaskedLM(config)

    print(f"=> model params: {sum(p.numel() for p in model.parameters())}")

    # 获取modulelist
    mlist = model.bert.encoder.layer
    # print(mlist)  # debug
    # setattr(model, "layers", mlist)  # maphsge4
    # print("yes!!")  # debug
    # print(model.layers)  # debug

    # For multiprocessing distributed, DistributedDataParallel constructor
    # should always set the single device scope, otherwise,
    # DistributedDataParallel will use all available devices.
    model.cuda(device_id)
    cudnn.benchmark = True
    # model = DistributedDataParallel(model, device_ids=[device_id])
    # define loss function (criterion) and optimizer
    criterion = nn.CrossEntropyLoss().cuda(device_id)
    optimizer = SGD(
        model.parameters(), lr, momentum=momentum, weight_decay=weight_decay
    )
    return model, criterion, optimizer


def initialize_data_loader(
        data_dir, batch_size, num_data_workers, iterations
) -> Tuple[DataLoader, DataLoader]:
    traindir = os.path.join(data_dir, "train")
    valdir = os.path.join(data_dir, "val")
    # normalize = transforms.Normalize(
    #     mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
    # )
    #     train_dataset = datasets.ImageFolder(
    #         traindir,
    #         transforms.Compose(
    #             [
    #                 transforms.RandomResizedCrop(224),
    #                 transforms.RandomHorizontalFlip(),
    #                 transforms.ToTensor(),
    #                 normalize,
    #             ]
    #         ),
    #     )

    from torch.utils.data import Dataset, DataLoader
    class RandomDataset(Dataset):
        def __init__(self, length, batch_size, seq_len=512):
            self.len = length
            self.batch_size = batch_size
            self.seq_len = seq_len
            self.data = torch.randint(3, 30500, (length, seq_len))  # vocab_size: 30522

        def __getitem__(self, index):
            # return self.data[:, :, :, index]
            input_ids = self.data[index]
            mask_labels = torch.randint(3, 30500, (int(self.seq_len * 0.15),))  # vocab_size: 30522
            return (input_ids, mask_labels)

        def __len__(self):
            return self.len

    train_dataset = RandomDataset(36718, batch_size)

    # train_sampler = ElasticDistributedSampler(train_dataset)
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        num_workers=num_data_workers,
        pin_memory=True,
        # sampler=train_sampler,
    )
    #     val_loader = DataLoader(
    #         datasets.ImageFolder(
    #             valdir,
    #             transforms.Compose(
    #                 [
    #                     transforms.Resize(256),
    #                     transforms.CenterCrop(224),
    #                     transforms.ToTensor(),
    #                     normalize,
    #                 ]
    #             ),
    #         ),
    #         batch_size=batch_size,
    #         shuffle=False,
    #         num_workers=num_data_workers,
    #         pin_memory=True,
    #     )
    val_dataset = RandomDataset(batch_size * 1000, batch_size)
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        num_workers=num_data_workers,
        pin_memory=True,
        # sampler=train_sampler,
    )
    return train_loader, val_loader


def load_checkpoint(
        checkpoint_file: str,
        device_id: int,
        arch: str,
        model: DistributedDataParallel,
        optimizer,  # SGD
) -> State:
    """
    Loads a local checkpoint (if any). Otherwise, checks to see if any of
    the neighbors have a non-zero state. If so, restore the state
    from the rank that has the most up-to-date checkpoint.

    .. note:: when your job has access to a globally visible persistent storage
              (e.g. nfs mount, S3) you can simply have all workers load
              from the most recent checkpoint from such storage. Since this
              example is expected to run on vanilla hosts (with no shared
              storage) the checkpoints are written to local disk, hence
              we have the extra logic to broadcast the checkpoint from a
              surviving node.
    """

    state = State(arch, model, optimizer)

    if os.path.isfile(checkpoint_file):
        print(f"=> loading checkpoint file: {checkpoint_file}")
        state.load(checkpoint_file, device_id)
        print(f"=> loaded checkpoint file: {checkpoint_file}")

    # logic below is unnecessary when the checkpoint is visible on all nodes!
    # create a temporary cpu pg to broadcast most up-to-date checkpoint
    # with tmp_process_group(backend="gloo") as pg:
    #     rank = dist.get_rank(group=pg)

    #     # get rank that has the largest state.epoch
    #     epochs = torch.zeros(dist.get_world_size(), dtype=torch.int32)
    #     epochs[rank] = state.epoch
    #     dist.all_reduce(epochs, op=dist.ReduceOp.SUM, group=pg)
    #     t_max_epoch, t_max_rank = torch.max(epochs, dim=0)
    #     max_epoch = t_max_epoch.item()
    #     max_rank = t_max_rank.item()

    #     # max_epoch == -1 means no one has checkpointed return base state
    #     if max_epoch == -1:
    #         print(f"=> no workers have checkpoints, starting from epoch 0")
    #         return state

    #     # broadcast the state from max_rank (which has the most up-to-date state)
    #     # pickle the snapshot, convert it into a byte-blob tensor
    #     # then broadcast it, unpickle it and apply the snapshot
    #     print(f"=> using checkpoint from rank: {max_rank}, max_epoch: {max_epoch}")

    #     with io.BytesIO() as f:
    #         torch.save(state.capture_snapshot(), f)
    #         raw_blob = numpy.frombuffer(f.getvalue(), dtype=numpy.uint8)

    #     blob_len = torch.tensor(len(raw_blob))
    #     dist.broadcast(blob_len, src=max_rank, group=pg)
    #     print(f"=> checkpoint broadcast size is: {blob_len}")

    #     if rank != max_rank:
    #         blob = torch.zeros(blob_len.item(), dtype=torch.uint8)
    #     else:
    #         blob = torch.as_tensor(raw_blob, dtype=torch.uint8)

    #     dist.broadcast(blob, src=max_rank, group=pg)
    #     print(f"=> done broadcasting checkpoint")

    #     if rank != max_rank:
    #         with io.BytesIO(blob.numpy()) as f:
    #             snapshot = torch.load(f)
    #         state.apply_snapshot(snapshot, device_id)

    #     # wait till everyone has loaded the checkpoint
    #     dist.barrier(group=pg)

    print(f"=> done restoring from previous checkpoint")
    return state


@contextmanager
def tmp_process_group(backend):
    cpu_pg = dist.new_group(backend=backend)
    try:
        yield cpu_pg
    finally:
        dist.destroy_process_group(cpu_pg)


def save_checkpoint(state: State, is_best: bool, filename: str):
    checkpoint_dir = os.path.dirname(filename)
    os.makedirs(checkpoint_dir, exist_ok=True)

    # save to tmp, then commit by moving the file in case the job
    # gets interrupted while writing the checkpoint
    tmp_filename = filename + ".tmp"
    torch.save(state.capture_snapshot(), tmp_filename)
    os.rename(tmp_filename, filename)
    print(f"=> saved checkpoint for epoch {state.epoch} at {filename}")
    if is_best:
        best = os.path.join(checkpoint_dir, "model_best.pth.tar")
        print(f"=> best model found at epoch {state.epoch} saving to {best}")
        shutil.copyfile(filename, best)


def train(
        train_loader: DataLoader,
        model: DistributedDataParallel,
        criterion,  # nn.CrossEntropyLoss
        optimizer,  # SGD,
        epoch: int,
        device_id: int,
        print_freq: int,
        batch_size: int,
):
    batch_time = AverageMeter("Time", ":6.3f")
    data_time = AverageMeter("Data", ":6.3f")
    losses = AverageMeter("Loss", ":.4e")
    top1 = AverageMeter("Acc@1", ":6.2f")
    top5 = AverageMeter("Acc@5", ":6.2f")
    progress = ProgressMeter(
        len(train_loader),
        [batch_time, data_time, losses, top1, top5],
        prefix="Epoch: [{}]".format(epoch),
    )

    # switch to train mode
    model.train()

    end = time.time()
    # target = torch.LongTensor(batch_size).random_(1000).cuda()

    prof = FlopsProfiler(model)  # add profiler
    # prof_step = len(train_loader) // 3  # 整除3，所以会在33%的时候输出profile！
    prof_step = 10  # debug

    for i, (images, target, batch_idx) in enumerate(train_loader):
        if batch_idx == prof_step:  # add profile
            prof.start_profile()
        # measure data loading time
        data_time.update(time.time() - end)

        images = images.cuda(device_id, non_blocking=True)
        target = target.cuda(device_id, non_blocking=True)

        # compute output
        output = model(images).logits  # output: [batch_size, seq_len, vocab_size]
        # target: [batch_size, seq_len*0.15]

        if batch_idx == prof_step:  # add profile
            prof.print_model_profile(profile_step=batch_idx)
            prof.end_profile()

        loss = criterion(output[:, :int(512 * 0.15), :].permute((0, 2, 1)), target)

        # measure accuracy and record loss
        # acc1, acc5 = accuracy(output, target, topk=(1, 5))
        losses.update(loss.item(), images.size(0))
        # top1.update(acc1[0], images.size(0))
        # top5.update(acc5[0], images.size(0))

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % print_freq == 0:
            progress.display(i)


def validate(
        val_loader: DataLoader,
        model: DistributedDataParallel,
        criterion,  # nn.CrossEntropyLoss
        device_id: int,
        print_freq: int,
):
    batch_time = AverageMeter("Time", ":6.3f")
    losses = AverageMeter("Loss", ":.4e")
    top1 = AverageMeter("Acc@1", ":6.2f")
    top5 = AverageMeter("Acc@5", ":6.2f")
    progress = ProgressMeter(
        len(val_loader), [batch_time], prefix="Test: "
    )

    # switch to evaluate mode
    model.eval()

    prof = FlopsProfiler(model)  # add profiler
    # prof_step = len(val_loader) // 3  # 整除3，所以会在33%的时候输出profile！
    prof_step = 100  # debug

    with torch.no_grad():
        end = time.time()
        for i, (images, target) in enumerate(val_loader):

            if i == prof_step:  # add profile
                prof.start_profile()

            if device_id is not None:
                images = images.cuda(device_id, non_blocking=True)
            target = target.cuda(device_id, non_blocking=True)

            # compute output
            output = model(images)
            # print(model)
            loss = criterion(output[0], target)
            print("loss: ", loss)  # debug

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
                # print(torch.cuda.memory_allocated(device=torch.device("cuda")))  # 显存量

            if i == prof_step:
                return 999

        # TODO: this should also be done with the ProgressMeter
        # print(
        #     " * Acc@1 {top1.avg:.3f} Acc@5 {top5.avg:.3f}".format(top1=top1, top5=top5)
        # )

    return top1.avg


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
    def __init__(self, num_batches: int, meters: List[AverageMeter], prefix: str = ""):
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


def adjust_learning_rate(optimizer, epoch: int, lr: float) -> None:
    """
    Sets the learning rate to the initial LR decayed by 10 every 30 epochs
    """
    learning_rate = lr * (0.1 ** (epoch // 30))
    for param_group in optimizer.param_groups:
        param_group["lr"] = learning_rate


def accuracy(output, target, topk=(1,)):
    """
    Computes the accuracy over the k top predictions for the specified values of k
    """
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].reshape(1, -1).view(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res


if __name__ == "__main__":
    time1 = time.time()
    seed_all(1)
    main()
    time2 = time.time()
    time_all = time2 - time1
    print('The total time cost is: {}s'.format(time_all))
