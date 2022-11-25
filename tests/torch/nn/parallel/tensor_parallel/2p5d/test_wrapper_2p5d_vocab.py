"""
torchrun --nproc_per_node=8 tests/torch/nn/parallel/tensor_parallel/2p5d/test_wrapper_2p5d_vocab.py
"""
import time

import torch
import torch.distributed as dist
from datasets import load_dataset
from torch.optim import Adam
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, GPT2Config

import oslo
import wandb
from oslo.torch.distributed import ParallelContext, ParallelMode
from oslo.torch.nn.parallel.tensor_parallel import TensorParallel
from oslo.transformers.models.gpt2.modeling_gpt2 import GPT2LMHeadModel


def latency_trace(func):
    def wrapper(*args, **kwargs):
        start = time.time()
        result = func(*args, **kwargs)
        end = time.time()
        return result, end - start

    return wrapper


@latency_trace
def fsw(func, *args, **kwargs):
    return func(*args, **kwargs).loss


@latency_trace
def bw(tensors):
    return tensors.backward()


tp_size = 8
batch_size = 16
model_name = "gpt2"

model_name = "gpt2"
mkwargs = {}
dataset_name = "squad"

# parallel context 생성
parallel_context = ParallelContext.from_torch(
    data_parallel_size=1,
    pipeline_parallel_size=1,
    tensor_parallel_size=tp_size,
    tensor_parallel_mode=ParallelMode.TENSOR_2P5D,
    tensor_parallel_depth=2,
)

torch.set_printoptions(sci_mode=False)
torch.manual_seed(0)

# 토크나이저 생성
tokenizer = AutoTokenizer.from_pretrained(model_name, **mkwargs)
tokenizer.pad_token = tokenizer.eos_token

# 모델 생성 및 병렬화 수행
model_no_tp = GPT2LMHeadModel(GPT2Config.from_pretrained(model_name)).cuda()
model_tp = GPT2LMHeadModel(GPT2Config.from_pretrained(model_name))
wrapper_tp = TensorParallel(model_tp, parallel_context)

oslo.ready(model_tp, parallel_context)

if dist.get_rank() == 0:
    print(wrapper_tp)

# 옵티마이저 생성
optimizer_tp = Adam(wrapper_tp.parameters(), lr=3e-5)
optimizer_no_tp = Adam(model_no_tp.parameters(), lr=3e-5)

# 데이터셋 생성
batch_size = 16
datasets = load_dataset(dataset_name).data["train"]["context"]
datasets = [str(sample) for sample in datasets[:500]]
dataloader = DataLoader(datasets, batch_size=batch_size)

# 모니터링 생성
if dist.get_rank() == 0:
    wandb.init(project="oslo", name=f"{model_name}_tp2p5d_bs{batch_size}")
    cur = time.time()

# 모니터링 생성 대기
dist.barrier()

# 학습 시작
for data in dataloader:
    optimizer_tp.zero_grad()
    optimizer_no_tp.zero_grad()

    inputs = tokenizer(
        data,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=512,
    ).to("cuda")

    loss_no_tp, notp_fw_time = fsw(model_no_tp, **inputs, labels=inputs["input_ids"])
    loss_tp, tp_fw_time = fsw(wrapper_tp, **inputs, labels=inputs["input_ids"])

    _, notp_bw_time = bw(loss_no_tp)
    _, tp_bw_time = bw(loss_tp)

    if dist.get_rank() == 0:
        wandb.log(
            {
                "tp_loss": loss_tp,
                "notp_loss": loss_no_tp,
                "tp.forward.time:": tp_fw_time,
                "tp.backward.time:": tp_bw_time,
                "notp.forward.time:": notp_fw_time,
                "notp.backward.time:": notp_bw_time,
            }
        )

dist.barrier()
