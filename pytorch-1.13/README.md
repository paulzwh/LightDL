# [PyTorch](https://pytorch.org/) Template

PyTorch code for training deep learning.

|         | verison |
| ------- | ------- |
| python  | 3.10    |
| pytorch | 1.13.1  |

## TODO
+ [x] distributed trainer
+ [x] getting started & requirements
+ [x] usage example with a random dataset
+ [x] set determinism
+ [ ] gradient accumulation
+ [ ] gradient clipping

## Getting Started

1. Create [conda](https://docs.conda.io/) environment,

```shell
conda create -n torch113 python=3.10
conda activate torch113
```

2. Install PyTorch, see more in [PyTorch - Get Started](https://pytorch.org/get-started/),
```shell
conda install pytorch==1.13.1 torchvision==0.14.1 pytorch-cuda=11.6 numpy=1.26 -c pytorch -c nvidia
```

3. Install requirements,
```shell
pip install -r requirements.txt
```

# Training

+ Train on single GPU

```bash
# from scrach
CUDA_VISIBLE_DEVICES=0 python train_example.py --batch_size 2 --max_epochs 8 --val_interval 1 --amp --output ./output/example_$(date "+%y%m%d%H%M%S")

# from scrach with determinism
CUDA_VISIBLE_DEVICES=0 python train_example.py --batch_size 2 --max_epochs 8 --val_interval 1 --seed 3407 --amp --output ./output/example_$(date "+%y%m%d%H%M%S")

# from scrach without AMP
CUDA_VISIBLE_DEVICES=0 python train_example.py --batch_size 2 --max_epochs 8 --val_interval 1 --output ./output/example_$(date "+%y%m%d%H%M%S")

# without output (DEBUG only)
CUDA_VISIBLE_DEVICES=0 python train_example.py --batch_size 2 --max_epochs 8 --val_interval 1 --amp

# resume from checkpoint
CUDA_VISIBLE_DEVICES=0 python train_example.py --batch_size 2 --max_epochs 8 --val_interval 1 --amp --output ./output/example_$(date "+%y%m%d%H%M%S") --resume ./output/example_240124194132/model_best.pt

# resume from checkpoint with determinism
CUDA_VISIBLE_DEVICES=0 python train_example.py --batch_size 2 --max_epochs 8 --val_interval 1 --amp --output ./output/example_$(date "+%y%m%d%H%M%S") --resume ./output/example_240124194132/model_best.pt --seed 3407 # must set the same seed
```

+ Train on multi GPUs (distributed)

```bash
# from scrach
CUDA_VISIBLE_DEVICES=0,1 python train_example.py --batch_size 2 --max_epochs 8 --val_interval 1 --distributed --amp --output ./output/example_$(date "+%y%m%d%H%M%S")

# from scrach with determinism
CUDA_VISIBLE_DEVICES=0,1 python train_example.py --batch_size 2 --max_epochs 8 --val_interval 1 --distributed --seed 3407 --amp --output ./output/example_$(date "+%y%m%d%H%M%S")

# from scrach without AMP
CUDA_VISIBLE_DEVICES=0,1 python train_example.py --batch_size 2 --max_epochs 8 --val_interval 1 --distributed --output ./output/example_$(date "+%y%m%d%H%M%S")

# without output (DEBUG only)
CUDA_VISIBLE_DEVICES=0,1 python train_example.py --batch_size 2 --max_epochs 8 --val_interval 1 --amp --distributed

# resume from checkpoint
CUDA_VISIBLE_DEVICES=0,1 python train_example.py --batch_size 2 --max_epochs 8 --val_interval 1 --distributed --amp --output ./output/example_$(date "+%y%m%d%H%M%S") --resume ./output/example_240124193738/model_best.pt

# resume from checkpoint with determinism
CUDA_VISIBLE_DEVICES=0,1 python train_example.py --batch_size 2 --max_epochs 8 --val_interval 1 --distributed --amp --output ./output/example_$(date "+%y%m%d%H%M%S") --resume ./output/example_240124193738/model_best.pt --seed 3407 # must set the same seed
```

+ Other args

```bash
--nnodes 1                 # number of nodes for distributed training
--node_rank 0              # node rank for distributed training
--master_addr "localhost"  # master address for init distributed
--master_port "62674"      # master address for init distributed
--workers 4                # number of workers for dataloader
```

## Thanks

+ [PyTorch/Tutorials/Getting Started with Distributed Data Parallel](https://pytorch.org/tutorials/intermediate/ddp_tutorial.html)
+ [PyTorch/Docs/CUDA Automatic Mixed Precision examples](https://pytorch.org/docs/stable/notes/amp_examples.html)
+ [Pytorch/Docs/Reproducibility](https://pytorch.org/docs/stable/notes/randomness.html)
+ [Swin-Transformer](https://github.com/microsoft/Swin-Transformer)
+ [Swin-UNETR](https://github.com/Project-MONAI/research-contributions/tree/main/SwinUNETR/BTCV)