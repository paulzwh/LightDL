# [PyTorch](https://pytorch.org/) Template

pytorch code for training deep learning

## TODO
+ [x] distributed trainer
+ [x] getting started & requirements
+ [x] usage example with a random dataset
+ [x] set determinism
+ [ ] gradient accumulation
+ [ ] gradient clipping

## Getting Started

1. Create [conda](https://docs.conda.io/) environment, `python>=3.10`

```shell
conda create -n pytorch python=3.10
```

2. Install PyTorch, see more in [PyTorch - Get Started](https://pytorch.org/get-started/)
```shell
conda install pytorch==1.11.0 torchvision==0.12.0 cudatoolkit=11.3 -c pytorch
```

Tested Environments:

| python | pytorch | cuda | os |
| ---- | ---- | ---- | ---- |
| 3.10 | 1.11.0 | 11.3 | Ubuntu 16.04 |
| 3.10 | 1.13.1 | 11.6 | Ubuntu 20.04 |

3. Install requirements
```shell
pip install -r requirements.txt

# Optional
pip install -r requirements_optional.txt
```

# Training

+ Train on single GPU

```bash
# from scrach
CUDA_VISIBLE_DEVICES=0 python train_example.py --batch_size 2 --max_epochs 8 --val_every 1 --amp --output ./output/example_$(date "+%y%m%d%H%M%S")

# from scrach with determinism
CUDA_VISIBLE_DEVICES=0 python train_example.py --batch_size 2 --max_epochs 8 --val_every 1 --seed 3407 --amp --output ./output/example_$(date "+%y%m%d%H%M%S")

# from scrach without AMP
CUDA_VISIBLE_DEVICES=0 python train_example.py --batch_size 2 --max_epochs 8 --val_every 1 --output ./output/example_$(date "+%y%m%d%H%M%S")

# without output (DEBUG only)
CUDA_VISIBLE_DEVICES=0 python train_example.py --batch_size 2 --max_epochs 8 --val_every 1 --amp

# resume from checkpoint
CUDA_VISIBLE_DEVICES=0 python train_example.py --batch_size 2 --max_epochs 8 --val_every 1 --amp --output ./output/example_$(date "+%y%m%d%H%M%S") --resume ./output/example_240124194132/model_best.pt

# resume from checkpoint with determinism
CUDA_VISIBLE_DEVICES=0 python train_example.py --batch_size 2 --max_epochs 8 --val_every 1 --amp --output ./output/example_$(date "+%y%m%d%H%M%S") --resume ./output/example_240124194132/model_best.pt --seed 3407 # must set the same seed
```

+ Train on multi GPUs (distributed)

```bash
# from scrach
CUDA_VISIBLE_DEVICES=0,1 python train_example.py --batch_size 2 --max_epochs 8 --val_every 1 --distributed --amp --output ./output/example_$(date "+%y%m%d%H%M%S")

# from scrach with determinism
CUDA_VISIBLE_DEVICES=0,1 python train_example.py --batch_size 2 --max_epochs 8 --val_every 1 --distributed --seed 3407 --amp --output ./output/example_$(date "+%y%m%d%H%M%S")

# from scrach without AMP
CUDA_VISIBLE_DEVICES=0,1 python train_example.py --batch_size 2 --max_epochs 8 --val_every 1 --distributed --output ./output/example_$(date "+%y%m%d%H%M%S")

# without output (DEBUG only)
CUDA_VISIBLE_DEVICES=0,1 python train_example.py --batch_size 2 --max_epochs 8 --val_every 1 --amp --distributed

# resume from checkpoint
CUDA_VISIBLE_DEVICES=0,1 python train_example.py --batch_size 2 --max_epochs 8 --val_every 1 --distributed --amp --output ./output/example_$(date "+%y%m%d%H%M%S") --resume ./output/example_240124193738/model_best.pt

# resume from checkpoint with determinism
CUDA_VISIBLE_DEVICES=0,1 python train_example.py --batch_size 2 --max_epochs 8 --val_every 1 --distributed --amp --output ./output/example_$(date "+%y%m%d%H%M%S") --resume ./output/example_240124193738/model_best.pt --seed 3407 # must set the same seed
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

[PyTorch - Tutorials - Getting Started with Distributed Data Parallel](https://pytorch.org/tutorials/intermediate/ddp_tutorial.html), [PyTorch - Docs - CUDA Automatic Mixed Precision examples](https://pytorch.org/docs/stable/notes/amp_examples.html), [Swin-Transformer](https://github.com/microsoft/Swin-Transformer), [Swin-UNETR](https://github.com/Project-MONAI/research-contributions/tree/main/SwinUNETR/BTCV)