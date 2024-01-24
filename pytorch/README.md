# Pytorch Template

pytorch code for training deep learning

## TODO
+ [x] distributed trainer
+ [x] getting started & requirements
+ [x] usage example with a random dataset
+ [ ] set determinism
+ [ ] gradient accumulation
+ [ ] gradient clipping

## Getting Started

1. Create [conda](https://docs.conda.io/) environment, `python>=3.10`

```shell
conda create -n pytorch python=3.10
```

2. Install [Pytorch](https://pytorch.org/get-started/previous-versions/)
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

# from scrach without AMP
CUDA_VISIBLE_DEVICES=0 python train_example.py --batch_size 2 --max_epochs 8 --val_every 1 --output ./output/example_$(date "+%y%m%d%H%M%S")

# without output (DEBUG only)
CUDA_VISIBLE_DEVICES=0 python train_example.py --batch_size 2 --max_epochs 8 --val_every 1 --amp

# resume from checkpoint
CUDA_VISIBLE_DEVICES=0 python train_example.py --batch_size 2 --max_epochs 8 --val_every 1 --amp --output ./output/example_$(date "+%y%m%d%H%M%S") --resume ./output/example_240124194132/model_best.pt
```

+ Train on multi GPUs (distributed)

```bash
# from scrach
CUDA_VISIBLE_DEVICES=0,1 python train_example.py --batch_size 2 --max_epochs 8 --val_every 1 --distributed --amp --output ./output/example_$(date "+%y%m%d%H%M%S")

# from scrach without AMP
CUDA_VISIBLE_DEVICES=0,1 python train_example.py --batch_size 2 --max_epochs 8 --val_every 1 --distributed --output ./output/example_$(date "+%y%m%d%H%M%S")

# resume from checkpoint
CUDA_VISIBLE_DEVICES=0,1 python train_example.py --batch_size 2 --max_epochs 8 --val_every 1 --distributed --amp --output ./output/example_$(date "+%y%m%d%H%M%S") --resume ./output/example_240124193738/model_best.pt
```

## Thanks

[GETTING STARTED WITH DISTRIBUTED DATA PARALLEL](https://pytorch.org/tutorials/intermediate/ddp_tutorial.html), [CUDA AUTOMATIC MIXED PRECISION EXAMPLES](https://pytorch.org/docs/stable/notes/amp_examples.html), [Swin-Transformer](https://github.com/microsoft/Swin-Transformer), [Swin-UNETR](https://github.com/Project-MONAI/research-contributions/tree/main/SwinUNETR/BTCV)