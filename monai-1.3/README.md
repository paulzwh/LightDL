# [MONAI](https://docs.monai.io/) Template

MONAI template for training of medical deep learning, modified from our pure [Pytorch Template](../pytorch).

## TODO
+ [x] copy from [Pytorch Template](../pytorch) and modify Dataloader
+ [x] modify `trainer.py` to MONAI-based implementation
+ [ ] usage example

## Getting Started

1. Create [conda](https://docs.conda.io/) environment, `python>=3.10`

```shell
conda create -n pytorch python=3.10
```

2. Install PyTorch, see more in [PyTorch - Get Started](https://pytorch.org/get-started/)
```shell
conda install pytorch==1.13.1 torchvision==0.14.1 pytorch-cuda=11.6 -c pytorch -c nvidia
```

3. Install MONAI, see more in [MONAI - Installation Guide](https://docs.monai.io/en/stable/installation.html)
```shell
pip install monai==1.3.0
```

3. Install requirements
```shell
pip install -r requirements.txt

# Optional
pip install -r requirements_optional.txt
```

## Thanks
+ [Swin-UNETR](https://github.com/Project-MONAI/research-contributions/tree/main/SwinUNETR/BTCV)
+ [Project-MONAI/tutorials](https://github.com/Project-MONAI/tutorials)