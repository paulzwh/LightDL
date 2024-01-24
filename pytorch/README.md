# Pytorch Template

pytorch code for training deep learning

## TODO
+ [x] distributed trainer
+ [x] getting started & requirements
+ [ ] usage example

## Getting Started

1. Create [conda](https://docs.conda.io/) environment, `python>=3.10`

```shell
conda create -n pytorch python=3.10
```

2. Install [Pytorch](https://pytorch.org/get-started/previous-versions/)
```shell
conda install pytorch==1.11.0 torchvision==0.12.0 cudatoolkit=11.3 -c pytorch
```

This template has been tested using pytorch 1.11.0 (on Ubuntu 16.04) and pytorch 1.13.1 (on Ubuntu 20.04)

3. Install requirements
```shell
pip install -r requirements.txt

# Optional
pip install -r requirements_optional.txt
```

## Thanks

[GETTING STARTED WITH DISTRIBUTED DATA PARALLEL](https://pytorch.org/tutorials/intermediate/ddp_tutorial.html), [CUDA AUTOMATIC MIXED PRECISION EXAMPLES](https://pytorch.org/docs/stable/notes/amp_examples.html), [Swin-Transformer](https://github.com/microsoft/Swin-Transformer), 