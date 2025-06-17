# [Paddle](https://www.paddlepaddle.org.cn/) Template

Paddle code for training deep learning.

|        | verison |
| ------ | ------- |
| python | 3.11    |
| paddle | 2.6     |

## TODO
+ [x] getting started & requirements
+ [ ] distributed trainer
+ [ ] usage example with a random dataset
+ [ ] set determinism

## Getting Started

1. Create [conda](https://docs.conda.io/) environment,

```shell
conda create -n paddle26 python=3.11 -c conda-forge
conda activate paddle26
```

1. Install Paddle, see more in PaddlePaddle - Install ([EN](https://www.paddlepaddle.org.cn/en/install/quick?docurl=/documentation/docs/zh/install/conda/linux-conda_en.html) | [中](https://www.paddlepaddle.org.cn/install/quick?docurl=/documentation/docs/zh/install/conda/linux-conda.html)),
```shell
conda install paddlepaddle-gpu==2.6.2 cudatoolkit=11.7 -c Paddle -c conda-forge
```

1. Install requirements,
```shell
pip install -r requirements.txt
```
