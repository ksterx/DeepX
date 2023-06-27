# Vision

Reimplemetation of computer vision algorithms in Pytorch and Lightning.

## Installation

- Docker installation:

```bash
./build.sh
```

- PyPI installation:

```bash
pip install -e .
```

## Usage

- Starting a container:

```bash
./up.sh
```

- Classifications with ResNet18:

```bash
cd vision/nn/classification
python train.py --model resnet18 --dataset cifar10 --batch_size 128 --epochs 200 --test True
```
