# DeepX

*Deep learning algorithms implemented with PyTorch and Lightning.*

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## Features

- Backbone

  - [x] ResNet
  - [ ] Vision Transformer
  - [ ] Transformer

- Tasks
  - Vision
    - Classification
    - Segmentation
      - [x] UNet
    - Object Detection
  - Language
    - [ ] Language Model
    - [ ] Text Classification

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

### Starting a container

```bash
./up.sh
```

### Training

#### Classification

- Terminal

```bash
cd /tasks
python train.py --task classification --model resnet18 --dataset cifar10 --batch_size 128 --epochs 200 --debug
tensorboard --logdir experiments
```

- Python

```python
from deepx.tasks import train
from deepx.nn import MLP

model = MLP([64, 128, 10])

train("classification", model, "cifar10", batch_size=128, epochs=200, debug=True)
# train("classification", "resnet18", "cifar10", batch_size=128, epochs=200, is_test=True,)
```

#### Segmetation

- Terminal

```bash
cd deepx/tasks
python train.py --task segmentation --model unet --dataset voc --batch_size 128 --epochs 200 --debug
```

- Python

```python
from deepx.tasks import train

train("segmentation", "unet", "voc", batch_size=128, epochs=200, debug=True)
```

### Inference

```bash
cd deepx/tasks
python inference.py
```

![inference](./docs/app.png)

## Development

### Profiling

```bash
python -m cProfile -o profile.prof train.py <args>
```
