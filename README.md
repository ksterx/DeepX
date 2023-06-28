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

### Examples

#### Training

- Classifications with ResNet18:

```bash
cd vision/nn/classification
python train.py --model resnet18 --dataset cifar10 --batch_size 128 --epochs 200 --test True
tensorboard --logdir experiments
```

or

```python
from vision.nn.classification import train
train("resnet18", "cifar10", batch_size=128, epochs=200, is_test=True)
```

#### Inference

```bash
cd vision/tasks
python inference.py
```

![inference](./docs/app.png)
