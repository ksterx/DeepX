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

- Classification

Terminal

```bash
cd vision/tasks
python train.py --task classification --model resnet18 --dataset cifar10 --batch_size 128 --epochs 200 --debug
tensorboard --logdir experiments
```

Python

```python
from vision.tasks import train
from vision.nn import MLP

model = MLP([64, 128, 10])

train("classification", model, "cifar10", batch_size=128, epochs=200, debug=True)
# train("classification", "resnet18", "cifar10", batch_size=128, epochs=200, is_test=True,)
```

- Segmetation

Terminal

```bash
cd vision/tasks
python train.py --task segmentation --model unet --dataset voc --batch_size 128 --epochs 200 --debug
```

Python

```python
from vision.tasks import train

train("segmentation", "unet", "voc", batch_size=128, epochs=200, debug=True)
```

#### Inference

```bash
cd vision/tasks
python inference.py
```

![inference](./docs/app.png)
