# Vision

Reimplemetation of computer vision algorithms in Pytorch and Lightning.

## Usage example

- Classifications with ResNet18:

```bash
cd vision/nn/classification
python train.py --model resnet18 --dataset cifar10 --batch_size 128 --epochs 200 --test True
```
