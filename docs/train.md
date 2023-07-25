# Training examples

## Image generation

### MNIST

```bash
cd /workspace/experiments/training
python imggenerator.py dataset=mnist machine=docker task=imggen
```

![mnist](./images/mnist.gif)

### LFW

```bash
cd /workspace/experiments/training
python imggenerator.py dataset=lfw machine=docker task=imggen
```

![lfw](./images/lfw.gif)
