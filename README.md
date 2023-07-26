# DeepX

*Deep learning algorithms implemented with PyTorch and Lightning.*

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## Features

- Backbone

  - [x] ResNet
  - [ ] Vision Transformer
  - [x] Transformer

- Algorithms

  - Vision
    - Classification
      - [x] ResNet
    - Segmentation
      - [x] UNet
    - Object Detection
      - [ ] YOLO
      - [ ] SSD
    - Image Generation
      - [ ] VAE
      - [x] DCGAN
      - [ ] Diffusion Models
  - Language
    - [x] Language Model
    - [ ] Text Classification
    - [ ] Translation

## Documentation
Documentation is available [here!!](https://ksterx.github.io/DeepX/)

## Installation

- Docker installation:

```bash
cd envs
docker compose up -d
```

- PyPI installation:

```bash
pip install -e .
```

## Usage

- Starting a container:

```bash
docker exec -it deepx bash
```

- Experiment Tracking:

Access `http://localhost:5000` in your browser.

## Development

- Profiling

```bash
cd experiments/training
python -m cProfile -o profile.prof <task>.py <args>
snakeviz profile.prof
```
