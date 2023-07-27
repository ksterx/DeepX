from typing import TypeVar

from torch import nn, optim

ModuleType = TypeVar("ModuleType", bound=nn.Module)
OptimType = TypeVar("OptimType", bound=optim.Optimizer)
LRSchedulerType = TypeVar("LRSchedulerType", bound=optim.lr_scheduler._LRScheduler)
