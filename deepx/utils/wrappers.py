# %%
import inspect
from logging import getLogger

logger = getLogger(__name__)


def watch_kwargs(func):
    def wrapper(*args, **kwargs):
        sig = inspect.signature(func)
        params = sig.parameters
        for k, v in kwargs.items():
            if k not in params:
                logger.warning(
                    f"{k} = {v} is passed to {func.__module__}/{func.__name__}, but it passes through."
                )
        func(*args, **kwargs)

    return wrapper
