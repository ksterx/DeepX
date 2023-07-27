from abc import ABC, abstractmethod
from tkinter import filedialog


class App(ABC):
    def __init__(self, initialdir):
        self.initialdir = initialdir

    @abstractmethod
    def launch(self):
        raise NotImplementedError

    def set_ckpt_path(self):
        path = filedialog.askopenfilename(
            initialdir=self.initialdir,
            title="Select checkpoint file",
        )
        return path
