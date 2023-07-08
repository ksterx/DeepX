from torchtext.datasets import PennTreebank

from ..langmodel import LangModelDM


class PennTreebankDM(LangModelDM):
    NAME = "penn"

    def setup(self, stage=None):
        self.train_data, self.val_data, self.test_data = PennTreebank(
            self.data_dir, split=("train", "valid", "test")
        )
