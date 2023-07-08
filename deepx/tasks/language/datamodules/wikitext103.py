from torchtext.datasets import WikiText103

from ..langmodel import LangModelDM


class WikiText103DM(LangModelDM):
    NAME = "wikitext103"

    def setup(self, stage=None):
        self.train_data, self.valid_data, self.test_data = WikiText103(
            self.data_dir, split=("train", "valid", "test")
        )
