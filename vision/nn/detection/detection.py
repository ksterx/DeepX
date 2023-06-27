from .. import BaseModel


class DetectionModel(BaseModel):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        pass

    def training_step(self, batch, batch_idx):
        pass
