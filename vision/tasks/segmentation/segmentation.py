import lightning as L


class SegmentationModel(L.LightningModule):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        pass
