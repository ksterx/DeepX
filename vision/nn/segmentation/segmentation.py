from .. import BaseModel


class SegmentationModel(BaseModel):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        pass
