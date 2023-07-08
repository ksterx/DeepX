from torch import nn


class MLP(nn.Module):
    NAME = "mlp"

    def __init__(
        self,
        channels_in_layers: list[int],
        activation: nn.Module = nn.ReLU(),
        output_activation: nn.Module | None = None,
        flatten: bool = True,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.layers = nn.ModuleList()
        if flatten:
            self.layers.append(nn.Flatten())
        for i in range(len(channels_in_layers) - 1):
            self.layers.append(nn.Linear(channels_in_layers[i], channels_in_layers[i + 1]))
        self.activation = activation
        self.output_activation = output_activation
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x):
        for layer in self.layers[:-1]:
            x = layer(x)
            x = self.activation(x)
            x = self.dropout(x)
        x = self.layers[-1](x)
        if self.output_activation is not None:
            x = self.output_activation(x)
        return x
