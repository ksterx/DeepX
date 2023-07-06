import math

import torch
from torch import nn

from deepx.nn.core import MLP


class Attention(nn.Module):
    def __init__(self, embed_dim: int, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        self.scale = math.sqrt(embed_dim)
        self.softmax = nn.Softmax(dim=-1)
        self.w_q = nn.Linear(embed_dim, embed_dim)
        self.w_k = nn.Linear(embed_dim, embed_dim)
        self.w_v = nn.Linear(embed_dim, embed_dim)

    def forward(self, query: torch.Tensor, key: torch.Tensor, value: torch.Tensor) -> torch.Tensor:
        query = self.w_q(query)
        key = self.w_k(key)
        value = self.w_v(value)
        scores = query @ key.transpose(-2, -1) / self.scale
        scores = self.softmax(scores)
        scores = self.dropout(scores)
        return scores @ value


class SelfAttention(Attention):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return super().forward(x, x, x)


class MultiHeadAttention(nn.Module):
    def __init__(self, embed_dim: int, num_heads: int, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        self.heads = nn.ModuleList([SelfAttention(embed_dim, dropout) for _ in range(num_heads)])
        self.w = nn.Linear(embed_dim * num_heads, embed_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self._concat_heads(x)
        x = self.w(x)
        x = self.dropout(x)
        return x

    def _concat_heads(self, x: torch.Tensor) -> torch.Tensor:
        return torch.cat([head(x) for head in self.heads], dim=1)


class PositionalEncoding(nn.Module):
    def __init__(self, embed_dim: int, dropout: float = 0.1, max_len: int = 5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        pe = torch.zeros(max_len, embed_dim)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, embed_dim, 2).float() * (-math.log(10000.0) / embed_dim)
        )
        pe[:, 0::2] = torch.sin(position * div_term)  # even indices
        pe[:, 1::2] = torch.cos(position * div_term)  # odd indices
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer("pe", pe)

    def forward(self, x):
        x = x + self.pe[: x.size(0), :]
        return self.dropout(x)


class TransformerEncoderBlock(nn.Module):
    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        channels: list[int],
        dropout: float = 0.1,
    ):
        super().__init__()
        self.attention = MultiHeadAttention(embed_dim, num_heads, dropout)
        self.dropout = nn.Dropout(p=dropout)
        self.norm1 = nn.LayerNorm(embed_dim)
        self.fc = MLP(channels, dropout=dropout, activation=nn.GELU())
        self.norm2 = nn.LayerNorm(embed_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attention(x)
        x = self.dropout(x)
        x = self.norm1(x)
        x = x + self.fc(x)
        x = self.norm2(x)
        return x


class TransformerDecoderBlock(nn.Module):
    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        channels: list[int],
        dropout: float = 0.1,
    ):
        super().__init__()
        self.masked_attention = MultiHeadAttention(embed_dim, num_heads, dropout)
        self.dropout = nn.Dropout(p=dropout)
        self.norm1 = nn.LayerNorm(embed_dim)
        self.attention = MultiHeadAttention(embed_dim, num_heads, dropout)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.fc = MLP(channels, dropout=dropout, activation=nn.GELU())
        self.norm3 = nn.LayerNorm(embed_dim)

    def forward(self, x: torch.Tensor, enc_out: torch.Tensor) -> torch.Tensor:
        x = x + self.masked_attention(x)
        x = self.dropout(x)
        x = self.norm1(x)
        x = x + self.attention(x, enc_out)
        x = self.dropout(x)
        x = self.norm2(x)
        x = x + self.fc(x)
        x = self.norm3(x)
        return x


class TransformerEncoder(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        embed_dim: int,
        num_heads: int,
        channels: list[int],
        num_layers: int = 4,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, embed_dim)
        self.pe = PositionalEncoding(embed_dim, dropout)
        self.layers = nn.ModuleList(
            [
                TransformerEncoderBlock(embed_dim, num_heads, channels, dropout)
                for _ in range(num_layers)
            ]
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for layer in self.layers:
            x = layer(x)
        return x


class TransformerDecoder(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        embed_dim: int,
        num_heads: int,
        channels: list[int],
        num_layers: int = 4,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, embed_dim)
        self.pe = PositionalEncoding(embed_dim, dropout)
        self.layers = nn.ModuleList(
            [
                TransformerDecoderBlock(embed_dim, num_heads, channels, dropout)
                for _ in range(num_layers)
            ]
        )

    def forward(self, x: torch.Tensor, enc_out: torch.Tensor) -> torch.Tensor:
        x = self.embed(x)
        x = self.pe(x)
        for layer in self.layers:
            x = layer(x, enc_out)
        return x


class Transformer(nn.Module):
    def __init__(
        self,
        enc_vocab_size: int,
        dec_vocab_size: int,
        embed_dim: int,
        num_heads: int,
        channels: list[int],
        num_layers: int,
        dropout: float,
        head: nn.Module | None = None,
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.dec_vocab_size = dec_vocab_size

        self.encoder = TransformerEncoder(
            enc_vocab_size, embed_dim, num_heads, channels, num_layers, dropout
        )
        self.decoder = TransformerDecoder(
            dec_vocab_size, embed_dim, num_heads, channels, num_layers, dropout
        )
        if head is None:
            self.head = self._make_head()

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        enc_out = self.encoder(x)
        dec_out = self.decoder(y, enc_out)
        out = self.head(dec_out)
        return out

    def _make_head(self):
        raise NotImplementedError


class LangModelTransformer(Transformer):
    def __init__(
        self,
        vocab_size: int,
        embed_dim: int,
        num_heads: int,
        channels: list[int],
        num_layers: int,
        dropout: float,
    ):
        super().__init__(
            vocab_size,
            vocab_size,
            embed_dim,
            num_heads,
            channels,
            num_layers,
            dropout,
            head=nn.Linear(embed_dim, vocab_size),
        )
        self.vocab_size = vocab_size

    def _make_head(self):
        """Returns a probability distribution over the vocabulary."""
        return nn.Linear(self.embed_dim, self.vocab_size)


class ClassificationTransformer(Transformer):
    def __init__(
        self,
        enc_vocab_size: int,
        dec_vocab_size: int,
        num_classes: int,
        embed_dim: int,
        num_heads: int,
        channels: list[int],
        num_layers: int,
        dropout: float,
    ):
        super().__init__(
            enc_vocab_size,
            dec_vocab_size,
            embed_dim,
            num_heads,
            channels,
            num_layers,
            dropout,
            head=nn.Linear(embed_dim, dec_vocab_size),
        )
        self.num_classes = num_classes

    def _make_head(self):
        """Returns a probability distribution over the vocabulary."""
        return nn.Linear(self.embed_dim, self.num_classes)


def visualize_attention(x: torch.Tensor, y: torch.Tensor):
    import matplotlib.pyplot as plt

    plt.figure(figsize=(16, 16))
    plt.imshow(x @ y.transpose(-2, -1))
    plt.show()
