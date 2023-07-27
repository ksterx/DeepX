import math

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from .mlp import MLP


class Attention(nn.Module):
    def __init__(self, embed_dim: int, num_heads: int = 1, dropout: float = 0.0):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        self.scale = math.sqrt(embed_dim)
        head_dim = embed_dim // num_heads
        assert (
            head_dim * num_heads == embed_dim
        ), "embed_dim must be divisible by num_heads"
        self.w_q = nn.Linear(embed_dim, head_dim, bias=False)
        self.w_k = nn.Linear(embed_dim, head_dim, bias=False)
        self.w_v = nn.Linear(embed_dim, head_dim, bias=False)

    def forward(
        self,
        query: Tensor,
        key: Tensor,
        value: Tensor,
        mask: Tensor | None = None,
    ) -> tuple[Tensor, Tensor]:
        """Computes the attention scores.

        Args:
            query: (batch_size, seq_len, embed_dim)
            key: (batch_size, seq_len, embed_dim)
            value: (batch_size, seq_len, embed_dim)
            mask: (batch_size, seq_len, seq_len)

        Returns:
            attention and similarity: (batch_size, seq_len, embed_dim)
        """
        q = self.w_q(query)  # (batch_size, seq_len, embed_dim)
        k = self.w_k(key)  # (batch_size, seq_len, embed_dim)
        v = self.w_v(value)  # (batch_size, seq_len, embed_dim)
        sim = self.scaled_dot_product(q, k, mask)  # (batch_size, seq_len, seq_len)
        out = self.dropout(sim)  # (batch_size, seq_len, seq_len)
        score = out @ v  # (batch_size, seq_len, embed_dim)
        return score, sim

    @staticmethod
    def scaled_dot_product(
        query: Tensor,
        key: Tensor,
        mask: Tensor | None = None,
    ) -> Tensor:
        """Computes the similarity between the query and key.

        Args:
            mask: -inf for masked values, 0 otherwise

        Returns:
            similarity: (batch_size, seq_len, embed_dim)
        """
        num_channels = query.shape[-1]
        assert query.shape == key.shape, "query and key must have the same shape"

        similarity = (
            query @ key.transpose(-2, -1) / num_channels
        )  # (batch_size, seq_len, seq_len)
        if mask is not None:
            similarity = similarity + mask
        similarity = F.softmax(similarity, dim=-1)  # (batch_size, seq_len, seq_len)
        return similarity


class SelfAttention(Attention):
    def forward(self, x: Tensor, mask: Tensor | None = None) -> tuple[Tensor, Tensor]:  # type: ignore
        return super().forward(x, x, x, mask)


class MultiHeadAttention(nn.Module):
    def __init__(self, embed_dim: int, num_heads: int, dropout: float = 0.0):
        super().__init__()

        self.head_dim = embed_dim // num_heads
        assert (
            self.head_dim * num_heads == embed_dim
        ), "embed_dim must be divisible by num_heads"

        self.dropout = nn.Dropout(p=dropout)
        self.heads = nn.ModuleList(
            [Attention(embed_dim, num_heads, dropout) for _ in range(num_heads)]
        )
        self.fc = nn.Linear(embed_dim, embed_dim)
        self.relu = nn.ReLU()

    def forward(
        self, query: Tensor, key: Tensor, value: Tensor, mask: Tensor | None = None
    ) -> tuple[Tensor, Tensor]:
        """Computes the multi-head attention scores.

        Args:
            query (Tensor): (batch_size, seq_len, embed_dim)
            key (Tensor): (batch_size, seq_len, embed_dim)
            value (Tensor): (batch_size, seq_len, embed_dim)
            mask (Tensor): (batch_size, seq_len, seq_len)

        Returns:
            attention (Tensor): (batch_size, seq_len, embed_dim)
        """
        x, sim = self._concat_heads(query, key, value, mask)
        x = self.dropout(x)
        x = self.fc(x)
        x = self.relu(x)
        return x, sim

    def _concat_heads(self, query, key, value, mask) -> tuple[Tensor, Tensor]:
        out = [head(query, key, value, mask) for head in self.heads]
        score = torch.cat([x[0] for x in out], dim=-1)
        sim = torch.cat([x[1] for x in out], dim=-1)
        return score, sim


class MultiHeadSelfAttention(MultiHeadAttention):
    def forward(self, x: Tensor, mask: Tensor | None = None) -> tuple[Tensor, Tensor]:  # type: ignore
        return super().forward(x, x, x, mask)


class PositionalEncoding(nn.Module):
    def __init__(self, embed_dim: int, dropout: float = 0.0, max_len: int = 5000):
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
        hidden_dim: int,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.attention = MultiHeadSelfAttention(embed_dim, num_heads, dropout)
        self.dropout = nn.Dropout(p=dropout)
        self.norm1 = nn.LayerNorm(embed_dim)
        self.fc = MLP(
            [embed_dim, hidden_dim, embed_dim],
            dropout=dropout,
            activation=nn.GELU(),
            flatten=False,
        )
        self.norm2 = nn.LayerNorm(embed_dim)

    def forward(self, x: Tensor, mask: Tensor | None = None) -> tuple[Tensor, Tensor]:
        score, sim = self.attention(x, mask)
        x = x + score
        x = self.dropout(x)
        x = self.norm1(x)
        x = x + self.fc(x)
        x = self.norm2(x)
        return x, sim


class TransformerDecoderBlock(nn.Module):
    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        hidden_dim: int,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.masked_attention = MultiHeadSelfAttention(embed_dim, num_heads, dropout)
        self.dropout = nn.Dropout(p=dropout)
        self.norm1 = nn.LayerNorm(embed_dim)
        self.attention = MultiHeadAttention(embed_dim, num_heads, dropout)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.fc = MLP(
            [embed_dim, hidden_dim, embed_dim], dropout=dropout, activation=nn.GELU()
        )
        self.norm3 = nn.LayerNorm(embed_dim)

    def forward(
        self, x: Tensor, enc_out: Tensor, tgt_mask: Tensor | None = None
    ) -> tuple[Tensor, Tensor]:
        score, sim = self.masked_attention(x, mask=tgt_mask)
        x = x + score
        x = self.dropout(x)
        x = self.norm1(x)
        x = x + self.attention(query=x, key=enc_out, value=enc_out, mask=None)
        x = self.dropout(x)
        x = self.norm2(x)
        x = x + self.fc(x)
        x = self.norm3(x)
        return x, sim


class TransformerEncoder(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        embed_dim: int,
        num_heads: int,
        hidden_dim: int,
        num_blocks: int = 6,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, embed_dim)
        self.pe = PositionalEncoding(embed_dim, dropout)
        self.blocks = nn.ModuleList(
            [
                TransformerEncoderBlock(embed_dim, num_heads, hidden_dim, dropout)
                for _ in range(num_blocks)
            ]
        )

    def forward(
        self, x: Tensor, mask: Tensor | None = None
    ) -> tuple[Tensor, list[Tensor]]:
        x = self.embed(x)
        x = self.pe(x)
        sims = []
        for block in self.blocks:
            x, sim = block(x, mask)
            sims.append(sim)
        return x, sims


class TransformerDecoder(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        embed_dim: int,
        num_heads: int,
        hidden_dim: int,
        num_blocks: int = 6,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, embed_dim)
        self.pe = PositionalEncoding(embed_dim, dropout)
        self.blocks = nn.ModuleList(
            [
                TransformerDecoderBlock(embed_dim, num_heads, hidden_dim, dropout)
                for _ in range(num_blocks)
            ]
        )

    def forward(
        self, x: Tensor, enc_out: Tensor, mask: Tensor | None = None
    ) -> tuple[Tensor, list[Tensor]]:
        x = self.embed(x)
        x = self.pe(x)
        sims = []
        for block in self.blocks:
            x, sim = block(x, enc_out, mask)
            sims.append(sim)
        return x, sims


class Transformer(nn.Module):
    def __init__(
        self,
        enc_vocab_size: int,
        dec_vocab_size: int,
        embed_dim: int,
        num_heads: int,
        hidden_dim: int,
        num_enc_blocks: int = 6,
        num_dec_blocks: int = 6,
        dropout: float = 0.0,
        head: nn.Module | None = None,
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.dec_vocab_size = dec_vocab_size

        self.encoder = TransformerEncoder(
            enc_vocab_size, embed_dim, num_heads, hidden_dim, num_enc_blocks, dropout
        )
        self.decoder = TransformerDecoder(
            dec_vocab_size, embed_dim, num_heads, hidden_dim, num_dec_blocks, dropout
        )
        if head is None:
            self.head = self._make_head()

    def forward(
        self,
        src: Tensor,
        tgt: Tensor,
        src_mask: Tensor | None = None,
        tgt_mask: Tensor | None = None,
    ) -> Tensor:
        enc_out, enc_sims = self.encoder(src, mask=src_mask)
        dec_out, dec_sims = self.decoder(tgt, enc_out, mask=tgt_mask)
        return self.head(dec_out)

    def _make_head(self):
        raise NotImplementedError


class LangModelTransformer(TransformerEncoder):
    NAME = "lmtransformer"

    def __init__(
        self,
        vocab_size: int,
        embed_dim: int,
        num_heads: int = 8,
        hidden_dim: int = 2048,
        num_blocks: int = 6,
        dropout: float = 0.0,
    ):
        super().__init__(
            vocab_size,
            embed_dim,
            num_heads,
            hidden_dim,
            num_blocks,
            dropout,
        )
        self.vocab_size = vocab_size
        self.head = nn.Linear(embed_dim, vocab_size)

    def forward(
        self, x: Tensor, mask: Tensor | None = None
    ) -> tuple[Tensor, list[Tensor]]:
        x = self.embed(x)
        x = self.pe(x)
        sims = []
        for block in self.blocks:
            x, sim = block(x, mask)
            sims.append(sim)
        return self.head(x), sims


class ClassificationTransformer(Transformer):
    def __init__(
        self,
        enc_vocab_size: int,
        dec_vocab_size: int,
        num_classes: int,
        embed_dim: int,
        num_heads: int,
        hidden_dim: int,
        num_enc_blocks: int,
        num_dec_blocks: int,
        dropout: float,
    ):
        super().__init__(
            enc_vocab_size,
            dec_vocab_size,
            embed_dim,
            num_heads,
            hidden_dim,
            num_enc_blocks,
            num_dec_blocks,
            dropout,
            head=nn.Linear(embed_dim, dec_vocab_size),
        )
        self.num_classes = num_classes

    def _make_head(self):
        """Returns a probability distribution over the vocabulary."""
        return nn.Linear(self.embed_dim, self.num_classes)


class TranslationTransformer(Transformer):
    def __init__(
        self,
        enc_vocab_size: int,
        dec_vocab_size: int,
        embed_dim: int,
        num_heads: int,
        hidden_dim: int,
        num_enc_blocks: int,
        num_dec_blocks: int,
        dropout: float,
    ):
        super().__init__(
            enc_vocab_size,
            dec_vocab_size,
            embed_dim,
            num_heads,
            hidden_dim,
            num_enc_blocks,
            num_dec_blocks,
            dropout,
            head=nn.Linear(embed_dim, dec_vocab_size),
        )

    def _make_head(self):
        """Returns a probability distribution over the vocabulary."""
        return nn.Linear(self.embed_dim, self.dec_vocab_size)


def visualize_attention(x: Tensor, y: Tensor):
    import matplotlib.pyplot as plt

    plt.figure(figsize=(16, 16))
    plt.imshow(x @ y.transpose(-2, -1))
    plt.show()
