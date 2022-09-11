import torch
import torch.nn as nn

from attention import MultiHeadedSelfAttention
from encoder import TransformerBlock, Encoder


class DecoderBlock(nn.Module):
    def __init__(self, embedding_size, n_heads, forward_expansion, dropout, device):
        super().__init__()
        self.attention = MultiHeadedSelfAttention(embedding_size, n_heads)
        self.norm = nn.LayerNorm(embedding_size)
        self.transformer_block = TransformerBlock(
            embedding_size, n_heads, dropout, forward_expansion
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, key, value, src_mask, target_mask):
        attention = self.attention(x, x, x, target_mask)
        query = self.dropout(self.norm(attention + x))
        out = self.transformer_block(key, query, value, src_mask)
        return out


class Decoder(nn.Module):
    def __init__(
        self,
        target_vocab_size,
        embedding_size,
        num_layers,
        n_heads,
        forward_expansion,
        dropout,
        device,
        max_length,
    ):
        super().__init__()
        self.device = device
        self.word_embedding = nn.Embedding(target_vocab_size, embedding_size)
        self.position_embedding = nn.Embedding(max_length, embedding_size)
        self.layers = nn.ModuleList(
            [
                DecoderBlock(
                    embedding_size, n_heads, forward_expansion, dropout, device
                )
                for _ in range(num_layers)
            ]
        )
        self.fc_out = nn.Linear(embedding_size, target_vocab_size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, encoder_out, src_mask, target_mask):
        N, seq_len = x.shape
        positions = torch.arange(0, seq_len).expand(N, seq_len).to(self.device)
        x = self.dropout(self.word_embedding(x) + self.position_embedding(positions))
        for layer in self.layers:
            x = layer(x, encoder_out, encoder_out, src_mask, target_mask)
        out = self.fc_out(x)
        return out


if __name__ == "__main__":
    device = "mps"
    enc = Encoder(10, 256, 4, 8, device, 1, 0, 10).to(device)
    dec = Decoder(10, 256, 4, 1, 1, 0, device, 10).to(device)
    x = torch.randint(0, 10, (2, 9)).to(device)
    target = torch.randint(0, 10, (2, 9)).to(device)
    N, target_len = target.shape
    src_mask = (x != 0).unsqueeze(1).unsqueeze(2).to(device)
    target_mask = (
        torch.tril(torch.ones(target_len, target_len))
        .expand(N, 1, target_len, target_len)
        .to(device)
    )
    enc_out = enc(x, src_mask)
    out = dec(x, enc_out, src_mask, target_mask)
