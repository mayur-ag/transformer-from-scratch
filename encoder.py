import torch
import torch.nn as nn

from attention import MultiHeadedSelfAttention


class TransformerBlock(nn.Module):
    def __init__(self, embedding_size, n_heads, dropout, forward_expansion=1):
        super().__init__()
        self.attention = MultiHeadedSelfAttention(embedding_size, n_heads)
        self.norm1 = nn.LayerNorm(embedding_size)
        self.norm2 = nn.LayerNorm(embedding_size)
        self.feed_forward = nn.Sequential(
            nn.Linear(embedding_size, forward_expansion * embedding_size),
            nn.ReLU(),
            nn.Linear(forward_expansion * embedding_size, embedding_size),
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, key, query, value, mask):
        attention = self.attention(key, query, value, mask)
        # first step
        out1 = self.dropout(self.norm1(attention + query))
        # second step, and skip connection
        out2 = self.norm2(self.feed_forward(out1) + out1)
        out2 = self.dropout(out2)
        return out2


class Encoder(nn.Module):
    def __init__(
        self,
        src_vocab_size,
        embedding_size,
        num_layers,
        n_heads,
        device,
        forward_expansion,
        dropout,
        max_length,
    ):
        super().__init__()
        self.embedding_size = embedding_size
        self.device = device
        self.word_embedding = nn.Embedding(src_vocab_size, embedding_size).to(
            self.device
        )
        self.position_embedding = nn.Embedding(max_length, embedding_size).to(
            self.device
        )
        self.layers = nn.ModuleList(
            [
                TransformerBlock(embedding_size, n_heads, dropout, forward_expansion)
                for _ in range(num_layers)
            ]
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask):
        N, seq_length = x.shape
        positions = torch.arange(0, seq_length).expand(N, seq_length).to(self.device)
        # make the embeddings aware of the positions
        out = self.word_embedding(x) + self.position_embedding(positions)
        out = self.dropout(out)
        for layer in self.layers:
            out = layer(out, out, out, mask)
        return out


if __name__ == "__main__":
    device = "cpu"
    enc = Encoder(10, 256, 4, 8, device, 1, 0, 10).to(device)

    src = torch.randint(0, 10, (4, 9)).to(device)
    src_mask = (src != 0).unsqueeze(1).unsqueeze(2).to(device)
    # target = torch.randint(0, 10, (2, 9)).to(device)
    # N, target_len = target.shape
    # target_mask = (
    #     torch.tril(torch.ones(target_len, target_len))
    #     .expand(N, 1, target_len, target_len)
    #     .to(device)
    # )
    out = enc(src, src_mask)
