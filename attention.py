import torch
import torch.nn as nn


class MultiHeadedSelfAttention(nn.Module):
    def __init__(self, embedding_size, n_heads):
        super().__init__()
        self.embedding_size = embedding_size
        self.n_heads = n_heads
        self.head_dim = self.embedding_size // self.n_heads
        assert self.embedding_size == self.head_dim * self.n_heads, (
            f"Embedding size doesn't match "
            f" {self.embedding_size} != {self.n_heads} x {self.head_dim}"
        )
        self.keys = nn.Linear(self.head_dim, self.head_dim, bias=False)
        self.values = nn.Linear(self.head_dim, self.head_dim, bias=False)
        self.queries = nn.Linear(self.head_dim, self.head_dim, bias=False)
        self.fc_out = nn.Linear(self.embedding_size, self.embedding_size)

    def forward(
        self,
        key: torch.Tensor,
        query: torch.Tensor,
        value: torch.Tensor,
        mask: torch.Tensor,
    ):
        # batch size
        N = query.shape[0]
        key_len, query_len, value_len = key.shape[1], query.shape[1], value.shape[1]
        # split embedding into self.head pieces
        key = key.reshape(N, key_len, self.n_heads, self.head_dim)
        query = key.reshape(N, query_len, self.n_heads, self.head_dim)
        value = key.reshape(N, value_len, self.n_heads, self.head_dim)
        # perform forward pass to learn the weights
        key = self.keys(key)
        query = self.queries(query)
        value = self.values(value)
        # multiple query and key to calculate the energy
        # energy shape: N x num_heads x query_len x key_len
        energy = torch.einsum("nqhd, nkhd -> nhqk", query, key)
        # wherever mask is zero, we need to ignore those values
        # after taking attention -inf values will be 0
        if mask is not None:
            energy = energy.masked_fill(mask == 0, float("-1e20"))
        # calculate attention across key_len dimension
        attention = torch.softmax(energy / self.embedding_size ** 1 / 2, dim=3)
        # attend to values
        # attention = N x num_heads x query_len x key_len
        # values = N x value_len x n_heads x head_dim
        out = torch.einsum("nhql, nlhd -> nqhd", attention, value)
        # concatenate the additional dimension that was added
        out = out.reshape(N, query_len, self.embedding_size)
        # pass through Linear layer
        out = self.fc_out(out)
        return out
