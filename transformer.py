import torch
import torch.nn as nn

from decoder import Decoder
from encoder import Encoder


class Transformer(nn.Module):
    def __init__(
        self,
        src_vocab_size,
        target_vocab_size,
        src_pad_idx,
        target_pad_idx,
        embedding_size=256,
        num_layers=6,
        forward_expansion=4,
        n_heads=8,
        dropout=0,
        device="cpu",
        max_length=100,
    ):
        super().__init__()
        self.encoder = Encoder(
            src_vocab_size,
            embedding_size,
            num_layers,
            n_heads,
            device,
            forward_expansion,
            dropout,
            max_length,
        ).to(device)
        self.decoder = Decoder(
            target_vocab_size,
            embedding_size,
            num_layers,
            n_heads,
            forward_expansion,
            dropout,
            device,
            max_length,
        ).to(device)
        self.src_pad_idx = src_pad_idx
        self.target_pad_idx = target_pad_idx
        self.device = device

    def make_src_mask(self, src: torch.Tensor):
        return (src != self.src_pad_idx).unsqueeze(1).unsqueeze(2).to(self.device)

    def make_target_mask(self, target: torch.Tensor):
        N, target_len = target.shape
        target_mask = (
            torch.tril(torch.ones(target_len, target_len))
            .expand(N, 1, target_len, target_len)
            .to(self.device)
        )
        return target_mask

    def forward(self, src, target):
        src_mask = self.make_src_mask(src)
        target_mask = self.make_target_mask(target)
        enc_src = self.encoder(src, src_mask)
        out = self.decoder(target, enc_src, src_mask, target_mask)
        return out
