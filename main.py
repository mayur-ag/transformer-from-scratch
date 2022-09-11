import torch

from transformer import Transformer


def train():
    device = (
        torch.device("mps")
        if torch.backends.mps.is_available()
        else torch.device("cpu")
    )
    print("Device:", device)
    x = torch.randint(0, 10, (2, 9)).to(device)
    target = torch.randint(0, 10, (2, 9)).to(device)
    src_pad_index = 0
    target_pad_index = 0
    src_vocab_size = 10
    target_vocab_size = 10
    model = Transformer(
        src_vocab_size,
        target_vocab_size,
        src_pad_index,
        target_pad_index,
        device=device,
    ).to(device)
    print(x)
    print(target)
    out = model(x, target)
    print(out.shape)


if __name__ == "__main__":
    train()
