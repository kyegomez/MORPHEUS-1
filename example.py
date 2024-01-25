import torch

from morpheus_torch import MorpheusDecoder

model = MorpheusDecoder(
    dim=128,
    heads=4,
    depth=2,
    dim_head=32,
    dropout=0.1,
    num_channels=32,
    conv_channels=32,
    kernel_size=3,
    in_channels=1,
    out_channels=32,
    stride=1,
    padding=1,
    ff_mult=4,
)

frmi = torch.randn(1, 1, 32, 32, 32)
eeg = torch.randn(1, 32, 128)

output = model(frmi, eeg)
print(output.shape)
