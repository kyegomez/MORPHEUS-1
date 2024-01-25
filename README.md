[![Multi-Modality](agorabanner.png)](https://discord.gg/qUtxnK2NMf)

# Morpheus 1

![Morphesus transformer](morpheus.jpeg)

Implementation of "MORPHEUS-1" from Prophetic AI and "The world’s first multi-modal generative ultrasonic transformer designed to induce and stabilize lucid dreams. "





## Installation

You can install the package using pip

```bash
pip install morpheus-torch
```

# Usage
- The input is FRMI and EEG tensors.

- FRMI shape is (batch_size, in_channels, D, H, W)

- EEG Embedding is [batch_size, channels, time_samples]

```python
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


```



### Code Quality 🧹

We providehandy commands inside the `Makefile`, namely:

- `make style` to format the code
- `make check_code_quality` to check code quality (PEP8 basically)
- `black .`
- `ruff . --fix`

# License
MIT

# Todo
- [ ] Implement the scatter in the end of the decoder to output spatial outputs

- [ ] Implement a full model with the depth of the decoder layers

- [ ] Change all the MHAs to Multi Query Attentions

- [ ] Double check popular brain scan EEG and FRMI AI papers to double check tensor shape

