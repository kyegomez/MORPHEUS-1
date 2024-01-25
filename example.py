# Importing the torch library
import torch

# Importing the Morpheus model from the morpheus_torch package
from morpheus_torch.model import Morpheus

# Creating an instance of the Morpheus model with specified parameters
model = Morpheus(
    dim=128,                # Dimension of the model
    heads=4,                # Number of attention heads
    depth=2,                # Number of transformer layers
    dim_head=32,            # Dimension of each attention head
    dropout=0.1,            # Dropout rate
    num_channels=32,        # Number of input channels
    conv_channels=32,       # Number of channels in convolutional layers
    kernel_size=3,          # Kernel size for convolutional layers
    in_channels=1,          # Number of input channels for convolutional layers
    out_channels=32,        # Number of output channels for convolutional layers
    stride=1,               # Stride for convolutional layers
    padding=1,              # Padding for convolutional layers
    ff_mult=4,              # Multiplier for feed-forward layer dimension
)

# Creating random tensors for input data
frmi = torch.randn(1, 1, 32, 32, 32)    # Random tensor for FRMI data
eeg = torch.randn(1, 32, 128)           # Random tensor for EEG data

# Passing the input data through the model to get the output
output = model(frmi, eeg)

# Printing the shape of the output tensor
print(output.shape)
