from torch import nn, Tensor
from zeta.nn import (
    MultiheadAttention,
    FeedForward,
    MultiQueryAttention,
)
from einops import rearrange, reduce


def threed_to_text(
    x: Tensor, max_seq_len: int, dim: int, flatten: bool = False
):
    """
    Converts a 3D tensor to text representation.

    Args:
        x (Tensor): The input tensor of shape (batch_size, sequence_length, input_dim).
        max_seq_len (int): The maximum sequence length of the output tensor.
        dim (int): The dimension of the intermediate tensor.
        flatten (bool, optional): Whether to flatten the intermediate tensor. Defaults to False.

    Returns:
        Tensor: The output tensor of shape (batch_size, max_seq_len, input_dim).
    """
    b, s, d = x.shape

    x = nn.Linear(d, dim)(x)

    x = rearrange(x, "b s d -> b d s")
    x = nn.Linear(s, max_seq_len)(x)
    x = rearrange(x, "b d s -> b s d")
    return x



def scatter3d_to_4d_spatial(x: Tensor, dim: int):
    """
    Scatters a 3D tensor into a 4D spatial tensor using einops.

    Args:
        x (Tensor): The input tensor of shape (b, s, d), where b is the batch size, s is the spatial dimension, and d is the feature dimension.
        dim (int): The dimension along which to scatter the tensor.

    Returns:
        Tensor: The scattered 4D spatial tensor of shape (b, (s*s1), d), where s1 is the new spatial dimension after scattering.
    """
    b, s, d = x.shape
    
    # Scatter the 3D tensor into a 4D spatial tensor
    x = rearrange(x, "b s d -> b (s s1) d")
    
    return x



class EEGConvEmbeddings(nn.Module):
    def __init__(
        self,
        num_channels,
        conv_channels,
        kernel_size,
        stride=1,
        padding=0,
    ):
        """
        Initializes the EEGConvEmbeddings module.

        Args:
        - num_channels (int): Number of EEG channels in the input data.
        - conv_channels (int): Number of output channels for the convolutional layer.
        - kernel_size (int): Size of the convolutional kernel.
        - stride (int, optional): Stride of the convolution. Default: 1.
        - padding (int, optional): Padding added to both sides of the input. Default: 0.
        """
        super(EEGConvEmbeddings, self).__init__()

        self.conv1 = nn.Conv1d(
            in_channels=num_channels,
            out_channels=conv_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
        )

        # Additional layers and operations can be added here

    def forward(self, x):
        """
        Forward pass of the EEGConvEmbeddings module.

        Args:
        - x (Tensor): Input tensor of shape (batch_size, num_channels, time_samples)

        Returns:
        - Tensor: Output tensor after convolution
        """
        x = self.conv1(x)
        return x


class FMRIEmbedding(nn.Module):
    def __init__(
        self,
        in_channels=1,
        out_channels=32,
        kernel_size=3,
        stride=1,
        padding=1,
    ):
        """
        Initializes an fMRI Embedding Network.

        Args:
        - in_channels (int): Number of input channels (scans/modalities).
        - out_channels (int): Number of output channels for the convolutional layer.
        - kernel_size (int): Size of the convolutional kernels.
        - stride (int): Stride of the convolutions.
        - padding (int): Padding added to the input.

        Example:
        model = fMRIEmbeddingNet()
        x = torch.randn(1, 1, 32, 32, 32)
        input_tensor = torch.randn(8, 1, 64, 64, 64)  # 8 fMRI scans
        output_tensor = model(input_tensor)
        print(output_tensor.shape)  # torch.Size([8, 32, 64, 64, 64])


        """
        super(FMRIEmbedding, self).__init__()

        self.conv1 = nn.Conv3d(
            in_channels, out_channels, kernel_size, stride, padding
        )
        # Additional layers can be added here as needed

    def forward(self, x):
        """
        Forward pass of the fMRI Embedding Network.

        Args:
        - x (Tensor): Input tensor of shape (batch_size, in_channels, D, H, W)

        Returns:
        - Tensor: Output embedding tensor
        """
        x = self.conv1(x)
        # Additional operations can be added here as needed
        return x


class MorpheusEncoder(nn.Module):
    """
    MorpheusEncoder is a module that performs encoding on EEG data using multi-head attention and feed-forward networks.

    Args:
        dim (int): The dimension of the input data.
        heads (int): The number of attention heads.
        depth (int): The number of layers in the encoder.
        dim_head (int): The dimension of each attention head.
        dropout (int): The dropout rate.
        num_channels (int): The number of input channels in the EEG data.
        conv_channels (int): The number of output channels after the convolutional layer.
        kernel_size (int): The size of the convolutional kernel.
        stride (int, optional): The stride of the convolutional layer. Defaults to 1.
        padding (int, optional): The padding size for the convolutional layer. Defaults to 0.
        ff_mult (int, optional): The multiplier for the feed-forward network hidden dimension. Defaults to 4.

    Attributes:
        dim (int): The dimension of the input data.
        heads (int): The number of attention heads.
        depth (int): The number of layers in the encoder.
        dim_head (int): The dimension of each attention head.
        dropout (int): The dropout rate.
        num_channels (int): The number of input channels in the EEG data.
        conv_channels (int): The number of output channels after the convolutional layer.
        kernel_size (int): The size of the convolutional kernel.
        stride (int): The stride of the convolutional layer.
        padding (int): The padding size for the convolutional layer.
        ff_mult (int): The multiplier for the feed-forward network hidden dimension.
        mha (MultiheadAttention): The multi-head attention module.
        ffn (FeedForward): The feed-forward network module.
        eeg_embedding (EEGConvEmbeddings): The EEG convolutional embedding module.

    """

    def __init__(
        self,
        dim: int,
        heads: int,
        depth: int,
        dim_head: int,
        dropout: int,
        num_channels,
        conv_channels,
        kernel_size,
        stride=1,
        padding=0,
        ff_mult: int = 4,
        *args,
        **kwargs,
    ):
        super(MorpheusEncoder, self).__init__()
        self.dim = dim
        self.heads = heads
        self.depth = depth
        self.dim_head = dim_head
        self.dropout = dropout
        self.num_channels = num_channels
        self.conv_channels = conv_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.ff_mult = ff_mult

        self.mha = MultiheadAttention(
            dim,
            heads,
            dropout,
            subln=True,
        )

        self.ffn = FeedForward(dim, dim, ff_mult, *args, **kwargs)

        self.eeg_embedding = EEGConvEmbeddings(
            num_channels, conv_channels, kernel_size, stride, padding
        )

    def forward(self, x: Tensor) -> Tensor:
        """
        Forward pass of the MorpheusEncoder module.

        Args:
            x (Tensor): The input tensor of shape (batch_size, seq_len, dim).

        Returns:
            Tensor: The output tensor of shape (batch_size, seq_len, dim).

        """
        x = self.eeg_embedding(x)
        # print(x.shape)

        x = self.mha(x, x, x) + x

        x = self.ffn(x) + x

        return x


class MorpheusDecoder(nn.Module):
    """
    MorpheusDecoder is a module that performs decoding in the Morpheus model.

    Args:
        dim (int): The dimension of the input and output tensors.
        heads (int): The number of attention heads.
        depth (int): The number of layers in the decoder.
        dim_head (int): The dimension of each attention head.
        dropout (int): The dropout rate.
        num_channels: The number of channels in the input tensor.
        conv_channels: The number of channels in the convolutional layers.
        kernel_size: The size of the convolutional kernel.
        in_channels: The number of input channels for the FMRI embedding.
        out_channels: The number of output channels for the FMRI embedding.
        stride (int, optional): The stride of the convolutional layers. Defaults to 1.
        padding (int, optional): The padding size for the convolutional layers. Defaults to 0.
        ff_mult (int, optional): The multiplier for the feed-forward network dimension. Defaults to 4.

    Attributes:
        dim (int): The dimension of the input and output tensors.
        heads (int): The number of attention heads.
        depth (int): The number of layers in the decoder.
        dim_head (int): The dimension of each attention head.
        dropout (int): The dropout rate.
        num_channels: The number of channels in the input tensor.
        conv_channels: The number of channels in the convolutional layers.
        kernel_size: The size of the convolutional kernel.
        stride (int): The stride of the convolutional layers.
        padding (int): The padding size for the convolutional layers.
        ff_mult (int): The multiplier for the feed-forward network dimension.
        frmi_embedding (nn.Linear): The linear layer for FRMI embedding.
        masked_attn (MultiQueryAttention): The masked attention module.
        mha (MultiheadAttention): The multihead attention module.
        frmni_embedding (FMRIEmbedding): The FMRI embedding module.
        ffn (FeedForward): The feed-forward network module.
        proj (nn.Linear): The linear layer for projection to original dimension.
        softmax (nn.Softmax): The softmax activation function.
        encoder (MorpheusEncoder): The MorpheusEncoder module.

    """

    def __init__(
        self,
        dim: int,
        heads: int,
        depth: int,
        dim_head: int,
        dropout: int,
        num_channels,
        conv_channels,
        kernel_size,
        in_channels,
        out_channels,
        stride=1,
        padding=0,
        ff_mult: int = 4,
        *args,
        **kwargs,
    ):
        super(MorpheusDecoder, self).__init__()
        self.dim = dim
        self.heads = heads
        self.depth = depth
        self.dim_head = dim_head
        self.dropout = dropout
        self.num_channels = num_channels
        self.conv_channels = conv_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.ff_mult = ff_mult

        self.frmi_embedding = nn.Linear(num_channels, dim)

        self.masked_attn = MultiQueryAttention(
            dim,
            heads,
        )

        self.mha = MultiheadAttention(
            dim,
            heads,
            dropout,
            subln=True,
        )

        self.frmni_embedding = FMRIEmbedding(
            in_channels, out_channels, kernel_size, stride, padding
        )

        self.ffn = FeedForward(dim, dim, ff_mult, *args, **kwargs)

        self.proj = nn.Linear(dim, num_channels)

        self.softmax = nn.Softmax(1)

        self.encoder = MorpheusEncoder(
            dim,
            heads,
            depth,
            dim_head,
            dropout,
            num_channels,
            conv_channels,
            kernel_size,
            stride,
            padding,
            ff_mult,
            *args,
            **kwargs,
        )

    def forward(self, frmi: Tensor, eeg: Tensor) -> Tensor:
        """
        Perform forward pass through the MorpheusDecoder.

        Args:
            frmi (Tensor): The FRMI input tensor.
            eeg (Tensor): The EEG input tensor.

        Returns:
            Tensor: The output tensor after decoding.

        """
        # X = FRMI of shapef
        # # MRI data is represented as a 4D tensor: [batch_size, channels, depth, height, width].
        # # EEG data is represented as a 3D tensor: [batch_size, channels, time_samples].
        x = self.frmi_embedding(frmi)

        # Rearrange to text dimension
        x = reduce(x, "b c d h w -> b (h w) (c d)", "sum")

        # Rearrange tensor to be compatible with attn
        x = threed_to_text(x, self.num_channels, self.dim)

        # Masked Attention
        x, _, _ = self.masked_attn(x)

        # EEG Encoder
        eeg = self.encoder(eeg)

        # Multihead Attention
        x = self.mha(x, eeg, x) + x

        # Feed Forward
        x = self.ffn(x) + x

        # Projection to original dimension
        x = self.proj(x)

        # Softmax
        x = self.softmax(x)

        return x


class Morpheus(nn.Module):
    """
    Morpheus model implementation.

    Args:
        dim (int): Dimension of the model.
        heads (int): Number of attention heads.
        depth (int): Number of layers in the model.
        dim_head (int): Dimension of each attention head.
        dropout (int): Dropout rate.
        num_channels: Number of input channels.
        conv_channels: Number of channels in the convolutional layers.
        kernel_size: Size of the convolutional kernel.
        in_channels: Number of input channels for the convolutional layers.
        out_channels: Number of output channels for the convolutional layers.
        stride (int, optional): Stride value for the convolutional layers. Defaults to 1.
        padding (int, optional): Padding value for the convolutional layers. Defaults to 0.
        ff_mult (int, optional): Multiplier for the feed-forward layer dimension. Defaults to 4.

    Attributes:
        dim (int): Dimension of the model.
        heads (int): Number of attention heads.
        depth (int): Number of layers in the model.
        dim_head (int): Dimension of each attention head.
        dropout (int): Dropout rate.
        num_channels: Number of input channels.
        conv_channels: Number of channels in the convolutional layers.
        kernel_size: Size of the convolutional kernel.
        stride (int): Stride value for the convolutional layers.
        padding (int): Padding value for the convolutional layers.
        ff_mult (int): Multiplier for the feed-forward layer dimension.
        layers (nn.ModuleList): List of MorpheusDecoder layers.
        norm (nn.LayerNorm): Layer normalization module.

    Examples:
        >>> model = Morpheus(
        ...     dim=128,
        ...     heads=4,
        ...     depth=2,
        ...     dim_head=32,
        ...     dropout=0.1,
        ...     num_channels=32,
        ...     conv_channels=32,
        ...     kernel_size=3,
        ...     in_channels=1,
        ...     out_channels=32,
        ...     stride=1,
        ...     padding=1,
        ...     ff_mult=4,
        ... )
        >>> frmi = torch.randn(1, 1, 32, 32, 32)
        >>> eeg = torch.randn(1, 32, 128)
        >>> output = model(frmi, eeg)
        >>> print(output.shape)


    """

    def __init__(
        self,
        dim: int,
        heads: int,
        depth: int,
        dim_head: int,
        dropout: int,
        num_channels,
        conv_channels,
        kernel_size,
        in_channels,
        out_channels,
        stride=1,
        padding=0,
        ff_mult: int = 4,
        scatter: bool = False,
        *args,
        **kwargs,
    ):
        super(Morpheus, self).__init__()
        self.dim = dim
        self.heads = heads
        self.depth = depth
        self.dim_head = dim_head
        self.dropout = dropout
        self.num_channels = num_channels
        self.conv_channels = conv_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.ff_mult = ff_mult
        self.scatter = scatter

        self.layers = nn.ModuleList([])

        for _ in range(depth):
            self.layers.append(
                MorpheusDecoder(
                    dim,
                    heads,
                    depth,
                    dim_head,
                    dropout,
                    num_channels,
                    conv_channels,
                    kernel_size,
                    in_channels,
                    out_channels,
                    stride,
                    padding,
                    ff_mult,
                    *args,
                    **kwargs,
                )
            )

        self.norm = nn.LayerNorm(num_channels)

    def forward(self, frmi: Tensor, eeg: Tensor) -> Tensor:
        """
        Forward pass of the Morpheus model.

        Args:
            frmi (Tensor): Input tensor for the frmi modality.
            eeg (Tensor): Input tensor for the eeg modality.

        Returns:
            Tensor: Output tensor of the model.

        """
        for layer in self.layers:
            x = layer(frmi, eeg)
            x = self.norm(x)
        
        if self.scatter:
            # Scatter the tensor to 4d spatial tensor
            s1 = x.shape[1]
            x = rearrange(x, "b (s s1) d -> b s s1 d", s1=s1)
    
        return x
