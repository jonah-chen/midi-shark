import torch
from torch.nn import Module
from torch.nn import Sequential
from torch.nn import Conv2d, BatchNorm2d, ReLU, AdaptiveAvgPool2d, Upsample
import torch.nn.functional as F
import math


class SepearableConv2d(Module):
    """
    Seperable Convolution layer consisting of:
        Depthwise Convolution: a single convolution filter applied to 
            each channel.

        Pointwise Convolution: convolution with a kernel size 1 to extract
            output features of the layer.

    Note that batch normalization and activation function is applied after
    the depthwise convolution.
    """

    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size=3,
                 stride=1,
                 padding=0,
                 dilation=1,
                 groups=1,
                 depth_bias=False,
                 point_bias=False,
                 padding_mode='zeros'
                 ):
        """
        Args:
            depth_bias (bool, optional): Use a bias for the depthwise 
                                         convolution. Defaults to False.
            point_bias (bool, optional): Use a bias for the pointwise 
                                         convolution. Defaults to False.

            Other Args: refer to the documentation of torch.nn.Conv2d. 
        """
        super(SepearableConv2d, self).__init__()
        self.depth_conv = Conv2d(in_channels=in_channels,
                                 out_channels=in_channels,
                                 kernel_size=kernel_size,
                                 stride=stride,
                                 padding=padding,
                                 dilation=dilation,
                                 groups=in_channels,
                                 bias=depth_bias,
                                 padding_mode=padding_mode)

        self.point_conv = Conv2d(in_channels=in_channels,
                                 out_channels=out_channels,
                                 kernel_size=1,
                                 stride=1,
                                 padding=0,
                                 dilation=1,
                                 groups=groups,
                                 bias=point_bias,
                                 padding_mode=padding_mode)

        self.batch_norm = BatchNorm2d(in_channels)

    def forward(self, x):
        return self.point_conv(F.relu(self.batch_norm(self.depth_conv(x))))


class ConvBlockNoSkip(Module):
    """
    Block of 3 convolutional layers with an identity skip connection.
    """
    def __init__(self, in_channels, sizes, stride3=1):
        """
        Args:
            in_channels (int): The number of input channels.
            sizes (int or tuple): The number output channels for each of the
                three layers. If one int is specified, then the number of
                output channels for all three layers are equal. 
            stride3 (int, optional): The stride of the final layer in the 
                block. Defaults to 1.
        """
        super(ConvBlockNoSkip, self).__init__()

        if type(sizes) is int:
            sizes = (sizes, sizes, sizes)

        self.layer1 = SepearableConv2d(in_channels, sizes[0], padding=1)
        self.layer2 = SepearableConv2d(sizes[0], sizes[1], padding=1)
        self.layer3 = SepearableConv2d(
            sizes[1], sizes[2], padding=1, stride=stride3)

        self.batch_norm1 = BatchNorm2d(sizes[0])
        self.batch_norm2 = BatchNorm2d(sizes[1])
        self.batch_norm3 = BatchNorm2d(sizes[2])

    def forward(self, x):
        y = F.relu(self.batch_norm1(self.layer1(x)))
        y = F.relu(self.batch_norm2(self.layer2(y)))
        y = self.batch_norm3(self.layer3(y))
        return F.relu(x + y)


class ConvBlock(ConvBlockNoSkip):
    """
    Block of 3 convolutional layers with an single convolutional layer 
    as the skip connection. 
    """
    def __init__(self, in_channels, sizes, stride3=2):
        """
        Args:
            in_channels (int): The number of input channels.
            sizes (int or tuple): The number output channels for each of the
                three layers. If one int is specified, then the number of
                output channels for all three layers are equal. 
            stride3 (int, optional): The stride of the final layer in the 
                block. Defaults to 2.
        """
        super(ConvBlock, self).__init__(in_channels, sizes, stride3)

        if type(sizes) is int:
            sizes = (sizes, sizes, sizes)

        self.skip1 = Conv2d(in_channels, sizes[2],
                            kernel_size=1, stride=stride3, padding=0, bias=False)
        self.batch_norm4 = BatchNorm2d(sizes[2])

    def forward(self, x):
        y = self.batch_norm4(self.skip1(x))
        x = F.relu(self.batch_norm1(self.layer1(x)))
        x = F.relu(self.batch_norm2(self.layer2(x)))
        return y + self.batch_norm3(self.layer3(x))


class ConvBlockReturnSkip(ConvBlock):
    """
    Block of 3 convolutional layers with an single convolutional layer 
    as the skip connection. 

    WARNING: Do NOT use directly in a sequential model as `forward` returns
    two Tensors, which cause sequential to crash.
    """
    def __init__(self, in_channels, sizes, stride3=2):
        super(ConvBlockReturnSkip, self).__init__(in_channels, sizes, stride3)

    def forward(self, x):
        y = self.batch_norm4(self.skip1(x))
        x = F.relu(self.batch_norm1(self.layer1(x)))
        x = F.relu(self.batch_norm2(self.layer2(x)))
        return y + self.batch_norm3(self.layer3(x)), x


class SpacialPyramidPooling(Module):
    """
    A Spacial Pyramid Pooling module for multi-scale feature extraction
    which includes the following layers concatonated:
        - 1x1 Conv for local features.
        - 3x3 Conv with dilation rate 6 for small features.
        - 3x3 Conv with dilation rate 12 for medium features.
        - 3x3 Conv with dilation rate 18 for large features.
        - Image pooling layer for global features.
    following the design described in `Encoder-Decoder with Atrous Separable
    Convolution for Semantic Image Segmentation`.
    """
    def __init__(self, in_channels=2048, out_channels=256):
        """
        Args:
            in_channels (int, optional): . Defaults to 2048.
            out_channels (int, optional): Number of output channels for each
            feature. Defaults to 256.
        """
        super(SpacialPyramidPooling, self).__init__()

        # Direct 1x1 convolution layer
        self.conv1 = Conv2d(in_channels, out_channels,
                            kernel_size=1, bias=False)
        self.batch_norm1 = BatchNorm2d(out_channels)

        # Atrous convolution rate=6
        self.conv6 = Conv2d(in_channels, out_channels,
                            kernel_size=3, dilation=6, 
                            padding=6, bias=False)
        self.batch_norm6 = BatchNorm2d(out_channels)

        # Atrous convolution rate=12
        self.conv12 = Conv2d(in_channels, out_channels,
                             kernel_size=3, dilation=12, 
                             padding=12, bias=False)
        self.batch_norm12 = BatchNorm2d(out_channels)

        # Atrous convolution rate=18
        self.conv18 = Conv2d(in_channels, out_channels,
                             kernel_size=3, dilation=18, 
                             padding=18, bias=False)
        self.batch_norm18 = BatchNorm2d(out_channels)

        # Image pooling
        self.img_pool = AdaptiveAvgPool2d(1)
        self.img_pool_conv = Conv2d(
            in_channels, out_channels, kernel_size=1, bias=False)
        self.img_pool_bn = BatchNorm2d(out_channels)

    def forward(self, x):
        # Direct 1x1 convolution
        y1 = F.relu(self.batch_norm1(self.conv1(x)))

        # Atrous convolution at different rates
        y6 = F.relu(self.batch_norm6(self.conv6(x)))
        y12 = F.relu(self.batch_norm12(self.conv12(x)))
        y18 = F.relu(self.batch_norm18(self.conv18(x)))

        # Image pooling
        w, h = x.shape[2], x.shape[3]
        x = self.img_pool(x)
        x = F.relu(self.img_pool_bn(self.img_pool_conv(x)))
        x = F.interpolate(x, size=(w, h))

        return torch.cat((y1, y6, y12, y18, x), 1)


class DeepLabv3Encoder(Module):
    """
    The encoder part of the semantic segmentation model described in 
    `Encoder-Decoder with Atrous Separable Convolution for Semantic
     Image Segmentation`.
    """
    def __init__(self):
        super(DeepLabv3Encoder, self).__init__()

        self.entry_flow1 = Sequential(
            Conv2d(3, 32, kernel_size=3, stride=2, padding=1,
                   bias=False),
            BatchNorm2d(32),
            ReLU(),
            Conv2d(32, 64, kernel_size=3, padding=1,
                   bias=False),
            BatchNorm2d(64),
            ReLU(),
            ConvBlock(64, 128),
            ReLU()
        )

        self.entry_4x = ConvBlockReturnSkip(128, 256)

        self.entry_flow2 = Sequential(
            ReLU(),
            ConvBlock(256, 728),
            ReLU()
        )

        self.middle_flow = Sequential(*[
            ConvBlockNoSkip(728, 728) for _ in range(16)
        ])

        self.exit_flow = Sequential(
            ConvBlock(728, (728, 1024, 1024), stride3=1),
            SepearableConv2d(1024, 1536, padding=2, dilation=2),
            BatchNorm2d(1536),
            ReLU(),
            SepearableConv2d(1536, 1536, padding=2, dilation=2),
            BatchNorm2d(1536),
            ReLU(),
            SepearableConv2d(1536, 2048, padding=2, dilation=2),
            BatchNorm2d(2048),
            ReLU()
        )

        self.spp = SpacialPyramidPooling()

        self.encode_conv = Conv2d(256*5, 256, kernel_size=1, bias=False)
        self.encode_bn = BatchNorm2d(256)

        self.low_encode_conv = Conv2d(256, 48, kernel_size=1, bias=False)
        self.low_encode_bn = BatchNorm2d(48)

    def forward(self, x):
        x = self.entry_flow1(x)
        x, y = self.entry_4x(x)
        x = self.entry_flow2(x)
        x = self.middle_flow(x)
        x = self.exit_flow(x)
        x = self.spp(x)

        x = F.relu(self.encode_bn(self.encode_conv(x)))
        x = F.interpolate(x, scale_factor=4, mode="bilinear")

        y = F.relu(self.low_encode_bn(self.low_encode_conv(y)))

        x = torch.cat((x, y), 1)

        return F.interpolate(x, scale_factor=4, mode="bilinear")


class ImageDecoder(Module):
    """
    The decoder part of the semantic segmentation model described in 
    `Encoder-Decoder with Atrous Separable Convolution for Semantic
     Image Segmentation`.
    """
    def __init__(self, classes):
        super(ImageDecoder, self).__init__()
        self.params = Sequential(
            Conv2d(256+48, 256, kernel_size=3, padding=1, bias=False),
            BatchNorm2d(256), 
            ReLU(),
            Conv2d(256, 256, kernel_size=3, padding=1, bias=False),
            BatchNorm2d(256),
            ReLU(),
            Conv2d(256, classes, kernel_size=1)
        )


    def forward(self, x):
        return self.params(x)

if __name__ == '__main__':
    if torch.cuda.is_available():  # Use GPU if and only if available
        torch.set_default_tensor_type('torch.cuda.FloatTensor')
        torch.set_default_dtype(torch.float32)

    dl = DeepLabv3Encoder()
    dld = ImageDecoder(10)
    test = torch.rand(2, 3, 512, 512)
    print(dld(dl(test)))
    print(dld(dl(test)).shape)
