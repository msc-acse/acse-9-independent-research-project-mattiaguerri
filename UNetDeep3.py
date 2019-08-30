import torch
import torch.nn as nn
import numpy as np


# 1x1 convolution with padding = 0.
def con1x1_0(inCha, outCha, kernel_size=1, stride=1, padding=0, bias=True):
    return nn.Conv2d(inCha, outCha, kernel_size=kernel_size, stride=stride,
                     padding=padding, bias=bias)


# 3x3 convolution with padding = 0.
def con3x3_0(inCha, outCha, kernel_size=3, stride=1, padding=0, bias=True):
    return nn.Conv2d(inCha, outCha, kernel_size=kernel_size, stride=stride,
                     padding=padding, bias=bias)


# 3x3 convolution with padding = 1.
def con3x3_1(inCha, outCha, kernel_size=3, stride=1, padding=1, bias=True):
    return nn.Conv2d(inCha, outCha, kernel_size=kernel_size, stride=stride,
                     padding=padding, bias=bias)


# 2x2 transposed convolution.
def traCon2x2(inCha, outCha, kernel_size, stride, padding,
              output_padding, bias=True):
    return nn.ConvTranspose2d(inCha, outCha, kernel_size=kernel_size,
                              stride=stride, padding=padding,
                              output_padding=output_padding, bias=bias)


# ReLU Activation function.
def actReLU():
    return nn.ReLU(inplace=True)


class DownBlock(nn.Module):
    """
    Block of the downsampling (encoding) path.

    Parameters
    ----------
    inCha : integer
        Input channels for the convolution.
    outCha : integer
        Output channels of the convolution (number of kernels).
    pooling : bool
        Whether to perform the pooling or not at the end of the block.
    poolPad : tuple of two integers
        Padding to be used in the pooling.

    Methods
    -------
    forward : forward pass into the block
    """
    def __init__(self, inCha, outCha, pooling, poolPad):
        super(DownBlock, self).__init__()

        self.pooling = pooling

        self.con0 = con3x3_1(inCha, outCha)
        self.con1 = con3x3_1(outCha, outCha)
        self.con2 = con3x3_1(outCha, outCha)

        self.act = actReLU()

        if self.pooling:
            self.pool = nn.MaxPool2d(kernel_size=2, stride=2, padding=poolPad)

    def forward(self, x):
        """
        Perform the forward pass through the block.

        Parameters
        ----------
        x : torch tensor
            4D tensor, output of the block.
                first dimension = batch size
                second dimension = number of channels
                third dimension = number of columns
                fourth dimension = number of rows

        Returns
        -------
        x : torch tensor
            4D tensor output by the block.
              first dimension = batch size
              second dimension = number of channels
              third dimension = number of columns
              fourth dimension = number of rows
        x_0 : torch tensor
            4D tensor output by the block before pooling. Dimensions as above.
        """
        x = self.act(self.con0(x))
        x = self.act(self.con1(x))
        x = self.act(self.con2(x))
        xOut = x
        if self.pooling:
            x = self.pool(x)

        # print("\n DownBlock Output Shape = ", x.shape)

        return x, xOut


class UpBlock(nn.Module):
    """
    Block of the upsampling (decoding) path.

    Parameters
    ----------
    inCha : integer
        Input channels for the convolution.
    outCha : integer
        Output channels of the convolution (number of kernels).
    tranPad : tuple of two intergers
        Padding to be used in the transposed convolution.
    finConv : bool
        Define whether to do the last 1x1 convolution or not.

    Methods
    -------
    forward : forward pass into the block
    """
    def __init__(self, inCha, outCha, tranPad, finConv=False, finOutCha=1):
        super(UpBlock, self).__init__()

        self.traCon = traCon2x2(inCha, outCha, kernel_size=2, stride=2,
                                padding=tranPad, output_padding=(0, 0))
        self.con0 = con3x3_1(2*outCha, outCha)
        self.con1 = con3x3_1(outCha, outCha)
        self.con2 = con3x3_1(outCha, outCha)

        self.act = actReLU()

        self.finConv = finConv
        if self.finConv:
            self.con3 = con1x1_0(outCha, finOutCha)

    def forward(self, xDown, x):
        """
        Perform the forward pass through the block.

        Parameters
        ----------
        xDown : torch tensor
            4D tensor generated by the corresding block in the
            downsampling path.
              first dimension = batch size
              second dimension = number of channels
              third dimension = number of columns
              fourth dimension = number of rows
        x : torch tensor
            4D tensor, output of the block.
              first dimension = batch size
              second dimension = number of channels
              third dimension = number of columns
              fourth dimension = number of rows

        Returns
        -------
        x : torch tensor
            4D tensor output by the block.
              first dimension = batch size
              second dimension = number of channels
              third dimension = number of columns
              fourth dimension = number of rows
        """
        xUp = self.traCon(x)
        x = torch.cat((xUp, xDown), 1)

        x = self.act((self.con0(x)))
        x = self.act((self.con1(x)))
        x = self.act((self.con2(x)))

        if self.finConv:
            x = self.con3(x)

        # print("\n     UpBlock Output Shape = ", x.shape)

        return x


class UNetDeep3(nn.Module):
    """
    The network is constituted by a downsampling (encoding) path
    and an upsampling (decoding) path.
    Each block performs three convolutions.
    The last block of the decoder also performs a 1x1 convolution.

    Parameters
    ----------
    inCha : integer
        Number of channels in the input tensor.
    finOutCha : integer
        Number of channels of the network output.
    depths : list of integer
        First number is the number of input channels.
        The remaining numbers give the number of kernels used
        in the convolutions performed in the downsampling and
        upsampling paths.
        The length of this list, minus one,
        gives the number of downsampling blocks.
        The number of Upsampling blocks is len(depths)-2.
    inWidth : integer
        Number of columns of the input image.
    inHeight : integer
        Number of rows of the input image.

    Methods
    -------
    forward : forward pass into the network
    """
    def __init__(self, inCha, finOutCha, depths, inWidth, inHeight):
        super(UNetDeep3, self).__init__()

        # If input height and/or width are odd, the input has been padded.
        # Correct the input width and heigth.
        if inWidth % 2 != 0:
            inWidth += 1
        if inHeight % 2 != 0:
            inHeight += 1

        # Adapt the list depths.
        depths.insert(0, inCha)
        self.depths = depths

        self.down_convs = []
        self.up_convs = []

        # Build the encoder.
        testList0 = [inWidth]
        testList1 = [inHeight]
        numDowns = len(depths)-1  # number of downsampling blocks
        pad0 = 0
        pad1 = 0
        for i in range(numDowns):
            pooling = True if i < numDowns-1 else False
            # avoid odd num. using padding.
            if (pooling and i < numDowns - 2):
                if (inWidth/2) % 2 == 0:
                    pad0 = 0
                    inWidth //= 2
                    testList0.append(inWidth)
                else:
                    pad0 = 1
                    inWidth = inWidth // 2 + 1
                    testList0.append(inWidth)
                if (inHeight/2) % 2 == 0:
                    pad1 = 0
                    inHeight //= 2
                    testList1.append(inHeight)
                else:
                    pad1 = 1
                    inHeight = inHeight // 2 + 1
                    testList1.append(inHeight)
            # do not avoid odd numbers at the bottom.
            if (pooling and i == numDowns - 2):
                pad0 = 0
                inWidth //= 2
                testList0.append(inWidth)
                pad1 = 0
                inHeight //= 2
                testList1.append(inHeight)
            block = DownBlock(depths[i], depths[i+1], pooling, (pad0, pad1))
            self.down_convs.append(block)

        # Build the decoder.
        testList0.reverse()
        testList1.reverse()
        numUps = len(depths) - 2  # number of upsampling blocks
        self.depths.reverse()  # reverse the list of the depths
        for i in range(numUps):
            if testList0[i]*2 == testList0[i+1]:
                pad0 = 0
            else:
                pad0 = 1
            if testList1[i]*2 == testList1[i+1]:
                pad1 = 0
            else:
                pad1 = 1
            if i < (numUps - 1):
                block = UpBlock(depths[i], depths[i+1], (pad0, pad1))
                self.up_convs.append(block)
            else:
                block = UpBlock(depths[i], depths[i+1], (pad0, pad1),
                                finConv=True, finOutCha=finOutCha)
                self.up_convs.append(block)

        self.down_convs = nn.ModuleList(self.down_convs)
        self.up_convs = nn.ModuleList(self.up_convs)

    def forward(self, x):
        """
        Perform the forward pass through the network.

        Parameters
        ----------
        x : torch tensor
            4D tensor defining input to be fed into the network.
                first dimension = batch size
                second dimension = number of channels
                third dimension = number of columns of the input image
                fourth dimension = Number of rows of the input image

        Returns
        -------
        x : torch tensor
            4D tensor output by the network.
                first dimension = batch size
                second dimension = number of channels
                third dimension = number of columns of the target
                fourth dimension = number of rows of the target
        """
        encoder_outs = []

        # If input height and/or width are odd, make it even (pad it).
        padIt = False
        pad3 = 0
        pad2 = 0

        if x.shape[3] % 2 != 0:
            padIt = True
            pad3 = 1
        if x.shape[2] % 2 != 0:
            padIt = True
            pad2 = 1
        if padIt:
            p2d = (0, pad3, 0, pad2)
            x = nn.functional.pad(x, p2d, 'constant', 0)

        # Encoder.
        for i, module in enumerate(self.down_convs):
            x, x_0 = module(x)
            encoder_outs.append(x_0)

        # Decoder.
        encoder_outs.reverse()
        for i, module in enumerate(self.up_convs):
            x_0 = encoder_outs[i+1]
            x = module(x_0, x)

        # If the input has been padded, crop the output.
        if padIt:
            x = x[:, :, 0:(x.shape[2]-pad2), 0:(x.shape[3]-pad3)]

        return x
