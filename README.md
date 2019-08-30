# Neural Networks Applied to Signal Separation on Multi-Sensor Array Data

## Neural networks architectures developed for the Independent Research Project (ACSE 9)

Here are a set of U-shaped networks that have been developed and tested to perform seismic signal separation (a.k.a de-blending).
The code is written in Python 3.6 and the neural networks are implemented with the PyTorch platform (version 1.0.1.post2).

All the experiments presented in the Report have been run using UNetRes.py, UNetSkip.py, UNetInPlus.py and UNetDeep3.py.  

## Examples of how to instantiate the networks
Network in the script UNetRes.py can be instantiated as:

```
from UNetRes import UNet
blocks = [64, 96, 144, 216, 324, 486, 730, 1096, 1644]
inWidth = 636
inHeight = 1251
model = UNet(input_channels, output_channels, blocks, inWidth, inHeight)
```

This will generate the reference model, with 9 down-blocks and 8 down-blocks.  
input_channels: integer, is the number of channels in the input tensor.  
output_channels: integer, is the number of channels in the output tensor.  
blocks: list of integers, gives the number of kernels in the convolutions for each block.  
The length of this list is also the number of down-blocks.  
The last two integers are the width and height of the input tensor. Its third and fourth dimensions.  
Note that these last two inputs are removed in the code used in production. They are introduced here only for developing reasons, in particular, keep my modifications to the machine learning pipeline to a minimum.  

Networks in UNetSkip.py, UNetInPlus.py and UNetDeep3.py can be used in a similar way, for example:

```
from UNetSkip import UNetSkip
blocks = [64, 96, 144, 216, 324, 486, 730, 1096, 1644]
inWidth = 636
inHeight = 1251
model = UNetSkip(input_channels, output_channels, blocks, inWidth, inHeight)
```

## Fourier Transform Loss
In the script fft.loss, the class FFTloss implements the loss function discussed in Section 3.3.2 of the Report.
It requires two inputs:  
signal_ndim: integer, 1 or 2, defines wheather to perform 1d or 2d FFT.  
normalized: bool, default False, whether to normalize or not FFT output.  

