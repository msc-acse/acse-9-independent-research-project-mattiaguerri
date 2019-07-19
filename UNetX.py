import torch
import torch.nn as nn
from torch.nn.modules.utils import _single, _pair, _triple
import sys
import numpy as np





# 1x1 convolution with padding = 0.
def con1x1_0(inCha, outCha, kernel_size = 1, stride = 1, padding = 0, bias = False):
    return nn.Conv2d(inCha, outCha, kernel_size=kernel_size, stride=stride, padding=padding, bias=bias)





# 3x3 convolution with padding = 0.
def con3x3_0(inCha, outCha, kernel_size = 3, stride = 1, padding = 0, bias = False):
    return nn.Conv2d(inCha, outCha, kernel_size=kernel_size, stride=stride, padding=padding, bias=bias)





# 3x3 convolution with padding = 1.
def con3x3_1(inCha, outCha, kernel_size = 3, stride = 1, padding = 1, bias = False):
    return nn.Conv2d(inCha, outCha, kernel_size=kernel_size, stride=stride, padding=padding, bias=bias)





# 2x2 transposed convolution.
def traCon2x2(inCha, outCha, kernel_size, stride, padding, output_padding):
    return nn.ConvTranspose2d(inCha, outCha, kernel_size=kernel_size, stride=stride, padding=padding, output_padding=output_padding)





# ReLU Activation function.
def actReLU():
    return nn.ReLU(inplace=True)





class DownBlock(nn.Module):
    """

    """
    def __init__(self, inCha, outCha, pooling, poolPad, blockInd):
        super(DownBlock, self).__init__()
        
        self.pooling = pooling
        
        self.con0 = con1x1_0(inCha, outCha)
        self.con1 = con3x3_1(inCha, outCha)
        self.con2 = con3x3_1(outCha, outCha)
        
        self.act = actReLU()
        
        if (self.pooling and blockInd%2==0):
            self.pool = nn.MaxPool2d(kernel_size = (1, 2), stride = (1, 2), padding=poolPad)
        elif (self.pooling and blockInd%2!=0):
            self.pool = nn.MaxPool2d(kernel_size = (2, 1), stride = (2, 1), padding=poolPad)
        
    def forward(self, x):
        """ 

        """
        xAdd = self.con0(x)
        
        x = self.act(self.con1(x))
        x = self.act(self.con2(x) + xAdd)
        
        x_0 = x
        if self.pooling:
            x = self.pool(x)
        
        print("\n DownBlock Output Shape = ", x.shape)
        
        return x, x_0

    
    
    
    
class UpBlock(nn.Module):
    """

    """
    def __init__(self, inCha, outCha, tranPad, finConv=False, finOutCha=1):
        super(UpBlock, self).__init__()
        
        self.finConv = finConv
        
        self.traCon = traCon2x2(inCha, outCha, kernel_size=2, stride=2, padding=tranPad, output_padding=(0, 0))
        
        self.con0 = con1x1_0(2*outCha, outCha)
        self.con1 = con3x3_1(2*outCha, outCha)
        self.con2 = con3x3_1(outCha, outCha)
        
        self.act = actReLU()
        
        if self.finConv:
            self.con3 = nn.Conv2d(outCha, finOutCha, kernel_size=1, stride=1, padding=0, bias=False)
        
    def forward(self, xDown, x):
        """ 

        """
        xUp = self.traCon(x)
        x = torch.cat((xUp, xDown), 1)
        xAdd = self.con0(x)
        
        x = self.act(self.con1(x))
        x = self.act(self.con2(x) + xAdd)
        
        if self.finConv:
            x = self.con3(x)
        
        print("\n     UpBlock Output Shape = ", x.shape)
            
        return x
    
    
    
    
    
class UNetX(nn.Module):
    """

    """
    def __init__(self, inCha, finOutCha, depths, inWidth, inHeight):
        super(UNetX, self).__init__()
        
        # If input height and/or width are odd, the input has been padded.
        # Correct the input width and heigth.
        if inWidth%2 != 0:
            inWidth += 1
        if inHeight%2 != 0:
            inHeight += 1
        
        # Adapt the list depths.
        depths.insert(0, inCha)
        self.depths = depths
    
        self.down_convs = []
        self.up_convs = []
        
        
#         # Build the encoder.
#         testList0 = [inWidth]
#         testList1 = [inHeight]
#         numDowns = len(depths)-1 # number of downsampling blocks
#         pad0 = 0
#         pad1 = 0
#         for i in range(numDowns):
#             pooling = True if i < numDowns-1 else False
#             if (pooling and i<numDowns-2): # avoid odd numbers using padding. 
#                 if (inWidth/2)%2==0:
#                     pad0 = 0
#                     inWidth //= 2
#                     testList0.append(inWidth)
#                 else:
#                     pad0 = 1
#                     inWidth = inWidth//2+1
#                     testList0.append(inWidth)
#                 if (inHeight/2)%2==0:
#                     pad1 = 0
#                     inHeight //= 2
#                     testList1.append(inHeight)
#                 else:
#                     pad1 = 1
#                     inHeight = inHeight//2+1
#                     testList1.append(inHeight)
#             if (pooling and i==numDowns-2): # do not avoid odd numbers at the bottom. 
#                 pad0 = 0
#                 inWidth //= 2
#                 testList0.append(inWidth)
#                 pad1 = 0
#                 inHeight //= 2
#                 testList1.append(inHeight)


        # Build the encoder.
        testList0 = [inWidth]
        testList1 = [inHeight]
        numDowns = len(depths)-1 # number of downsampling blocks
        pad0 = 0
        pad1 = 0
        for i in range(numDowns):
            pooling = True if i < numDowns-1 else False
            if (pooling and i<numDowns-2): # avoid odd numbers using padding.
                if i%2==0:
                    pad0 = 0
                    inWidth = inWidth
                    testList0.append(inWidth)
                    if (inHeight/2)%2==0:
                        pad1 = 0
                        inHeight //= 2
                        testList1.append(inHeight)
                    else:
                        pad1 = 1
                        inHeight = inHeight//2+1
                        testList1.append(inHeight)
                else:
                    pad1 = 0
                    inHeight = inHeight
                    testList1.append(inHeight)
                    if (inWidth/2)%2==0:
                        pad0 = 0
                        inWidth //= 2
                        testList0.append(inWidth)
                    else:
                        pad0 = 1
                        inWidth = inWidth//2+1
                        testList0.append(inWidth)
            if (pooling and i==numDowns-2): # do not avoid odd numbers at the bottom.
                if i%2==0:
                    pad0 = 0
                    inWidth = inWidth
                    testList0.append(inWidth)
                    pad1 = 0
                    inHeight //= 2
                    testList1.append(inHeight)
                else:
                    pad0 = 0
                    inWidth //= 2
                    testList0.append(inWidth)
                    pad1 = 0
                    inHeight = inHeight
                    testList1.append(inHeight)
                    
            
            block = DownBlock(depths[i], depths[i+1], pooling, (pad0,pad1), i)
            self.down_convs.append(block)
        
        
#         # Build the decoder.
#         testList0.reverse()
#         testList1.reverse()
#         numUps = len(depths)-2 # number of upsampling blocks
#         self.depths.reverse() # reverse the list of the depths
#         for i in range(numUps):
#             if testList0[i]*2==testList0[i+1]:
#                 pad0 = 0
#             else:
#                 pad0 = 1
#             if testList1[i]*2==testList1[i+1]:
#                 pad1 = 0
#             else:
#                 pad1 = 1
#             if i<(numUps-1):
#                 block = UpBlock(depths[i], depths[i+1], (pad0, pad1))
#                 self.up_convs.append(block)
#             else:
#                 block = UpBlock(depths[i], depths[i+1], (pad0, pad1), finConv=True, finOutCha=finOutCha)
#                 self.up_convs.append(block)
        
        
        # Build the decoder.
        testList0.reverse()
        testList1.reverse()
        numUps = len(depths)-2 # number of upsampling blocks
        self.depths.reverse() # reverse the list of the depths
        for i in range(numUps):
            if testList0[i]*2==testList0[i+1]:
                pad0 = 0
            else:
                pad0 = 1
            if testList1[i]*2==testList1[i+1]:
                pad1 = 0
            else:
                pad1 = 1
            if i<(numUps-1):
                block = UpBlock(depths[i], depths[i+1], (pad0, pad1))
                self.up_convs.append(block)
            else:
                block = UpBlock(depths[i], depths[i+1], (pad0, pad1), finConv=True, finOutCha=finOutCha)
                self.up_convs.append(block)
        
        
        self.down_convs = nn.ModuleList(self.down_convs)
        self.up_convs = nn.ModuleList(self.up_convs)
        
        
    def forward(self, x):
        """

        """
        encoder_outs = []
        
        # If input height and/or width are odd, make it even (pad it).
        padIt = False
        pad3 = 0
        pad2 = 0
        #shape3 = x.shape[3]
        #shape2 = x.shape[2]
        if x.shape[3] != 0:
            padIt = True
            pad3 = 1
        if x.shape[2]%2 != 0:
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
        
        
        
        
        
    
    
    
    
    
    
    
    
    
    