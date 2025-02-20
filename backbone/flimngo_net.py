import math
import torch
import torch.nn as nn
import torch.nn.init
import torch.nn.functional as F
import torchvision
import functools # used by RRDBNet
from torchvision.transforms import Resize

# Model selection function
def GetModel(opt):
    
    if opt.model.lower() == 'flimngo':
        net = FLIMngo(opt)
    else:
        print("model undefined")    
        return None
    
    if not opt.cpu:
        net.cuda()

    return net

# Conv-BN-SiLU block
class CBL(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding):
        super(CBL, self).__init__()

        conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, bias=False)
        bn = nn.BatchNorm2d(out_channels)

        self.cbl = nn.Sequential(
            conv,
            bn,
            
            nn.SiLU(inplace=True)
        )

    def forward(self, x):
        return self.cbl(x)

# Residual bottleneck block
class Bottleneck(nn.Module):
    """
    Parameters:
        in_channels (int): number of channel of the input tensor
        out_channels (int): number of channel of the output tensor
        width_multiple (float): it controls the number of channels (and weights)
                                of all the convolutions beside the
                                first and last one. If closer to 0,
                                the simpler the modelIf closer to 1,
                                the model becomes more complex
    """
    def __init__(self, in_channels, out_channels, width_multiple=1):
        super(Bottleneck, self).__init__()
        c_ = int(width_multiple*in_channels)
        self.c1 = CBL(in_channels, c_, kernel_size=1, stride=1, padding=0)
        self.c2 = CBL(c_, out_channels, kernel_size=3, stride=1, padding=1)

    def forward(self, x):
        return self.c2(self.c1(x)) + x

# C3 block with residual bottlenecks
class C3(nn.Module):
    """
    Parameters:
        in_channels (int): number of channel of the input tensor
        out_channels (int): number of channel of the output tensor
        width_multiple (float): it controls the number of channels (and weights)
                                of all the convolutions beside the
                                first and last one. If closer to 0,
                                the simpler the modelIf closer to 1,
                                the model becomes more complex
        depth (int): it controls the number of times the bottleneck (residual block)
                        is repeated within the C3 block
        backbone (bool): if True, self.seq will be composed by bottlenecks 1, if False
                            it will be composed by bottlenecks 2 (check in the image linked below)
        https://user-images.githubusercontent.com/31005897/172404576-c260dcf9-76bb-4bc8-b6a9-f2d987792583.png
    """
    def __init__(self, in_channels, out_channels, width_multiple=1, depth=1, backbone=True):
        super(C3, self).__init__()
        c_ = int(width_multiple*in_channels)

        self.c1 = CBL(in_channels, c_, kernel_size=1, stride=1, padding=0)
        self.c_skipped = CBL(in_channels,  c_, kernel_size=1, stride=1, padding=0)
        if backbone:
            self.seq = nn.Sequential(
                *[Bottleneck(c_, c_, width_multiple=1) for _ in range(depth)]
            )
        else:
            self.seq = nn.Sequential(
                *[nn.Sequential(
                    CBL(c_, c_, 1, 1, 0),
                    CBL(c_, c_, 3, 1, 1)
                ) for _ in range(depth)]
            )
        self.c_out = CBL(c_ * 2, out_channels, kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        x = torch.cat([self.seq(self.c1(x)), self.c_skipped(x)], dim=1)
        return self.c_out(x)   

# Spatial Pyramid Pooling Fast (SPPF)
class SPPF(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(SPPF, self).__init__()

        c_ = int(in_channels//2)

        self.c1 = CBL(in_channels, c_, 1, 1, 0)
        self.pool = nn.MaxPool2d(kernel_size=5, stride=1, padding=2)
        self.c_out = CBL(c_ * 4, out_channels, 1, 1, 0)

    def forward(self, x):
        x = self.c1(x)
        pool1 = self.pool(x)
        pool2 = self.pool(pool1)
        pool3 = self.pool(pool2)

        return self.c_out(torch.cat([x, pool1, pool2, pool3], dim=1))

# FLIMngo model
class FLIMngo(nn.Module):
    def __init__(self, opt):
        super(FLIMngo, self).__init__()
        n_in_channels = opt.n_in_channels
        self.width_multiple = opt.width_multiple
        self.first_out = 86
        im_size = opt.imageSize
        self.bin_width = opt.bin_width

        # default time resolution training data have been simulated with
        default_bin_width = 0.09765625 
        self.bin_width_ratio = self.bin_width/default_bin_width

        # Define backbone module
        self.head = nn.Sequential(
            CBL(n_in_channels, 2*self.first_out, kernel_size=5, stride=1, padding=2),
            CBL(2*self.first_out, 2*self.first_out, kernel_size=3, stride=2, padding=1)
        )
        
        # Encoder
        self.down_one = nn.Sequential(
            C3(in_channels=2*self.first_out, out_channels=2*self.first_out, width_multiple=self.width_multiple, depth=4),
            CBL(2*self.first_out, 4*self.first_out, kernel_size=3, stride=2, padding=1)
        )

        self.down_two = nn.Sequential(
            C3(in_channels=4*self.first_out, out_channels=4*self.first_out, width_multiple=self.width_multiple, depth=4),
            CBL(4*self.first_out, 8*self.first_out, kernel_size=3, stride=2, padding=1),
            SPPF(8*self.first_out, 8*self.first_out)
        )

        # Decoder
        self.up_one = nn.Sequential(
            CBL(8*self.first_out, 4*self.first_out, kernel_size=1, stride=1, padding=0),
            Resize((int(im_size/4), int(im_size/4)))
        )

        self.up_two = nn.Sequential(
            CBL(8*self.first_out, 4*self.first_out, kernel_size=1, stride=1, padding=0),
            C3(in_channels=4*self.first_out, out_channels=4*self.first_out, width_multiple=self.width_multiple, depth=4),
            CBL(4*self.first_out, 2*self.first_out, kernel_size=1, stride=1, padding=0),
            Resize((int(im_size/2), int(im_size/2)))
        )

        self.tail = nn.Sequential(
            CBL(4*self.first_out, 2*self.first_out, kernel_size=1, stride=1, padding=0),
            C3(in_channels=2*self.first_out, out_channels=2*self.first_out, width_multiple=self.width_multiple, depth=4),
            Resize((int(im_size), int(im_size))),
            CBL(2*self.first_out, self.first_out, kernel_size=1, stride=1, padding=0),
            C3(in_channels=self.first_out, out_channels=self.first_out, width_multiple=self.width_multiple, depth=2),
            nn.Conv2d(in_channels=self.first_out, out_channels=1, kernel_size=3, stride=1, padding=1, bias=False), 
            nn.ReLU(inplace=True)
        )
    
    def forward(self, x):
        """Forward pass with U-Net-like skip connections"""
        # x = [256 256 256]

        # Encoder
        pool1 = self.head(x) # First downsampling --> [172, 128, 128]
        pool2 = self.down_one(pool1) # Second downsampling --> [344, 64, 64]
        pool3 = self.down_two(pool2) # Third downsampling --> [688, 32, 32]

        # Decoder
        concat_1 = torch.cat([self.up_one(pool3), pool2], dim=1) # First upsampling --> [688, 64, 64]
        concat_2 = torch.cat([self.up_two(concat_1), pool1], dim=1) # Second upsampling --> [344, 128, 128]
        
        output = self.tail(concat_2) # Final processing --> [1, 256, 256]

        return output.squeeze() * self.bin_width_ratio # Squeeze to remove extra dimensions and adust output based on input bin width
   