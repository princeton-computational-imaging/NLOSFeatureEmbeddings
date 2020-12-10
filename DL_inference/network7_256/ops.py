

import torch.nn.functional as F
import torch.nn as nn
import torch

import numpy as np


#############################################################################
class Interpsacle2d(nn.Module):

    def __init__(self, factor=2, gain=1, align_corners=False):
        """
            the first upsample method in G_synthesis.
        :param factor:
        :param gain:
        """
        super(Interpsacle2d, self).__init__()
        self.gain = gain
        self.factor = factor
        self.align_corners = align_corners

    def forward(self, x):
        if self.gain != 1:
            x = x * self.gain
        
        x = nn.functional.interpolate(x, scale_factor=self.factor, mode='bilinear', align_corners=self.align_corners)
        
        return x


######################################################################
class ResConv3D(nn.Module):

    def __init__(self, nf0, inplace=False):
        super(ResConv3D, self).__init__()
        
        self.tmp = nn.Sequential(
                
                nn.ReplicationPad3d(1),
                nn.Conv3d(nf0 * 1,
                          nf0 * 1,
                          kernel_size=[3, 3, 3],
                          padding=0,
                          stride=[1, 1, 1],
                          bias=True),
                
                nn.LeakyReLU(negative_slope=0.2, inplace=inplace),
                # nn.Dropout3d(0.1, inplace),
                
                nn.ReplicationPad3d(1),
                nn.Conv3d(nf0 * 1,
                          nf0 * 1,
                          kernel_size=[3, 3, 3],
                          padding=0,
                          stride=[1, 1, 1],
                          bias=True),
        )
        
        self.inplace = inplace

    def forward(self, x):
        re = F.leaky_relu(self.tmp(x) + x, negative_slope=0.2, inplace=self.inplace)
        return re


class ResConv2D(nn.Module):

    def __init__(self, nf0, inplace=False):
        super(ResConv2D, self).__init__()
        
        self.tmp = nn.Sequential(
                
                nn.ReplicationPad2d(1),
                nn.Conv2d(nf0 * 1,
                          nf0 * 1,
                          kernel_size=[3, 3],
                          padding=0,
                          stride=[1, 1],
                          bias=True),
                
                nn.LeakyReLU(negative_slope=0.2, inplace=inplace),
                # nn.Dropout3d(0.1, inplace),
                
                nn.ReplicationPad2d(1),
                nn.Conv2d(nf0 * 1,
                          nf0 * 1,
                          kernel_size=[3, 3],
                          padding=0,
                          stride=[1, 1],
                          bias=True),
        )
        
        self.inplace = inplace

    def forward(self, x):
        re = F.leaky_relu(self.tmp(x) + x, negative_slope=0.2, inplace=self.inplace)
        return re


#####################################################################
if __name__ == '__main__':
    
    import numpy as np
    
    btf = torch.from_numpy(np.random.rand(10, 16, 256, 256).astype(np.float32)).cuda()
    
    scaledownlayer = Interpsacle2d(0.5)
    ctf = scaledownlayer(btf)
    
    scaleuplayer = Interpsacle2d(2.0)
    dtf = scaleuplayer(ctf)
    
    print(dtf.shape)
    
