

# License: BSD
# Author: Ghassen Hamrouni

from __future__ import print_function
from __future__ import division

import torch
import torch.nn as nn
import torch.optim as optim

import numpy as np
import cv2

from config import get_args
args = get_args()

import sys
sys.path.append('../%s' % args.netfolder)
from deepvoxel import DeepVoxels as Net

############################################
# Make experiments reproducible
_ = torch.manual_seed(1234569527)
np.random.seed(123456)

mode = args.mode
in_dim = args.in_dim
out_dim = in_dim * 2
model = Net(img_sidelength=256,
    in_channels=in_dim,
    out_channels=out_dim,
    nf0=args.dim,
    grid_dim=args.grid,
    mode=mode,
    bin_len=0.0096,
    # addraw=args.addraw > 0,
    # raytracing=args.raytracing > 0
    )

def rmnan(tensor):
    tensor[torch.isnan(tensor)] = 0
    tensor[torch.isinf(tensor)] = 0
    return tensor

for p in model.parameters():
    p.register_hook(lambda grad: rmnan(grad))

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device_ids = [0]
model = model.to(device)
model.todev(device)

criterion = nn.BCELoss()
mse = nn.MSELoss()


#################################################################
if __name__ == '__main__':
    
    prefix = '%s_bike_mode-%s_dim-%d_trainframe-%d_lr-%.5f_addraw-%d_raytracing-%d' % (args.netsvfolder, mode, args.dim, args.frame, args.lr, args.addraw, args.raytracing)
    
    prefix = '%s_mode-%s_dim-%d_trainframe-%d_lr-%.5f_raytracing-%d' % (args.netsvfolder, mode, args.dim, args.frame, args.lr, args.raytracing)

    prefix = '%s_%s_mode-%s_dim-%d_trainframe-%d_tres-%d_confocal-%d_lr-%.5f' % (args.netsvfolder, args.netfolder, mode, args.dim, args.frame, args.tres, args.confocal, args.lr)

    prefix = '%s_%s_mode-%s_dim-%d_trainnum-%d_trainframe-%d_gtsz-%d-%d_tres-%d_confocal-%d_lr-%.5f' % \
            (args.netsvfolder, args.netfolder, mode, args.dim, args.datanum, args.frame, args.w, args.h, args.tres, args.confocal, args.lr)

    rootfolder = args.datafolder
    datafolder = '%s/datasets' % (folder, prefix)
    modelsvdir = '%s/%s' % (folder, prefix)
    
    epochstart = args.epochbe
    epochsv = args.epochsv
    epochnum = args.epoch
    from testrealcall import testreal
    for epochbe in range(58, 59):
        if epochbe >= -1:
            model.load_state_dict(torch.load('%s/%d.pth' % (modelsvdir, epochbe), map_location='cpu'))
            print('data loaded %d' % epochbe)
            with torch.no_grad():
                testreal(model, epochbe, args.tres, args.in_dim, device, datafolder, '.')

