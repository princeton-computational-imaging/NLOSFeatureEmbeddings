

import torch
import torch.nn as nn
import torch.nn.functional as F

import  numpy as np
import sys
sys.path.append('../utils')
from helper import definePsf, resamplingOperator, \
waveconvparam, waveconv


class phasor2(nn.Module):
    
    def __init__(self, spatial=256, crop=512, \
                 bin_len=0.01, wall_size=2.0, \
                 sampling_coeff=2.0, \
                 cycles=5):
        super(phasor2, self).__init__()
        
        self.spatial_grid = spatial
        self.crop = crop
        assert 2 ** int(np.log2(crop)) == crop
        
        self.bin_len = bin_len
        self.wall_size = wall_size
        
        self.sampling_coeff = sampling_coeff
        self.cycles = cycles
        
        self.parpareparam()
    
    #####################################################
    def parpareparam(self):
        
        self.c = 3e8
        self.width = self.wall_size / 2.0;
        self.bin_resolution = self.bin_len / self.c
        self.trange = self.crop * self.c * self.bin_resolution
        
        ########################################################3
        temprol_grid = self.crop
        sptial_grid = self.spatial_grid
        
        wall_size = self.wall_size
        bin_resolution = self.bin_resolution
        
        sampling_coeff = self.sampling_coeff
        cycles = self.cycles
        
        ######################################################
        # Step 0: define virtual wavelet properties
        # s_lamda_limit = wall_size / (sptial_grid - 1);  # sample spacing on the wall
        # sampling_coeff = 2;  # scale the size of the virtual wavelength (usually 2, optionally 3 for noisy scenes)
        # virtual_wavelength = sampling_coeff * (s_lamda_limit * 2);  # virtual wavelength in units of cm
        # cycles = 5;  # number of wave cycles in the wavelet, typically 4-6
        
        s_lamda_limit = wall_size / (sptial_grid - 1);  # sample spacing on the wall
        virtual_wavelength = sampling_coeff * (s_lamda_limit * 2);  # virtual wavelength in units of cm
        self.virtual_wavelength = virtual_wavelength
        
        virtual_cos_wave_k, virtual_sin_wave_k = \
        waveconvparam(bin_resolution, virtual_wavelength, cycles)
        
        virtual_cos_sin_wave_2xk = np.stack([virtual_cos_wave_k, virtual_sin_wave_k], axis=0)
        
        # use pytorch conv to replace matlab conv
        self.virtual_cos_sin_wave_inv_2x1xk = torch.from_numpy(virtual_cos_sin_wave_2xk[:, ::-1].copy()).unsqueeze(1)
        
        ###################################################
        slope = self.width / self.trange
        psf = definePsf(sptial_grid, temprol_grid, slope)
        fpsf = np.fft.fftn(psf)
        # lct
        # invpsf = np.conjugate(fpsf) / (1 / self.snr + np.real(fpsf) ** 2 + np.imag(fpsf) ** 2)
        # bp
        invpsf = np.conjugate(fpsf)
        
        self.invpsf_real = torch.from_numpy(np.real(invpsf).astype(np.float32)).unsqueeze(0)
        self.invpsf_imag = torch.from_numpy(np.imag(invpsf).astype(np.float32)).unsqueeze(0)
        
        ######################################################
        mtx_MxM, mtxi_MxM = resamplingOperator(temprol_grid)
        self.mtx_MxM = torch.from_numpy(mtx_MxM.astype(np.float32))
        self.mtxi_MxM = torch.from_numpy(mtxi_MxM.astype(np.float32))
        
    def todev(self, dev, dnum):
        
        self.virtual_cos_sin_wave_inv_2x1xk_todev = self.virtual_cos_sin_wave_inv_2x1xk.to(dev)
        # self.datapad_2Dx2Tx2Hx2W = torch.zeros((2 * dnum, 2 * self.crop, 2 * self.spatial_grid, 2 * self.spatial_grid), dtype=torch.float32, device=dev)
        
        self.mtx_MxM_todev = self.mtx_MxM.to(dev)
        self.mtxi_MxM_todev = self.mtxi_MxM.to(dev)
        
        self.invpsf_real_todev = self.invpsf_real.to(dev)
        self.invpsf_imag_todev = self.invpsf_imag.to(dev)
        
    def forward(self, feture_bxdxtxhxw, tbes, tens):
        
        # 1 padd data with zero
        bnum, dnum, tnum, hnum, wnum = feture_bxdxtxhxw.shape
        for tbe, ten in zip(tbes, tens):
            assert tbe >= 0
            assert ten <= self.crop
        dev = feture_bxdxtxhxw.device
        
        featpad_bxdxtxhxw = []
        for i in range(bnum):
            featpad_1xdxt1xhxw = torch.zeros((1, dnum, tbes[i], hnum, wnum), dtype=torch.float32, device=dev)
            featpad_1xdxt2xhxw = torch.zeros((1, dnum, self.crop - tens[i], hnum, wnum), dtype=torch.float32, device=dev)
            featpad_1xdxtxhxw = torch.cat([featpad_1xdxt1xhxw, feture_bxdxtxhxw[i:i + 1], featpad_1xdxt2xhxw], dim=2)
            featpad_bxdxtxhxw.append(featpad_1xdxtxhxw)
        featpad_bxdxtxhxw = torch.cat(featpad_bxdxtxhxw, dim=0)
        
        # 2 params
        assert hnum == wnum
        assert hnum == self.spatial_grid
        sptial_grid = hnum
        temprol_grid = self.crop
        tnum = self.crop
        
        ####################################################
        # 3 run lct
        # assert bnum == 1
        data_BDxTxHxW = featpad_bxdxtxhxw.view(bnum * dnum, tnum, hnum, wnum)
        
        ############################################################
        # Step 1: convolve measurement volume with virtual wave
        
        data_BDxHxWxT = data_BDxTxHxW.permute(0, 2, 3, 1)
        data_BDHWx1xT = data_BDxHxWxT.reshape(-1, 1, tnum)
        knum = self.virtual_cos_sin_wave_inv_2x1xk.shape[2]
        phasor_data_cos_sin_BDHWx2x1T = F.conv1d(data_BDHWx1xT, self.virtual_cos_sin_wave_inv_2x1xk_todev, padding=knum // 2)
        if knum % 2 == 0:
            data_BDHWx2xT = phasor_data_cos_sin_BDHWx2x1T[:, :, 1:]
        else:
            data_BDHWx2xT = phasor_data_cos_sin_BDHWx2x1T
        
        data_BDxHxWx2xT = data_BDHWx2xT.reshape(bnum * dnum, hnum, wnum, 2, tnum)
        data_2xBDxTxHxW = data_BDxHxWx2xT.permute(3, 0, 4, 1, 2)
        data_2BDxTxHxW = data_2xBDxTxHxW.reshape(2 * bnum * dnum, tnum, hnum, wnum)
        
        #############################################################    
        # Step 2: transform virtual wavefield into LCT domain
        
        left = self.mtx_MxM_todev
        right = data_2BDxTxHxW.view(2 * bnum * dnum, temprol_grid, -1)
        tmp = torch.matmul(left, right)
        tmp2 = tmp.view(2 * bnum * dnum, temprol_grid, sptial_grid, sptial_grid)
        
        
        ###############################################33
        batch = data_2BDxTxHxW.shape[0]
        
        res = []
        for i in range(batch):
            
            dataslice = torch.zeros((1, 2 * temprol_grid, 2 * sptial_grid, 2 * sptial_grid), dtype=torch.float32, device=dev)
            dataslice[:, :temprol_grid, :sptial_grid, :sptial_grid] = tmp2[i:i + 1]
            
            datafre = torch.rfft(dataslice, 3, onesided=False)
            datafre_real = datafre[:, :, :, :, 0]
            datafre_imag = datafre[:, :, :, :, 1]
            
            re_real = datafre_real * self.invpsf_real_todev - datafre_imag * self.invpsf_imag_todev
            re_imag = datafre_real * self.invpsf_imag_todev + datafre_imag * self.invpsf_real_todev
            refre = torch.stack([re_real, re_imag], dim=4)
            re = torch.ifft(refre, 3)
            volumn_1xTxHxWx2 = re[:, :temprol_grid, :sptial_grid, :sptial_grid, :]
            res.append(volumn_1xTxHxWx2)
        
        volumn_2BDxTxHxWx2 = torch.cat(res, 0)
        
        ########################################################################
        # Step 4: compute phasor field magnitude and inverse LCT
        
        cos_real = volumn_2BDxTxHxWx2[:bnum * dnum, :, :, :, 0]
        cos_imag = volumn_2BDxTxHxWx2[:bnum * dnum, :, :, :, 1]
        
        sin_real = volumn_2BDxTxHxWx2[bnum * dnum:, :, :, :, 0]
        sin_imag = volumn_2BDxTxHxWx2[bnum * dnum:, :, :, :, 1]
        
        sum_real = cos_real ** 2 - cos_imag ** 2 + sin_real ** 2 - sin_imag ** 2
        sum_image = 2 * cos_real * cos_imag + 2 * sin_real * sin_imag
        
        tmp = (torch.sqrt(sum_real ** 2 + sum_image ** 2) + sum_real) / 2
        # numerical issue
        tmp = F.relu(tmp, inplace=False)
        sqrt_sum_real = torch.sqrt(tmp)
        
        #####################################################################
        left = self.mtxi_MxM_todev
        right = sqrt_sum_real.view(bnum * dnum, temprol_grid, -1)
        tmp = torch.matmul(left, right)
        tmp2 = tmp.view(bnum * dnum, temprol_grid, sptial_grid, sptial_grid)
        
        ########################################################################
        # do we force to be > 0?
        # volumn_BDxTxHxW = F.relu(tmp2, inplace=False)
        volumn_BDxTxHxW = tmp2
        
        volumn_BxDxTxHxW = volumn_BDxTxHxW.view(bnum, dnum, self.crop, hnum, wnum)
        
        return volumn_BxDxTxHxW


if __name__ == '__main__':
    
    import os
    import cv2
    import numpy as np
    
    syn = False
    if syn:
        fd = '/home/wenzheng/largestore/nlos-phasor/data/car';
        ims = []
        tbe = -1
        tlen = 0
        for i in range(512):
            name = '%s/2-%d.png' % (fd, i)
            if not os.path.isfile(name):
                ims.append(np.zeros((256, 256), dtype=np.uint8))
                continue
            
            if tbe < 0:
                tbe = i
            tlen += 1
            
            im = cv2.imread(name)
            imgt = im[:256, :256, :]
            im = im[:256, -256:, :]
            imgray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
            ims.append(imgray)
        
        rect_data_txhxw = np.array(ims, dtype=np.float32) / 255.0
        rect_data_hxwxt = np.transpose(rect_data_txhxw, [1, 2, 0])
        
        sptial_grid = 256
        crop = 512
        bin_len = 0.01
    else:
        
        from scipy.io import loadmat
        
        data = loadmat('/home/wenzheng/largestore/nlos-phasor/nlos-fk-master/statue.mat')
        rect_data_hxwxt = data['data']
        
        sptial_grid = 512
        crop = 512
        bin_len = 32e-12 * 3e8  # 0.01
        tbe = 0
        tlen = crop
    
    K = 2
    temds = True
    for k in range(K):
        rect_data_hxwxt = rect_data_hxwxt[::2, :, :] + rect_data_hxwxt[1::2, :, :]
        rect_data_hxwxt = rect_data_hxwxt[:, ::2, :] + rect_data_hxwxt[:, 1::2, :]
        sptial_grid = sptial_grid // 2
        
        if temds:
            rect_data_hxwxt = rect_data_hxwxt[:, :, ::2] + rect_data_hxwxt[:, :, 1::2]
            crop = crop // 2
            bin_len = bin_len * 2 
        
    rect_data_dxhxwxt = np.expand_dims(rect_data_hxwxt, axis=0)
    rect_data_bxdxhxwxt = np.expand_dims(rect_data_dxhxwxt, axis=0)
    
    bnum = 1
    dnum = 1
    rect_data_bxdxhxwxt = np.tile(rect_data_bxdxhxwxt, [bnum, dnum, 1, 1, 1])
    rect_data_bxdxhxwxt = torch.from_numpy(rect_data_bxdxhxwxt).cuda()
    
    dev = 'cuda'
    
    #####################################################################
    lctlayer = phasor(spatial=sptial_grid, crop=crop, bin_len=bin_len, sampling_coeff=2.0, cycles=5)
    lctlayer.todev(dev, dnum)
    
    if temds:
        tbe = tbe // (2 ** K)
        tlen = tlen // (2 ** K)
    
    for i in range(10):
        print(i)
        re = lctlayer(rect_data_bxdxhxwxt[:, :, :, :, tbe:tbe + tlen].permute(0, 1, 4, 2, 3), \
                      [tbe, tbe, tbe], [tbe + tlen, tbe + tlen, tbe + tlen])
    
    volumn_MxNxN = re.detach().cpu().numpy()[0, -1]
    zdim = volumn_MxNxN.shape[0] * 100 // 128
    volumn_MxNxN = volumn_MxNxN[:zdim]
    print('volumn min, %f' % volumn_MxNxN.min())
    print('volumn max, %f' % volumn_MxNxN.max())
    
    volumn_MxNxN[volumn_MxNxN < 0] = 0
    front_view = np.max(volumn_MxNxN, axis=0)
    cv2.imshow("re", front_view / np.max(front_view))
    # cv2.imshow("gt", imgt)
    cv2.waitKey()
    
    volumn_ZxYxX = volumn_MxNxN
    volumn_ZxYxX = volumn_ZxYxX / np.max(volumn_ZxYxX)
    for i, frame in enumerate(volumn_ZxYxX):
        print(i)
        cv2.imshow("re1", frame)
        cv2.imshow("re2", frame / np.max(frame))
        cv2.waitKey(0)
    
