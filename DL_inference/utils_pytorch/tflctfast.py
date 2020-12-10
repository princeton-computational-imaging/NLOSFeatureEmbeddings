

import torch
import torch.nn as nn
import torch.nn.functional as F

import  numpy as np
import sys
sys.path.append('../utils')
from helper import definePsf, resamplingOperator, \
filterLaplacian


class lct(nn.Module):
    
    def __init__(self, spatial=256, crop=512, \
                 bin_len=0.01, wall_size=2.0, \
                 method='lct', material='diffuse'):
        super(lct, self).__init__()
        
        self.spatial_grid = spatial
        self.crop = crop
        assert 2 ** int(np.log2(crop)) == crop
        
        self.bin_len = bin_len
        self.wall_size = wall_size
        
        #############################################################
        self.method = method
        self.material = material
        
        self.parpareparam()
    
    #####################################################
    def parpareparam(self):
        
        self.c = 3e8
        self.width = self.wall_size / 2.0;
        self.bin_resolution = self.bin_len / self.c
        self.trange = self.crop * self.c * self.bin_resolution
        
        # maybe learnable?
        self.snr = 1e-1
        
        ########################################################3
        temprol_grid = self.crop
        sptial_grid = self.spatial_grid
        
        # 0-1
        gridz_M = np.arange(temprol_grid, dtype=np.float32)
        gridz_M = gridz_M / (temprol_grid - 1)
        gridz_1xMx1x1 = gridz_M.reshape(1, -1, 1, 1)
        self.gridz_1xMx1x1 = torch.from_numpy(gridz_1xMx1x1.astype(np.float32))
        
        ###################################################
        slope = self.width / self.trange
        psf = definePsf(sptial_grid, temprol_grid, slope)
        fpsf = np.fft.fftn(psf)
        
        if self.method == 'lct':
            invpsf = np.conjugate(fpsf) / (1 / self.snr + np.real(fpsf) ** 2 + np.imag(fpsf) ** 2)
        elif self.method == 'bp':
            invpsf = np.conjugate(fpsf)
        
        self.invpsf_real = torch.from_numpy(np.real(invpsf).astype(np.float32)).unsqueeze(0)
        self.invpsf_imag = torch.from_numpy(np.imag(invpsf).astype(np.float32)).unsqueeze(0)
        
        ######################################################
        mtx_MxM, mtxi_MxM = resamplingOperator(temprol_grid)
        self.mtx_MxM = torch.from_numpy(mtx_MxM.astype(np.float32))
        self.mtxi_MxM = torch.from_numpy(mtxi_MxM.astype(np.float32))
        
        #############################################################
        if self.method == 'bp':
            lapw_kxkxk = filterLaplacian()
            k = lapw_kxkxk.shape[0]
            self.pad = nn.ReplicationPad3d(2)
            self.lapw = torch.from_numpy(lapw_kxkxk).reshape(1, 1, k, k, k)
        
    def todev(self, dev, dnum):
        self.gridz_1xMx1x1_todev = self.gridz_1xMx1x1.to(dev)
        self.datapad_Dx2Tx2Hx2W = torch.zeros((dnum, 2 * self.crop, 2 * self.spatial_grid, 2 * self.spatial_grid), dtype=torch.float32, device=dev)
        self.datazero_Dx2Tx2Hx2W = torch.zeros((dnum, 2 * self.crop, 2 * self.spatial_grid, 2 * self.spatial_grid), dtype=torch.float32, device=dev)
        
        self.mtx_MxM_todev = self.mtx_MxM.to(dev)
        self.mtxi_MxM_todev = self.mtxi_MxM.to(dev)
        
        self.invpsf_real_todev = self.invpsf_real.to(dev)
        self.invpsf_imag_todev = self.invpsf_imag.to(dev)
        
        if self.method == 'bp':
            self.lapw_todev = self.lapw.to(dev)
    
    def forward(self, feture_bxdxtxhxw):
        
        # 1 padd data with zero
        bnum, dnum, tnum, hnum, wnum = feture_bxdxtxhxw.shape
        dev = feture_bxdxtxhxw.device
        
        '''
        for tbe, ten in zip(tbes, tens):
            assert tbe >= 0
            assert ten <= self.crop
        featpad_bxdxtxhxw = []
        for i in range(bnum):
            featpad_1xdxt1xhxw = torch.zeros((1, dnum, tbes[i], hnum, wnum), dtype=torch.float32, device=dev)
            featpad_1xdxt2xhxw = torch.zeros((1, dnum, self.crop - tens[i], hnum, wnum), dtype=torch.float32, device=dev)
            featpad_1xdxtxhxw = torch.cat([featpad_1xdxt1xhxw, feture_bxdxtxhxw[i:i + 1], featpad_1xdxt2xhxw], dim=2)
            featpad_bxdxtxhxw.append(featpad_1xdxtxhxw)
        featpad_bxdxtxhxw = torch.cat(featpad_bxdxtxhxw, dim=0)
        '''
        # full dim
        assert tnum == self.crop
        featpad_bxdxtxhxw = feture_bxdxtxhxw
        
        # 2 params
        assert hnum == wnum
        assert hnum == self.spatial_grid
        sptial_grid = hnum
        temprol_grid = self.crop
        
        ####################################################
        # 3 run lct
        # assert bnum == 1
        data_BDxTxHxW = featpad_bxdxtxhxw.view(bnum * dnum, self.crop, hnum, wnum)
        
        gridz_1xMx1x1 = self.gridz_1xMx1x1_todev
        if self.material == 'diffuse':
            data_BDxTxHxW = data_BDxTxHxW * (gridz_1xMx1x1 ** 4)
        elif self.material == 'specular':
            data_BDxTxHxW = data_BDxTxHxW * (gridz_1xMx1x1 ** 2)
        
        ###############################################################
        # datapad_BDx2Tx2Hx2W = torch.zeros((bnum * dnum, 2 * temprol_grid, 2 * sptial_grid, 2 * sptial_grid), dtype=torch.float32, device=dev)
        datapad_Dx2Tx2Hx2W = self.datapad_Dx2Tx2Hx2W
        
        '''
        # create new variable
        datapad_BDx2Tx2Hx2W = datapad_Dx2Tx2Hx2W.repeat(bnum, 1, 1, 1)
        '''
        assert bnum == 1
        datapad_BDx2Tx2Hx2W = datapad_Dx2Tx2Hx2W
        
        left = self.mtx_MxM_todev
        right = data_BDxTxHxW.view(bnum * dnum, temprol_grid, -1)
        tmp = torch.matmul(left, right)
        tmp2 = tmp.view(bnum * dnum, temprol_grid, sptial_grid, sptial_grid)
        
        datapad_BDx2Tx2Hx2W[:, :temprol_grid, :sptial_grid, :sptial_grid] = tmp2 # .detach()
        
        ####################################################################################
        # datapad_BDx2Tx2Hx2Wx2 = torch.stack([datapad_BDx2Tx2Hx2W, torch.zeros_like(datapad_BDx2Tx2Hx2W)], dim=4)
        # datafre = torch.rfft(datapad_BDx2Tx2Hx2W, 3, onesided=False)
        datazero_Dx2Tx2Hx2W = self.datazero_Dx2Tx2Hx2W
        datazero_BDx2Tx2Hx2W = datazero_Dx2Tx2Hx2W
        datapad_BDx2Tx2Hx2Wx2 = torch.stack([datapad_BDx2Tx2Hx2W, datazero_BDx2Tx2Hx2W], dim=4)
        datafre = torch.fft(datapad_BDx2Tx2Hx2Wx2, 3)
        
        datafre_real = datafre[:, :, :, :, 0]
        datafre_imag = datafre[:, :, :, :, 1]
        
        re_real = datafre_real * self.invpsf_real_todev - datafre_imag * self.invpsf_imag_todev
        re_imag = datafre_real * self.invpsf_imag_todev + datafre_imag * self.invpsf_real_todev
        refre = torch.stack([re_real, re_imag], dim=4)
        re = torch.ifft(refre, 3)
        
        volumn_BDxTxHxW = re[:, :temprol_grid, :sptial_grid, :sptial_grid, 0]
        
        #########################################################################
        left = self.mtxi_MxM_todev
        right = volumn_BDxTxHxW.reshape(bnum * dnum, temprol_grid, -1)
        tmp = torch.matmul(left, right)
        tmp2 = tmp.view(bnum * dnum, temprol_grid, sptial_grid, sptial_grid)
        
        # volumn_BDxTxHxW = F.relu(tmp2, inplace=False)
        volumn_BDxTxHxW = tmp2
        
        volumn_BxDxTxHxW = volumn_BDxTxHxW.view(bnum, dnum, self.crop, hnum, wnum)
        
        return volumn_BxDxTxHxW


if __name__ == '__main__':
    
    import os
    import cv2
    import numpy as np
    
    '''
    fd = '/u6/a/wenzheng/remote2/code-nlos-git/OccludedSceneRep-2/code/pytorch-wz/dataloader_light22_bbox';
    ims = []
    tbe = -1
    for i in range(512):
        name = '%s/2-%d.png' % (fd, i)
        if not os.path.isfile(name):
            ims.append(np.zeros((256, 256), dtype=np.uint8))
            continue
        
        if tbe < 0:
            tbe = i
        
        im = cv2.imread(name)
        imgt = im[:256, :256, :]
        im = im[:256, -256:, :]
        imgray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
        ims.append(imgray)
    
    rect_data_txhxw = np.array(ims, dtype=np.float32) / 255.0
    rect_data_hxwxt = np.transpose(rect_data_txhxw, [1, 2, 0])
    '''
    
    from scipy.io import loadmat
    
    data = loadmat('/home/wenzheng/largestore/nlos-phasor/nlos-fk-master/statue.mat')
    rect_data_hxwxt = data['data']
    # rect_data_hxwxt = np.zeros((512, 512, 512), dtype=np.float32)
    
    sptial_grid = 512
    crop = 512
    bin_len = 32e-12 * 3e8  # 0.01
    
    K = 2
    temds = False
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
    lctlayer = lct(spatial=sptial_grid, crop=crop, bin_len=bin_len,
                   method='lct')
    lctlayer = lctlayer.to(dev).eval()
    lctlayer.todev(dev, dnum)
    
    tbe = 0 // (2 ** K)
    if temds:
        tlen = 512 // (2 ** K)
    else:
        tlen = 512
    
    import time
    for i in range(10):
        t1 = time.time()
        re = lctlayer(rect_data_bxdxhxwxt[:, :, :, :, tbe:tbe + tlen].permute(0, 1, 4, 2, 3))  # , \
                      # [tbe, tbe, tbe], [tbe + tlen, tbe + tlen, tbe + tlen])
        print("cost time: ", time.time() - t1)
        
    def export_onnx_model(model, input_shape, onnx_path, input_names=None, output_names=None, dynamic_axes=None):
        inputs = torch.ones(*input_shape).to(dev)
        model(inputs)
        torch.onnx.export(model, inputs, onnx_path, input_names=input_names, output_names=output_names)
    
    input = rect_data_bxdxhxwxt.permute(0, 1, 4, 2, 3)
    inputnp = input.detach().cpu().numpy()
    np.save(file='ni.npy', arr=inputnp)
    shape = input.shape
    print(shape)
    export_onnx_model(lctlayer, shape, 'test.onnx')
    
    
    
    volumn_MxNxN = re.detach().cpu().numpy()[0, -1]
    
    # get rid of bad points
    zdim = volumn_MxNxN.shape[0] * 100 // 128
    volumn_MxNxN = volumn_MxNxN[:zdim]
    print('volumn min, %f' % volumn_MxNxN.min())
    print('volumn max, %f' % volumn_MxNxN.max())
    # volumn_MxNxN[:5] = 0
    # volumn_MxNxN[-5:] = 0
    
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
    
