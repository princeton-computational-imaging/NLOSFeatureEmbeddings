

import numpy as np
import cv2
from helper import definePsf, resamplingOperator


##########################################################
def lct(meas_hxwxt, wall_size, crop, bin_len):
    
    c = 3e8
    width = wall_size / 2.0;
    bin_resolution = bin_len / c
    assert 2 ** int(np.log2(crop)) == crop
    
    snr = 1e-1
    
    ###########################################
    meas_hxwxt = meas_hxwxt[:, :, :crop]  # HxWxT
    sptial_grid = meas_hxwxt.shape[0]  # H, N
    temprol_grid = meas_hxwxt.shape[2]  # T, M
    trange = temprol_grid * c * bin_resolution
    
    #########################################################
    # 0-1
    gridz_M = np.arange(temprol_grid, dtype=np.float32)
    gridz_M = gridz_M / (temprol_grid - 1)
    gridz_MxNxN = np.tile(gridz_M.reshape(-1, 1, 1), [1, sptial_grid, sptial_grid])
    
    ###################################################
    slope = width / trange
    psf = definePsf(sptial_grid, temprol_grid, slope)
    fpsf = np.fft.fftn(psf)
    invpsf = np.conjugate(fpsf) / (1 / snr + np.real(fpsf) ** 2 + np.imag(fpsf) ** 2)
    # invpsf = np.conjugate(fpsf)
    
    mtx_MxM, mtxi_MxM = resamplingOperator(temprol_grid)
    
    #############################################################
    # diffuse
    data_TxHxW = np.transpose(meas_hxwxt, [2, 0, 1])
    data_TxHxW = data_TxHxW * (gridz_MxNxN ** 4)
    
    datapad_2Tx2Hx2W = np.zeros(shape=(2 * temprol_grid, 2 * sptial_grid, 2 * sptial_grid), dtype=np.float32)
    
    left = mtx_MxM
    right = data_TxHxW.reshape(temprol_grid, -1)
    tmp = np.matmul(left, right).reshape(temprol_grid, sptial_grid, sptial_grid)
    datapad_2Tx2Hx2W[:temprol_grid, :sptial_grid, :sptial_grid] = tmp
    
    datafre = np.fft.fftn(datapad_2Tx2Hx2W)
    volumn_2Mx2Nx2N = np.fft.ifftn(datafre * invpsf)
    volumn_2Mx2Nx2N = np.real(volumn_2Mx2Nx2N)
    volumn_ZxYxX = volumn_2Mx2Nx2N[:temprol_grid, :sptial_grid, :sptial_grid]
    
    left = mtxi_MxM
    right = volumn_ZxYxX.reshape(temprol_grid, -1)
    tmp = np.matmul(left, right).reshape(temprol_grid, sptial_grid, sptial_grid)
    volumn_ZxYxX = tmp
    
    ################################
    volumn_ZxYxX[volumn_ZxYxX < 0] = 0
    
    dim = volumn_ZxYxX.shape[0] * 100 // 128
    volumn_ZxYxX = volumn_ZxYxX[:dim]
    volumn_ZxYxX = volumn_ZxYxX / np.max(volumn_ZxYxX)
    
    front_view_HxW = np.max(volumn_ZxYxX, axis=0)
    cv2.imshow("re3", front_view_HxW / np.max(front_view_HxW))
    # cv2.imshow('gt', imgt)
    cv2.waitKey()
    
    for frame in volumn_ZxYxX:
        cv2.imshow("re1", frame)
        cv2.imshow("re2", frame / np.max(frame))
        cv2.waitKey(0)
    
    
########################################################
if __name__ == '__main__':
    
    import os
    
    '''
    fd = '/u6/a/wenzheng/remote2/code-nlos-git/OccludedSceneRep-2/code/pytorch-wz/dataloader_lct';
    ims = []
    tbe = -1
    for i in range(512):
        name = '%s/1-%d.png' % (fd, i)
        if not os.path.isfile(name):
            ims.append(np.zeros((256, 256), dtype=np.uint8))
            continue
        
        if tbe < 0:
            tbe = i
        
        im = cv2.imread(name)
        imgt = im[:, :256, :]
        im = im[:, -256:, :]
        imgray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
        ims.append(imgray)
    
    rect_data_txhxw = np.array(ims, dtype=np.float32) / 255.0
    rect_data_hxwxt = np.transpose(rect_data_txhxw, [1, 2, 0])
    '''
    
    from scipy.io import loadmat
    
    data = loadmat('/home/wenzheng/largestore/nlos-phasor/realdata/bike0.mat')
    rect_data_hxwxt = data['measlr']
    
    crop = 512
    bin_len = 32e-12 * 3e8  # 0.01
    
    K = 0
    for k in range(K):
        rect_data_hxwxt = rect_data_hxwxt[::2, :, :] + rect_data_hxwxt[1::2, :, :]
        rect_data_hxwxt = rect_data_hxwxt[:, ::2, :] + rect_data_hxwxt[:, 1::2, :]
        
        rect_data_hxwxt = rect_data_hxwxt[:, :, ::2] + rect_data_hxwxt[:, :, 1::2]
        crop = crop // 2
        bin_len = bin_len * 2
        
    lct(rect_data_hxwxt, wall_size=2.0, crop=crop, bin_len=bin_len)

