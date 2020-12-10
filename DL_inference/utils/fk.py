

import numpy as np
import cv2
import scipy.interpolate as si
from helper import roll_1


def lct(meas_hxwxt, wall_size, crop, bin_len):
    
    c = 3e8
    width = wall_size / 2.0;
    bin_resolution = bin_len / c
    assert 2 ** int(np.log2(crop)) == crop
    
    ###########################################
    meas_hxwxt = meas_hxwxt[:, :, :crop]  # HxWxT
    sptial_grid = meas_hxwxt.shape[0]  # H, N
    temprol_grid = meas_hxwxt.shape[2]  # T, M
    trange = temprol_grid * c * bin_resolution
    
    ############################################################
    # 0-1
    gridz_M = np.arange(temprol_grid, dtype=np.float32)
    gridz_M = gridz_M / (temprol_grid - 1)
    gridz_1xMx1x1 = gridz_M.reshape(1, -1, 1, 1)
    
    ###############################################################
    # axis
    zdim = np.arange(2 * temprol_grid, dtype=np.float32)
    xdim = np.arange(2 * sptial_grid, dtype=np.float32)
    
    zdim = (zdim - temprol_grid) / temprol_grid
    xdim = (xdim - sptial_grid) / sptial_grid
    ydim = xdim
    
    [gridy_2Nx2Nx2M, gridx_2Nx2Nx2M, gridz_2Nx2Nx2M] = np.meshgrid(xdim, ydim, zdim)
    gridz_2Mx2Nx2N = np.transpose(gridz_2Nx2Nx2M, [2, 1, 0])
    gridy_2Mx2Nx2N = np.transpose(gridy_2Nx2Nx2M, [2, 1, 0])
    gridx_2Mx2Nx2N = np.transpose(gridx_2Nx2Nx2M, [2, 1, 0])
    
    ##########################################################3
    M = temprol_grid
    N = sptial_grid
    
    fkrange = ((N * trange) / (M * width * 4)) ** 2
    gridznew = fkrange * (gridx_2Mx2Nx2N ** 2 + gridy_2Mx2Nx2N ** 2) + gridz_2Mx2Nx2N ** 2
    gridznew = np.sqrt(gridznew)
    
    ###################################################
    # padd data
    data_TxHxW = np.transpose(meas_hxwxt, [2, 0, 1])
    data_TxHxW = data_TxHxW * (gridz_1xMx1x1[0] ** 2)
    data_TxHxW = np.sqrt(data_TxHxW)
    
    datapad_2Tx2Hx2W = np.zeros(shape=(2 * temprol_grid, 2 * sptial_grid, 2 * sptial_grid), dtype=np.float32)
    datapad_2Tx2Hx2W[:temprol_grid, :sptial_grid, :sptial_grid] = data_TxHxW
    
    # fft
    datafre = np.fft.fftn(datapad_2Tx2Hx2W)
    # shift
    '''
    datafre = np.roll(datafre, shift=temprol_grid, axis=0)
    datafre = np.roll(datafre, shift=sptial_grid, axis=1)
    datafre = np.roll(datafre, shift=sptial_grid, axis=2)
    '''
    datafre = roll_1(datafre, dim=1, n=temprol_grid)
    datafre = roll_1(datafre, dim=2, n=sptial_grid)
    datafre = roll_1(datafre, dim=3, n=sptial_grid)
    # stolt trick
    
    # interpolate
    tvol = si.interpn(points=(zdim, ydim, xdim), values=datafre, \
                      xi=np.stack([gridznew, gridy_2Mx2Nx2N, gridx_2Mx2Nx2N], axis=3), \
                      method='linear', bounds_error=False, fill_value=0)
    dim = np.where(zdim > 0)[0][0]
    print('zzeropos %d' % dim)
    tvol[:dim, :, :] = 0
    
    gridznew = np.maximum(gridznew, 1e-8)
    tvol = tvol * np.abs(gridz_2Mx2Nx2N) / gridznew
    
    #########################################################
    # 0-1
    datafre = tvol
    '''
    datafre = np.roll(datafre, shift=temprol_grid, axis=0)
    datafre = np.roll(datafre, shift=sptial_grid, axis=1)
    datafre = np.roll(datafre, shift=sptial_grid, axis=2)
    '''
    datafre = roll_1(datafre, dim=1, n=temprol_grid)
    datafre = roll_1(datafre, dim=2, n=sptial_grid)
    datafre = roll_1(datafre, dim=3, n=sptial_grid)
    
    volumn_2Mx2Nx2N = np.fft.ifftn(datafre)
    volumn_ZxYxX = volumn_2Mx2Nx2N[:temprol_grid, :sptial_grid, :sptial_grid]
    volumn_ZxYxX = np.real(volumn_ZxYxX) ** 2 + np.imag(volumn_ZxYxX) ** 2
    
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
    
    data = loadmat('/home/wenzheng/largestore/nlos-phasor/nlos-fk-master/statue.mat')
    rect_data_hxwxt = data['data']
    
    crop = 512
    bin_len = 32e-12 * 3e8  # 0.01
    
    K = 3
    for k in range(K):
        rect_data_hxwxt = rect_data_hxwxt[::2, :, :] + rect_data_hxwxt[1::2, :, :]
        rect_data_hxwxt = rect_data_hxwxt[:, ::2, :] + rect_data_hxwxt[:, 1::2, :]
        
        '''
        rect_data_hxwxt = rect_data_hxwxt[:, :, ::2] + rect_data_hxwxt[:, :, 1::2]
        crop = crop // 2
        bin_len = bin_len * 2
        '''
        
    lct(rect_data_hxwxt, wall_size=2.0, crop=crop, bin_len=bin_len)

