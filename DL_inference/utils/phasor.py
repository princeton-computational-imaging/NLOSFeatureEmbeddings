

import numpy as np
import cv2
from helper import definePsf, resamplingOperator, waveconv


def phasor(meas_hxwxt, wall_size, crop, bin_len):
    
    c = 3e8
    width = wall_size / 2.0;
    bin_resolution = bin_len / c
    assert 2 ** int(np.log2(crop)) == crop
    
    ###########################################
    meas_hxwxt = meas_hxwxt[:, :, :crop]  # HxWxT
    sptial_grid = meas_hxwxt.shape[0]  # H, N
    temprol_grid = meas_hxwxt.shape[2]  # T, M
    trange = temprol_grid * c * bin_resolution
    
    ###################################################
    slope = width / trange
    psf = definePsf(sptial_grid, temprol_grid, slope)
    fpsf = np.fft.fftn(psf)
    # invpsf = np.conjugate(fpsf) / (1 / snr + np.real(fpsf) ** 2 + np.imag(fpsf) ** 2)
    invpsf = np.conjugate(fpsf)
    
    mtx_MxM, mtxi_MxM = resamplingOperator(temprol_grid)
    
    #############################################################
    # Step 0: define virtual wavelet properties
    s_lamda_limit = wall_size / (sptial_grid - 1);  # sample spacing on the wall
    sampling_coeff = 2;  # scale the size of the virtual wavelength (usually 2, optionally 3 for noisy scenes)
    virtual_wavelength = sampling_coeff * (s_lamda_limit * 2);  # virtual wavelength in units of cm
    cycles = 5;  # number of wave cycles in the wavelet, typically 4-6
    
    ###########################################################
    data_TxHxW = np.transpose(meas_hxwxt, [2, 0, 1])
    
    ############################################################
    # Step 1: convolve measurement volume with virtual wave
    phasor_data_cos, phasor_data_sin = waveconv(bin_resolution, virtual_wavelength, cycles, data_TxHxW);
    # phasor_data_cos = single(phasor_data_cos);
    # phasor_data_sin = single(phasor_data_sin);
    
    #############################################################    
    # Step 2: transform virtual wavefield into LCT domain
    M = temprol_grid
    N = sptial_grid
    phasor_tdata_cos = np.zeros((2 * M, 2 * N, 2 * N), dtype=np.float32);
    phasor_tdata_sin = np.zeros((2 * M, 2 * N, 2 * N), dtype=np.float32);
    
    left = mtx_MxM
    right = phasor_data_cos.reshape(temprol_grid, -1)
    tmp = np.matmul(left, right).reshape(temprol_grid, sptial_grid, sptial_grid)
    phasor_tdata_cos[:temprol_grid, :sptial_grid, :sptial_grid] = tmp
    
    right = phasor_data_sin.reshape(temprol_grid, -1)
    tmp = np.matmul(left, right).reshape(temprol_grid, sptial_grid, sptial_grid)
    phasor_tdata_sin[:temprol_grid, :sptial_grid, :sptial_grid] = tmp
    
    ###################################################################
    # Step 3: convolve with backprojection kernel
    '''
    tvol_phasorbp_sin = ifftn(fftn(phasor_tdata_sin).*bp_psf);
    tvol_phasorbp_sin = tvol_phasorbp_sin(1:end./2,1:end./2,1:end./2);
    phasor_tdata_cos = ifftn(fftn(phasor_tdata_cos).*bp_psf);       
    phasor_tdata_cos = phasor_tdata_cos(1:end./2,1:end./2,1:end./2);
    '''
    
    datafre = np.fft.fftn(phasor_tdata_sin)
    tvol_phasorbp_sin = np.fft.ifftn(datafre * invpsf)
    tvol_phasorbp_sin = tvol_phasorbp_sin[:M, :N, :N]
    
    datafre = np.fft.fftn(phasor_tdata_cos)
    tvol_phasorbp_cos = np.fft.ifftn(datafre * invpsf)
    tvol_phasorbp_cos = tvol_phasorbp_cos[:M, :N, :N]
    
    ###############################################################
    # Step 4: compute phasor field magnitude and inverse LCT
    '''
    tvol = sqrt(tvol_phasorbp_sin.^2 + phasor_tdata_cos.^2);
    vol  = reshape(mtxi*tvol(:,:),[M N N]);
    vol  = max(real(vol),0);
    '''
    
    tvol = np.sqrt(tvol_phasorbp_cos ** 2 + tvol_phasorbp_sin ** 2)
    
    left = mtxi_MxM
    right = tvol.reshape(temprol_grid, -1)
    tmp = np.matmul(left, right).reshape(temprol_grid, sptial_grid, sptial_grid)
    volumn_ZxYxX = np.real(tmp)
    volumn_ZxYxX[volumn_ZxYxX < 0] = 0
    
    # volumn_ZxYxX[-10:] = 0 
    
    #######################################################33
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
    
    data = loadmat('/home/wenzheng/largestore/nlos-phasor/realdata/resolution0.mat')
    rect_data_hxwxt = data['measlr']
    
    crop = 512
    bin_len = 32e-12 * 3e8  # 0.01
    
    K = 1
    for k in range(K):
        '''
        rect_data_hxwxt = rect_data_hxwxt[::2, :, :] + rect_data_hxwxt[1::2, :, :]
        rect_data_hxwxt = rect_data_hxwxt[:, ::2, :] + rect_data_hxwxt[:, 1::2, :]
        '''
        rect_data_hxwxt = rect_data_hxwxt[:, :, ::2] + rect_data_hxwxt[:, :, 1::2]
        crop = crop // 2
        bin_len = bin_len * 2
        
    phasor(rect_data_hxwxt, wall_size=2.0, crop=crop, bin_len=bin_len)

