

import  numpy as np
import cv2
import scipy.sparse as ssp

 
debug = False
# still not very sure about coordinate!


#########################################################
def filterLaplacian():
    
    hszie = 5;
    std1 = 1.0
    
    lim = (hszie - 1) // 2
    std2 = std1 ** 2
    
    dims = np.arange(-lim, lim + 1, dtype=np.float32)
    [y, x, z] = np.meshgrid(dims, dims, dims)
    w = np.exp(-(x ** 2 + y ** 2 + z ** 2) / (2 * std2))
    w = w / np.sum(w)
    
    w1 = w * (x ** 2 + y ** 2 + z ** 2 - 3 * std2)
    w1 = w1 / (std2 ** 2)
    
    # w = w1 - np.sum(w1) / (hszie ** 3)
    w = w1 - np.mean(w1)
    
    return w


def resamplingOperator(temprol_grid):
    
    M = temprol_grid
    row = M ** 2 
    col = M
    assert 2 ** int(np.log2(M)) == M
    
    # 1 to M^2
    x = np.arange(row, dtype=np.float32)
    x = x + 1
    
    rowidx = np.arange(row)
    # 0 to M-1
    colidx = np.ceil(np.sqrt(x)) - 1
    data = np.ones_like(rowidx, dtype=np.float32)
    mtx1 = ssp.csr_matrix((data, (rowidx, colidx)), shape=(row, col), dtype=np.float32)
    mtx2 = ssp.spdiags(data=[1.0 / np.sqrt(x)], diags=[0], m=row, n=row)
    
    mtx = mtx2.dot(mtx1)
    K = int(np.log2(M))
    for _ in np.arange(K):
        mtx = 0.5 * (mtx[0::2, :] + mtx[1::2])
        # mtxi = 0.5 * (mtxi[:, 0::2] + mtxi[:, 1::2])
    
    mtxi = np.transpose(mtx)
    
    if debug:
        print(mtx.toarray())
    
    if debug:
        print(mtxi.toarray())
    
    return mtx.toarray(), mtxi.toarray()
    
    
def definePsf(sptial_grid, temprol_grid, slope):
    
    # slop is time_range / wall_size
    N = sptial_grid
    M = temprol_grid
    
    # -1 to 1
    x_2N = np.arange(2 * sptial_grid, dtype=np.float32)
    x_2N = x_2N / (2 * sptial_grid - 1) * 2 - 1
    
    # here, x and y are symetric
    # it doesn't mater y is postive or negative
    y_2N = x_2N
    
    # 0 to 2
    z_2M = np.arange(2 * temprol_grid, dtype=np.float32)
    z_2M = z_2M / (2 * temprol_grid - 1) * 2
    
    # grid axis, also in hxwxt
    # that's why x is the second axis
    # y is the first axis
    [gridy_2Nx2Nx2M, gridx_2Nx2Nx2M, gridz_2Nx2Nx2M] = np.meshgrid(x_2N, y_2N, z_2M)
    
    # dst
    a_2Nx2NX2M = (4 * slope) ** 2 * (gridx_2Nx2Nx2M ** 2 + gridy_2Nx2Nx2M ** 2) - gridz_2Nx2Nx2M
    b_2Nx2NX2M = np.abs(a_2Nx2NX2M)
    
    # min data
    c_2Nx2NX2M = np.min(b_2Nx2NX2M, axis=2, keepdims=True)
    
    # should be a ellipse
    d_2Nx2NX2M = np.abs(b_2Nx2NX2M - c_2Nx2NX2M) < 1e-8
    d_2Nx2NX2M = d_2Nx2NX2M.astype(np.float32)
    if debug:
        cv2.imshow("0", d_2Nx2NX2M[:, :, 0])
        cv2.imshow("1", d_2Nx2NX2M[:, :, M // 2 - 1])
        cv2.imshow("2", d_2Nx2NX2M[:, :, M - 1])
        cv2.waitKey()
    
    # norm
    e_2Nx2NX2M = d_2Nx2NX2M / np.sqrt(np.sum(d_2Nx2NX2M))
    
    # shift
    f1_2Nx2NX2M = np.roll(e_2Nx2NX2M, shift=N, axis=0)
    f2_2Nx2NX2M = np.roll(f1_2Nx2NX2M, shift=N, axis=1)
    if debug:
        cv2.imshow("0", f2_2Nx2NX2M[:, :, 0] * 256)
        cv2.imshow("1", f2_2Nx2NX2M[:, :, M // 2 - 1] * 256)
        cv2.imshow("2", f2_2Nx2NX2M[:, :, M - 1] * 256)
        cv2.waitKey()
    
    psf_2Mx2Nx2N = np.transpose(f2_2Nx2NX2M, [2, 0, 1])
    
    return psf_2Mx2Nx2N


#########################################################################
def roll_1(x_bxtxhxwx2, dim, n):
    if dim == 1:
        a = np.concatenate((x_bxtxhxwx2[-n:], x_bxtxhxwx2[:-n]), axis=dim - 1)
    if dim == 2:
        a = np.concatenate((x_bxtxhxwx2[:, -n:], x_bxtxhxwx2[:, :-n]), axis=dim - 1)
    if dim == 3:
        a = np.concatenate((x_bxtxhxwx2[:, :, -n:], x_bxtxhxwx2[:, :, :-n]), axis=dim - 1)
    return a


##################################################################################
def gaussianwin(L, alpha):
    
    N = L - 1;
    Nhalf = N / 2.0
    n_k = np.arange(N + 1, dtype=np.float32) - Nhalf;
    w_k = np.exp(-0.5 * (alpha * n_k / Nhalf) ** 2)
    
    return w_k


def waveconvparam(bin_resolution, virtual_wavelength, cycles):
    
    c = 3e8;
    s_z = bin_resolution * c;
    samples = int(round(cycles * virtual_wavelength / (bin_resolution * c)));
    num_cycles = samples * s_z / virtual_wavelength;
    sigma = 0.3;
    
    # generate sin/cos signals
    grids_k = np.arange(samples, dtype=np.float32) + 1
    sin_wave_k = np.sin(2 * np.pi * (num_cycles * grids_k) / samples);
    cos_wave_k = np.cos(2 * np.pi * (num_cycles * grids_k) / samples);
    
    # window = single(gausswin(samples, 1/sigma));  
    window = gaussianwin(samples, 1.0 / sigma)
    virtual_sin_wave_k = sin_wave_k * window
    virtual_cos_wave_k = cos_wave_k * window
    
    return virtual_cos_wave_k, virtual_sin_wave_k


def waveconv(bin_resolution, virtual_wavelength, cycles, data_txhxw):
    
    c = 3e8;
    s_z = bin_resolution * c;
    samples = int(round(cycles * virtual_wavelength / (bin_resolution * c)));
    num_cycles = samples * s_z / virtual_wavelength;
    sigma = 0.3;
    
    # generate sin/cos signals
    grids_k = np.arange(samples, dtype=np.float32) + 1
    sin_wave_k = np.sin(2 * np.pi * (num_cycles * grids_k) / samples);
    cos_wave_k = np.cos(2 * np.pi * (num_cycles * grids_k) / samples);
    
    # window = single(gausswin(samples, 1/sigma));  
    window = gaussianwin(samples, 1.0 / sigma)
    virtual_sin_wave_k = sin_wave_k * window
    virtual_cos_wave_k = cos_wave_k * window
    
    wave_sin = np.zeros_like(data_txhxw)
    wave_cos = np.zeros_like(data_txhxw)
    
    # conv
    M, N, _ = data_txhxw.shape
    for i in range(N):
        for j in range(N):
            data_t = data_txhxw[:, i, j]
            real = np.convolve(data_t, v=virtual_sin_wave_k, mode='same')
            image = np.convolve(data_t, v=virtual_cos_wave_k, mode='same')
            wave_sin[:, i, j] = real
            wave_cos[:, i, j] = image
    
    return wave_cos, wave_sin

