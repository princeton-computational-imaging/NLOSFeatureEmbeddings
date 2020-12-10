

import torch
import torch.nn as nn
import torch.nn.functional as F

import  numpy as np
from utils import init_mats 

from tffk import lct_fk
from tflct import lct
from tfphasor import phasor
from tfphasor2 import phasor2
from cv2.cv2 import imshow
from tffkfast import lct_fk_fast


################################################################
class diffmodule(nn.Module):
    
    def __init__(self, spatial=256, crop=512, \
                 bin_len=0.01, wall_size=2.0, \
                 mview=False, mode='fk', \
                 align_corners=False, \
                 material='diffuse', \
                 sampling_coeff=2.0, cycles=5):
        super(diffmodule, self).__init__()
        
        self.mode = mode
        if mode == 'fk':
            self.model = lct_fk(spatial, crop, bin_len, wall_size, align_corners)
        
        elif mode == 'fkfast':
            self.model = lct_fk_fast(spatial, crop, bin_len, wall_size, align_corners)
        
        elif mode == 'lct' or mode == 'bp':
            self.model = lct(spatial, crop, bin_len, wall_size, \
                             method=mode, material=material)
        
        elif mode == 'phasor':
            self.model = phasor(spatial, crop, bin_len, wall_size, \
                                sampling_coeff, cycles)
        
        elif mode == 'phasor2':
            self.model = phasor2(spatial, crop, bin_len, wall_size, \
                                sampling_coeff, cycles)
        
        else:
            print('error!')
            
        self.align_corners = align_corners

        ################################################
        # multiple view
        self.mview = mview
        if mview:
            
            # resample
            self.grids = self.initgrid(1, sdim=spatial, tdim=crop)
            
            # crop z
            trange = self.model.trange
            gridslonger = self.grids[0]
            gridzlonger = (gridslonger[:, :, :, :, 2:] + 1) / 2
            gridzshorter = gridzlonger / ((trange / 2) / wall_size)
            gridzshorter = gridzshorter * 2 - 1
            self.gridsshorter = torch.cat([gridslonger[:, :, :, :, :2], gridzshorter], dim=4)
    
    def change_bin_len(self, bin_len):
        assert self.mode == 'fk' or  self.mode == 'fkfast'
        self.model.change_bin_len(bin_len)
        
    def todev(self, dev, dnum):
        self.model.todev(dev, dnum)
        
        if self.mview:
            self.grids_todev = [d.to(dev) for d in self.grids]
            self.gridsshorter_todev = self.gridsshorter.to(dev)
    
    def initgrid(self, channel, tdim, sdim):
        
        self.mats = init_mats(False)
        
        tgt_size = torch.Size((1, channel, tdim, sdim, sdim))
        p_in_bx3x4 = self.mats[0:1]
        grid = F.affine_grid(p_in_bx3x4, tgt_size, align_corners=self.align_corners)
        
        def rev(grid):
            grid[:, :, :, :, 1] = -grid[:, :, :, :, 1]
            grid[:, :, :, :, 2] = -grid[:, :, :, :, 2]
            return grid
        
        grid = rev(grid)
        grids = []
        for i in range(26):
            j = i
            mat = self.mats[j][:3, :3]
            gridnew = torch.matmul(grid, mat)
            gridnew = rev(gridnew)
            grids.append(gridnew)
        
        return grids
    
    def forward(self, feture_bxdxtxhxw, tbes, tens, views=None):
        
        volumn_BxDxTxHxW = self.model(feture_bxdxtxhxw, tbes, tens)
        
        if views is None:
            return volumn_BxDxTxHxW
        else:
            assert self.mview
            
            volumnz_BxDxTxHxW = F.grid_sample(volumn_BxDxTxHxW, self.gridsshorter_todev.repeat(bnum, 1, 1, 1, 1), \
                                              mode='bilinear', padding_mode='zeros', \
                                              align_corners=self.align_corners)
            
            if False:
                import glob
                views = range(26)
                fd = "/home/wenzheng/largestore/nlos-phasor/data/shine_0.0000-rot_-4.2247_101.9157_11.9376-shift_0.1647_-0.1210_-0.0659"
                
                depscore = 0
                depscoretop5 = 0
                for vid in views:
                    imviewnames = glob.glob('%s/confocal-%d-*.hdr' % (fd, vid))
                    imview = cv2.imread(imviewnames[0], -1)
                    imview = imview / np.max(imview)
                    h, w, _ = imview.shape
                    imview = cv2.resize(imview, (w // 2 ** K, h // 2 ** K))
                    
                    grid_bxdxhxwx3 = self.grids[vid].to(dev).repeat(bnum, 1, 1, 1, 1)
                    volumn_MxNxN = F.grid_sample(volumnz_BxDxTxHxW, grid_bxdxhxwx3, \
                                                 mode='bilinear', padding_mode='zeros', \
                                                 align_corners=self.align_corners)
                    
                    volumn_MxNxN = volumn_MxNxN.detach().cpu().numpy()[0, -1]
                    front_view = np.max(volumn_MxNxN, axis=0)
                    front_view = front_view / np.max(front_view)
                    imview = imview * 0.2 + 0.8 * np.tile(np.expand_dims(front_view, axis=2), [1, 1, 3])
                    cv2.imshow("re", imview)
                    # cv2.waitKey()
                    
                    imviewnames = glob.glob('%s/depth-%d-*.png' % (fd, vid))
                    imview = cv2.imread(imviewnames[0], 0)
                    immask = (imview > 0).astype(np.uint8) * 255
                    
                    h, w = imview.shape
                    imview = cv2.resize(imview, (w // 2 ** K, h // 2 ** K))
                    immask = cv2.resize(immask, (w // 2 ** K, h // 2 ** K))
                    immask = immask == 255
                    
                    # 0 - 256
                    depgt = imview.copy().astype(np.int)
                    
                    front_dep = np.argmax(volumn_MxNxN, axis=0)
                    front_dep = volumn_MxNxN.shape[0] - 1 - front_dep
                    # convert matlab into python
                    front_dep = front_dep - 1
                    front_dep[~immask] = 0
                    
                    # pre
                    deppre = front_dep.copy()
                    strr = ''
                    for k in range(-2, 3):
                        strr = '%s%.5f\t' % (strr, np.mean(deppre[immask] == (depgt[immask] + k)))
                        if k == 0:
                            depscore += np.mean(deppre[immask] == (depgt[immask] + k))
                    strr = '%s top5 %.5f' % (strr, np.mean(np.abs(deppre[immask] - (depgt[immask]) < 2.5)))
                    depscoretop5 += np.mean(np.abs(deppre[immask] - (depgt[immask]) < 2.5))
                    print(strr)
                    
                    cv2.imshow("depgt", imview)
                    cv2.imshow("deppre", front_dep.astype(np.float32) / volumn_MxNxN.shape[0])
                    cv2.waitKey(0)
                print(depscore / 26, depscoretop5 / 26)
                cv2.waitKey(0)
            
            grid_bxdxhxwx3 = torch.cat([self.grids_todev[vid] for vid in views])
            volumnz_BxDxTxHxW = F.grid_sample(volumnz_BxDxTxHxW, grid_bxdxhxwx3, \
                                              mode='bilinear', padding_mode='zeros', \
                                              align_corners=self.align_corners)
            
            return volumnz_BxDxTxHxW


######################################3333
def gather_numpy2(data, dim, index):
    """
    Gathers values along an axis specified by dim.
    For a 3-D tensor the output is specified by:
        out[i][j][k] = input[index[i][j][k]][j][k]  # if dim == 0
        out[i][j][k] = input[i][index[i][j][k]][k]  # if dim == 1
        out[i][j][k] = input[i][j][index[i][j][k]]  # if dim == 2

    :param dim: The axis along which to index
    :param index: A tensor of indices of elements to gather
    :return: tensor of gathered values
    """
    idx_xsection_shape = index.shape[:dim] + index.shape[dim + 1:]
    self_xsection_shape = data.shape[:dim] + data.shape[dim + 1:]
    if idx_xsection_shape != self_xsection_shape:
        raise ValueError("Except for dimension " + str(dim) + 
                         ", all dimensions of index and data should be the same size")
    if index.dtype != np.dtype('int_'):
        raise TypeError("The values of index must be integers")
    data_swaped = np.swapaxes(data, 0, dim)
    index_swaped = np.swapaxes(index, 0, dim)
    gathered = np.choose(index_swaped, data_swaped)
    return np.swapaxes(gathered, 0, dim)


def gather_numpy(a, dim, index):
    expanded_index = [index if dim == i else np.arange(a.shape[i]).reshape([-1 if i == j else 1 for j in range(a.ndim)]) for i in range(a.ndim)]
    
    re = []
    
    a_dxhxw = a
    index_1xhxw = index
    for i in range(a_dxhxw.ndim):
        if dim == i:
            tmp = index_1xhxw
        else:
            tmp = np.arange(a.shape[i])
            reshapdim = [-1 if i == j else 1 for j in range(a_dxhxw.ndim)]
            tmp = tmp.reshape(reshapdim)
        re.append(tmp)
    
    re1 = a[re]
    re2 = a[expanded_index]
    return re1

##################################################
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
        
        video = False
        hdr = True
        mat = False
        if video:
            videofile = '/home/wenzheng/Desktop/zzzzz/mnist-renders/0/1/shine_0.0000-rot_0.0000_0.0000_0.0000-shift_0.0000_0.0000_0.4000/video-confocal-gray-full.mp4'
            cap = cv2.VideoCapture(videofile)
            assert cap.isOpened() 

            ims = []
            i = 0
            # Read until video is completed
            while(cap.isOpened()):
                print(i)
                ret, frame = cap.read()
                if ret == True:
                    imgray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                    ims.append(imgray)
                    i += 1
                else:
                    break
             
            # When everything done, release the video capture object
            # cap.close()
            cap.release()
            
            # for fangyin's code, you should change it!!!
            assert i == 600
            ims = ims[:512]
            tbe = 0
            tlen = 512
        elif hdr:
             
            hfrfile = '/home/wenzheng/Desktop/zzzzz/render-bike/shine_0.0000-rot_5.1729_-90.9157_0.8179-shift_0.1812_0.0535_-0.0183/light-1-1277.0513-4.4600-100.0000.hdr'
            # hfrfile = '/home/wenzheng/Desktop/zzzzz/render-mnist/shine_0.0000-rot_-0.9894_9.2027_-24.2743-shift_0.0966_-0.0485_-0.2049/light-1-971.8722-4.9000-100.0000.hdr'
            # hfrfile = '/home/wenzheng/Desktop/zzzzz/render-bike-multi/ff1fb5a9dd255b059036436a751dc6c2_5e4fe8a57130744dece0b0a75f5fc9f6_42b5ea257418825b5c11b538bde9e3f1/shine_0.0000-rot_0.0000_0.0000_0.0000-shift_0.0000_0.0000_0.0000/light-1-12949.6777-4.6600-0.6500.hdr'
            # hfrfile = '/home/wenzheng/Desktop/zzzzz/101/shine_400.7502-rot_-3.1685_8.4930_107.4384-shift_0.3293_-0.2420_-0.1318/light-1-5.2906-4.0600-100.0000.hdr'
            hfrfile = '/home/wenzheng/Desktop/zzzzz/render-bike/0/mesh/shine_0.0000-rot_5.1729_-90.9157_0.8179-shift_0.1812_0.0535_-0.0183/light-2-hdr2spad.hdr'
            
            im = cv2.imread(hfrfile, -1);
            imgray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
            imgray = imgray.reshape(600, 256, 256);
            ims = imgray[:512]
            
            immax = np.max(ims)
            for i, im in enumerate(ims):
                print(i)
                imshow("", im / immax)
                cv2.waitKey(1)
                
            imsum = np.sum(ims, axis=0)
            imshow("steady", imsum / np.max(imsum))
            cv2.waitKey()
            tbe = 0
            tlen = 512
            
            if False:
                '''
                from scipy import ndimage
                a = ndimage.gaussian_filter
                sig = 1.0
                ims = a(ims, sigma=sig, mode='reflect', cval=0.0)
                '''
                from scipy import ndimage
                a = ndimage.filters.gaussian_filter1d
                a = ndimage.filters.uniform_filter1d
                # 30 - 40 pico
                b = np.random.randint(11) + 30
                blen = b * 1e-12 * 3e8
                bgrid = blen / 0.01
                sig = 1
                axis = 0
                ims = a(ims, size=3, axis=0, mode='reflect', cval=0.0)
        elif mat:
            from scipy.io import loadmat
            data = loadmat(file_name='/home/wenzheng/largestore/nlos-phasor/hdr2spad/code/newplanargau2.mat')
            imgray = np.transpose(data['hdr_spad'], [2, 0, 1])
            ims = imgray[:512]
            # ims = ims + np.random.randn(512, 256, 256).astype(np.float32) * 0.03
            tbe = 0
            tlen = 512
            
        else:
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
        
        '''
        data = loadmat('/home/wenzheng/largestore/nlos-phasor/nlos-fk-master/statue.mat')
        rect_data_hxwxt = data['data']
        '''
        data = loadmat(file_name='/home/wenzheng/largestore/nlos-phasor/realdata/bike0.mat')
        rect_data_hxwxt = data['measlr']
        rect_data_hxwxt = rect_data_hxwxt / rect_data_hxwxt.max()
        
        sptial_grid = 256
        crop = 512
        bin_len = 32e-12 * 3e8  # 0.01
        tbe = 0
        tlen = crop
    
    K = 1
    temds =0
    
    if False:
        if temds:
            shape = [1.0 / 2 ** K, 1.0 / 2 ** K, 1.0 / 2 ** K]
            sptial_grid = sptial_grid // 2 ** K
            crop = crop // 2 ** K
            bin_len = bin_len * 2 ** K
        else:
            shape = [1.0 / 2 ** K, 1.0 / 2 ** K, 1.0]
            sptial_grid = sptial_grid // 2 ** K
        import scipy
        from scipy.ndimage import zoom
        rect_data_hxwxt = zoom(rect_data_hxwxt, shape)
    
    else:
        for k in range(K):
            rect_data_hxwxt = rect_data_hxwxt[::2, :, :] + rect_data_hxwxt[1::2, :, :]  # np.maximum(rect_data_hxwxt[::2, :, :], rect_data_hxwxt[1::2, :, :]) 
            rect_data_hxwxt = rect_data_hxwxt[:, ::2, :] + rect_data_hxwxt[:, 1::2, :]  # np.maximum(rect_data_hxwxt[:, ::2, :], rect_data_hxwxt[:, 1::2, :])
            sptial_grid = sptial_grid // 2
        
        for k in range(temds):
                rect_data_hxwxt = rect_data_hxwxt[:, :, ::2] + rect_data_hxwxt[:, :, 1::2]  # np.maximum(rect_data_hxwxt[:, :, ::2], rect_data_hxwxt[:, :, 1::2])
                crop = crop // 2
                bin_len = bin_len * 2
                tbe = tbe // 2
                tlen = tlen // 2
    
    rect_data_dxhxwxt = np.expand_dims(rect_data_hxwxt, axis=0)
    rect_data_bxdxhxwxt = np.expand_dims(rect_data_dxhxwxt, axis=0)
    print(rect_data_hxwxt.max())
    np.save(file='a.npy', arr=rect_data_hxwxt)
    
    dev = 'cuda'
    
    bnum = 1
    dnum = 1
    rect_data_bxdxhxwxt = np.tile(rect_data_bxdxhxwxt, [bnum, dnum, 1, 1, 1])
    rect_data_bxdxhxwxt = torch.from_numpy(rect_data_bxdxhxwxt).to(dev)
    
    #####################################################################
    lctlayer = diffmodule(spatial=sptial_grid, crop=crop, bin_len=bin_len, \
                          mode='lct' , mview=False,
                          sampling_coeff=2.0, \
                          cycles=5)
    lctlayer.todev(dev, dnum)
    # lctlayer.change_bin_len(0.0096 * 2)
    for i in range(1):
        print('%d, tbe%d, ten%d' % (i, tbe, tbe + tlen))
        re = lctlayer(rect_data_bxdxhxwxt[:, :, :, :, tbe:tbe + tlen].permute(0, 1, 4, 2, 3), \
                      [tbe, tbe, tbe], [tbe + tlen, tbe + tlen, tbe + tlen], None)
    
    volumn_MxNxN = re.detach().cpu().numpy()[0, -1]
    zdim = volumn_MxNxN.shape[0] * 100 // 128
    volumn_MxNxN = volumn_MxNxN[:zdim]
    print('volumn min, %f' % volumn_MxNxN.min())
    print('volumn max, %f' % volumn_MxNxN.max())
    print(volumn_MxNxN.shape)
    
    volumn_MxNxN[volumn_MxNxN < 0] = 0
    front_view = np.max(volumn_MxNxN, axis=0)
        
    re = front_view / np.max(front_view)
    re = cv2.resize(re, (256, 256))
    re2 = re * 2
    re2[re2 > 1] = 1
    cv2.imshow("re", re)
    cv2.imshow("re2", re2)
    cv2.imshow("re256", re)
    # cv2.imshow("gt", imgt)
    
    if False:
        imgt = cv2.imread('/home/wenzheng/Desktop/zzzzz/mnist-renders-600/0/1/shine_0.0000-rot_12.4859_4.0714_78.2269-shift_-0.2150_0.0642_0.0098/confocal-0-0.4961-0.0000.png');
        imgt = imgt.astype(np.float32) / 255.0
        impre = cv2.resize(front_view / np.max(front_view), (256, 256))
        imgt = 0.5 * imgt + 0.5 * np.expand_dims(impre, axis=2)
        cv2.imshow("gt", imgt)
    cv2.waitKey()
    
    volumn_ZxYxX = volumn_MxNxN
    volumn_ZxYxX = volumn_ZxYxX / np.max(volumn_ZxYxX)
    
    front_index = np.argmax(volumn_ZxYxX, axis=0)
    front_index_show = front_index.astype(np.uint8)
    cv2.imshow("dep", front_index_show)
    cv2.waitKey()
    
    front_index = np.expand_dims(front_index, axis=0)
    
    d = volumn_ZxYxX.shape[0]
    layers = []
    for i in range(-5, 6):
        idx = front_index + i
        idx[idx < 0] = 0
        idx[idx >= d] = d - 1 
        map = gather_numpy(volumn_ZxYxX, 0, idx)
        layers.append(map)
    layers = np.concatenate(layers, axis=0)
    for i, frame in enumerate(layers):
        print(i)
        frame = cv2.resize(frame, (256, 256))
        cv2.imshow("layer1", 5 * frame)
        cv2.imshow("layer2", frame / np.max(frame))
        cv2.waitKey(0)
        
        
    for i, frame in enumerate(volumn_ZxYxX):
        print(i)
        cv2.imshow("re1", 2 * cv2.resize(frame, (256, 256)))
        cv2.imshow("re2", frame / np.max(frame))
        cv2.waitKey(0)
    
