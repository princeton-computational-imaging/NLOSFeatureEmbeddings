

# License: BSD
# Author: Ghassen Hamrouni

from __future__ import print_function
from __future__ import division

import torch
import torch.nn as nn

import numpy as np
import cv2


########################################################
# Training dataset

def crop_video(video, framenum, res):
    if framenum == 512:
        return video, 0, 512
    
    sumdata = np.zeros((len(video),), dtype=np.float32)
    for i, frame in enumerate(video):
        sumdata[i] = np.sum(frame)
    
    idxs = []
    vals = []
    for i in range((len(video) - framenum) // res):
        idxs.append([i * res, i * res + framenum])
        vals.append(np.sum(sumdata[i * res:i * res + framenum]))
    # find max
    valmaxid = np.argmax(vals)
    tbe, ten = idxs[valmaxid]
    return video[tbe:ten], tbe, ten


##############################################
import scipy.io as sio
'''
matref = sio.loadmat('/u6/a/wenzheng/remote2/code-nlos-git/nlos-fk-master/bike0.mat')
matref = np.array(matref['measlr'], dtype=np.float32)
matmaxref = np.max(matref)
'''

def loadrealdata(datafolder, name, scale, tres=2, isdebug=False):
   
    matmaxref = 252.0
   
    tlen = 256
    # name = args.realname  # 'discoball'
    # scale = args.realscale
    
    if name == 'bike':
        tbe2 = 280
        # scale = 1.0
    if name == 'dragon':
        tbe2 = 256
        # scale = 1.0
    if name == 'statue':
        tbe2 = 188
        # scale = 1.0
    if name == 'resolution':
        tbe2 = 180
        # scale = 1
    if name == 'discoball':
        tbe2 = 188
        # scale = 1
    if name == 'teaser':
        tbe2 = 180
        tlen = 512 - tres
        # scale = 1
    
    # tbe2 = 0
    # tlen = 512
    folders = ['/scratch/gobi2/wenzheng/nlos-trans/realdata/%s0.mat' % name]
    # folders = ['/u6/a/wenzheng/remote2/code-nlos-git/nlos-fk-master/%s0.mat' % name]
    folders = ['%s/%s0.mat' % (datafolder, name)]

    mats = []
    tbes = []
    tens = []
    for df in folders:
        mat = sio.loadmat(df)
        mat = np.array(mat['measlr'], dtype=np.float32)
        mat = np.transpose(mat, [2, 0, 1]);
        
        matmax = np.max(mat)
        print('maxmat%s %.2f, maxmatbike %.2f, scale%.2f' % \
              (name, matmax, matmaxref, matmax / matmaxref))
        
        mat = mat / matmax
        '''
        if mat99 > 0:
            mat = mat / mat99
            mat[mat > 1] = 1
            '''
        
        if isdebug:
            tbe = 0
            imcrop = mat
            for i in range(imcrop.shape[0]):
                    
                impatch = imcrop[i]
                impatch2 = imcrop[i]
                print('%.2f, %.2f' % ((i + tbe) / 100, np.mean(impatch)))
                
                # impatch = impatch / np.max(im)
                # impatchlog = impatch ** 0.3
                # impatch2 = (impatch > 0).astype(np.float32)
                impatch = np.concatenate((impatch, np.ones_like(impatch[:, :1]), impatch2 \
                                          ), axis=1)
                # cv2.destroyAllWindows()
                # cv2.imshow("im", (impatch > 0).astype(np.float32))
                cv2.imshow("im", impatch)
                key = cv2.waitKey(33)
                if key == ord('s'):
                    key = cv2.waitKey(0)
                # cv2.imwrite('1-%d.png' % (i + tbe), (impatch * 255).astype(np.uint8))
            cv2.waitKey(0)
        
        matcrop, tbe, ten = crop_video(mat, tlen, tres)
        print('tbe %d, ten %d' % (tbe, ten))
        
        # assert tbe2 >= tbe
        if tbe2 > tbe:
            matcrop = matcrop[tbe2 - tbe:]
            tbe = tbe2
        
        if isdebug:
            imcrop = matcrop
            for i in range(imcrop.shape[0]):
                    
                impatch = imcrop[i]
                impatch2 = imcrop[i]
                print('%.2f, %.2f' % ((i + tbe) / 100, np.mean(impatch)))
                
                # impatch = impatch / np.max(im)
                # impatchlog = impatch ** 0.3
                # impatch2 = (impatch > 0).astype(np.float32)
                impatch = np.concatenate((impatch, np.ones_like(impatch[:, :1]), impatch2 \
                                          ), axis=1)
                # cv2.destroyAllWindows()
                # cv2.imshow("im", (impatch > 0).astype(np.float32))
                cv2.imshow("im", impatch)
                cv2.waitKey(33)
                # cv2.imwrite('1-%d.png' % (i + tbe), (impatch * 255).astype(np.uint8))
                
            cv2.waitKey(0)
        mats.append(matcrop)
        tbes.append(tbe)
        tens.append(ten)
        
        return mats, tbes, tens


#########################################################
def testreal(model, epoch, tres=2, in_dim=1, dev='cuda', datafolder='.', svdir='.'):
    model.eval()
    
    ims = []
    
    for ni, name in enumerate(['bike', 'dragon', 'statue', 'resolution', 'discoball', 'teaser']):
        
        mats, tbes, tens = loadrealdata(datafolder, name, scale=-1, tres=tres, isdebug=False)
        mat0 = mats[0]
        # matmax = np.max(mat0)
        # scales = [-1, 99.9, 99.5, 99.0, 98.0]
        # for scale in scales:
        if True:
            scales = [-1, -1, -1, -1, -1, 99.9]
            scale = scales[ni]
            mat = mat0
            if scale > 0:
                mat99 = np.percentile(mat, scale)
                print('0.999 %.5f' % mat99)
                # scale = matmax / matmaxref
                if mat99 > 0:
                    mat = mat / mat99
                    mat[mat > 1] = 1
            
            batch_idx = 0
            datanp_txhxw = mat
            tbe = tbes[batch_idx]
            ten = tens[batch_idx]
            
            tbe = [tbe // tres]
            ten = [ten // tres]
    
            #########################################################
            data_bxcxdxhxw = torch.from_numpy(datanp_txhxw).unsqueeze(0).unsqueeze(0).repeat(1, in_dim, 1, 1, 1)
            
            data_bxcxdxhxw = \
            data_bxcxdxhxw.to(dev)
            
            if torch.min(data_bxcxdxhxw) >= 0:
                print('data, max %.5f, min %.5f' % (torch.max(data_bxcxdxhxw), torch.min(data_bxcxdxhxw)))
                data_bxcxdxhxw = data_bxcxdxhxw * 2 - 1
            
            dim = data_bxcxdxhxw.shape[2] // 2
            output = model(data_bxcxdxhxw, tbe, ten)
            output = torch.clamp(output, -1, 1)
            w = output.shape[-1]
            
            tmp = (output[:, :in_dim] + 1) / 2
            tmpmax = tmp.max()
            tmp = tmp / (tmpmax.view(-1, 1, 1, 1) + 1e-8)
            output[:, :in_dim] = tmp * 2 - 1

            # output[:, -in_dim:] = output[:, -in_dim:].mean(1, keepdim=True).repeat(1, 3, 1, 1)
            data = np.concatenate([output[:, :in_dim].detach().cpu().numpy(),
                                   output[:, -in_dim:].detach().cpu().numpy()], axis=3)
            ims.append(data)
    
    ims = np.concatenate(ims, axis=0)
    datanum = len(ims)
    row = len(scales)
    row = 1
    a = np.zeros([0, w * 2 * row, in_dim])
    for i in range(datanum // row):
        imslice = [ims[d:d + 1] for d in range(i * row, i * row + row)]  # ims[i * 4:i * 4 + 4]
        imslice = np.concatenate(imslice, axis=3)
        a = np.concatenate((a, np.transpose(imslice[0], [1, 2, 0])), axis=0)
        
    '''
        complct = cv2.imread('/u6/a/wenzheng/remote2/code-nlos-git/nlos-fk-master/%s-lct.png'%name)
        compfk = cv2.imread('/u6/a/wenzheng/remote2/code-nlos-git/nlos-fk-master/%s-fk.png'%name)
    '''
    '''
        adep = a[:, 256:, :]
        a = a[:, :256, :]
        a = a / np.max(np.abs(a))
        a = (a * 255).astype(np.uint8)
        adep = (adep * 255).astype(np.uint8)
    '''
    a = (a + 1) / 2
    im = a * 255
    cv2.imwrite('%s/testreal-%d-%d.png' % (svdir, epoch, batch_idx), im)

