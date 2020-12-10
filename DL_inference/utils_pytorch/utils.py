

import torch
import torch.nn as nn

import cv2
import numpy as np

################################################################################
'''
def num_divisible_by_2(number):
    i = 0
    while not number % 2:
        number = number // 2
        i += 1
    return i
'''


################################################################################
def restrctuctre(nf0, norm, inplace=False):
    
    tmp = nn.Sequential(
            norm(nf0 * 1, affine=True),
            nn.ReLU(inplace),
            # nn.Dropout3d(0.1, inplace),
            
            nn.ReplicationPad3d(1),
            nn.Conv3d(nf0 * 1,
                      nf0 * 1,
                      kernel_size=[3, 3, 3],
                      padding=0,
                      stride=[1, 1, 1],
                      bias=False),
            
            norm(nf0 * 1, affine=True),
            nn.ReLU(inplace),
            # nn.Dropout3d(0.1, inplace),
            
            nn.ReplicationPad3d(1),
            nn.Conv3d(nf0 * 1,
                      nf0 * 1,
                      kernel_size=[3, 3, 3],
                      padding=0,
                      stride=[1, 1, 1],
                      bias=False),
        )
    
    return tmp

def restrctuctre3d(nf0, norm, inplace=False):
    
    tmp = nn.Sequential(
            norm(nf0 * 1, affine=True),
            nn.ReLU(inplace),
            # nn.Dropout3d(0.1, inplace),
            
            nn.ReplicationPad3d(1),
            nn.Conv3d(nf0 * 1,
                      nf0 * 1,
                      kernel_size=[3, 3, 3],
                      padding=0,
                      stride=[1, 1, 1],
                      bias=False),
            
            norm(nf0 * 1, affine=True),
            nn.ReLU(inplace),
            # nn.Dropout3d(0.1, inplace),
            
            nn.ReplicationPad3d(1),
            nn.Conv3d(nf0 * 1,
                      nf0 * 1,
                      kernel_size=[3, 3, 3],
                      padding=0,
                      stride=[1, 1, 1],
                      bias=False),
        )
    
    return tmp


def restrctuctre2d(nf0, norm, kernel=3, pad=1):
    
    tmp = nn.Sequential(
            norm(nf0 * 1, affine=True),
            nn.ReLU(False),
            # nn.Dropout3d(0.1, False),
            
            nn.ReflectionPad2d(pad),
            nn.Conv2d(nf0 * 1,
                      nf0 * 1,
                      kernel_size=[kernel, kernel],
                      padding=0,
                      stride=[1, 1],
                      bias=False),
            
            norm(nf0 * 1, affine=True),
            nn.ReLU(False),
            # nn.Dropout3d(0.1, False),
            
            nn.ReflectionPad2d(pad),
            nn.Conv2d(nf0 * 1,
                      nf0 * 1,
                      kernel_size=[kernel, kernel],
                      padding=0,
                      stride=[1, 1],
                      bias=False),
        )
    
    return tmp


####################################################
def init_mats(trans=False):
    
    mats = torch.Tensor([[1, 0, 0, 0, 1, 0, 0, 0, 1],
                         [0.760836, 0.0447155, -0.647402, 0.0447155, 0.99164,
                          0.121042, 0.647402, -0.121042, 0.752475],
                         [0.828744, -0.106611, 0.54938, -0.106611, 0.933632,
                          0.342001, -0.54938, -0.342001, 0.762376],
                         [0.983521, -0.0590013, -0.170898, -0.0590013,
                          0.788757, -0.611867, 0.170898, 0.611867, 0.772277],
                         [0.95545, 0.0878597, -0.281775, 0.0878597, 0.826729,
                          0.555698, 0.281775, -0.555698, 0.782178],
                         [0.817867, 0.0685338, 0.571311, 0.0685338,
                          0.974212, -0.214976, -0.571311, 0.214976, 0.792079],
                         [0.829463, -0.0684608, -0.55435, -0.0684608,
                          0.972517, -0.22254, 0.55435, 0.22254, 0.80198],
                         [0.964787, -0.0733777, 0.252591, -0.0733777,
                          0.847094, 0.526352, -0.252591, -0.526352, 0.811881],
                         [0.985013, 0.0494608, 0.165237, 0.0494608,
                          0.836769, -0.545317, -0.165237, 0.545317, 0.821782],
                         [0.87543, 0.0738207, -0.477675, 0.0738207, 0.956254,
                          0.283071, 0.477675, -0.283071, 0.831683],
                         [0.848254, -0.0318132, 0.528634, -0.0318132,
                          0.99333, 0.110827, -0.528634, -0.110827, 0.841584],
                         [0.94951, -0.070351, -0.305746, -0.070351,
                          0.901975, -0.426019, 0.305746, 0.426019, 0.851485],
                         [0.998042, 0.0163562, -0.060365, 0.0163562,
                          0.863344, 0.504351, 0.060365, -0.504351, 0.861386],
                         [0.925979, 0.0636266, 0.372175, 0.0636266,
                          0.945308, -0.319913, -0.372175, 0.319913, 0.871287],
                         [0.881307, -0.00375067, -0.47253, -0.00375067,
                          0.999882, -0.0149318, 0.47253, 0.0149318, 0.881188],
                         [0.94422, -0.0544394, 0.324784, -0.0544394, 0.946869,
                          0.316979, -0.324784, -0.316979, 0.891089],
                         [0.99969, -0.00552741, -0.0242579, -0.00552741,
                          0.9013, -0.433161, 0.0242579, 0.433161, 0.90099],
                         [0.964273, 0.0436712, -0.261286, 0.0436712,
                          0.946618, 0.319386, 0.261286, -0.319386, 0.910891],
                         [0.922411, 0.0112091, 0.386046, 0.0112091,
                          0.998381, -0.0557716, -0.386046, 0.0557716, 0.920792],
                         [0.95267, -0.0322514, -0.302292, -0.0322514,
                          0.978024, -0.205985, 0.302292, 0.205985, 0.930693],
                         [0.996886, -0.0132402, 0.0777394, -0.0132402,
                          0.943708, 0.330514, -0.0777394, -0.330514, 0.940594],
                         [0.988176, 0.0211075, 0.151861, 0.0211075,
                          0.962319, -0.271104, -0.151861, 0.271104, 0.950495],
                         [0.964281, 0.0117793, -0.264621, 0.0117793, 0.996116,
                          0.0872645, 0.264621, -0.0872645, 0.960396],
                         [0.975304, -0.0111198, 0.220587, -0.0111198, 0.994993,
                          0.0993228, -0.220587, -0.0993228, 0.970297],
                         [0.99691, -0.00718622, -0.0782245, -0.00718622,
                          0.983288, -0.181914, 0.0782245, 0.181914, 0.980198],
                         [0.998926, 0.00307839, -0.0462215, 0.00307839, 0.991173,
                          0.132543, 0.0462215, -0.132543, 0.990099]])
    n_view = mats.shape[0]
    selfmats = torch.zeros(n_view, 3, 4)
    selfmats[..., : 3] = mats.reshape(n_view, 3, 3).transpose(1, 2)
    
    ###########################################################
    mtxs = []
    mtxs.append(np.eye(3))
    
    samplenum = 25
    ratio = 0.5
    sam2 = int(samplenum / ratio / ratio)
    sambe = sam2 - samplenum
    
    for i in range(samplenum):
        
        n = sambe + i + 1.0
        N = sam2 + 1.0
        
        zn = n / N
        r = np.sqrt(1 - zn * zn)
        
        phi = (np.sqrt(5.0) - 1.0) / 2.0
        angle = 2.0 * np.pi * n * phi
        xn = r * np.cos(angle)
        yn = r * np.sin(angle)
        
        zaxis = np.array([0, 0, 1], dtype=np.float64)
        newaxis = np.array([xn, yn, zn], dtype=np.float64)
        costheta = zn
        theta = np.arccos(costheta)
        # print(theta / np.pi * 180)
        
        rotaxis = np.cross(zaxis, newaxis)
        rotaxis = rotaxis / np.sqrt(np.sum(rotaxis ** 2))
        
        rotvec = rotaxis * theta
        mtx = cv2.Rodrigues(rotvec)[0]
        if trans:
            mtx = mtx.T
        mtxs.append(mtx)
    
    mtxs = torch.from_numpy(np.array(mtxs, dtype=np.float32))
    mtxs_bx3x4 = torch.cat([mtxs, torch.zeros_like(mtxs[:, :, :1])], dim=2)
    
    print(torch.max((mtxs_bx3x4 - selfmats).abs()))
    
    return mtxs_bx3x4


if __name__ == '__main__':
    
    init_mats(False)

