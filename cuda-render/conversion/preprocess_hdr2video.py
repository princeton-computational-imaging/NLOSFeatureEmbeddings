

from __future__ import print_function
from __future__ import division

import os
import glob

import cv2
import numpy as np
from scipy import ndimage

from torch.utils.data import Dataset, DataLoader
from torch.hub import tqdm

# make it reproducible
np.random.seed(123456)


#######################################################
class DataProvider(Dataset):
    """
    Class for the data provider
    """

    def __init__(self, datafolder, shininess, \
                 timeimszs, \
                 confocal=1, \
                 timebe=0, timeen=6, \
                 time_cropbe=0, time_cropen=6, \
                 loadnum=-1, \
                 mode='train', datadebug=False):
        
        self.mode = mode
        self.datadebug = datadebug
        
        self.datafolder = datafolder
        self.shinness = shininess
        
        ##########################################
        self.timeimszs = timeimszs
        self.timebe = timebe
        self.timeen = timeen
        self.time_cropbe = time_cropbe
        self.time_cropen = time_cropen
        
        ###########################################
        self.modeldirs = []
        for fol in datafolder:
            for shi in shininess:
                modeldirs = glob.glob('%s/%d/*' % (fol, shi))
                for modeldir in modeldirs:
                    rotdirs = glob.glob('%s/shin*' % (modeldir))
                    self.modeldirs.extend(rotdirs)
        
        self.imnum = len(self.modeldirs)
        
        #######################################################
        self.transient2video = True
        self.videoname = 'video'
        self.videoconfocal = confocal
        self.gray = True
        self.clip = False
        self.clipratio = 99.7
        if self.videoconfocal == 1:
            self.videoname = '%s-confocal' % self.videoname
        elif self.videoconfocal == 2:
            self.videoname = '%s-confocalspad' % self.videoname
        
        if self.gray:
            timeimszs[-1] = 1
            self.videoname = '%s-gray' % self.videoname
        else:
            self.videoname = '%scolor' % self.videoname
            
        if self.clip:
            self.videoname = '%s-clip%.2f' % (self.videoname, self.clipratio)
        else:
            self.videoname = '%s-full' % (self.videoname,)
        
        self.videoformat = 'mp4'
        
        # direct exit after preprocess
        # exit(0)
        
        #########################################################
        # assert len(self.modeldirs) == 1  # for optimization, it is 1
        self.imszs = timeimszs  # t h w 3
        self.hei = timeimszs[1]
        self.wei = timeimszs[2]
        self.color = timeimszs[3]
        
        self.timebe = timebe
        self.timeen = timeen
        self.time_cropbe = time_cropbe
        self.time_cropen = time_cropen

    def __len__(self):
        return self.imnum

    def __getitem__(self, idx):
        return self.prepare_instance(idx)
    
    def prepare_instance(self, idx):
        
        flga = False
        # print('progress %.5f' % (idx / self.imnum))
        
        # let's do the preprocession
        if self.transient2video:
            
            rotshiftdir = self.modeldirs[idx]
            
            videofile = '%s/%s.%s' % (rotshiftdir, self.videoname, self.videoformat)
            if not os.path.isfile(videofile):
                flga = self.preprocess(rotshiftdir, \
                                       self.timeimszs, self.time_cropbe, \
                                       self.time_cropen, videofile)
                # print('suc signal %d' % int(flga))
            else:
                # print('already done')
                # redo it
                cmd = 'rm %s' % videofile
                os.system(cmd)
                flga = self.preprocess(rotshiftdir, \
                                       self.timeimszs, self.time_cropbe, \
                                       self.time_cropen, videofile)
                '''
                flga = True
                '''
        
        return flga
    
    def load_hdr(self, rotshiftdir, imszs, time_cropbe, time_cropen):
        
        if self.videoconfocal > 0:
            name = glob.glob('%s/light-%d-*.hdr' % (rotshiftdir, self.videoconfocal))
        else:
            name = glob.glob('%s/light-0-*.hdr' % rotshiftdir)
            
        if len(name) != 1:
            print('bad file')
            return None
        
        imname = name[0]

        if self.videoconfocal <= 1:
            params = os.path.split(imname)[1][:-4].split('-')
            mindist = float(params[-1])
            maxdist = float(params[-2])
            
            # care baout 1.5 to 4.5
            if mindist < time_cropbe or maxdist >= time_cropen:
                print('bad dist')
                return None
        
        try:
            # imgtfile = cv2.FileStorage(imname, flags=cv2.FileStorage_READ)
            # im = imgtfile.getFirstTopLevelNode().mat()
            im = cv2.imread(imname, -1)
            im = im / np.max(im)
            
            if self.gray:
                im = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
                im = im / np.max(im)
            
            # it should be txhxwx3
            im = im.reshape(*imszs)
        except:
            print('bad capture')
            return None
        
        ##################################################################
        # crop meaningful region
        timebe = int(100 * time_cropbe)
        timeen = int(100 * time_cropen)
        imcrop = im[timebe:timeen, :, :, :]
        
        if self.clip:
            # do we need to do the fixed normalization?
            data90 = np.percentile(imcrop, self.clipratio)
            maxdata = np.max(imcrop)
            meandata = np.mean(imcrop)
            # data90 = maxdata
            imnorm = imcrop / data90
            imnorm[imnorm > 1] = 1
        else:
            imnorm = imcrop
        
        # smooth to get rid of the artifacts
        a = ndimage.filters.uniform_filter
        a = ndimage.gaussian_filter
        
        if False and (self.videoconfocal <= 1):
            sig = 0.7
            if self.gray:
                imsmooth = a(imnorm[:, :, :, 0], sigma=sig, mode='reflect', cval=0.0)
            else:
                imb = a(imnorm[:, :, :, 0], sigma=sig, mode='reflect', cval=0.0)
                img = a(imnorm[:, :, :, 1], sigma=sig, mode='reflect', cval=0.0)
                imr = a(imnorm[:, :, :, 2], sigma=sig, mode='reflect', cval=0.0)
                imsmooth = np.stack([imb, img, imr], axis=3)
        else:
            if self.gray:
                imsmooth = imnorm[:, :, :, 0]
            else:
                imsmooth = imnorm
        
        return imsmooth
    
    def preprocess(self, modeldir, imszs, time_cropbe, time_cropen, videofile):
        
        # print(modeldir)
        imsmooth = self.load_hdr(modeldir, imszs, time_cropbe, time_cropen)
        if imsmooth is None:
            return False
        
        ###########################################################################
        frame_width = imsmooth.shape[2]
        frame_height = imsmooth.shape[1]
        
        if self.gray:
            imsmooth = np.tile(np.expand_dims(imsmooth, axis=3), [1, 1, 1, 3])
         
        # Define the codec and create VideoWriter object.The output is stored in 'outpy.avi' file.
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(videofile, \
                              fourcc, \
                              20, \
                              (frame_width, frame_height))
        
        # write it into videos
        imlists = [100, 110, 120, 130, 140, 150, 160, 170, 180, 190, 200, 210, 220, 230, 240, 250]
        for i in range(imsmooth.shape[0]):
            impatch = imsmooth[i]
            if i in imlists:
                cv2.imwrite(videofile.replace('.mp4',  '_%d.png'%i), impatch / impatch.max() *255)
            frame = (impatch * 255).astype(np.uint8)
            out.write(frame)
        impatch = np.mean(imsmooth, 0)
        print(impatch.shape)
        cv2.imwrite(videofile.replace('.mp4',  '_all.png'), impatch / impatch.max() *255)
        out.release()
        
        return True


def collate_fn(batch_list):

    return batch_list


def get_data_loaders(folders, shininess=[0], imszs=[600, 256, 256, 3], \
                    confocal=1, \
                     timebe=0, timeen=6, \
                     time_cropbe=0, time_cropen=6, \
                     mode='train', bs=4, numworkers=32):

    print('Building dataloaders')

    dataset_train = DataProvider(folders, shininess, imszs, \
                                confocal, \
                                 timebe, timeen, \
                                 time_cropbe, time_cropen, \
                                 mode=mode, datadebug=False)
    
    # always true
    shuffle = True
    if mode == 'train_val' or mode == 'test':
        shuffle = False

    train_loader = DataLoader(dataset_train, batch_size=bs, \
                              shuffle=shuffle, num_workers=numworkers, collate_fn=collate_fn)

    print('train num {}'.format(len(dataset_train)))
    print('train iter'.format(len(train_loader)))

    return train_loader


##############################################
if __name__ == '__main__':
    
    folders = ['/home/wenzheng/largestore/projects-clean/Learned-Feature-Embeddings-for-Non-Line-of-Sight-Imaging-and-Recognition/data/bunny-renders']
    
    imszs = [600, 256, 256, 3]
    timebin = 100
    
    '''
    rootfd = '/u6/a/wenzheng/remote2/datasets/shapenet'
    fds = glob.glob('%s/*-multiclassrender' % rootfd)
    '''
    
    '''
    import tqdm
    for i in tqdm.tqdm(np.arange(1000)):
        print(i)
    '''
    
    for conf in [1]:
        train_loader = get_data_loaders(folders, \
                                       imszs=imszs, \
                                        confocal=conf, \
                                        mode='test', \
                                        bs=1, numworkers=0)
    
        ###############################################
        for data in train_loader:
            print(data)

