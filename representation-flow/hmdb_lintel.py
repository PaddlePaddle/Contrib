import paddle.fluid as fluid

import numpy as np
import random

import os

os.environ['ffmpegpath'] = '/home/aistudio/work/FFmpeg'
os.environ['PATH'] = os.environ['PATH'] + ':' + os.environ['ffmpegpath'] +'/bin'
os.environ['LD_LIBRARY_PATH'] = os.environ['LD_LIBRARY_PATH'] + ':' + os.environ['ffmpegpath'] +'/lib'
os.environ['CPATH'] = os.environ['ffmpegpath'] +'/include'
os.environ['LIBRARY_PATH'] = os.environ['LIBRARY_PATH'] + ':' + os.environ['ffmpegpath'] +'/lib'

import lintel


class HMDB(object):

    def __init__(self, split_file, root, mode='rgb', length=16, model='2d', random=False, c2i={}):
        self.class_to_id = c2i
        self.id_to_class = []
        for i in range(len(c2i.keys())):
            for k in c2i.keys():
                if c2i[k] == i:
                    self.id_to_class.append(k)
        cid = 0
        self.data = []
        self.model = model
        self.size = 112

        with open(split_file, 'r') as f:
            for l in f.readlines():
                if len(l) <= 5:
                    continue
                v,c = l.strip().split(' ')
                v = v.split('.')[0]+'.mp4'   #mode+'_'+v.split('.')[0]+'.avi'
                if c not in self.class_to_id:
                    self.class_to_id[c] = cid
                    self.id_to_class.append(c)
                    cid += 1
                v = os.path.join(c,v)
                vidmp4 = os.path.join(root, v)
                if(os.path.exists(vidmp4)):
                    self.data.append([vidmp4, self.class_to_id[c]])

        self.split_file = split_file
        self.root = root
        self.mode = mode
        self.length = length
        self.random = random
    def shuffdata(self):
        def get_all():
            for i in range(len(self.data)):
                yield self.data[i]

        shuffreader = fluid.io.shuffle(get_all, len(self.data))
        i = 0
        for e in shuffreader():
            self.data[i] = e
            i += 1

    def __getitem__(self, index):
        vid, cls = self.data[index]
        with open(vid, 'rb') as f:
            enc_vid = f.read()

        for i in range(10):
            df, w, h, temp = lintel.loadvid(enc_vid, should_random_seek=self.random, width=0,height=0,num_frames=self.length*2)
            df = np.frombuffer(df, dtype=np.uint8)
            xp = np.mean(df)
            if(xp > 1):
                break

            
        #print('[lintel]', xp)
            
        w=w//2
        h=h//2
        
        # center crop
        if not self.random:
            i = int(round((h-self.size)/2.))
            j = int(round((w-self.size)/2.))
            df = np.reshape(df, newshape=(self.length*2, h*2, w*2, 3))[::2,::2,::2,:][:, i:-i, j:-j, :]
        else:
            th = self.size
            tw = self.size
            i = random.randint(0, h - th) if h!=th else 0
            j = random.randint(0, w - tw) if w!=tw else 0
            df = np.reshape(df, newshape=(self.length*2, h*2, w*2, 3))[::2,::2,::2,:][:, i:i+th, j:j+tw, :]
            
        if self.mode == 'flow':
            #print(df[:,:,:,1:].mean())
            #exit()
            # only take the 2 channels corresponding to flow (x,y)
            df = df[:,:,:,1:]
            if self.model == '2d':
                # this should be redone...
                # stack 10 along channel axis
                df = np.asarray([df[:10],df[2:12],df[4:14]]) # gives 3x10xHxWx2
                df = df.transpose(0,1,4,2,3).reshape(3,20,self.size,self.size).transpose(0,2,3,1)
            
                
        df = 1-2*(df.astype(np.float32)/255)
        #print('[lintel test]', df.shape)
        if self.model == '2d':
            # 2d -> return TxCxHxW
            df = df.transpose([0,3,1,2])
            #print('[lintel test]', df.shape)
            return df, cls
        # 3d -> return CxTxHxW
        return df.transpose([3,0,1,2]), cls


    def __len__(self):
        return len(self.data)


########### Test Only ###########
import time

if __name__ == '__main__':
    DS = HMDB
    dataseta = DS('/home/aistudio/work/representation-flow/data/hmdb/split1_train.txt', '/home/aistudio/data/data49479/', model='2d', mode='rgb', length=16)
    
    start = time.time()
    for i in range(10):
        print(i)
        dataseta[i]
    print(len(dataseta))

    end = time.time()

    print('[Time spend]', 1000*(end-start))
