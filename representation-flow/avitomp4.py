import os
import argparse

parser = argparse.ArgumentParser()

parser.add_argument('-data_dir', type=str, default='data/data/')

root = data_path = args.data_dir       ##  path for train/test data.
with open("data/hmdb/split_train.txt", 'r') as f:
    
    i = 0
    for l in f.readlines():
        if len(l) <= 5:
            continue
        v,c = l.strip().split(' ')
        v = v.split('.')[0]    #+'.avi'   #mode+'_'+v.split('.')[0]+'.avi'
        vidavi = v + '.avi'
        vidavi = os.path.join(c,vidavi)
        vidavi = os.path.join(root, vidavi)
        vidmp4 = v + '.mp4'
        vidmp4 = os.path.join(c,vidmp4)
        vidmp4 = os.path.join(root, vidmp4)
        #print(vidavi, vidmp4)
        if(not os.path.exists(vidmp4)):
            os.system('ffmpeg -i '+vidavi+' '+vidmp4)
            i +=1
    

with open("data/hmdb/split_test.txt", 'r') as f:
    
    i = 0
    for l in f.readlines():
        if len(l) <= 5:
            continue
        v,c = l.strip().split(' ')
        v = v.split('.')[0]    #+'.avi'   #mode+'_'+v.split('.')[0]+'.avi'
        vidavi = v + '.avi'
        vidavi = os.path.join(c,vidavi)
        vidavi = os.path.join(root, vidavi)
        vidmp4 = v + '.mp4'
        vidmp4 = os.path.join(c,vidmp4)
        vidmp4 = os.path.join(root, vidmp4)
        #print(vidavi, vidmp4)
        if(not os.path.exists(vidmp4)):
            os.system('ffmpeg -i '+vidavi+' '+vidmp4)
            i +=1