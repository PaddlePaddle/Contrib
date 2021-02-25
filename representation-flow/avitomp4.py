import os
import argparse

parser = argparse.ArgumentParser()

parser.add_argument('-video_dir', type=str, default='data/')
parser.add_argument('-file_dir', type=str, default='data/hmdb/')
args = parser.parse_args()

video_path = args.video_dir       ##  path for train/test video data.
file_path = args.file_dir         ##  path for txt file.

for ifile in range(1, 4):
    with open(file_path + "split_train"  + str(ifile) + ".txt", 'r') as f:
        root = video_path + 'data/'
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
        

    with open(file_path + "split_train"  + str(ifile) + ".txt", 'r') as f:
        
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