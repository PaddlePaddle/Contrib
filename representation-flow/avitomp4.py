
import os

with open("/home/aistudio/work/representation-flow/data/hmdb/split1_train.txt", 'r') as f:
    root = '/home/aistudio/data/data49479/'
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
    

with open("/home/aistudio/work/representation-flow/data/hmdb/split1_test.txt", 'r') as f:
    root = '/home/aistudio/data/data49479/'
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