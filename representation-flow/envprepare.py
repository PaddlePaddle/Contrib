import os 
import time
import zipfile


#f = zipfile.ZipFile("/home/aistudio/data/data49479/hmdb51_org.zip",'r')
#for file in f.namelist():
#    f.extract(file)

os.system('cd /home/aistudio/work/FFmpeg && make && make install')
time.sleep(60)

os.environ['ffmpegpath'] = '/home/aistudio/work/FFmpeg'
os.environ['PATH'] = os.environ['PATH'] + ':' + os.environ['ffmpegpath'] +'/bin'
os.environ['LD_LIBRARY_PATH'] = os.environ['LD_LIBRARY_PATH'] + ':' + os.environ['ffmpegpath'] +'/lib'
os.environ['CPATH'] = os.environ['ffmpegpath'] +'/include'
os.environ['LIBRARY_PATH'] = os.environ['LIBRARY_PATH'] + ':' + os.environ['ffmpegpath'] +'/lib'

os.system('cd lintel/ && pip install --editable . --user')

filenames = os.listdir("/home/aistudio/data/data49479/")

#for file_name in filenames:
 #   if os.path.splitext(file_name)[1] == '.zip':
  #      f = zipfile.ZipFile(file_name,'r')
   #     for file in f.namelist():
    #        f.extract(file)
            