# Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# This code is based on https://github.com/craston/MARS/blob/master/utils1/extract_frames.py

'''
For HMDB51 and UCF101 datasets:

Code extracts frames from video at a rate of 25fps and scaling the
larger dimension of the frame is scaled to 256 pixels.
After extraction of all frames write a "done" file to signify proper completion
of frame extraction.

Usage:
  python extract_frames.py video_dir frame_dir
  
  video_dir => path of video files
  frame_dir => path of extracted jpg frames

'''
#paddle
import sys, os, pdb
import numpy as np
import subprocess
from tqdm import tqdm
       

def extract(vid_dir, frame_dir, start, end, redo=False):
  class_list = sorted(os.listdir(vid_dir))[start:end]

  print("Classes =", class_list)
  
  for ic,cls in enumerate(class_list): 
    vlist = sorted(os.listdir(vid_dir + cls))
    print("")
    print(ic+1, len(class_list), cls, len(vlist))
    print("")
    for v in tqdm(vlist):
      outdir = os.path.join(frame_dir, cls, v[:-4])
      
      # Checking if frames already extracted
      if os.path.isfile( os.path.join(outdir, 'done') ) and not redo: continue
      try:  
        os.system('mkdir -p "%s"'%(outdir))
        # check if horizontal or vertical scaling factor
        o = subprocess.check_output('ffprobe -v error -show_entries stream=width,height -of default=noprint_wrappers=1 "%s"'%(os.path.join(vid_dir, cls, v)), shell=True).decode('utf-8')
        lines = o.splitlines()
        width = int(lines[0].split('=')[1])
        height = int(lines[1].split('=')[1])
        resize_str = '-1:256' if width>height else '256:-1'

        # extract frames
        os.system('ffmpeg -i "%s" -r 25 -q:v 2 -vf "scale=%s" "%s"  > /dev/null 2>&1'%( os.path.join(vid_dir, cls, v), resize_str, os.path.join(outdir, '%05d.jpg')))
        nframes = len([ fname for fname in os.listdir(outdir) if fname.endswith('.jpg') and len(fname)==9])
        if nframes==0: raise Exception 

        os.system('touch "%s"'%(os.path.join(outdir, 'done') ))
      except:
        print("ERROR", cls, v)

if __name__ == '__main__':
  vid_dir   = sys.argv[1]
  frame_dir = sys.argv[2]
  start     = int(sys.argv[3])
  end       = int(sys.argv[4])
  extract(vid_dir, frame_dir, start, end, redo=True)