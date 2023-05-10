import os
import numpy as np
import cv2
import glob
import argparse
import subprocess

parser = argparse.ArgumentParser()
parser.add_argument("--folder", type=str, default='/home/lyh/Source/CGI2023/')
args = parser.parse_args()

original_path = args.folder + 'original'
sr_path = args.folder + 'inputs'
fou_path = args.folder + 'results'
tmp_path = args.folder + 'tmp'
save_image_path = args.folder + 'foutput'
save_video_path = args.folder + 'fvideo'

os.makedirs(tmp_path, exist_ok=True)

def get_image_list(root):
  path = os.path.join(root, '*.png')
  return list(glob.glob(path))
  
def get_first_image(root):
  path = os.path.join(root, '*_00001.png')
  return list(glob.glob(path)) 
  
first_file = get_first_image(fou_path)
video_name = first_file[0].split('/')[-1]
video_name = video_name.split('_')[0]

original_lists = get_image_list(original_path)
sr_lists = get_image_list(sr_path)
fou_lists = get_image_list(fou_path)

lists = original_lists + sr_lists + fou_lists

for image in lists:
  name = image.split('/')[-1]
  fold = image.replace('/' + name ,'')
  fold = fold.replace(args.folder,'')
  
  img = cv2.imread(image)

  if fold == 'original':
    save_path = image.replace('original', 'tmp')
    w = int(3 * img.shape[1])
    h = int(3 * img.shape[0])
    img = cv2.resize(img, (w, h), interpolation=cv2.INTER_CUBIC)
    img = img[:, :1622, :]
    cv2.imwrite(save_path, img)
  else:
    img = img[:, :1622, :]
    cv2.imwrite(image, img)
    
 
 
for image in fou_lists:
  original_path = image.replace('results', 'tmp')
  sr_path = image.replace('results', 'inputs')
  save_path = image.replace('results', 'foutput')
  original = cv2.imread(original_path)
  sr = cv2.imread(sr_path)
  fou = cv2.imread(image) 
  
  original = np.pad(original, ((50,50),(20,20),(0,0)), 'constant', constant_values=(0,0))
  sr = np.pad(sr, ((50,50),(20,20),(0,0)), 'constant', constant_values=(0,0))
  fou = np.pad(fou, ((50,50),(20,20),(0,0)), 'constant', constant_values=(0,0))
  
  out_1 = np.concatenate((original,sr), axis=1)
  out = np.concatenate((out_1,fou), axis=1)
  
  cv2.imwrite(save_path, out)
  

cmd = "ffmpeg -r 20 -i " + save_image_path + '/' + video_name + '_%05d.png -vcodec libx264 -crf 15 -pix_fmt yuv420p ' \
+ save_video_path + '/' + video_name + '.mp4'

subprocess.call(cmd, shell=True)
  
    
    
  
  

  
  
  
  
  
  
  

