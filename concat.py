import os
import glob
import numpy as np
import cv2
import argparse
import subprocess

parser = argparse.ArgumentParser()
parser.add_argument("--input", type=str, default='/home/lyh/Source/CGI2023/original/')
args = parser.parse_args()

def get_image_list(root):
  ppath = os.path.join(root, '*.png')
  jpath = os.path.join(root, '*.jpg')
  return list(glob.glob(ppath) + glob.glob(jpath))
  
def get_first_image(root):
  ppath = os.path.join(root, '*_00001.png')
  jpath = os.path.join(root, '*_00001.jpg')
  return list(glob.glob(ppath) + glob.glob(jpath))
  
lists = get_image_list(args.input)

save_folder = args.input.replace('original/', 'foutput/')
os.makedirs(save_folder, exist_ok=True)

save_video_folder = args.input.replace('original/', 'fvideo/')
os.makedirs(save_video_folder, exist_ok=True)

for image in lists:
  denoise_path = image.replace('original/', 'doriginal/')
  sr_path = image.replace('original/', 'inputs/')
  fou_path = image.replace('original/', 'results/')
  save_path = image.replace('original/', 'foutput/')
  inp = cv2.imread(image)
  denoise = cv2.imread(denoise_path)
  sr = cv2.imread(sr_path)
  fou = cv2.imread(fou_path)
  
  w = int(fou.shape[1])
  h = int(fou.shape[0])
  inp = cv2.resize(inp, (w, h), interpolation=cv2.INTER_CUBIC)
  denoise = cv2.resize(denoise, (w, h), interpolation=cv2.INTER_CUBIC)
  
  # w = int((fou.shape[1] - inp.shape[1]) / 2)
  # h = int((fou.shape[0] - inp.shape[0]) / 2) 
  # inp = np.pad(inp, ((h,h),(w,w),(0,0)), 'constant', constant_values=(0,0))
  
  inp = np.pad(inp, ((20,20),(50,20),(0,0)), 'constant', constant_values=(0,0))
  denoise = np.pad(denoise, ((20,20),(20,50),(0,0)), 'constant', constant_values=(0,0))
  sr = np.pad(sr, ((20,20),(50,20),(0,0)), 'constant', constant_values=(0,0))
  fou = np.pad(fou, ((20,20),(20,50),(0,0)), 'constant', constant_values=(0,0))
  
  # w = int(inp.shape[1] / 2)
  # h = int(inp.shape[0])
  
  # cv2.putText(inp,"Input", (w-40,h-20), cv2.FONT_HERSHEY_COMPLEX, 1.5,(255,255,255), 3, cv2.LINE_AA)
  # cv2.putText(denoise,"Denoised", (w-75,h-20), cv2.FONT_HERSHEY_COMPLEX, 1.5,(255,255,255), 3, cv2.LINE_AA)
  # cv2.putText(sr,"Super-resolved", (w-135,h-20), cv2.FONT_HERSHEY_COMPLEX, 1.5,(255,255,255), 3, cv2.LINE_AA)
  # cv2.putText(fou,"Remastered", (w-105,h-20), cv2.FONT_HERSHEY_COMPLEX, 1.5,(255,255,255), 3, cv2.LINE_AA)
  
  out_1 = np.concatenate((inp, denoise), axis=1)
  out_2 = np.concatenate((sr, fou), axis=1)
  out = np.concatenate((out_1, out_2), axis=0)
  
  # out_1 = np.concatenate((inp,sr), axis=1)
  # out = np.concatenate((out_1,fou), axis=1)
  
  # out = np.concatenate((inp, fou), axis=1)
  cv2.imwrite(save_path, out)
  
video_lists = get_first_image(save_folder)
for image in video_lists:
  ext = image.split('.')[-1]
  name = image.split('/')[-1]
  img_path = image.replace(name, '')
  name = name.split('_')[0]
  
  cmd = "ffmpeg -r 20 -f image2 -i " + img_path + name + '_%05d.' + ext + ' -vcodec libx264 -crf 15 -pix_fmt yuv420p ' + save_video_folder + name + '.mp4'
  subprocess.call(cmd, shell=True)
  

  







  

  
  
  

   