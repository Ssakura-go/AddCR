import numpy as np
import cv2
from PIL import Image
from PIL import ImageDraw
from PIL import ImageFont

root= '/home/lyh/Source/CGI2023/zmnwd_00301.png'
img = cv2.imread(root)
print(img.shape)
w = int(img.shape[1] / 2)
img = np.pad(img, ((0,50),(0,0),(0,0)),'constant',constant_values=(0))
h = int(img.shape[0])

  
#  cv2.putText(inp,"Input", (w-40,h-20), cv2.FONT_HERSHEY_COMPLEX, 1,(255,255,255), 1, cv2.LINE_AA)
#  cv2.putText(inp,"Denoised", (w-75,h-20), cv2.FONT_HERSHEY_COMPLEX, 1,(255,255,255), 1, cv2.LINE_AA)
#  cv2.putText(inp,"Super-resolved", (w-135,h-20), cv2.FONT_HERSHEY_COMPLEX, 1,(255,255,255), 1, cv2.LINE_AA)
#  cv2.putText(inp,"Remastered", (w-105,h-20), cv2.FONT_HERSHEY_COMPLEX, 1,(255,255,255), 1, cv2.LINE_AA)

cv2.putText(img,"Remastered", (w-105,h-20), cv2.FONT_HERSHEY_COMPLEX, 1,(255,255,255), 1, cv2.LINE_AA)
save_path = root.replace('.png', '_text.png')
cv2.imwrite(save_path, img)