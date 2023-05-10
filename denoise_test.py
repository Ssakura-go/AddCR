# -*- coding: gbk -*-
import argparse
import os
import copy
import torch
import cv2
import numpy as np
import glob
from torch import nn
from baseunet import Unet
from torch.autograd import Variable

# set the parameters
parser = argparse.ArgumentParser()
parser.add_argument("--input", type=str, default='/home/lyh/Source/CGI2023/original/')
parser.add_argument("--output", type=str, default='/home/lyh/Source/CGI2023/doriginal/')
args = parser.parse_args()


def get_image_list(root):
    ppath = os.path.join(root, '*.png')
    jpath = os.path.join(root, '*.jpg')
    return list(glob.glob(ppath) + glob.glob(jpath))


tests_path = get_image_list(args.input)
save_path_folder = args.output + '/'
if not os.path.exists(save_path_folder):
    os.makedirs(save_path_folder)
len = len(tests_path)
print("len:{}".format(len))

if __name__ == "__main__":
    print("             *      *                ")
    print("              *    *                 ")


    # configuration of device and set the dp for network
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    torch.cuda.manual_seed_all(123)
    net = Unet().to(device=device, dtype=torch.float32)
    net = nn.DataParallel(net, device_ids=[0])

    # count the number of parameters
    # number_parameters = sum(map(lambda x: x.numel(), net.parameters()))
    # print(number_parameters)

    # load the pre-trained model
    checkpoint = torch.load('weights/multiunetThree_epoch_999.pth', map_location=device)
    net.load_state_dict(checkpoint['model_state_dict'])

    # set the evaluation mode
    net.eval()
    sum = 0.0

    # compute the avg time-comsuimg: step 1
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)


    for test_path in tests_path:

        name = test_path.split('/')[-1]

        img = cv2.imread(test_path).astype(np.float32)

        # the size of images should be 32x, if not, pad automatically
        w = float(img.shape[1])
        h = float(img.shape[0])
        ww = w / 32
        wwr = w // 32
        padw = 0
        padh = 0
        if ww != wwr:
            padw = int((wwr + 1 - ww) * 16)
        hh = h / 32
        hhr = h // 32
        if hh != hhr:
            padh = int((hhr + 1 - hh) * 16)
        img = np.pad(img, ((padh, padh), (padw, padw), (0, 0)), 'edge')

        # normalize and convert array to tensor
        img = cv2.normalize(img, None, 0, 1, cv2.NORM_MINMAX).transpose(2, 0, 1)
        img_tensor = torch.from_numpy(img).to(device)
        img_tensor = Variable(torch.unsqueeze(img_tensor, dim=0).float(), requires_grad=False)

        # compute the avg time-comsuimg: step 2
        start.record()
        pred_image, pred_line = net(img_tensor)
        end.record()

        # compute the avg time-comsuimg: step 3
        torch.cuda.synchronize()
        sum += start.elapsed_time(end)

        # convert tensor to array (must be convert cuda to cpu first)
        pred_image = Variable(torch.squeeze(pred_image, dim=0).float(), requires_grad=False)
        pred_line = Variable(torch.squeeze(pred_line, dim=0).float(), requires_grad=False)
        pred_image = pred_image.cpu()
        pred_line = pred_line.cpu()
        result = pred_image.numpy()

        result = result.transpose(1,2,0)
        result = result * 255.0

        # when the size of image is not 32x, we have to crop it to the original size
        result = result[int(padh):int(result.shape[0]-padh), int(padw):int(result.shape[1]-padw), :]
        save_path = save_path_folder + name
        cv2.imwrite(save_path, result)



    # compute the avg time-comsuimg: step 3
    avgtime = sum / len
    avgtime /= 1000.0
    print("*- Denoising {}-*".format(avgtime))
    print("                |                    ")
    print("                |                    ")
