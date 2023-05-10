import glob
import os
import time
import torch
import cv2
import numpy as np
import argparse
from resblock import Model
from torch import nn
from torch.autograd import Variable

parser = argparse.ArgumentParser()
parser.add_argument("--input", type=str, default='/home/lyh/Source/CGI2023/inputs/')
parser.add_argument("--output", type=str, default='/home/lyh/Source/CGI2023/results/')
args = parser.parse_args()

def get_image_list(root):
    ppath = os.path.join(root, '*.png')
    jpath = os.path.join(root, '*.jpg')
    return list(glob.glob(ppath) + glob.glob(jpath))

test_path = get_image_list(args.input)
len = len(test_path)
save_path_folder = args.output
os.makedirs(save_path_folder, exist_ok=True)

if __name__ == '__main__':

    os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    torch.cuda.manual_seed_all(123)

    model = Model().to(device=device, dtype=torch.float32)
    number_parameters = sum(map(lambda x: x.numel(), model.parameters()))
    # print(number_parameters)
    model = nn.DataParallel(model, device_ids=[0])

    # plan-1
    checkpoint = torch.load('/home/lyh/Source/CGI2023/weights/epoch_30.pth', map_location=device)

    model.load_state_dict(checkpoint['model_state_dict'])
    with torch.no_grad():
        model.eval()
    sum = 0.0

    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)

    for img_path in test_path:
        name = img_path.split('/')[-1]
        img = cv2.imread(img_path).astype(np.float32)

        img = cv2.normalize(img, None, 0, 1, cv2.NORM_MINMAX).transpose(2,0,1)
        img_tensor = torch.from_numpy(img).to(device)
        img_tensor = Variable(torch.unsqueeze(img_tensor, dim=0).float(), requires_grad=False)

        start.record()
        pred_image, pred_line = model(img_tensor)
        end.record()

        torch.cuda.synchronize()
        sum += start.elapsed_time(end)

        pred_image = Variable(torch.squeeze(pred_image, dim=0).float(), requires_grad=False)
        pred_image = pred_image.cpu()
        result = pred_image.numpy()
        result = result.transpose(1,2,0)
        result = result * 255.0

        save_path = save_path_folder + name
        cv2.imwrite(save_path, result)

    avgtime = sum / len
    avgtime /= 1000.0
    print("*----Color {}---*".format(avgtime))
    print("                |                    ")
    print("                |                    ")
    print("                |                    ")
    print("                |                    ")
    print('\n')
    print("               Done!                 ")
