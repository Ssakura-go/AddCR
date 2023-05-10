import os
import subprocess
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--input", type=str, default='/home/lyh/Source/CGI2023/inputs/')
parser.add_argument("--output", type=str, default='/home/lyh/Source/CGI2023/results/')
parser.add_argument("--tmp", type=str, default='/home/lyh/Source/CGI2023/doriginal/')
parser.add_argument("--cuda_num", type=str, default='0')
args = parser.parse_args()

os.makedirs(args.tmp, exist_ok=True)

denoise_cmd = 'CUDA_VISIBLE_DEVICES=' + args.cuda_num + ' python denoise_test.py --input '\
              + args.input + ' --output ' + args.output
cp_cmd = 'cp -r ' + args.output + '* ' + args.tmp
mv_cmd = 'mv ' + args.output + '* ' +  args.input

sr_cmd = 'CUDA_VISIBLE_DEVICES=' + args.cuda_num + ' python sr_test.py --input '\
              + args.input + ' --output ' + args.output
color_cmd = 'CUDA_VISIBLE_DEVICES=' + args.cuda_num + ' python color_test.py --input '\
              + args.input + ' --output ' + args.output

subprocess.call(denoise_cmd, shell=True)
subprocess.call(cp_cmd, shell=True)
subprocess.call(mv_cmd, shell=True)
subprocess.call(sr_cmd, shell=True)
subprocess.call(mv_cmd, shell=True)
subprocess.call(color_cmd, shell=True)

