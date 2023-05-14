# AddCR
This repository contains the source code for AddCR: A Data-Driven Cartoon Remastering

# Requirements
This code has been tested with Pytorch 1.8.1 and Cuda 11.1

```
conda create -n addcr python==3.8.3

conda activate addcr

pip install torch==1.8.1+cu111 torchvision==0.9.1+cu111 torchaudio==0.8.1 -f https://download.pytorch.org/whl/torch_stable.html

pip install basicsr opencv-python numpy

python setup.py develop
```

# Demos
You can demo a trained model on a sequence of frames

```
python test.py --cuda_num 0 --input /your-own-path/inputs/ --output /your-own-path/results/

```
--cuda_num: the number of your cuda

--input: the path on which you stored the unprocessed frames

--output: the path of final output
  
# TODO: Thanks for xintao 


