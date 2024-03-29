# Fast and Physically Enriched Deep Network for Joint Low-light Enhancement and Image Restoration - (FELI) #

This code originally is proposed for FELI for single image deblurring in low-light conditions. 

## Overview of FELI ##

![alt text](https://github.com/trungpsu1210/FELI/blob/main/Model.jpg)

## Structure of the code ##

1. models: Folders for proposed networks (FELI)
2. train.py: to train the models
3. test.py: to test the quantitative results (PSNR, SSIM, LPIPS (vgg))
4. predict.py: to generate the clear images and evaluate (NIQE and NRQM)
5. data_loader.py: to load the pair input and output data or single images
6. CCR.py: vgg loss function
7. configs.json: configuration parameter to train, test, and predict
8. utils: useful function using in the code

## Requirments ##

Pytorch 1.10.2

Python 3.7

Cudatoolkit=11.3

Deep learning libraries/frameworks: OpenCV, TensorBoard, timm, torchvision, pytorch_msssim...

To run the code, make sure all the files are in the corresponding folders and install the requirements.txt

## Citation ##

If you find this method useful for your work/research, please cite our paper:

```
@inproceedings{icassp2024TrungpsuFELI,
  author={Hoang, Trung and McElvain, Jon and Monga, Vishal},
  booktitle={2024 IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP)},
  title={Fast and Physically Enriched Deep Network for Joint Low-light Enhancement and Image Deblurring}, 
  year={2024}
  }
```
