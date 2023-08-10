import os
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from collections import OrderedDict
from utils import AverageMeter, write_img, chw_to_hwc
from data_loader import SingleLoader
from models import *
import cv2
import pyiqa

parser = argparse.ArgumentParser()
parser.add_argument('--model_test', default='testFELI_t', type=str, help='test model name')
parser.add_argument('--model_train', default='trainFELI_t', type=str, help='train model name')
parser.add_argument('--num_workers', default=16, type=int, help='number of workers')
parser.add_argument('--data_dir', default='/root/', type=str, help='path to dataset')
parser.add_argument('--save_dir', default='./saved_models/', type=str, help='path to models saving')
parser.add_argument('--result_dir', default='./results/', type=str, help='path to results saving')
parser.add_argument('--dataset', default='Dataset Blur', type=str, help='dataset name')
parser.add_argument('--testset', default='realworld_test', type=str, help='testset name')
parser.add_argument('--exp', default='lowlight_blur', type=str, help='experiment setting')
args = parser.parse_args()

def check_image_size(x, down_factor):
    _, _, h, w = x.size()
    mod_pad_h = (down_factor - h % down_factor) % down_factor
    mod_pad_w = (down_factor - w % down_factor) % down_factor
    x = F.pad(x, (0, mod_pad_w, 0, mod_pad_h), 'reflect')
    return x

def single(save_dir):
	state_dict = torch.load(save_dir)['state_dict']
	
	moduleA_state_dict = {k: v for k, v in state_dict.items() if k.startswith('module.LRNet.main_LRNet')}
		
	new_state_dict = OrderedDict()

	for k, v in moduleA_state_dict.items():
		name=k[24:]
		new_state_dict[name] = v

	return new_state_dict


def predict(test_loader, network, result_dir, niqe, nrqm):

	NIQE = AverageMeter()
	NRQM = AverageMeter()

	torch.cuda.empty_cache()

	network.eval()

	os.makedirs(os.path.join(result_dir, 'predict_imgs'), exist_ok=True)

	for idx, batch in enumerate(test_loader):
		input = batch['img'].cuda()
		filename = batch['filename'][0]


		with torch.no_grad():

			torch.cuda.empty_cache()

			H, W = input.shape[2:]
			input = check_image_size(input, down_factor)
			output, _, _ = network(input)
					
			output = output.clamp_(-1, 1)	
	
			# [-1, 1] to [0, 1]
			output = output * 0.5 + 0.5

			output = output[:,:,:H,:W]

			niqe_metric = float(niqe(output))
			nrqm_metric = float(nrqm(output))


		NIQE.update(niqe_metric)
		NRQM.update(nrqm_metric)


		print('Test: [{0}]\t'
			  'NIQE: {niqe.val:.04f} ({niqe.avg:.04f})\t'
			  'NRQM: {nrqm.val:.04f} ({nrqm.avg:.04f})\t'
			  'Name: {filename}'.format(idx, niqe=NIQE, nrqm=NRQM, filename=filename))

		out_img = chw_to_hwc(output.detach().cpu().squeeze(0).numpy())
		write_img(os.path.join(result_dir, 'predict_imgs', filename), out_img)


if __name__ == '__main__':
	network = eval(args.model_test)()
	network.cuda()
	saved_model_dir = os.path.join(args.save_dir, args.exp, args.model_train + '.pth')

	down_factor = 8

	if os.path.exists(saved_model_dir):
		print('==> Start predicting, current model name: ' + args.model_test)
		network.load_state_dict(single(saved_model_dir))
	else:
		print('==> No existing trained model!')
		exit(0)

	niqe = pyiqa.create_metric('niqe', device=torch.device("cuda"))
	nrqm = pyiqa.create_metric('nrqm', device=torch.device("cuda"))

	root_dir = os.path.join(args.data_dir, args.dataset, args.testset)
	test_dataset = SingleLoader(root_dir)
	test_loader = DataLoader(test_dataset,
							 batch_size=1,
							 num_workers=args.num_workers,
							 pin_memory=True)


	result_dir = os.path.join(args.result_dir, args.dataset, args.model_train)
	predict(test_loader, network, result_dir, niqe, nrqm)
