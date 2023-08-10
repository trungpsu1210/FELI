import os
import argparse
import json
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.cuda.amp import autocast, GradScaler
from torch.utils.data import DataLoader
from tensorboardX import SummaryWriter
from tqdm import tqdm
from utils import AverageMeter
from data_loader import PairLoader
from models import *
from models.CCR import *


parser = argparse.ArgumentParser()
parser.add_argument('--model', default='trainFELI_t', type=str, help='train model name')
parser.add_argument('--num_workers', default=16, type=int, help='number of workers')
parser.add_argument('--no_autocast', action='store_false', default=True, help='disable autocast')
parser.add_argument('--save_dir', default='./saved_models/', type=str, help='path to models saving')
parser.add_argument('--data_dir', default='/root/', type=str, help='path to dataset')
parser.add_argument('--log_dir', default='./logs/', type=str, help='path to logs')
parser.add_argument('--dataset', default='Dataset Blur', type=str, help='dataset name')
parser.add_argument('--exp', default='lowlight_blur', type=str, help='experiment setting')
parser.add_argument('--gpu', default='0', type=str, help='GPUs used for training')
parser.add_argument('--resume', type=bool, default=True, help='Continue Train')
args = parser.parse_args()

os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu


def train(train_loader, network, criterion, optimizer, scaler):
	losses = AverageMeter()

	torch.cuda.empty_cache()
	
	network.train()

	for batch in train_loader:
		source_img = batch['source'].cuda()
		target_img = batch['target'].cuda()

		with autocast(args.no_autocast):
			output, R_low, R_high, I_high_3, I_delta_3, lle_gt, I_re = network(source_img, target_img)
			
			main_loss = criterion[1](output, target_img)
			high_loss = criterion[0](R_high*I_high_3, lle_gt)
			mutal_high_loss = criterion[0](R_low*I_high_3, lle_gt)
			mutal_low_loss = criterion[0](R_high*I_delta_3, lle_gt)
			relight_loss = criterion[0](R_low*I_delta_3, lle_gt)
			equal_R_loss = criterion[0](R_low, R_high)
			input_loss = criterion[1](source_img, I_re)

			cr_loss = criterion[2](target_img, output, source_img, I_re)

			loss = main_loss + 0.5*input_loss + 0.2*(relight_loss + high_loss + 0.001*mutal_high_loss + 0.001*mutal_low_loss + 0.01*equal_R_loss) + 0.1*cr_loss

		losses.update(loss.item())

		optimizer.zero_grad()
		scaler.scale(loss).backward()
		scaler.step(optimizer)
		scaler.update()

	return losses.avg


def valid(val_loader, network):
	PSNR = AverageMeter()

	torch.cuda.empty_cache()

	network.eval()

	for batch in val_loader:
		source_img = batch['source'].cuda()
		target_img = batch['target'].cuda()

		with torch.no_grad():		
			output, R_low, R_high, I_high_3, I_delta_3, lle_gt, I_re = network(source_img, target_img)
			output = output.clamp_(-1, 1)		

		mse_loss = F.mse_loss(output * 0.5 + 0.5, target_img * 0.5 + 0.5, reduction='none').mean((1, 2, 3))
		psnr = 10 * torch.log10(1 / mse_loss).mean()
		PSNR.update(psnr.item(), source_img.size(0))

	return PSNR.avg


if __name__ == '__main__':
	
	setting_filename = 'configs.json'
	with open(setting_filename, 'r') as f:
		setting = json.load(f)

	network = eval(args.model.replace('-', '_'))()
	network = nn.DataParallel(network).cuda()

	criterion = []
	criterion.append(nn.L1Loss().cuda())
	criterion.append(nn.SmoothL1Loss().cuda())
	criterion.append(ContrastLoss(ablation=False))

	if setting['optimizer'] == 'adam':
		optimizer = torch.optim.Adam(network.parameters(), lr=setting['lr'])
	elif setting['optimizer'] == 'adamw':
		optimizer = torch.optim.AdamW(network.parameters(), lr=setting['lr'])
	else:
		raise Exception("ERROR: unsupported optimizer") 

	scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_mult=2, last_epoch=-1, 
                                                                     T_0=setting['step_T'],
                                                                     eta_min=setting['lr'] * 1e-3)

	scaler = GradScaler()

	dataset_dir = os.path.join(args.data_dir, args.dataset)
	train_dataset = PairLoader(dataset_dir, 'train', 'train', 
								setting['patch_size'], setting['edge_decay'], setting['only_h_flip'])
	train_loader = DataLoader(train_dataset,
                              batch_size=setting['batch_size'],
                              shuffle=True,
                              num_workers=args.num_workers,
                              pin_memory=True,
                              drop_last=True)
	val_dataset = PairLoader(dataset_dir, 'test', setting['valid_mode'], 
							  setting['patch_size'])
	val_loader = DataLoader(val_dataset,
                            batch_size=int(setting['batch_size']/1),
                            num_workers=args.num_workers,
                            pin_memory=True)

	save_dir = os.path.join(args.save_dir, args.exp)
	os.makedirs(save_dir, exist_ok=True)

	losses = []
	start_epoch = 0
	best_psnr = 0
	psnrs = []

	if args.resume and os.path.exists(os.path.join(save_dir, args.model+'.pth')):
		ckp = torch.load(os.path.join(save_dir, args.model+'.pth'))
		network.load_state_dict(ckp['state_dict'])
		start_epoch = ckp['epoch']
		best_psnr = ckp['best_psnr']
		psnrs = ckp['psnrs']
		print(f'start_step: {start_epoch} continue to train --- from best psnr {best_psnr}')
	else:
		print('==> Start training, current model name: ' + args.model)


	writer = SummaryWriter(log_dir=os.path.join(args.log_dir, args.exp, args.model))
	
	for epoch in tqdm(range(setting['epochs'] + 1)):
			loss = train(train_loader, network, criterion, optimizer, scaler)

			writer.add_scalar('train_loss', loss, epoch)

			scheduler.step()

			if epoch % setting['eval_freq'] == 0:
				avg_psnr = valid(val_loader, network)
				
				writer.add_scalar('valid_psnr', avg_psnr, epoch)
				print(avg_psnr)

				if avg_psnr > best_psnr:
					best_psnr = avg_psnr
					torch.save({'epoch': epoch,
                            'best_psnr': best_psnr,
                            'psnrs': psnrs,
                            'state_dict': network.state_dict()},
                           os.path.join(save_dir, args.model + '_' + str(epoch)+'.pth'))
					
					print(f'\n Models saved at epoch: {epoch} | best_psnr: {best_psnr:.4f}')
				
				writer.add_scalar('best_psnr', best_psnr, epoch)
