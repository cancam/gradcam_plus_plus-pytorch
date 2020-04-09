import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
from torchvision import datasets, models, transforms

import numpy as np
import time
import os
import copy
import pdb

from mmcv.parallel import MMDataParallel

# PATHS
DATA_DIR = '/home/cancam/imgworkspace/gradcam_plus_plus-pytorch/data/coco/fine-tune'
MODEL_DIR = '/home/cancam/imgworkspace/gradcam_plus_plus-pytorch/fine-tuning/models'

class fineTune():

	def __init__(self, work_dir, model_name, num_classes, batch_size, input_size = 224):
		
		self.model = None
		self.params_to_update = None		
		self.model_name = model_name
		self.num_classes = num_classes
		self.batch_size = batch_size
		self.input_size = input_size
		self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
		
		self.data_dir = os.path.join(work_dir, 'data/coco/fine_tune')
		self.save_dir = os.path.join(work_dir, 'fine-tuning/models')

		self.data_transforms = {
			'train': transforms.Compose([
			transforms.RandomResizedCrop(input_size),
			transforms.RandomHorizontalFlip(),
			transforms.ToTensor(),
			transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
			]),
			'val': transforms.Compose([
			transforms.Resize(input_size),
			transforms.CenterCrop(input_size),
			transforms.ToTensor(),
			transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
			]),
		}
		self.init_dataloaders(os.path.join(work_dir, self.data_dir))

	def set_params(self, tune_all_params):
		if not tune_all_params:
			for param in self.model.parameters():
				param.requires_grad = False

	def get_params(self, tune_all_params):
		self.params_to_update = self.model.parameters()		
		if not tune_all_params:
			self.params_to_update = []			
			for name, param in self.model.named_parameters():
				if param.requires_grad == True:
					self.params_to_update.append(param)
					print("\t", name)
		else:
			for name, param in self.model.named_parameters():
				if param.requires_grad == True:
					print("\t", name) 
		
	def get_model(self):
		print(self.model)

	def init_model(self, tune_all_params, from_scratch = True):
	##set model whether pre-trained or not and set parameters of the model to
	##train.
		input_size = 0
		#pdb.set_trace()
		if self.model_name=='resnet':
			self.model = models.resnet50(pretrained=not from_scratch)
			self.set_params(tune_all_params)
			num_features = self.model.fc.in_features
			self.model.fc = nn.Linear(num_features, self.num_classes)
			input_size = 224
		elif self.model_name=='vgg':
			self.model = models.vgg16_bn(pretrained=not from_scratch)
			self.set_params(tune_all_params)
			num_features = self.model.classifier[6].in_features
			self.model.classifier[6] = nn.Linear(num_features, self.num_classes)
			input_size = 224
		elif self.model_name=='densenet':
			self.model = models.densenet121(pretrained=not from_scratch)
			self.set_params(tune_all_params)
			num_features = self.model.classifier.in_features
			self.model.classifier = nn.Linear(num_features, self.num_classes)
			input_size = 224
		else:
			print("Invalid model name, exiting.")
			exit()

		# send model to processing device.
		self.model = self.model.cuda()
                #self.model = MMDataParallel(self.model, device_ids=range(1)).cuda()
		self.get_params(tune_all_params)

	def init_dataloaders(self, data_path):
		dataset = {x: datasets.ImageFolder(os.path.join(data_path, x), \
				   self.data_transforms[x]) \
				   for x in ['train', 'val']}
		self.dataloaders = {x: torch.utils.data.DataLoader(dataset[x], \
						   batch_size = self.batch_size, \
						   shuffle = True) \
                           for x in ['train', 'val']}
	def train_model(self, lr=0.01, decay_factor=0.1, epochs=15, momentum = 0.9):
		# get optimizer
		optimizer = optim.SGD(self.params_to_update, lr, momentum)
		criterion = nn.CrossEntropyLoss()
		# get current time		
		val_acc_history = []
		best_model = copy.deepcopy(self.model.state_dict())
		best_acc = 0.0
		
		for epoch in range(epochs):
			print('Epoch {}/{}'.format(epoch, epochs-1))
			if epoch == 30 or epoch == 60:
				for g in optimizer.param_groups:
					g['lr'] = g['lr'] * decay_factor
				print('LR*{}'.format(decay_factor))
		
			for phase in ['train', 'val']:
				if phase == 'train':
					self.model.train()
				else:
					self.model.eval()
				
				running_loss = 0.0
				running_corrects = 0
				# get current time for current epoch.
				since = time.time()
				for inputs, labels in self.dataloaders[phase]:
					inputs = inputs.to(self.device)
					labels = labels.to(self.device)

					optimizer.zero_grad()
					with torch.set_grad_enabled(phase == 'train'):
						outputs = self.model(inputs)
						loss = criterion(outputs, labels)
						_, preds = torch.max(outputs,1)
						if phase == 'train':
							loss.backward()
							optimizer.step()
					running_loss += loss.item() * inputs.size(0)
					running_corrects += torch.sum(preds == labels.data)
				# compute epoch stats.
				time_elapsed = time.time()-since
				epoch_loss = running_loss / len(self.dataloaders[phase].dataset)
				epoch_acc = running_corrects.double() / len(self.dataloaders[phase].dataset)
				
				print('{} Loss: {:.4f}, Acc: {:.4f}, Time: {:.4f}'\
					  .format(phase, epoch_loss, epoch_acc, time_elapsed))

				if phase == 'val' and epoch_acc > best_acc:
					model_name = 'epoch_{}.pth'.format(epoch)
					model_name = os.path.join(self.save_dir, model_name)
					best_acc = epoch_acc
					best_model = copy.deepcopy(self.model.state_dict())
					torch.save(self.model.state_dict(), model_name)

				if phase == 'val':
					val_acc_history.append(epoch_acc)
		model_name = 'last.pth'
		model_name = os.path.join(save_dir, model_name)
		torch.save(self.model.state_dict(), model_name)
		
