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


class fineTune():

	def __init__(self, model_name, num_classes, batch_size, num_epochs):

		self.model_name = model_name
		self.num_classes = num_classes
		self.batch_size = batch_size
		self.num_epochs = num_epochs

	def set_parameters(self, tune_all_params):
		if not tune_all_params:
			for param in self.model_.parameters():
				param.requires_grad = False
	
	def get_model(self):
		print(self.model)

	def init_model(self, tune_all_params = True, from_scratch = True):
		self.model = None
		input_size = 0

		if self.model_name = 'resnet':
			self.model = models.resnet18(pretrained=not from_scratch)
			set_parameters(tune_all_params)
			num_features = self.model.fc.in_features
			self.model.fc = nn.Linear(num_features, self.num_classes)
			input_size = 224
		elif self.model_name = 'vgg':
			self.model = models.vgg16_bn(pretrained=not from_scratch)
			set_parameters(tune_all_params)
			num_features = self.model.classifier[6].in_features
			self.model.classifier[6] = nn.Linear(num_features, num_classes)
			input_size = 224
		elif self.model_name = 'densenet':
			self.model = models.densenet121(pretrained=not from_scratch)
			set_parameters(tune_all_params)
			num_features = self.model.classifier.in_features
			self.model.classifier = nn.Linear(num_features, num_classes)
			input_size = 224
		else
			print("Invalid model name, exiting.")
			exit()
			
# PATHS
DATA_DIR = '/home/cancam/imgworkspace/gradcam_plus_plus-pytorch/data/coco/fine-tune'
MODEL_DIR = '/home/cancam/imgworkspace/gradcam_plus_plus-pytorch/fine-tuning/models'

