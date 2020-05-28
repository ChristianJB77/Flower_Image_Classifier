'''Save and load function for trained model's checkpoints'''
#Library imports
import numpy as np
import os

import torch
import torchvision
from torchvision import datasets, transforms, models

import argparse

'''Save checkpoint function'''
def save_cp(model, optimizer, hidden_units, arch, classifier, epochs, learning_rate):
    cp = {'input_size': 25088,
      'output_size': 102,
      'hidden_units': hidden_units,
      'arch': arch,
      'batch_size': 64,
      'classifier': classifier,
      'epochs': epochs,
      'learning_rate': learning_rate,
      'print_steps': 1,
      'class_to_idx': model.class_to_idx,  #Map Flower id's to names
      'state_dict': model.state_dict(),
      'optimizer': optimizer.state_dict()}

    torch.save(cp, 'cp.pth')

'''Load checkpoint function'''
def load_cp(file):
    cp = torch.load(file, map_location=lambda storage, loc: storage)
    learning_rate = cp['learning_rate']
    model = getattr(torchvision.models, cp['arch'])(pretrained='True')
    model.classifier = cp['classifier']
    model.epochs = cp['epochs']
    model.load_state_dict(cp['state_dict'])
    model.print_steps = cp['print_steps']
    model.class_to_idx = cp['class_to_idx']

    return model
