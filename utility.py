'''Utility code for data loading and processing'''
#Library imports
import numpy as np

import torch
from torch import nn
from torch import optim
import torch.nn.functional as F
import torchvision
from torchvision import datasets, transforms, models

import json

'''Data loading and transformation'''
def flower_data():
    #Data locations
    data_dir = 'flowers'
    train_dir = data_dir + '/train'
    valid_dir = data_dir + '/valid'
    test_dir = data_dir + '/test'

    #Transform data to Pytorch tensors
    train_trans = transforms.Compose([transforms.RandomRotation(30),
                                      transforms.RandomResizedCrop(224),
                                      transforms.RandomHorizontalFlip(),
                                      transforms.ToTensor(),
                                      transforms.Normalize([0.485, 0.456, 0.406],
                                                          [0.229, 0.224, 0.225])])

    valid_trans = transforms.Compose([transforms.Resize(255),
                                    transforms.CenterCrop(224),
                                    transforms.ToTensor(),
                                    transforms.Normalize([0.485, 0.456, 0.406],
                                                          [0.229, 0.224, 0.225])])

    test_trans = transforms.Compose([transforms.Resize(255),
                                    transforms.CenterCrop(224),
                                    transforms.ToTensor(),
                                    transforms.Normalize([0.485, 0.456, 0.406],
                                                          [0.229, 0.224, 0.225])])

    #Load data
    train_data = datasets.ImageFolder(data_dir + '/train', transform=train_trans)
    valid_data = datasets.ImageFolder(data_dir + '/valid', transform=valid_trans)
    test_data = datasets.ImageFolder(data_dir + '/test', transform=test_trans)

    #Generator for batch iteration
    batch = 64
    trainloader = torch.utils.data.DataLoader(train_data, batch_size=batch, shuffle=True)
    validloader = torch.utils.data.DataLoader(valid_data, batch_size=batch)
    testloader = torch.utils.data.DataLoader(test_data, batch_size=batch)

    #Combined data lists
    image_data = [train_data, valid_data, test_data]
    loaders = [trainloader, validloader, testloader]

    return image_data, loaders

'''Name dict'''
def name_dict(category_names):
    #Save flower dict
    with open(category_names, 'r') as f:
        cat_to_name = json.load(f)
    return cat_to_name
