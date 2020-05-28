#Library Imports
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import os
import random

import torch
from torch import nn
from torch import optim
import torch.nn.functional as F
import torchvision
from torchvision import datasets, transforms, models

import argparse

'''Import own functions'''
from utility import name_dict
from save_load import load_cp

'''Terminal command line input functions'''
def get_input_args():
    parser = argparse.ArgumentParser(description = 'Predict input options')
    parser.add_argument('--cp', action = 'store', default = 'cp.pth',help = 'Load checkpoint')
    parser.add_argument('--filepath', dest = 'filepath', default = None, help = 'Image path')
    parser.add_argument('--top_k', dest = 'top_k', default = '3', help = 'Top K predicitons')
    parser.add_argument('--category_names', dest = 'category_names', default = 'cat_to_name.json', help = 'Flower dict')
    parser.add_argument('--gpu', action = 'store_true', default = True)

    return parser.parse_args()


'''Process image (open, resizes and convert to ndarray)'''
def process_image(image):
    ''' Scales, crops, and normalizes a PIL image for a PyTorch model,
        returns an Numpy array
    '''
    #Open, resize and crop
    img = Image.open(image)
    img = img.resize((256, 256))
    crop_delta = (256-224) / 2
    img = img.crop((crop_delta, crop_delta, 256 - crop_delta, 256 - crop_delta))
    #Color channel normalization
    img = np.array(img)/255 #255 color channels
    img = (img - np.array([0.485, 0.456, 0.406])) / np.array([0.229, 0.224, 0.225])

    return img.transpose(2, 0, 1)


'''Image prediciton'''
def predict(image_path, model, topk, gpu):
    ''' Predict the class (or classes) of an image using a trained deep learning model.
    '''
    #Use GPU, if available and user input on GPU
    device = torch.device("cuda" if torch.cuda.is_available() and gpu == True else "cpu")
    model.to(device);

    #Turn off droput
    model.eval()
    #Tranfsorm image to pytorch tensor
    img = process_image(image_path) #numpy ndarray
    img = torch.from_numpy(np.array([img])).float() #Tensor

    #Forward pass to get top results
    with torch.no_grad():
        #img to GPU(if available)
        img = img.to(device)
        
        log_ps = model(img)
        ps = torch.exp(log_ps).data
        top_ps = torch.topk(ps, topk)[0].tolist()[0]
        top_classes = torch.topk(ps, topk)[1].tolist()[0]

    #Convert indices to class labels (classes)
    #Get index list of model
    index = []
    for i in range(len(model.class_to_idx.items())):
        index.append(list(model.class_to_idx.items())[i][0])
    #Transfer index to classes
    classes = []
    for i in range(topk):
        classes.append(index[top_classes[i]])

    return top_ps, classes

'''main function'''
def main():
    #Get command line input arguments
    in_arg = get_input_args()
    gpu = in_arg.gpu
    cp = in_arg.cp
    image_path = in_arg.filepath
    topk = int(in_arg.top_k)
    category_names = in_arg.category_names
    #Get flower dict
    cat_to_name = name_dict(category_names)

    #Load trained model for prediciiton
    model_prediction = load_cp(cp)

    #Define example random image, if user doesn't select image in command line
    if image_path == None:
        dummy_folder = '20'
        path = 'flowers/test/' + dummy_folder + '/'
        img = random.choice(os.listdir(path))
        image_path = path + str(img)
        top_ps, classes = predict(image_path, model_prediction, topk, gpu)
        dummy = cat_to_name[dummy_folder]
        print('Image for predicition: {}'.format(dummy.title()))
    else:
        top_ps, classes = predict(image_path, model_prediction, topk, gpu)
        print('Image for predicition: {}'.format(image_path))
    #Print topk image classes and its probabilities
    labels = []
    for i in classes:
        label = cat_to_name[i]
        labels.append(label)

    print('Top classes: {}'.format(classes))
    print('Labels: {}'.format(labels))
    print('Probabilities: {}'.format(top_ps))


if __name__ == '__main__':
    main()
