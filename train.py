'''Training of new ntwork architectures'''
#Library imports
import numpy as np

import torch
from torch import nn
from torch import optim
import torch.nn.functional as F
import torchvision
from torchvision import datasets, transforms, models

from time import time
import argparse

'''Import own functions'''
from utility import name_dict, flower_data
from save_load import save_cp


'''Terminal command line input functions'''
def get_input_args():
    parser = argparse.ArgumentParser(description = 'Training input options')
    parser.add_argument('--dir', action = 'store', help = 'Directory to save checkpoint')
    parser.add_argument('--arch', dest = 'arch', default = 'vgg13', choices = ['vgg16', 'vgg13'], help = 'CNN model architecture')
    parser.add_argument('--learning_rate', dest = 'learning_rate', default = '0.001', help = 'Learning rate')
    parser.add_argument('--hidden_units', dest = 'hidden_units', default = '2048', help = 'Hidden units')
    parser.add_argument('--epochs', dest = 'epochs', default = '5', help = 'Epochs')
    parser.add_argument('--gpu', action = 'store_true', default = True)

    return parser.parse_args()

'''Training model'''
def train_model(model, criterion, optimizer, epochs, loaders, gpu):
    #Use GPU, if available and user input on GPU
    #Move model to GPU (if available)
    device = torch.device("cuda" if torch.cuda.is_available() and gpu == True else "cpu")
    model.to(device);

    #Train model classifier parameters
    steps = 0
    running_loss = 0
    print_steps = 50

    #Start time calculation
    start_time = time()

    #1. Training
    for e in range(epochs):
        for images, labels in loaders[0]:#loads complete batch
        #labels are only an index now, to be translated by dict later
        #Train classifier
            model.train()
            steps += 1
            #Data to GPU
            images, labels = images.to(device), labels.to(device)
            #Clear gradients
            optimizer.zero_grad()
            #Forward pass
            log_ps = model(images)
            #Loss calculation (y_hat, y)
            loss = criterion(log_ps, labels)
            #Backward pass
            loss.backward()
            #Update classifier weights
            optimizer.step()
            #Add loss
            running_loss += loss.item()


    #2. Validation
            if steps % print_steps == 0:
                model.eval()
                valid_loss = 0
                accuracy = 0
                #Validation = no gradient descent update
                #Turn off droput for validation
                with torch.no_grad():
                    for images, labels in loaders[1]:
                        #Move data to GPU
                        images, labels = images.to(device), labels.to(device)
                        #Forward pass
                        log_ps = model(images)
                        #Test loss, no backward pass, therefore direct update
                        valid_loss += criterion(log_ps, labels)
                        #Get propability of LofSoftmax
                        ps = torch.exp(log_ps)
                        #Get most likely class
                        top_p, top_class = ps.topk(1, dim=1)
                        #Prediction validation
                        equals = top_class == labels.view(*top_class.shape)
                        #Accuracy mean of batch
                        accuracy += torch.mean(equals.type(torch.FloatTensor)).item()


                #Total time
                end_time = time()
                total_time = end_time - start_time

                print("Time: {:.1f}s for {} batch_size".format(total_time, print_steps),
                      "Epoch {}/{}".format(e+1, epochs),
                     "Training loss: {:.3f}".format(running_loss / print_steps),
                     "Validation_loss: {:.3f}".format(valid_loss / len(loaders[1])),
                     "Accuracy: {:.1f}%".format(accuracy*100 / len(loaders[1])))
                #Set running loss back to 0
                running_loss = 0
                model.train()


'''main function'''
def main():
    #Get command line input arguments
    in_arg = get_input_args()
    gpu = in_arg.gpu
    dir = in_arg.dir
    arch = in_arg.arch
    learning_rate = float(in_arg.learning_rate)
    hidden_units = int(in_arg.hidden_units)
    epochs = int(in_arg.epochs)
    #Device agnostic code to use CUDA automatically, if available
    device = torch.device("cuda" if torch.cuda.is_available() and gpu == True else "cpu")
    #Get flower data
    image_data, loaders = flower_data()

    #Get pre-trained model
    model = getattr(torchvision.models, arch)(pretrained='True')
    #Pre-trained CNN, only update of new classifier features
    for param in model.parameters():
        param.requires_grad = False

    #Flower classifier with 2 possible CNN architectures: VGG13 & VGG16
    #Both architecture have an equal numbe rof input features: 25088
    classifier = nn.Sequential(
                    nn.Linear(25088, hidden_units), nn.ReLU(), nn.Dropout(p=0.5),
                    nn.Linear(hidden_units, 102), nn.LogSoftmax(dim=1))
    
    #Update classifier of pre-trained model
    model.classifier = classifier
    #Train new classifier of pre-trained model with frozen feature paramters
    criterion = nn.NLLLoss()
    optimizer = optim.Adam(model.classifier.parameters(), lr=learning_rate)
    
    #Move model to GPU (if available)
    model.to(device)
    
    #Call train function
    train_model(model, criterion, optimizer, epochs, loaders, gpu)
    #Mapping of classes to indices
    model.class_to_idx = image_data[0].class_to_idx
    #Save checkpoint >> save_load.py
    save_cp(model, optimizer, hidden_units, arch, classifier, epochs, learning_rate)

# Call to main function to run the program
if __name__ == "__main__":
    main()
