#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Aug  6 15:24:47 2019

@author: matthew
"""

# bibliiotecas
import numpy as  np
import matplotlib.pyplot as plt
import time
import torch
from torch import nn, optim
from torchvision import datasets, transforms, models
import sys
import random
import os
import pickle

# Set random seem for reproducibility
manualSeed = 999

random.seed(manualSeed)
torch.manual_seed(manualSeed)

# checking GPU
if torch.cuda.is_available():
    device = 'cuda'
else:
    device = 'cpu'
    

# A dictionary with transformations for train, test and validation sets
image_transforms = { 
    'train': transforms.Compose([
        transforms.RandomChoice([transforms.RandomHorizontalFlip(), transforms.RandomAffine(120, shear=0.5),
                                transforms.RandomRotation(120), transforms.ColorJitter(0.5, 0.5, 0.5, 0.5),
                                transforms.RandomVerticalFlip()
                                ]),
        transforms.Resize(size=256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225])
    ]),
    'valid': transforms.Compose([
        transforms.Resize(size=256),
        transforms.CenterCrop(size=224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225])
    ]),
    'test': transforms.Compose([
        transforms.Resize(size=256),
        transforms.CenterCrop(size=224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225])
    ])
}   
 # This function will peform validation in order to avoid overfitting, and print accuracy.
def validate(valid_loss,  model, valid_loader, criterio, optimizer):
    correct_classes = list(np.zeros(len(valid_loader.dataset.classes))) 
    all_classes = list(np.zeros(len(valid_loader.dataset.classes)))
    with torch.no_grad():
        model.eval() # validation mode
        for images, labels in valid_loader:
            images, labels = images.to(device), labels.to(device)
            output = model.forward(images)
            loss = criterio(output, labels)
            valid_loss += loss.item()*images.size(0)
            
            prediction = torch.max(output, 1)[1] 
            correct = prediction == labels       
            for i in range(labels.shape[0]):     
              label = labels.data[i]            
              
              correct_classes[label] += correct[i].item()
              all_classes[label] += 1
            
        valid_loss = valid_loss/len(valid_loader.dataset)
        accuracy = (np.sum(correct_classes) / np.sum(all_classes))
        return valid_loss, accuracy

# training 
def training(train_loss, model, train_loader, criterio, optimizer):
    model.train() # training mode
    correct_classes = list(np.zeros(len(train_loader.dataset.classes))) 
    all_classes = list(np.zeros(len(train_loader.dataset.classes)))
    for data, target in train_loader:
        data, target = data.to(device), target.to(device) 
        optimizer.zero_grad() 
        output = model.forward(data)

        loss = criterio(output, target)
        loss.backward()
        optimizer.step()
            
        train_loss += loss.item()*data.size(0)
        
        prediction = torch.max(output, 1)[1] 
        correct = prediction == target       
        for i in range(target.shape[0]):     
            label = target.data[i]            
              
            correct_classes[label] += correct[i].item()
            all_classes[label] += 1
    train_loss = train_loss/len(train_loader.dataset)
    accuracy = (np.sum(correct_classes) / np.sum(all_classes))
    return train_loss, accuracy


def learn(epochs, model, optimizer, criterio, train_loader, valid_loader):
    valid_loss_min = np.inf
    start = time.time() # training time
    remain_ep = epochs
    val_acc = []
    train_acc = []
    train = []
    val = []
    
    print("training...")
    for ep in range(epochs):
        train_loss = 0.0 
        valid_loss = 0.0
        remain_ep -= 1
        
        
        train_loss, acc1 = training(train_loss, model, train_loader, criterio, optimizer)
        valid_loss, acc2 = validate(valid_loss,  model, valid_loader, criterio, optimizer)
        val_acc.append(acc2)
        train_acc.append(acc1)
            
        
        
        # saving to plot later
        train.append(train_loss)
        val.append(valid_loss)
        
        
        t_train = time.time() - start 
        if ep == 0:
          t_epoch = (t_train // 60) 
          
        print('training time {:.0f}m {:.0f}s    expected time: {:.0f} minutes'.format(t_train // 60, t_train % 60, t_epoch*remain_ep))
        print("------------------------------------------------------------------------------------------------")
        print('epoch: {} \t train_loss: {:.7f} \t valid_loss: {:.7f} \t val_accuracy: {:.3f} \t train_accuracy: {:.3f}'.format(ep+1, train_loss, valid_loss, acc2, acc1))
        
        if valid_loss <= valid_loss_min:
            print('saving model...')
            torch.save(model.state_dict(), 'resnet50_model.pt')
            valid_loss_min = valid_loss
            
    # saving uselful information
    with open("train_acc.txt", "wb") as fp0:
        pickle.dump(train_acc, fp0)
        
    with open("val_acc.txt", "wb") as fp:
        pickle.dump(val_acc, fp)
    
    with open("train.txt", "wb") as fp1:
        pickle.dump(train, fp1)
        
    with open("val.txt", "wb") as fp2:
        pickle.dump(val, fp2)
        
def train_val_acc_plot(train, val, train_acc, val_acc):
    fig = plt.figure(figsize=(10,10))
    ax = fig.add_subplot(221)
    ax.plot(train_acc)
    ax.set_title('Train Accuracy');
    ax = fig.add_subplot(222)
    ax.plot(val_acc,'r')
    ax.set_title('Valid Accuracy');
    ax = fig.add_subplot(223)
    ax.plot(train, 'r', label="tudo")
    ax.plot(val, 'y', label="nada")
    ax.set_title('Train and Validation Losses')
    fig.savefig("info")
    

def accuracy(model, test_loader, criterio):
    test_loss = 0.0
    correct_classes = list(np.zeros(len(test_loader.dataset.classes))) 
    all_classes = list(np.zeros(len(test_loader.dataset.classes)))
    with torch.no_grad():
        model.eval() 
        for data, target in test_loader:
            data, target = data.to(device), target.to(device) 
            # propagação
            output = model(data)
            loss = criterio(output, target)
        
            test_loss += loss.item()*data.size(0)
        
            prediction = torch.max(output, 1)[1] 
            correct = prediction == target       
            for i in range(target.shape[0]):     
              label = target.data[i]            
              
              correct_classes[label] += correct[i].item()
              all_classes[label] += 1
        
        
        test_loss = test_loss / len(test_loader.dataset)
        print('Test Loss : {:.6f}'.format(test_loss))  
      
        # metrica accurácia : classes corretamente classificadas sobre todas as classes
        print('Overall Accuracy : {:.2f}'.format(np.sum(correct_classes) / np.sum(all_classes)))
        return (np.sum(correct_classes) / np.sum(all_classes)), test_loss


    
def main(argv):
    
    # reading datasets
    train_data = datasets.ImageFolder("/home/matthew/Área de Trabalho/IC/datasets/__DATASETS__/data/all-idb-splitted/train", transform=image_transforms['train'])
    valid_data = datasets.ImageFolder("/home/matthew/Área de Trabalho/IC/datasets/__DATASETS__/data/all-idb-splitted/val", transform=image_transforms['valid'])
    test_data = datasets.ImageFolder("/home/matthew/Área de Trabalho/IC/datasets/__DATASETS__/data/all-idb-splitted/test", transform=image_transforms['test'])
    
    
    # dataloaders
    test_loader  = torch.utils.data.DataLoader(test_data,  batch_size=32, shuffle=True)
    train_loader = torch.utils.data.DataLoader(train_data,  batch_size=32, shuffle=True)
    valid_loader = torch.utils.data.DataLoader(valid_data,  batch_size=32, shuffle=True)
    
    # loading models
    model = models.resnet50(pretrained=True)
    
    for weights in model.parameters():
        weights.requires_grad = False
    
    fc_inputs = model.fc.in_features
 
    model.fc = nn.Sequential(
        nn.Linear(fc_inputs, 256),
        nn.ReLU(),
        nn.Dropout(0.4),
        nn.Linear(256, 2), 
        nn.LogSoftmax(dim=1) # For using NLLLoss()
    )
    

    criterio = nn.NLLLoss()
    
    optimizer = optim.Adam(model.parameters(), lr=0.01)
    model.to(device)
         
    
    
    
    os.chdir("/home/matthew/Área de Trabalho/IC/modelos/RESNET-50/online-aug/all-idb/")
    
    learn(80, model, optimizer, criterio, train_loader, valid_loader)
    
    model.load_state_dict(torch.load('resnet50_model.pt')) # carrengando melhor modelo
    with open("val.txt", "rb") as fp2:
        val = pickle.load(fp2)
    with open("train.txt", "rb") as fp:
        train = pickle.load(fp)
    with open("val_acc.txt", "rb") as fp3:
        val_acc = pickle.load(fp3)
    with open("train_acc.txt", "rb") as fp4:
        train_acc = pickle.load(fp4)
        
    test_accuracy, test_loss = accuracy(model, test_loader, criterio)  
    train_val_acc_plot(train, val, train_acc, val_acc) 
    with open("test_info.txt", "wb") as fp5:
        pickle.dump([test_accuracy,test_loss], fp5)
        

    
    
    
    
       

    
    
    
    
if __name__ == "__main__":
    main(sys.argv[1:])                                 