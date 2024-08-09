import os 
import numpy as np
import pandas as pd
from PIL import Image
import matplotlib.pyplot as plt

import torch
from torch import nn as nn


import torchvision
from torchvision import transforms

from torchinfo import summary

from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from dataset import train, val, ImageDataset, model_dataloader


# Training loop -> results dictionary
def training_loop(model, train_dataloader, val_dataloader, device, epochs, patience):
    # empty dict for restore results
    results = {"train_loss":[], "train_acc":[], "val_loss":[], "val_acc":[]}
    
    # hardcode loss_fn and optimizer
    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr = 0.0005)
    
    # loop through epochs
    for epoch in range(epochs):
        train_loss, train_acc = train(model = model, 
                                      dataloader = train_dataloader,
                                      loss_fn = loss_fn,
                                      optimizer = optimizer,
                                      device = device)
        
        val_loss, val_acc = val(model = model,
                                dataloader = val_dataloader,
                                loss_fn = loss_fn,
                                device = device)
        
        # print results for each epoch
        print(f"Epoch: {epoch+1}\n"
              f"Train loss: {train_loss:.4f} | Train accuracy: {(train_acc*100):.3f}%\n"
              f"Val loss: {val_loss:.4f} | Val accuracy: {(val_acc*100):.3f}%")
        
        # record results for each epoch
        results["train_loss"].append(train_loss)
        results["train_acc"].append(train_acc)
        results["val_loss"].append(val_loss)
        results["val_acc"].append(val_acc)
        
        # calculate average "val_loss" for early_stopping
        mean_val_loss = np.mean(results["val_loss"])
        best_val_loss = float("inf")
        num_no_improvement = 0
        if np.mean(mean_val_loss > best_val_loss):
            best_val_loss = mean_val_loss
        
            model_state_dict = model.state_dict()
            best_model.load_state_dict(model_state_dict)
        else:
            num_no_improvement +=1
    
        if num_no_improvement == patience:
            break
    
    # plt results after early_stopping
    plt.figure(figsize = (8, 4))
    plt.subplot(1, 2, 1)
    plt.title("Loss")
    plt.plot(results["train_loss"], label = "Train loss")
    plt.plot(results["val_loss"], label = "Val loss")
    plt.legend()
    
    plt.subplot(1, 2, 2)
    plt.title("Accuracy")
    plt.plot(results["train_acc"], label = "Train accuracy")
    plt.plot(results["val_acc"], label = "Val accuracy")
    plt.legend()
    
    return results


def testing_loop():
    # empty list store labels
    predict_label_list = []
    images = []
    image_files = []
    # eval mode
    resnet_model.eval()


    test_transform = transforms.Compose([
                transforms.Resize(size=232),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
                ])


    # Iterate over images in the test folder
    test_folder =  "./nycu-2023-deep-learning-final-project-1/achieve/test"
    image_files = sorted(os.listdir(test_folder), key=lambda x: int(x.split('.')[0]))
    for img_filename in image_files:
        if img_filename.endswith(".jpg"):
            img_path = os.path.join(test_folder, img_filename)
            images.append(img_path)
        

    for image in images:
        image_file = Image.open(image)
        if image_file.mode == "L":
            image_file = Image.merge("RGB", (image_file, image_file, image_file))
        transformed_image = test_transform(image_file)
        # add batch_size and device
        transformed_image = transformed_image.unsqueeze(dim = 0)
        # logits
        logits = resnet_model(transformed_image)
        # lables
        label = torch.argmax(logits).item()
        predict_label_list.append(label)

    return predict_label_list


if '__name__' == '__main__':
    resnet_weight = torchvision.models.ResNet50_Weights.DEFAULT

    resnet_model = torchvision.models.resnet50(weights = resnet_weight)

    for param in resnet_model.parameters():
        param.requires_grad = False

    # Custom output layer
    # resnet_model.fc
    custom_fc = nn.Sequential(
        nn.ReLU(),
        nn.Dropout(p = 0.5),
        nn.Linear(1000, 100))

    resnet_model.fc = nn.Sequential(
        resnet_model.fc,
        custom_fc
    )

    # Device
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Data augmentation
    # resnet_weight.transforms()
    resnet_transform = transforms.Compose([
        transforms.Resize(size = 232),
        transforms.ColorJitter(brightness = (0.8, 1.2)),
        transforms.RandomHorizontalFlip(p = 0.5),
        transforms.RandomRotation(degrees = 15),
        transforms.ToTensor(),
        transforms.Normalize(mean = [0.485, 0.456, 0.406], std = [0.229, 0.224, 0.225])
    ])

    resnet_train_dataloader, resnet_val_dataloader, resnet_test_dataloader = model_dataloader(weights = resnet_weight, 
                                                                                            transform = resnet_transform
                                                                                            )


    # Actual training ResNet model
    resnet_results = training_loop(model = resnet_model,
                                train_dataloader = resnet_train_dataloader,
                                val_dataloader = resnet_val_dataloader,
                                device = device,
                                epochs = 30,
                                patience = 5
                                )
    # Testing the model.
    pred = testing_loop()
    
