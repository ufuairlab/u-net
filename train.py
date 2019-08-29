#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Aug 29 17:53:30 2018

@author: caiom
"""

import numpy as np
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
import vgg_unet
import torch
from data_utils import SegmentationDataset
from torch.utils.data import DataLoader

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
gt_folder_train = 'Train/'
gt_folder_val = 'Val/'
model_name = 'segm2.pth'
patience = 30
plot_val = True
plot_train = True

max_epochs = 100

#Width x Height - MUST be divisible by 32
resolution = (640, 480)
class_weights = [1, 1, 1]
nClasses = 3

# Color in RGB
#class_to_color = {'Sky': (0, 0, 127), 'Ground': (0, 127, 127), 'Fruit': (0, 255, 0), 'Tree': (127, 0, 0)}
#class_to_id = {'Sky': 0, 'Ground': 1, 'Fruit': 2, 'Tree': 3}

class_to_color = {'Ground': (127, 0, 0) , 'Healthy': (0, 127, 127), 'Pest': (0, 255, 0)}
class_to_id = {'Ground': 0, 'Healthy': 1, 'Pest': 2}
id_to_class = {v: k for k, v in class_to_id.items()}

train_dataset = SegmentationDataset(gt_folder_train, gt_folder_train, True, class_to_id, resolution, True)
train_loader = DataLoader(train_dataset, batch_size=1, shuffle=True, num_workers=0, drop_last=True)

val_dataset = SegmentationDataset(gt_folder_val, gt_folder_val, False, class_to_id, resolution)
val_loader = DataLoader(val_dataset, batch_size=1, shuffle=True, num_workers=0, drop_last=False)


if plot_train:

    for i_batch, sample_batched in enumerate(train_loader):
    
            image_np = np.squeeze(sample_batched['image_original'].cpu().numpy())
            gt = np.squeeze(sample_batched['gt'].cpu().numpy())
                
            color_label = np.zeros((resolution[1], resolution[0], 3))
            
            for key, val in id_to_class.items():
                color_label[gt == key] = class_to_color[val]
                
            plt.figure()
            plt.imshow((image_np/255) * 0.5 + (color_label/255) * 0.5)
            plt.show()
            
            plt.figure()
            plt.imshow(color_label.astype(np.uint8))
            plt.show()

  
model = vgg_unet.UNetVgg(nClasses).to(device)

core_lr = 0.02
base_vgg_weight, base_vgg_bias, core_weight, core_bias = vgg_unet.UNetVgg.get_params_by_kind(model, 7)

optimizer = torch.optim.SGD([{'params': base_vgg_bias, 'lr': 0.000001}, 
                             {'params': base_vgg_weight, 'lr': 0.000001},
                             {'params': core_bias, 'lr': core_lr},
                             {'params': core_weight, 'lr': core_lr, 'weight_decay': 0.0005}], momentum=0.9)
    
lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 30, gamma=0.2)


best_val_acc = -1
best_epoch = 0

# Start training...
for epoch in range(max_epochs):
    
    print('Epoch %d starting...' % (epoch+1))
    
    lr_scheduler.step()
    
    model.train()
    
    mean_loss = 0.0
    
    n_correct = 0
    n_false = 0
    
    for i_batch, sample_batched in enumerate(train_loader):
    
    
        image = sample_batched['image'].to(device)
        gt = sample_batched['gt'].to(device)
    
        optimizer.zero_grad()
        output, total_loss = model.eval_net_with_loss(model, image, gt, class_weights, device)
        total_loss.backward()
        optimizer.step()
        
        mean_loss += total_loss.cpu().detach().numpy()
        
        # Measure accuracy
        
        gt = np.squeeze(sample_batched['gt'].cpu().numpy())
        
        label_out = torch.nn.functional.softmax(output, dim = 1)
        label_out = label_out.cpu().detach().numpy()
        label_out = np.squeeze(label_out)
        
        labels = np.argmax(label_out, axis=0)
        valid_mask = gt != -1
        curr_correct = np.sum(gt[valid_mask] == labels[valid_mask])
        curr_false = np.sum(valid_mask) - curr_correct
        n_correct += curr_correct
        n_false += curr_false
        
    mean_loss /= len(train_loader)
    train_acc = n_correct / (n_correct + n_false)
        
    print('Train loss: %f, train acc: %f' % (mean_loss, train_acc))
    
    
    n_correct = 0
    n_false = 0
    
    
    for i_batch, sample_batched in enumerate(val_loader):
    
    
        image = sample_batched['image'].to(device)
        image_np = np.squeeze(sample_batched['image_original'].cpu().numpy())
        gt = np.squeeze(sample_batched['gt'].cpu().numpy())
        
    
        label_out = model(image)
        label_out = torch.nn.functional.softmax(label_out, dim = 1)
        label_out = label_out.cpu().detach().numpy()
        label_out = np.squeeze(label_out)
        
        labels = np.argmax(label_out, axis=0)
        
        if plot_val:
            
            color_label = np.zeros((resolution[1], resolution[0], 3))
            
            for key, val in id_to_class.items():
                color_label[labels == key] = class_to_color[val]
                
            plt.figure()
            plt.imshow((image_np/255) * 0.5 + (color_label/255) * 0.5)
            plt.show()
            
            plt.figure()
            plt.imshow(color_label.astype(np.uint8))
            plt.show()
        
        valid_mask = gt != -1
        curr_correct = np.sum(gt[valid_mask] == labels[valid_mask])
        curr_false = np.sum(valid_mask) - curr_correct
        n_correct += curr_correct
        n_false += curr_false
        
        
    total_acc = n_correct / (n_correct + n_false)
    
    if best_val_acc < total_acc:
        best_val_acc = total_acc
        if epoch > 7:
            torch.save(model.state_dict(), model_name)
            print('New best validation acc. Saving...')
        best_epoch = epoch

    if (epoch - best_epoch) > patience:
        print("Fnishing training, best validation acc %f", best_val_acc)
        break
    
    print('Val acc: %f -- Best val acc: %f -- epoch %d.' % (total_acc, best_val_acc, best_epoch))
        
        
    
    
