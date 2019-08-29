#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jun 22 16:08:41 2018

@author: caiom
"""
import torch
import torchvision
import numpy as np

class UNetVgg(torch.nn.Module):
    """
    BorderNetwork is a NN that aims to detected border and classify occlusion.
    The architecture is a VGG without the last pool layer. After that we 
    have two paths, one for regression and one for classification (occlusion).
    """
    
    def __init__(self, nClasses):
        super(UNetVgg, self).__init__()
        
        vgg16pre = torchvision.models.vgg16(pretrained=True)
        self.vgg0 = torch.nn.Sequential(*list(vgg16pre.features.children())[:4])
        self.vgg1 = torch.nn.Sequential(*list(vgg16pre.features.children())[4:9])
        self.vgg2 = torch.nn.Sequential(*list(vgg16pre.features.children())[9:16])
        self.vgg3 = torch.nn.Sequential(*list(vgg16pre.features.children())[16:23])
        self.vgg4 = torch.nn.Sequential(*list(vgg16pre.features.children())[23:30])
        
        
        self.smooth0 = torch.nn.Sequential(
                torch.nn.Conv2d(128, 64, kernel_size=(3,3), stride=1, padding=(1, 1)),
                torch.nn.ReLU(True),
                torch.nn.Conv2d(64, 64, kernel_size=(3,3), stride=1, padding=(1, 1)),
                torch.nn.ReLU(True)
                )
        self.smooth1 = torch.nn.Sequential(
                torch.nn.Conv2d(256, 64, kernel_size=(3,3), stride=1, padding=(1, 1)),
                torch.nn.ReLU(True),
                torch.nn.Conv2d(64, 64, kernel_size=(3,3), stride=1, padding=(1, 1)),
                torch.nn.ReLU(True)
                )
        self.smooth2 = torch.nn.Sequential(
                torch.nn.Conv2d(512, 128, kernel_size=(3,3), stride=1, padding=(1, 1)),
                torch.nn.ReLU(True),
                torch.nn.Conv2d(128, 128, kernel_size=(3,3), stride=1, padding=(1, 1)),
                torch.nn.ReLU(True)
                )
        self.smooth3 = torch.nn.Sequential(
                torch.nn.Conv2d(1024, 256, kernel_size=(3,3), stride=1, padding=(1, 1)),
                torch.nn.ReLU(True),
                torch.nn.Conv2d(256, 256, kernel_size=(3,3), stride=1, padding=(1, 1)),
                torch.nn.ReLU(True)
                )
        
        
        self.final = torch.nn.Conv2d(64, nClasses, kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        """
        Args:
            x (torch.tensor): A tensor of size (batch, 3, H, W)
        Returns:
            reg_out (torch.tensor): A tensor with results of the regression (batch, 4).
            cls_out (torch.tensor): A tensor with results of the classification (batch, 2).
        """
        
        feat0 = self.vgg0(x)
        feat1 = self.vgg1(feat0)
        feat2 = self.vgg2(feat1)
        feat3 = self.vgg3(feat2)
        feat4 = self.vgg4(feat3)
        
        _,_,H,W = feat3.size()
        up3 = torch.nn.functional.interpolate(feat4, size=(H,W), mode='bilinear', align_corners=False)
        concat3 = torch.cat([feat3, up3], 1)
        end3 = self.smooth3(concat3)
        
        _,_,H,W = feat2.size()
        up2 = torch.nn.functional.interpolate(end3, size=(H,W), mode='bilinear', align_corners=False)
        concat2 = torch.cat([feat2, up2], 1)
        end2 = self.smooth2(concat2)
        
        _,_,H,W = feat1.size()
        up1 = torch.nn.functional.interpolate(end2, size=(H,W), mode='bilinear', align_corners=False)
        concat1 = torch.cat([feat1, up1], 1)
        end1 = self.smooth1(concat1)
        
        _,_,H,W = feat0.size()
        up0 = torch.nn.functional.interpolate(end1, size=(H,W), mode='bilinear', align_corners=False)
        concat0 = torch.cat([feat0, up0], 1)
        end0 = self.smooth0(concat0)
        
        return self.final(end0)
    
    
    @staticmethod
    def eval_net_with_loss(model, inp, gt, class_weights, device):
        """
        Evaluate network including loss.
        
        Args:
            model (torch.nn.Module): The model.
            inp (torch.tensor): A tensor (float32) of size (batch, 3, H, W)
            gt (torch.tensor): A tensor (long) of size (batch, 1, H, W) with the groud truth (0 to num_classes-1).
            class_weights (list of float): A list with len == num_classes.
            device (torch.device): device to perform computation
            
        Returns:
            out (torch.tensor): Network output.
            loss (torch.tensor): Tensor with the total loss.
                
        """
        weights = torch.from_numpy(np.array(class_weights, dtype=np.float32)).to(device)
        out = model(inp)
        
        softmax = torch.nn.functional.log_softmax(out, dim = 1)
        loss = torch.nn.functional.nll_loss(softmax, gt, ignore_index=-1, weight=weights)
            
        return (out, loss)
    
    @staticmethod
    def get_params_by_kind(model, n_base = 7):
    
        base_vgg_bias = []
        base_vgg_weight = []
        core_weight = []
        core_bias = []
    
        for name, param in model.named_parameters():
            if 'vgg' in name and ('weight' in name or 'bias' in name):
                vgglayer = int(name.split('.')[-2])
                
                if vgglayer <= n_base:
                    if 'bias' in name:
                        print('Adding %s to base vgg bias.' % (name))
                        base_vgg_bias.append(param)
                    else:
                        base_vgg_weight.append(param)
                        print('Adding %s to base vgg weight.' % (name))
                else:
                    if 'bias' in name:
                        print('Adding %s to core bias.' % (name))
                        core_bias.append(param)
                    else:
                        print('Adding %s to core weight.' % (name))
                        core_weight.append(param)
                        
            elif ('weight' in name or 'bias' in name):
                if 'bias' in name:
                    print('Adding %s to core bias.' % (name))
                    core_bias.append(param)
                else:
                    print('Adding %s to core weight.' % (name))
                    core_weight.append(param)
                    
        return (base_vgg_weight, base_vgg_bias, core_weight, core_bias)
    
# End class

