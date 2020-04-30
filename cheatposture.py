#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar  5 06:38:35 2020

@author: sanjanasrinivasareddy
"""

import torch
import torch.optim as optim
import torch.nn.functional as F
import torchvision
import torchvision.datasets as datasets
import torchvision.models as models
import torchvision.transforms as transforms
modelx = torchvision.models.alexnet(pretrained=False)
modelx.classifier[6] = torch.nn.Linear(modelx.classifier[6].in_features, 4)
modelx.load_state_dict(torch.load('/Users/sanjanasrinivasareddy/Desktop/best_model.pth'))
import cv2
import numpy as np

mean = 255.0 * np.array([0.485, 0.456, 0.406])
stdev = 255.0 * np.array([0.229, 0.224, 0.225])

normalize = torchvision.transforms.Normalize(mean, stdev)


def preprocess(x):
    x=cv2.resize(x,(224,224))
    x = cv2.cvtColor(x, cv2.COLOR_BGR2RGB)
    x = x.transpose((2, 0, 1))
    x = torch.from_numpy(x).float()
    x = normalize(x)
    x = x[None, ...]
    #x = preprocess(x)
    y = modelx(x)
    y = F.softmax(y, dim=1)
    y = y.flatten()
    cl=['student is bending ','student has his/her hands back','student has no observable cheating activity','student is turning completely']
    p=cl[list(y).index(max(y))]
    return p
#print(preprocess(cv2.imread('/Users/sanjanasrinivasareddy/Desktop/copy/copy/Bend/New FolderKang60.jpg')))