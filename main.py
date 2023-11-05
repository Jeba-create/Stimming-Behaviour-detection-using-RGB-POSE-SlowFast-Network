# Copyright (c) OpenMMLab. All rights reserved.
import argparse
import copy
import os
import os.path as osp
import time
import warnings

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import os
from os import listdir
from os.path import isfile, join

import json
import pandas as pd
from sklearn.model_selection import KFold

os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
torch.backends.cudnn.enabled = False


from Dataloaderrgbpose import ReadDataset
from modelrgbposeslowfast import RgbPoseSlowFastResNet34

def train(net, optimizer, criterion, dataloader_trn, device):

  net.train()
  total_trn,correct_trn=0,0
  train_loss = 0
  cnt=0
  for data1 in dataloader_trn:
    data1,data2,label1=data1['vid'].to(device),data1['pose'].to(device),data1['label'].to(device)
    #data1,label1=data1['vid'],data1['label'][0]
    #print("label1",label1,data1.shape,data2.shape)
    # zero the parameter gradient
    optimizer.zero_grad()

    # forward + backward + optimize
    outputs =net(data1,data2)
    loss = criterion(outputs, label1)

    # Calculate Loss
    train_loss += loss.item()

    loss.backward()
    torch.nn.utils.clip_grad_norm_(net.parameters(), max_norm=10)
    optimizer.step()  

    #compute accuracy
    _, predicted = torch.max(outputs, 1)
    total_trn += label1.size(0)
    correct_trn += (predicted == label1).sum().item()
    acc_trn=correct_trn /total_trn 

    #cnt+=1
    #if cnt>5:
      #break
  
  return train_loss,acc_trn

def validate(net, criterion, dataloader_val, device):
  net.eval()
  total_val,correct_val=0,0
  with torch.no_grad():
    val_loss = 0
    cnt=0
    for data1 in dataloader_val:
      data1,data2,label1=data1['vid'].to(device),data1['pose'].to(device),data1['label'].to(device)
      #data1,label1=data1['vid'],data1['label'][0]
   
      # forward 
      outputs =net(data1,data2)
      loss = criterion(outputs, label1)
      val_loss += loss.item()

      #compute accuracy
      _, predicted = torch.max(outputs, 1)
      total_val += label1.size(0)
      correct_val += (predicted == label1).sum().item()
      acc_val=correct_val /total_val

      #cnt+=1
      #if cnt>2:
        #break

  return val_loss,acc_val

def ReadAnntn(ann_file):
    video_infos = []
    with open(ann_file, 'r') as fin:
      for line in fin:
        line_split = line.strip().split()
        filename, label = line_split[0], line_split[1:]
        label = list(map(int, label))
        video_infos.append(dict(filename=filename, label=label))
    return video_infos

def reset_weights(m):
  '''
    Try resetting model weights to avoid
    weight leakage.
  '''
  for layer in m.children():
   if hasattr(layer, 'reset_parameters'):
    #print(f'Reset trainable parameters of layer = {layer}')
    layer.reset_parameters()    
  

# Configuration options
k_folds = 3
Epochs=250
NoOfFrms2slct_s=4
NoOfFrms2slct_f=20

left_kps=[1, 3, 5, 7, 9, 11, 13, 15]
right_kps=[2, 4, 6, 8, 10, 12, 14, 16]
skeletons=[(0, 1), (0, 2), (1, 3), (2, 4), (0, 5), (5, 7),(7, 9), (0, 6), (6, 8), 
    (8, 10), (5, 11), (11, 13),(13, 15), (6, 12), (12, 14), (14, 16), (11, 12)]

scalee=[56 ,56]
areaa_range=(0.56, 1.0)

logpath='/content/drive/My Drive/Autism/checkpoint/slowfast_ResNet34_limb_ssd_crsvld/'
if not os.path.exists(logpath):
	os.makedirs(logpath)

flname=logpath+'logfile.json'
df=open(flname, 'w+')
df1=open('logfile.json', 'w+')

ann_file_trn="ssd_train_video.txt"
vid_infor_trn=ReadAnntn(ann_file_trn) 
pose_infor_trn=pd.read_pickle(r'/content/ssd_anntn_train.pkl')
mypath_trn ='/content/videos_train_ssd'
vid_dataset_trn = ReadDataset(pose_infor_trn,mypath_trn,NoOfFrms2slct_s,NoOfFrms2slct_f,left_kp=left_kps,right_kp=right_kps,skeleton=skeletons,with_limb=True,scale=scalee,area_range=areaa_range,phase='train')
    
# For fold results
results = {}
  
# Set fixed random number seed
torch.manual_seed(42)

# Define the K-fold Cross Validator
kfold = KFold(n_splits=k_folds, shuffle=True)

# K-fold Cross Validation model evaluation
for fold, (train_ids, test_ids) in enumerate(kfold.split(vid_infor_trn)):
  print(f'FOLD {fold}')

  # Sample elements randomly from a given list of ids, no replacement.
  train_subsampler = torch.utils.data.SubsetRandomSampler(train_ids)
  test_subsampler = torch.utils.data.SubsetRandomSampler(test_ids)
  #print('train_ids, test_idss',train_ids, test_ids)

  # Define data loaders for training and testing data in this fold
  dataloader_trn = DataLoader(vid_dataset_trn , batch_size=2, sampler=train_subsampler, num_workers=2)  #
  dataloader_val = DataLoader(vid_dataset_trn , batch_size=2, sampler=test_subsampler, num_workers=2)  #

  device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
  net_sf=RgbPoseSlowFastResNet34(layers=[3,4,6,3],img_channels=3,pose_channels=17,num_classes=3,dropout_ratio=.5).to(device)
  net_sf.apply(reset_weights)

  criterion = nn.CrossEntropyLoss()
  optimizer = optim.SGD(net_sf.parameters(), lr=0.02, momentum=0.9,weight_decay=0.0003)     #,weight_decay=0.0003
  scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=Epochs, eta_min=0)

  min_val_loss = np.inf
  for epoch in range(Epochs):  # loop over the dataset multiple times
      
      #=====train======
      loss_trn,acc_trn=train(net_sf, optimizer, criterion, dataloader_trn, device)

      scheduler.step()

      #=====test======
      loss_val,acc_val=train(net_sf, optimizer, criterion, dataloader_val, device)

      print("Epoch:{}\{} Fold: {} lr: {:.4f} Train Loss: {:.4f} Val Loss:{:.4f} Train acc:{:.4f} Val acc:{:.4f}".format(epoch, Epochs,fold,optimizer.param_groups[0]['lr'], (loss_trn / len(dataloader_trn)),(loss_val / len(dataloader_val)),acc_trn,acc_val))

      save_path = f'./model-fold-{fold}.pth'
      if min_val_loss > loss_val:
        print(f'Validation Loss Decreased({min_val_loss:.6f}--->{loss_val:.6f}) \t Saving The Model')
        min_val_loss = loss_val
        filewrt=logpath+f'saved_model_best-{fold}.pth'
        # Saving State Dict
        torch.save(net_sf.state_dict(), filewrt)
      filewrt=logpath+f'saved_model_latest-{fold}.pth'
      # Saving State Dict
      torch.save(net_sf.state_dict(), filewrt)

      log_info={}  
      log_info['Epoch'] = epoch
      log_info['fold'] = fold
      log_info['lr:'] = optimizer.param_groups[0]['lr']
      log_info['Train Loss:'] = loss_trn / len(dataloader_trn)
      log_info['Val Loss:'] = loss_val / len(dataloader_val)
      log_info['Train acc'] = acc_trn 
      log_info['Val acc:'] = acc_val
      json_file = json.dumps(log_info)
      df.write(json_file)
      df.write('\n')  
      df1.write(json_file)
      df1.write('\n')

      results[fold] = acc_val
    
# Print fold results
print(f'K-FOLD CROSS VALIDATION RESULTS FOR {k_folds} FOLDS')
print('--------------------------------')
sum = 0.0
for key, value in results.items():
  print(f'Fold {key}: {value} %')
  sum += value
print(f'Average: {sum/len(results.items())} %')




    
