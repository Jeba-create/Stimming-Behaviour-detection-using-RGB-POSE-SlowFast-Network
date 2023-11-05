import cv2
from random import randrange
from torchvision import transforms as T
from PIL import Image

from torch.utils.data import Dataset, DataLoader

import random
import numpy as np
from random import randrange

from poseaugmentation import PoseCompact,resize,flip,RandomResizedCrop,CenterCrop,GeneratePoseTarget

class ReadDataset(Dataset):
 
    def __init__(self,video_infos,root_dir,NoOfFrms2slct_s,NoOfFrms2slct_f,left_kp,right_kp,skeleton,with_limb=False,scale=[56,56],area_range=(0.56, 1.0),phase='train'):
      super(ReadDataset).__init__()
      self.video_infos = video_infos
      self.root_dir = root_dir
      self.NoOfFrms2slct_s=NoOfFrms2slct_s
      self.NoOfFrms2slct_f=NoOfFrms2slct_f
      self.phase=phase
      self.left_kp=left_kp
      self.right_kp=right_kp
      self.scale=scale
      self.area_range=area_range
      self.with_limb=with_limb
      self.skeleton=skeleton

    # Function to find the number closest to n and divisible by m
    def closestNumber(self,n, m) :
        # Find the quotient
      q = int(n / m)
      # 1st possible closest number
      n1 = m * q
      return n1 

    def unifrmsampler(self,Nofrms,NoOfFrms2slct,mode):
        Frmstoselect=[]
        iu=0
        rangeIntrvl=int(Nofrms/NoOfFrms2slct)
        #print("rangeIntrvl",Nofrms,rangeIntrvl,NoOfFrms2slct,mode)
        for _ in range(NoOfFrms2slct):
          if mode=='train' or 'val':
            Frmstoselect.append(randrange(iu, iu+rangeIntrvl)) 
          #elif mode=='val':
            #Frmstoselect.append(i)
          iu+=rangeIntrvl
        return Frmstoselect

    def dataTransform(self,snpt,filena,mode):
        rndNo=randrange(0,10)
        procdvid=[]
        for i in range(len(snpt)):
            img=snpt[i]

            if i==0 and mode == 'train':
                Transform = T.Compose([T.ToPILImage(),T.Resize([224,224]),T.RandomHorizontalFlip(p=0.9),T.ToTensor(),T.Normalize(mean=[0.485,0.456,0.406], std=[0.229, 0.224,0.225])])
            elif i==1 and mode == 'train':
                Transform = T.Compose([T.ToPILImage(),T.Resize([256,256]),T.CenterCrop(224),T.ToTensor(),T.Normalize(mean=[0.485,0.456,0.406], std=[0.229, 0.224,0.225])])
            elif i==2 and mode == 'train': 
                Transform = T.Compose([T.ToPILImage(),T.CenterCrop(128),T.Resize([224,224]),T.ToTensor(),T.Normalize(mean=[0.485,0.456,0.406], std=[0.229, 0.224,0.225])])
            elif i==3 and mode == 'train': 
                Transform = T.Compose([T.ToPILImage(),T.Resize([224,224]),T.RandomRotation(degrees=15),T.ToTensor(),T.Normalize(mean=[0.485,0.456,0.406], std=[0.229, 0.224,0.225])])
            elif i==4 and mode == 'train': 
                Transform = T.Compose([T.ToPILImage(),T.Resize([224,224]),T.ColorJitter(),T.ToTensor(),T.Normalize(mean=[0.485,0.456,0.406], std=[0.229, 0.224,0.225])])
            elif i>4 and mode == 'train': 
                Transform = T.Compose([T.ToPILImage(),T.Resize([224,224]),T.ToTensor(),T.Normalize(mean=[0.485,0.456,0.406], std=[0.229, 0.224,0.225])])
            elif mode == 'val' :
                Transform = T.Compose([T.ToPILImage(),T.Resize([224,224]),T.ToTensor(),T.Normalize(mean=[0.485,0.456,0.406], std=[0.229, 0.224,0.225])])
            
            #print("preprocess",rndNo,mode,img.shape,len(snpt),filena)
            preprocess= Transform (img)
            
            procdvid.append(preprocess) 
        procdvid=torch.stack(procdvid)
        procdvid=torch.transpose(procdvid, 0, 1)
        #print("previd",len(procdvid),procdvid.shape) 
        return procdvid 

    def dataAug(self,pose,phase):

        #'keypoint','keypoint_score','frame_dir': '01_0003l03004_09032018_s_35_Yes', 'img_shape': (1080, 1920), 'original_shape': (1080, 1920), 'total_frames': 52, 'label': 1}    
 
        pose_smplr=UniformSampler(self.NoOfFrms2slct_f,self.phase)  #
        pose_smplr_rs=pose_smplr(pose)
        #print(pose_smplr_rs.keys())
        
        pose_compact=PoseCompact(hw_ratio=1)
        pose_compact_rs=pose_compact(pose_smplr_rs)
        #print("new_shape",pose_compact_rs['original_shape'],pose_compact_rs['img_shape'],pose_compact_rs['crop_quadruple'])
        
        pose_resize=resize([56 ,56])
        pose_resize_rs=pose_resize(pose_compact_rs)

        rndNo=randrange(0,10)

        if self.phase=='train' and rndNo==0:
            pose_flip=flip(left_kp=left_kps,right_kp=right_kps)
            pose_flip_rs=pose_flip(pose_resize_rs)

            pose_crop=RandomResizedCrop(areaa_range)
            pose_crop_rs=pose_crop(pose_flip_rs)
            #print("pose_crop", pose_crop_rs['crop_bbox'],pose_crop_rs['img_shape'])

            pose_resize=resize(scalee)
            pose_resize_rs=pose_resize(pose_crop_rs)
            #print("pose_resiz",pose_resize_rs['img_shape'])
        elif self.phase=='train' and rndNo==1:
            pose_flip=flip(left_kp=None,right_kp=None)
            pose_flip_rs=pose_flip(pose_resize_rs)

            pose_crop=RandomResizedCrop((0.56, 1.0))
            pose_crop_rs=pose_crop(pose_flip_rs)
            #print("pose_crop", pose_crop_rs['crop_bbox'],pose_crop_rs['img_shape'])

            pose_resize=resize(scalee)
            pose_resize_rs=pose_resize(pose_crop_rs)
            #print("pose_resiz",pose_resize_rs['img_shape'])  
        elif self.phase=='train' and rndNo==2:
            pose_flip=flip(left_kp=left_kps,right_kp=right_kps)
            pose_flip_rs=pose_flip(pose_resize_rs)

            pose_crop=RandomResizedCrop((0.28, 1.0))
            pose_crop_rs=pose_crop(pose_flip_rs)
            #print("pose_crop", pose_crop_rs['crop_bbox'],pose_crop_rs['img_shape'])

            pose_resize=resize(scalee)
            pose_resize_rs=pose_resize(pose_crop_rs)
            #print("pose_resiz",pose_resize_rs['img_shape']) 
        elif self.phase=='train' and rndNo==3:
            pose_flip=flip(left_kp=None,right_kp=None)
            pose_flip_rs=pose_flip(pose_compact_rs)

            pose_crop=CenterCrop(28)
            pose_crop_rs=pose_crop(pose_flip_rs)
            #print("pose_crop", pose_crop_rs['crop_bbox'],pose_crop_rs['img_shape'])

            pose_resize=resize(scalee)
            pose_resize_rs=pose_resize(pose_crop_rs)
            #print("pose_resiz",pose_resize_rs['img_shape']) 
        elif self.phase=='train' and rndNo==4:

            pose_flip=flip(left_kp=left_kps,right_kp=right_kps)
            pose_flip_rs=pose_flip(pose_compact_rs)

            pose_crop=CenterCrop(48)
            pose_crop_rs=pose_crop(pose_resize_rs)
            #print("pose_crop", pose_crop_rs['crop_bbox'],pose_crop_rs['img_shape'])

            pose_resize=resize(scalee)
            pose_resize_rs=pose_resize(pose_crop_rs)
            #print("pose_resiz",pose_resize_rs['img_shape']) 
        elif (self.phase=='train' and rndNo>4) or self.phase=='val': 
            pose_resize=resize(scalee)
            pose_resize_rs=pose_resize(pose_compact_rs)
            #print("pose_resiz",pose_resize_rs['img_shape'])


        pose_finall=GeneratePoseTarget()
        pose_final=pose_finall(pose_resize_rs)
        return pose_final


    def dataAugmentation(self,vid,filena,mode): 

        NoOfFrms = int(vid.get(cv2.CAP_PROP_FRAME_COUNT))
        Nofrms=self.closestNumber(NoOfFrms, NoOfFrms2slct_s) 
        #print("NoOfFrms,Nofrms",NoOfFrms,Nofrms,NoOfFrms2slct_s,filena)
        sltdfrms=self.unifrmsampler(Nofrms,NoOfFrms2slct_s,mode)

        snpt=[]
        for frN in range(len(sltdfrms)):  #NoOfFrms
            vid.set(1, sltdfrms[frN])
            ret, frame = vid.read()
            snpt.append(frame)
        trnfmvid=self.dataTransform(snpt,filena,mode)
        return trnfmvid

    def __len__(self):
        return len(self.video_infos)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        vidnme=self.video_infos[idx]['frame_dir'] + '.mp4'    #self.video_infos[idx]['filename'][-4:]

        vid_name = os.path.join(self.root_dir,vidnme)
        vid = cv2.VideoCapture(vid_name)
        #print("vid_name",vid_name)

        trsfmdvid=self.dataAugmentation(vid,vidnme,self.phase)
        
        pose_aug=self.dataAug(self.video_infos[idx],self.phase)
        #print("pose_aug",pose_aug['img_shape'])
        heatmap=pose_aug['imgs'][pose_aug['Frmstoselect']]

        #print("heatmap.shape",heatmap.shape)
        heatmap = torch.from_numpy(heatmap)
        heatmap=heatmap.permute(3,0,1,2)  
        
        label=self.video_infos[idx]['label']

        sample = {'vid': trsfmdvid, 'pose': heatmap,'label': label}

        return sample 