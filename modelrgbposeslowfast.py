import torch.nn.functional as nnf
import torch.nn as nn

class block3d(nn.Module):
	def __init__(self,in_channels,out_channels,identity_downsample=None,kernel_size=(1,1,1),stride=(1,1,1),padding=(0,0,0)):
		super(block3d,self).__init__()
		self.expansion=4
		self.conv1=nn.Conv3d(in_channels,out_channels,kernel_size=kernel_size,stride=(1,1,1),padding=padding, bias=False)
		self.bn1=nn.BatchNorm3d(out_channels, momentum=0,track_running_stats=False) 
		self.conv2=nn.Conv3d(out_channels,out_channels,kernel_size=(1,3,3),stride=stride,padding=(0, 1, 1), bias=False)
		self.bn2=nn.BatchNorm3d(out_channels, momentum=0,track_running_stats=False) 
		self.conv3=nn.Conv3d(out_channels,out_channels*self.expansion,kernel_size=(1, 1, 1),stride=(1,1,1),padding=(0, 0, 0), bias=False)
		self.bn3=nn.BatchNorm3d(out_channels*self.expansion, momentum=0,track_running_stats=False) 
		self.relu=nn.ReLU()
		self.identity_downsample=identity_downsample

	def forward(self,x):

		identity=x
		#print("x.shape000",x.shape)

		x=self.conv1(x)
		x=self.bn1(x)
		x=self.relu(x)
		#print("x.shape1111",x.shape)
		x=self.conv2(x)
		x=self.bn2(x)
		x=self.relu(x)
		#print("x.shape222",x.shape,identity.shape)
		x=self.conv3(x)
		x=self.bn3(x)
		x=self.relu(x)
		if self.identity_downsample is not None:
			identity=self.identity_downsample(identity)
			#print("identity.shape",identity.shape)
		#print("x.shape333",x.shape)
		x+=identity
		x=self.relu(x)
		return x


class RgbPoseSlowFast(nn.Module):
	def __init__(self,block3d,layers,img_channels,pose_channels,num_classes,dropout_ratio):
		super(RgbPoseSlowFast,self).__init__()
		self.in_channels=64
		#self.in_channels_fast=4
		self.conv1_slow=nn.Conv3d(img_channels,64,kernel_size=(1,7,7),stride=(1,2,2),padding=(0,3,3))
		self.bn1_slow=nn.BatchNorm3d(64) 
		self.relu_slow=nn.ReLU()
		self.maxpool_slow=nn.MaxPool3d(kernel_size=(1,3,3),stride=(1,2,2),padding=(0,1,1),dilation=1)

		self.conv1_fast=nn.Conv3d(pose_channels,32	,kernel_size=(1,7,7),stride=(1,1,1),padding=(0,3,3))
		self.bn1_fast=nn.BatchNorm3d(32) 
		self.relu_fast=nn.ReLU()

		#self.conv1_fast=nn.conv3d(in_channels_fast,out_channels,kernel_size=1,stride=1,padding=0)
		#self.bn1_fast=nn.BatchNorm3d(o)

		#if lateral is True:
			#lateral_downsample=nn.Sequential((nn.Conv2d(self.in_channels+lateral_channel,out_channels,kernel_size=1,stride=stride)),nn.BatchNorm2d(out_channels*4))

		#Resnet layers
		
		self.layer1_slow = self._make_layer(block3d,layers[0],out_channels=64,kernel_size=(1,1,1),stride=(1,1,1),padding=(0,0,0))
		self.layer2_slow = self._make_layer(block3d,layers[1],out_channels=128,kernel_size=(1,1,1),stride=(1,2,2),padding=(0,0,0))
		self.layer3_slow = self._make_layer(block3d,layers[2],out_channels=256,kernel_size=(3,1,1),stride=(1,2,2),padding=(1,0,0))
		self.layer4_slow = self._make_layer(block3d,layers[3],out_channels=512,kernel_size=(3,1,1),stride=(1,2,2),padding=(1,0,0))

		self.in_channels=int(64/2)
		self.layer2_fast = self._make_layer(block3d,layers[1],out_channels=32,kernel_size=(1,1,1),stride=(1,2,2),padding=(0,0,0))
		self.layer3_fast = self._make_layer(block3d,layers[2],out_channels=64,kernel_size=(3,1,1),stride=(1,2,2),padding=(1,0,0))
		self.layer4_fast = self._make_layer(block3d,layers[3],out_channels=128,kernel_size=(3,1,1),stride=(1,2,2),padding=(1,0,0))


		self.lateral1_slow=nn.Conv3d(256,32,kernel_size=(5,1,1),stride=(8,1,1),padding=(2,0,0))
		self.lateral2_slow=nn.Conv3d(512,128,kernel_size=(5,1,1),stride=(8,1,1),padding=(2,0,0))
		self.lateral3_slow=nn.Conv3d(1024,256,kernel_size=(5,1,1),stride=(8,1,1),padding=(2,0,0))

		self.lateral1_fast=nn.Conv3d(32,256,kernel_size=(5,1,1),stride=(5,1,1),padding=(2,0,0))
		self.lateral2_fast=nn.Conv3d(128,512,kernel_size=(5,1,1),stride=(5,1,1),padding=(2,0,0))
		self.lateral3_fast=nn.Conv3d(256,1024,kernel_size=(5,1,1),stride=(5,1,1),padding=(2,0,0))

		
		self.avgpool_slow=nn.AdaptiveAvgPool3d((1,1,1))
		self.avgpool_fast=nn.AdaptiveAvgPool3d((1,1,1))
		self.dropout = nn.Dropout(dropout_ratio)
		self.fc_cls_sf = nn.Linear(2560,num_classes)

		#self.soft_max= nn.Softmax(dim=1)

	def forward(self,x1,x2):
		x1=self.conv1_slow(x1)
		x1=self.bn1_slow(x1)
		x1=self.relu_slow(x1)
		x1=self.maxpool_slow(x1)
		x1=self.layer1_slow(x1)

		x2=self.conv1_fast(x2)
		x2=self.bn1_fast(x2)
		x2=self.relu_fast(x2)

		#print("x1.shape,x2.shape",x1.shape,x2.shape)
		x1=x1+self.lateral1_fast(x2)
		x2=self.lateral1_slow(x1)+x2

		x1=self.layer2_slow(x1)
		x2=self.layer2_fast(x2)
		
		x1=x1+self.lateral2_fast(x2)
		x2=self.lateral2_slow(x1)+x2

		#print("x1.shape......",x1.shape)
		x1=self.layer3_slow(x1)
		x2=self.layer3_fast(x2)

		#la=self.lateral3_fast(x2)
		#print("x1.shape,x2.shape,la.shape....",x1.shape,x2.shape,la.shape)

		x1=x1+self.lateral3_fast(x2)
		x2=self.lateral3_slow(x1)+x2

		x1=self.layer4_slow(x1)
		x2=self.layer4_fast(x2)
		
		# ([N, channel_fast, 1, 1, 1], [N, channel_slow, 1, 1, 1])
		x1=self.avgpool_slow(x1)
		x2=self.avgpool_fast(x2)
		#print("x1.shape,x2.shape",x1.shape,x2.shape)

		# [N, channel_fast + channel_slow, 1, 1, 1]
		x = torch.cat((x1, x2), dim=1)
		x = self.dropout(x)
		#print("x.shape",x.shape)
		# [N x C]
		x = x.view(x.size(0), -1)
		# [N x num_classes]
		cls_score =self.fc_cls_sf(x)
		#cls_score = self.soft_max(cls_score)

		return cls_score

	def _make_layer(self,block3d,num_residual_blocks,out_channels,kernel_size,stride,padding):
		identity_downsample=None
		layers=[]

		if stride != (1,1,1) or self.in_channels!=out_channels*4:
			identity_downsample=nn.Sequential((nn.Conv3d(self.in_channels,out_channels*4,kernel_size=(1,1,1),stride=stride)),nn.BatchNorm3d(out_channels*4))
		layers.append(block3d(self.in_channels,out_channels,identity_downsample,kernel_size=kernel_size,stride=stride,padding=padding))
		self.in_channels=out_channels*4

		for i in range(num_residual_blocks-1):
			layers.append(block3d(self.in_channels,out_channels,kernel_size=kernel_size,padding=padding))
		
		return nn.Sequential(*layers)	

def RgbPoseSlowFastResNet50(layers=[3,4,6,3],img_channels=3,pose_channels=17,num_classes=8,dropout_ratio=.5):
	return  RgbPoseSlowFast(block3d,layers,img_channels,pose_channels,num_classes,dropout_ratio) 

#----------------------

class block3dSmall(nn.Module):
  def __init__(self,in_channels,out_channels,identity_downsample=None,kernel_size=(1,3,3),stride=(1,1,1),padding=(0,1,1)):
    super(block3dSmall,self).__init__()
    self.expansion=4
    self.conv1=nn.Conv3d(in_channels,out_channels,kernel_size=kernel_size,stride=stride,padding=padding, bias=False)
    self.bn1=nn.BatchNorm3d(out_channels) 
    self.conv2=nn.Conv3d(out_channels,out_channels,kernel_size=(1,3,3),stride=(1,1,1),padding=(0, 1, 1), bias=False)
    self.bn2=nn.BatchNorm3d(out_channels) 
    self.relu=nn.ReLU()
    self.identity_downsample=identity_downsample

  def forward(self,x):

    identity=x
    #print("x.shape000",x.shape)

    x=self.conv1(x)
    x=self.bn1(x)
    x=self.relu(x)
    #print("x.shape1111",x.shape)
    x=self.conv2(x)
    x=self.bn2(x)
    x=self.relu(x)
    #print("x.shape222",x.shape,identity.shape)
    if self.identity_downsample is not None:
      identity=self.identity_downsample(identity)
      #print("identity.shape",identity.shape)
    #print("x.shape333,identity",x.shape,identity.shape)
    x+=identity
    x=self.relu(x)
    return x

class RgbPoseSlowFastSmall(nn.Module):
  def __init__(self,block3dSmall,layers,img_channels,pose_channels,num_classes,dropout_ratio):
    super(RgbPoseSlowFastSmall,self).__init__()
    self.in_channels=64
    #self.in_channels_fast=4
    self.conv1_slow=nn.Conv3d(img_channels,64,kernel_size=(1,7,7),stride=(1,2,2),padding=(0,3,3), bias=False)
    self.bn1_slow=nn.BatchNorm3d(64) 
    self.relu_slow=nn.ReLU()
    self.maxpool_slow=nn.MaxPool3d(kernel_size=(1,3,3),stride=(1,2,2),padding=(0,1,1),dilation=1)

    self.conv1_fast=nn.Conv3d(pose_channels,16,kernel_size=(1,7,7),stride=(1,1,1),padding=(0,3,3), bias=False)
    self.bn1_fast=nn.BatchNorm3d(16) 
    self.relu_fast=nn.ReLU()

    #Resnet layers
    
    self.layer1_slow = self._make_layer(block3dSmall,layers[0],out_channels=64,kernel_size=(1,3,3),stride=(1,1,1),padding=(0,1,1))
    self.layer2_slow = self._make_layer(block3dSmall,layers[1],out_channels=128,kernel_size=(1,3,3),stride=(1,2,2),padding=(0,1,1))
    self.layer3_slow = self._make_layer(block3dSmall,layers[2],out_channels=256,kernel_size=(3,3,3),stride=(1,2,2),padding=(1,1,1))
    self.layer4_slow = self._make_layer(block3dSmall,layers[3],out_channels=512,kernel_size=(3,3,3),stride=(1,2,2),padding=(1,1,1))

    #Resnet layers
    self.in_channels=int(32/2)
    #self.layer1_slow = self._make_layer(block3dSmall,layers[0],out_channels=64,kernel_size=(1,3,3),stride=(1,1,1),padding=(0,1,1))
    self.layer2_fast = self._make_layer(block3dSmall,layers[1],out_channels=32,kernel_size=(1,3,3),stride=(1,2,2),padding=(0,1,1))
    self.layer3_fast = self._make_layer(block3dSmall,layers[2],out_channels=64,kernel_size=(3,3,3),stride=(1,2,2),padding=(1,1,1))
    self.layer4_fast = self._make_layer(block3dSmall,layers[3],out_channels=128,kernel_size=(3,3,3),stride=(1,2,2),padding=(1,1,1))

    self.lateral1_slow=nn.Conv3d(64,16,kernel_size=(5,1,1),stride=(8,1,1),padding=(2,0,0))
    self.lateral2_slow=nn.Conv3d(128,32,kernel_size=(5,1,1),stride=(8,1,1),padding=(2,0,0))
    self.lateral3_slow=nn.Conv3d(256,64,kernel_size=(5,1,1),stride=(8,1,1),padding=(2,0,0))

    self.lateral1_fast=nn.Conv3d(16,64,kernel_size=(5,1,1),stride=(5,1,1),padding=(2,0,0))
    self.lateral2_fast=nn.Conv3d(32,128,kernel_size=(5,1,1),stride=(5,1,1),padding=(2,0,0))
    self.lateral3_fast=nn.Conv3d(64,256,kernel_size=(5,1,1),stride=(5,1,1),padding=(2,0,0))


    self.avgpool_slow=nn.AdaptiveAvgPool3d((1,1,1))
    self.avgpool_fast=nn.AdaptiveAvgPool3d((1,1,1))
    self.dropout = nn.Dropout(dropout_ratio)
    self.fc_cls_sf = nn.Linear(640,num_classes)

    #self.soft_max= nn.Softmax(dim=1)

  def forward(self,x1,x2):
    x1=self.conv1_slow(x1)
    x1=self.bn1_slow(x1)
    x1=self.relu_slow(x1)
    x1=self.maxpool_slow(x1)
    x1=self.layer1_slow(x1)
    
    x2=self.conv1_fast(x2)
    x2=self.bn1_fast(x2)
    x2=self.relu_fast(x2)
    
    #print("x1.shape,x2.shape",x1.shape,x2.shape)
    x1=x1+self.lateral1_fast(x2)
    x2=self.lateral1_slow(x1)+x2
    
    x1=self.layer2_slow(x1)
    x2=self.layer2_fast(x2)
    
    x1=x1+self.lateral2_fast(x2)
    x2=self.lateral2_slow(x1)+x2
    
    #print("x1.shape......",x1.shape)
    x1=self.layer3_slow(x1)
    x2=self.layer3_fast(x2)
    
    #la=self.lateral3_fast(x2)
    #print("x1.shape,x2.shape,la.shape....",x1.shape,x2.shape)
    
    x1=x1+self.lateral3_fast(x2)
    x2=self.lateral3_slow(x1)+x2
    
    x1=self.layer4_slow(x1)
    x2=self.layer4_fast(x2)
    
    # ([N, channel_fast, 1, 1, 1], [N, channel_slow, 1, 1, 1])
    x1=self.avgpool_slow(x1)
    x2=self.avgpool_fast(x2)
    #print("x1.shape,x2.shape",x1.shape,x2.shape)
    
    # [N, channel_fast + channel_slow, 1, 1, 1]
    x = torch.cat((x1, x2), dim=1)
    x = self.dropout(x)
    #print("x.shape",x.shape)
    # [N x C]
    x = x.view(x.size(0), -1)
    # [N x num_classes]
    cls_score =self.fc_cls_sf(x)
    #cls_score = self.soft_max(cls_score)

    return cls_score

  def _make_layer(self,block3dSmall,num_residual_blocks,out_channels,kernel_size,stride,padding):
    identity_downsample=None
    layers=[]
    
    if stride != (1,1,1) or self.in_channels!=out_channels*4:
      identity_downsample=nn.Sequential((nn.Conv3d(self.in_channels,out_channels,kernel_size=(1,3,3),stride=stride,padding=(0,1,1))),nn.BatchNorm3d(out_channels))
    layers.append(block3dSmall(self.in_channels,out_channels,identity_downsample,kernel_size=kernel_size,stride=stride,padding=padding))
    self.in_channels=out_channels
    for i in range(num_residual_blocks-1):
      layers.append(block3dSmall(self.in_channels,out_channels,kernel_size=kernel_size,padding=padding))
    return nn.Sequential(*layers) 

def RgbPoseSlowFastResNet34(layers=[3,4,6,3],img_channels=3,pose_channels=17,num_classes=8,dropout_ratio=0.5):
  return  RgbPoseSlowFastSmall(block3dSmall,layers,img_channels,pose_channels,num_classes,dropout_ratio)  