from dataclasses import replace
import os
from tqdm import tqdm
import numpy as np
import torch
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from tensorboardX import SummaryWriter
import time
import torch.nn as nn
import torch
import torchvision.transforms as transforms
from PIL import Image
from matplotlib import pyplot as plt
import torchvision.models as models
import torchsummary
import time
from torch.optim.lr_scheduler import ExponentialLR
from generateModel import generate_model
from MyDataset import weightGen, window_transform
from MyDataset import MyDatasets
from torch.utils.data import WeightedRandomSampler
from torch.optim.lr_scheduler import StepLR
from loss import FocalLoss,CELoss,CB_loss
from medcam import medcam
import SimpleITK as sitk
import torch.nn.functional as F
print('device =',device)
print(torch.cuda.get_device_name(0))
import torchio as tio
import os
import cv2
import numpy as np
import nibabel as nib
import SimpleITK as sitk
from PIL import Image



model = generate_model(model_type='resnet', model_depth=18,
                   input_W=112, input_H=112, input_D=112, resnet_shortcut='B',
                   no_cuda=False, gpu_id=[4],
                   nb_class=7)
model.to(device)


print("===========model finish======================")
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3,weight_decay=0.001)


criterion = nn.CrossEntropyLoss()
# alpha=[27.76,10,3.9,30.4,5.7,1,1.33]
# criterion=FocalLoss(7, alpha=None, gamma=2, use_alpha=False, size_average=False)


# scheduler = ExponentialLR(optimizer, gamma=0.99)
scheduler = StepLR(optimizer, step_size=10, gamma=0.5)
num_epochs =50
dirname='/home/kehao/bone_detection_classification/'
csvname1='./csv/box_train.csv'
csvname2='./csv/box_val_10_everyClass.csv'
csvname3='./csv/box_test.csv'
#######################################


weight1=torch.tensor(weightGen(csvname1))
weight2=torch.tensor(weightGen(csvname2))
weight3=torch.tensor(weightGen(csvname3))


######################################
train_set=MyDatasets(dirname,csvname1)
val_set=MyDatasets(dirname,csvname2)
test_set=MyDatasets(dirname,csvname3)
########################################
batch=1
sampler1=WeightedRandomSampler(weight1,len(weight1),replacement=True)
sampler2=WeightedRandomSampler(weight2,len(weight2),replacement=True)

# sampler1=WeightedRandomSampler(weight1,50,replacement=True)
# sampler2=WeightedRandomSampler(weight2,50,replacement=True)
sampler3=WeightedRandomSampler(weight3,len(weight3),replacement=True)


# sampler3=WeightedRandomSampler(weight3,100,replacement=True)


# train_loader = DataLoader(train_set, batch_size=batch, sampler=sampler1,num_workers=16,shuffle=False)
train_loader = DataLoader(train_set, batch_size=batch, num_workers=16,shuffle=True)
# val_loader=DataLoader(val_set, batch_size=batch, sampler=sampler2,num_workers=16,shuffle=False)
# test_loader=DataLoader(test_set, batch_size=batch, sampler=sampler3,shuffle=False)
val_loader=DataLoader(val_set, batch_size=batch, num_workers=16,shuffle=False)
test_loader=DataLoader(test_set, batch_size=batch,  num_workers=16,shuffle=False)
summaryWriter = SummaryWriter(log_dir='logs',comment='Linear')
##############################################################################


RESUME=True
start_epoch=0
if RESUME:
    path_checkpoint = "/data/kehao/models/checkpoint/ckpt_best_96.pth"  
    checkpoint = torch.load(path_checkpoint)  
    model.load_state_dict(checkpoint['net'])  
    optimizer.load_state_dict(checkpoint['optimizer'])  
    start_epoch = checkpoint['epoch']  


model = medcam.inject(model, backend='gcam',output_dir="attention_maps", label='best',save_maps=True)

def transform(inp):
    edge=112
    new_w=torch.linspace(-1, 1, edge).view(-1, 1,1).repeat(1,edge,edge)
    new_h = torch.linspace(-1, 1, edge).view(1,-1,1).repeat(edge,1,edge)
    new_d= torch.linspace(-1, 1, edge).view(1,1,-1).repeat(edge,edge,1)
    grid = torch.cat((new_d.unsqueeze(3), new_h.unsqueeze(3),new_w.unsqueeze(3)), dim=3)
    grid = grid.unsqueeze(0)
    outp = F.grid_sample(inp, grid=grid, mode='bilinear', padding_mode='zeros',align_corners=False )
    outp=outp.squeeze()
    return outp


path='/home/kehao/bone_detection_classification_mhd/attention_maps/module.layer4/'      
dirname2='/home/kehao/bone_detection_classification_mhd/attention_maps/hot_image'
dirname3='/home/kehao/bone_detection_classification_mhd/attention_maps/image_png'

# if __name__=='__main__':
#     print("hello,world")
#     model.eval()
#     for i,batch in tqdm(enumerate(val_loader)):
#         x = batch['source']['data']
#         label = batch['label']          
#         x=x.type(torch.FloatTensor).cuda()      
#         label = label.type(torch.LongTensor).cuda()        
#         label = label.type(torch.LongTensor).to(device)
#         label = torch.squeeze(label)
#         output=model(x)



if __name__=='__main__':
    for i,batch in tqdm(enumerate(val_loader)):
            x = batch['source']['data']
        # x=x.type(torch.FloatTensor).cuda() 
            x=np.array(x)
            path_r=path+'attention_map_'+str(i)+'_0_0.nii.gz'
            data=sitk.ReadImage(path_r)
            data=tio.ScalarImage.from_sitk(data)
            data=np.array(data)
            data=data.squeeze()
            data=torch.Tensor(data)
            data=data.unsqueeze(0)
            data=data.unsqueeze(1)
            data=transform(data)
            data=np.array(data)
            data=data.squeeze()
            #normalize
            data=(data-np.min(data))/float(np.max(data))
            y=data*x
            m=sitk.GetImageFromArray(y)

            # img = (x- x.min())/(x.max()-x.min())
            # img = img * 255
            # cv2.imwrite(dirname3 +'image'+str(i) +'.png', img)
            origin=sitk.GetImageFromArray(x)
            sitk.WriteImage(origin,dirname2+'/output_o'+str(i)+'.mhd')
            sitk.WriteImage(m,dirname2+'/output'+str(i)+'.mhd')

    # train()

         








