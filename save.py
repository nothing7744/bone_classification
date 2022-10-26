import SimpleITK as sitk
import torchio as tio
import torch
import numpy as np
import csv
import nibabel as nib
import torch.nn.functional as F
import pandas as pd
import os
from torchio.transforms import (
    RandomFlip,
    RandomAffine,
    RandomElasticDeformation,
    RandomNoise,
    RandomMotion,
    RandomBiasField,
    RescaleIntensity,
    Resample,
    ToCanonical,
    ZNormalization,
    CropOrPad,
    HistogramStandardization,
    OneOf,
    Compose,
)

dirname1='/data/kehao/dataset/trainSet/'
dirname2='/data/kehao/dataset/valSet/'
dirname3='/data/kehao/dataset/testSet/'
dirname4='/data/kehao/try/'


mapName1='./csv/train_data_map.csv'
mapName2='./csv/val_data_map.csv'
mapName3='./csv/test_data_map.csv'
mapName_label='./data/box_all_label.csv'

#read csv file and get the path



# Read the csv file


csvfile_label= pd.read_csv(mapName_label)
csvfile_label=np.array(csvfile_label)

csvfile_train= pd.read_csv(mapName1)
csvfile_train=np.array(csvfile_train)
# col_path_train=csvfile_train[:,0]

csvfile_val= pd.read_csv(mapName2)
csvfile_val=np.array(csvfile_val)
# col_path_val=csvfile_val[:,0]

csvfile_test= pd.read_csv(mapName3)
csvfile_test=np.array(csvfile_test)
# col_path_test=csvfile_test[:,0]

def window_transform(inp, windowWidth, windowCenter, normal=True):
    """
    return: trucated image according to window center and window width
    and normalized to [0,1]
    """
    minWindow = float(windowCenter) - 0.5*float(windowWidth)
    newing = (inp - minWindow)/float(windowWidth)
    newing[newing < 0] = 0
    newing[newing > 1] = 1
    if not normal:
        newing = (newing *255).astype('uint8')
    return newing


def transform(inp):
    edge=224
    new_w=torch.linspace(-1, 1, edge).view(-1, 1,1).repeat(1,edge,edge)
    new_h = torch.linspace(-1, 1, edge).view(1,-1,1).repeat(edge,1,edge)
    new_d= torch.linspace(-1, 1, edge).view(1,1,-1).repeat(edge,edge,1)
    grid = torch.cat((new_d.unsqueeze(3), new_h.unsqueeze(3),new_w.unsqueeze(3)), dim=3)
    grid = grid.unsqueeze(0)
    outp = F.grid_sample(inp, grid=grid, mode='bilinear', padding_mode='zeros',align_corners=False )
    outp=outp.squeeze()
    return outp


#when we write myDataset we can add the label
#save the box in our set
# transforms=[
#     ToCanonical(),
# ]
# transform=Compose(transforms)
# source=transform(source)
# # source.SetDirection=((1.0,0.0,0.0,0.0,1.0,0.0,0.0,0.0,1.0))

def PathtoImage(csvname,dirname):
    for i in range(len(csvname)-1):
        print("====================")
        data=csvname[i]
        source=sitk.ReadImage(data[0])
        print(source.GetDirection())
        source.SetDirection=((1.0,0.0,0.0,0.0,1.0,0.0,0.0,0.0,1.0))
        origin=source.GetOrigin()
        spacing=source.GetSpacing()
        grapCentCoord=[float(data[2])-origin[0],float(data[3])-origin[1],float(data[4])-origin[2]]        
        source=tio.ScalarImage.from_sitk(source)
        source=np.array(source)
        source=np.squeeze(source)
        edge=max(max(float(data[5]),float(data[6])),float(data[7]))
        sub_box=source[round((grapCentCoord[0]-edge*1.6/2)/spacing[0]):round((grapCentCoord[0]+edge*1.6/2)/spacing[0]),round((grapCentCoord[1]-edge*1.6/2)/spacing[1]):round((grapCentCoord[1]+edge*1.6/2)/spacing[1]),round((grapCentCoord[2]-edge*1.6/2)/spacing[2]):round((grapCentCoord[2]+edge*1.6/2)/spacing[2])]
        if(sub_box.shape[0]==0 or sub_box.shape[1]==0 or sub_box.shape[2]==0):
            pass
        else:
            img_t1=sitk.GetImageFromArray(sub_box)
            sitk.WriteImage(img_t1,dirname+str(data[20])+'/output'+str(i)+'.mhd')

            # img_t1 = nib.Nifti1Image(sub_box, np.eye(4))
            # nib.save(img_t1,dirname+str(data[20])+'/output'+str(i)+'.nii.gz')

            # subject = tio.Subject(
            #         source=tio.ScalarImage.from_sitk(sub_box),
            #         label= int(data[20]),  
            #     )     

       
 


if __name__=='__main__':

    PathtoImage(csvfile_train,dirname1)
    PathtoImage(csvfile_val,dirname2)
    PathtoImage(csvfile_test,dirname3)

    # PathtoImage(csvfile_label,dirname4)



    # def transform(self):
 
    #     if hp.aug: #数据增强开关
    #         training_transform = Compose([
    #         CropOrPad((hp.crop_or_pad_size), padding_mode='reflect'),#切割跟扩充
    #         # ToCanonical(),#规范坐标轴
    #         # RandomBiasField(),#随机 MRI 偏置场伪影
    #         ZNormalization(),#正则化
    #         # RandomNoise(),#高斯噪声
 
    #         # RandomGhosting(),#MRI 鬼影伪影
    #         # RandomMotion(),#运动伪影
    #         # RandomSpike(), #MRI 尖峰伪影
    #         # RandomBlur(),#随机大小的高斯滤波器模糊图像
    #         # RandomSwap(),#随机交换图像中的补丁
 
 
    #         RandomFlip(axes=(0,)),#反转图像中元素的顺序
    #         RandomAffine(),
 
    #         # OneOf({
    #         #     # RandomAffine(): 0.3,#倾斜
    #         #     # RandomElasticDeformation(): 0.1,#随机弹性变形
    #         # }),
    #         ])
 
 
    #     return training_transform








 

