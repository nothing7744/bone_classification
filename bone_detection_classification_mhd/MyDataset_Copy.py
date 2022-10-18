import os
import torch
import torchio as tio
from torch.utils.data import DataLoader
import numpy as np
import torch.nn as nn
#我们导入进行线性插值的库
import SimpleITK as sitk
import torch.nn.functional as F
from torch.utils.data import Dataset
import csv
import random




def weightGen(csvName):
    weight=[]
    with open(csvName,'r',newline='') as csvfile:
        reader=csv.reader(csvfile)
        for data in reader:
            if int(data[20])==0:
                weight.append(27.76)
            if int(data[20])==1:
                weight.append(10)
            if int(data[20])==2:
                weight.append(3.9)
            if int(data[20])==3:
                weight.append(30.4)
            if int(data[20])==4:
                weight.append(5.7)
            if int(data[20])==5:
                weight.append(1)
            if int(data[20])==6:
                weight.append(1.33)
    return weight


# print("=========================")
# print(weight1)


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


#adjust window and level
def window_transform(ct_array, windowWidth, windowCenter, normal=True):
    """
    return: trucated image according to window center and window width
    and normalized to [0,1]
    """
    minWindow = float(windowCenter) - 0.5*float(windowWidth)
    newing = (ct_array - minWindow)/float(windowWidth)
    newing[newing < 0] = 0
    newing[newing > 1] = 1
    #将值域转到0-255之间,例如要看头颅时， 我们只需将头颅的值域转换到 0-255 就行了
    if not normal:
        newing = (newing *255).astype('uint8')
    return newing


# #实现MyDatasets类
# we hope construct  a dataset instead of loading data according to the address every time


class MyDatasets(Dataset):
    def __init__(self, dir,csvname):
        self.list=[]
        self.data_dir = dir
        with open(os.path.join(dir, csvname), 'r') as fp:
            reader=csv.reader(fp)
            for data in reader:
                self.list.append(data)
            # 将所有图片的dir--label对都放入列表，如果要执行多个epoch，可以在这里多复制几遍，然后统一shuffle比较好
            # self.image_target_list = [(x[0].strip['['], int(x[21])) for x in str_list]


    def __getitem__(self, index):
        data = self.list[index]
        source=sitk.ReadImage(data[0])
        source.SetDirection=((1.0,0.0,0.0,0.0,1.0,0.0,0.0,0.0,1.0))
        origin=source.GetOrigin()
        spacing=source.GetSpacing()
        grapCentCoord=[float(data[2])-origin[0],float(data[3])-origin[1],float(data[4])-origin[2]]        
            # print(grapCentCoord)
        source=tio.ScalarImage.from_sitk(source)
        source=np.array(source)
        source=np.squeeze(source)
        edge=max(max(float(data[5]),float(data[6])),float(data[7]))
        sub_box=source[round((grapCentCoord[0]-edge*1.6/2)/spacing[0]):round((grapCentCoord[0]+edge*1.6/2)/spacing[0]),round((grapCentCoord[1]-edge*1.6/2)/spacing[1]):round((grapCentCoord[1]+edge*1.6/2)/spacing[1]),round((grapCentCoord[2]-edge*1.6/2)/spacing[2]):round((grapCentCoord[2]+edge*1.6/2)/spacing[2])]
        # sub_box=source[round((grapCentCoord[0]-float(data[5])*1.6/2)/spacing[0]):round((grapCentCoord[0]+float(data[5])*1.6/2)/spacing[0]),round((grapCentCoord[1]-float(data[6])*1.6/2)/spacing[1]):round((grapCentCoord[1]+float(data[6])*1.6/2)/spacing[1]),round((grapCentCoord[2]-float(data[7])*1.6/2)/spacing[2]):round((grapCentCoord[2]+float(data[7])*1.6/2)/spacing[2])]
        if(sub_box.shape[0]==0 or sub_box.shape[1]==0 or sub_box.shape[2]==0):
            idx=random.randint(1,100)
            return self.__getitem__(idx)
        else:
            sub_box=torch.Tensor(sub_box)
            sub_box=sub_box.unsqueeze(0)
            sub_box=sub_box.unsqueeze(1)
            sub_box=transform(sub_box)
            sub_box=sub_box.squeeze()
            sub_box=np.array(sub_box)
            sub_box=window_transform(sub_box,1500,300)
            sub_box=sitk.GetImageFromArray(sub_box)
            subject = tio.Subject(
                    source=tio.ScalarImage.from_sitk(sub_box),
                    label= int(data[20]),  
                )            
            return subject


    def __len__(self):
        return len(self.list)











