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


def augument(self,sub_box):
    transform_dict={
        tio.RandomAffine(degrees=(-5,5,-5,5,-5,5),center='image',scales=(0.9,1.1,0.9,1.1,0.9,1.1),translation=(-4,2,-2,4,-2,2),default_pad_value='mean'),
        tio.RandomSwap(patch_size=3,num_iterations=20),
        }
    transforms=tio.Compose(transform_dict)
    transformed=transforms(sub_box)
    return transformed

# #实现MyDatasets类
# we hope construct  a dataset instead of loading data according to the address every time
# we use transform data augumentation

class MyDatasets(Dataset):
    def __init__(self, dir,csvname,trans):
        self.list=[]
        self.data_dir = dir
        self.trans=trans
        with open(os.path.join(dir, csvname), 'r') as fp:
            reader=csv.reader(fp)
            for data in reader:
                self.list.append(data)
            # 将所有图片的dir--label对都放入列表，如果要执行多个epoch，可以在这里多复制几遍，然后统一shuffle比较好
            # self.image_target_list = [(x[0].strip['['], int(x[21])) for x in str_list]


    def __getitem__(self, index):
        data = self.list[index]
        sub_box=sitk.ReadImage(data[0])
        sub_box.SetDirection=((1.0,0.0,0.0,0.0,1.0,0.0,0.0,0.0,1.0))   
        # sub_box=tio.ScalarImage.from_sitk(sub_box)
        # sub_box=np.array(sub_box)
        sub_box=sitk.GetArrayFromImage(sub_box)
        #half of dataset using data augument half of dataset not
        # a=random.randint(1,10)
        # if self.trans==True and a>5:
        #     sub_box=self.tranfrom(sub_box)
        # else:
        #     pass
        # sub_box=self.tranfrom(sub_box)

        sub_box=np.squeeze(sub_box)
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
                    label= int(data[1]),  
                )            
        return subject


    def __len__(self):
        return len(self.list)


    def tranfrom(self,sub_box):
        transform_dict={
        tio.RandomAffine(degrees=(-5,5,-5,5,-5,5),center='image',scales=(0.9,1.1,0.9,1.1,0.9,1.1),translation=(-4,2,-2,4,-2,2),default_pad_value='mean'),
        tio.RandomSwap(patch_size=3,num_iterations=20),
        }
        transforms=tio.Compose(transform_dict)
        transformed=transforms(sub_box)
        return transformed

if __name__=="__main__":
    pass














