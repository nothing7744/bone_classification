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
from pathlib import Path



#you shouldn't make the sample's value too big
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


class My_dataset(Dataset): # 注意这里的父类继承
    def __init__(self):
        super().__init__()
        # 使用sin函数返回10000个时间序列,如果不自己构造数据，就使用numpy,pandas等读取自己的数据为x即可。
        # 以下数据组织这块既可以放在init方法里，也可以放在getitem方法里
        self.x = torch.randn(1000,3)
        self.y = self.x.sum(axis=1)
        self.src,  self.trg = [], []
        for i in range(1000):
            self.src.append(self.x[i])
            self.trg.append(self.y[i])
           
    def __getitem__(self, index):
        return self.src[index], self.trg[index]

    def __len__(self):
        return len(self.src)


#get the image form the data  dirname
class MedData_train(torch.utils.data.Dataset):
    def __init__(self, images_dir_0, images_dir_1):
        self.subjects = []
        images_dir_0 = Path(images_dir_0)
        self.image_paths_0 = sorted(images_dir_0.glob('.nii.gz'))
        images_dir_1 = Path(images_dir_1)
        self.image_paths_1 = sorted(images_dir_1.glob('.nii.gz'))
        #导入所有的医学图像
        for (image_path) in zip(self.image_paths_0):
            subject = tio.Subject(
                source=tio.ScalarImage(image_path),
                label= 0,
            )
            self.subjects.append(subject)
        for (image_path) in zip(self.image_paths_1):
            subject = tio.Subject(
                source=tio.ScalarImage(image_path),
                label= 1,
            )
        
            self.subjects.append(subject)
        self.transforms = self.transform()
        self.training_set = tio.SubjectsDataset(self.subjects, transform=self.transforms)
        # one_subject = self.training_set[0]
        # one_subject.plot()





#get all the box path
#==================train_set=====================
path0_tr= '/data/kehao/dataset/trainSet/0/'
path1_tr= '/data/kehao/dataset/trainSet/1/'
path2_tr= '/data/kehao/dataset/trainSet/2/'
path3_tr= '/data/kehao/dataset/trainSet/3/'
path4_tr= '/data/kehao/dataset/trainSet/4/'
path5_tr= '/data/kehao/dataset/trainSet/5/'
path5_tr= '/data/kehao/dataset/trainSet/6/'

#===================val_set======================
path0_val= '/data/kehao/dataset/valSet/0/'
path1_val= '/data/kehao/dataset/valSet/1/'
path2_val= '/data/kehao/dataset/valSet/2/'
path3_val= '/data/kehao/dataset/valSet/3/'
path4_val= '/data/kehao/dataset/valSet/4/'
path5_val= '/data/kehao/dataset/valSet/5/'
path5_val= '/data/kehao/dataset/valSet/6/'

#===================test_set======================
path0_tst= '/data/kehao/dataset/testSet/0/'
path1_tst= '/data/kehao/dataset/testSet/1/'
path2_tst= '/data/kehao/dataset/testSet/2/'
path3_tst= '/data/kehao/dataset/testSet/3/'
path4_tst= '/data/kehao/dataset/testSet/4/'
path5_tst= '/data/kehao/dataset/testSet/5/'
path5_tst= '/data/kehao/dataset/testSet/6/'

#======================save_name===================
name1='/home/kehao/bone_detection_classification/csv/box_train.csv'
name2='/home/kehao/bone_detection_classification/csv/box_val.csv'
name3='/home/kehao/bone_detection_classification/csv/box_test.csv'
if __name__=='__main__':
    with open(name1,'w',encoding='utf-8',newline='') as wt:
        writer=csv.writer(wt)
        for i in os.listdir(path1):
            print("==============")
            data=list()
            data.append(path1+i)
            data.append(1)
            writer.writerow(data)
            print(path1+i)














