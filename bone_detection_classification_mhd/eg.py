#split method use
# import os
# name='box_all_anno.csv'
# print(name.split('.')[1]=="csv")
#

##string.split('.')
import csv
import os
import torch
import torch.nn.functional as F
import pandas as pd
import numpy as np
import torchio as tio
import SimpleITK as sitk
# path0='/data/kehao/dataset/trainSet/0/'
# if __name__=="__main__":
#     for i in os.listdir(path0):
#         if i.split('.',1)[1]=="nii.gz":
#             print("==============")
#             print(path0+i)

# #the 5 class have some problem in classifation
mapName='/home/kehao/bone_detection_classification/csv/box_train_mhd.csv'
csvfile= pd.read_csv(mapName)
csvname=np.array(csvfile)
dirname='/data/kehao/try/'
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

if __name__=='__main__':
    for i in range(30):
        print("====================")
        data=csvname[i]
        sub_box=sitk.ReadImage(data[0])
        sub_box.SetDirection=((1.0,0.0,0.0,0.0,1.0,0.0,0.0,0.0,1.0))   
        # sub_box=tio.ScalarImage.from_sitk(sub_box)
        # sub_box=np.array(sub_box)
        sub_box=sitk.GetArrayFromImage(sub_box)
        sub_box=np.squeeze(sub_box)
        sub_box=torch.Tensor(sub_box)
        sub_box=sub_box.unsqueeze(0)
        sub_box=sub_box.unsqueeze(1)
        sub_box=transform(sub_box)
        sub_box=sub_box.squeeze()
        sub_box=np.array(sub_box)
        sub_box=window_transform(sub_box,1500,300)
        img_t1=sitk.GetImageFromArray(sub_box)
        sitk.WriteImage(img_t1,dirname+str(data[1])+'/output'+str(i)+'.mhd')


