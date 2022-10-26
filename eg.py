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
from matplotlib import pyplot as plt
from matplotlib import font_manager
import random
# path0='/data/kehao/dataset/trainSet/0/'
# if __name__=="__main__":
#     for i in os.listdir(path0):
#         if i.split('.',1)[1]=="nii.gz":
#             print("==============")
#             print(path0+i)

# #the 5 class have some problem in classifation
mapName='/home/kehao/bone_detection_classification_mhd/data/box_all_anno.csv'
csvfile= pd.read_csv(mapName)
csvname=np.array(csvfile)
length=len(csvname)
dirname='/data/kehao/try/'
#将所有的文件中的尺寸信息保存在一个list中



#根据list的尺寸绘制直方图
#之后在这个统计各个数据的数目
# a = [random.randint(80,140) for i in range(250)]
# print(a)
# print(max(a)-min(a))
 
# # 计算组数
# d = 3  # 组距
# num_bins = (max(a)-min(a))//d
 
# # 设置图形大小
# plt.figure(figsize=(20, 8), dpi=80)
# plt.hist(a, num_bins)
 
# # 设置x轴刻度
# plt.xticks(range(min(a), max(a)+d, d))
 
# # 设置网格
# plt.grid(alpha=0.4)
# plt.show()


def getList():
    list=[]
    for i in range(length-1):
        print("====================")
        data=csvname[i]
        sub_box=sitk.ReadImage(data[0])
        spacing=sub_box.GetSpacing()
        print(spacing)
        edge=max(max(float(data[5]),float(data[6])),float(data[7]))
        a=round(max(edge/spacing[0],max(edge/spacing[1],edge/spacing[2])))
        list.append(a)
    return list



if __name__=='__main__':
    list=getList()
    max_list=max(list)
    min_list=min(list)
    d=3
    num_bins=(max_list-min_list)//d
    plt.figure(figsize=(20, 8), dpi=80)
    plt.hist(list, num_bins)
    plt.xticks(range(min_list, max_list, d))
    plt.grid(alpha=0.4)
    plt.savefig("./data/bin.png")
    plt.show()




    





