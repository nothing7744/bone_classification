#split method use
# import os
# name='box_all_anno.csv'
# print(name.split('.')[1]=="csv")
#

##string.split('.')
import csv
import os
path0='/data/kehao/dataset/trainSet/0/'
if __name__=="__main__":
    for i in os.listdir(path0):
        if i.split('.',1)[1]=="nii.gz":
            print("==============")
            print(path0+i)

#the 5 class have some problem in classifation