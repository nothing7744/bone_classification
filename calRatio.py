import os
import csv
count_train={"血管瘤":0,"许莫氏结节":0,"骨岛":0,"终板炎":0,"混合性":0,"成骨性":0,"溶骨性":0}
count_val={"血管瘤":0,"许莫氏结节":0,"骨岛":0,"终板炎":0,"混合性":0,"成骨性":0,"溶骨性":0}
count_test={"血管瘤":0,"许莫氏结节":0,"骨岛":0,"终板炎":0,"混合性":0,"成骨性":0,"溶骨性":0}

mapName1='./csv/train_data_map.csv'
mapName2='./csv/val_data_map.csv'
mapName3='./csv/test_data_map.csv'

def count(filename,countname):
    with open(filename,'r',encoding='utf-8')as fp:
        reader = csv.reader(fp)
        for data in reader:
            if(int(data[20])==0):
                countname["血管瘤"]=countname["血管瘤"]+1
            if(int(data[20])==1):
                countname["许莫氏结节"]=countname["许莫氏结节"]+1
            if(int(data[20])==2):
                countname["骨岛"]=countname["骨岛"]+1
            if(int(data[20])==3):
                countname["终板炎"]=countname["终板炎"]+1
            if(int(data[20])==4):
                countname["混合性"]=countname["混合性"]+1
            if(int(data[20])==5):
                countname["成骨性"]=countname["成骨性"]+1
            if(int(data[20])==6):
                countname["溶骨性"]=countname["溶骨性"]+1
        return countname
count_train=count(mapName1,count_train)
count_val=count(mapName2,count_val)
count_test=count(mapName3,count_test)
print(count_train)
print(count_val)
print(count_test)
