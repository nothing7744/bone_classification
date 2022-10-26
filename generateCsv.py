import os
import csv
from pathlib import Path

filename1='./data/box_all_anno.csv'
filename2='./data/box_all_label.csv'

mapName1='./csv/train_data_map.csv'
mapName2='./csv/val_data_map.csv'
mapName3='./csv/test_data_map.csv'

train_set=set()
val_set=set()
test_set=set()
setAll=set()

#label our data firstly
def genearte_label():
    with open(filename2,'w',encoding='utf-8',newline='') as wt:
        writer=csv.writer(wt)
        with open(filename1,'r',encoding='utf-8')as fp:
            reader = csv.reader(fp)
            for data in reader:
                if(data[9]=="血管瘤"):
                    data.append(0)
                    writer.writerow(data)
                if(data[9]=="许莫氏结节"):
                    data.append(1)
                    writer.writerow(data)
                if(data[9]=="骨岛"):
                    data.append(2)
                    writer.writerow(data)
                if(data[9]=="终板炎"):
                    data.append(3)
                    writer.writerow(data)
                if(data[9]=="混合性"):
                    data.append(4)
                    writer.writerow(data)
                if(data[9]=="成骨性"):
                    data.append(5)
                    writer.writerow(data)
                if(data[9]=="溶骨性"):
                    data.append(6)
                    writer.writerow(data)

#generate label
# genearte_label()

#devide dataset into train_set,val_set and test_set
def generate(name,setname):
    with open(name,'r',encoding='utf-8') as fp:
        reader=csv.reader(fp)
        for data in reader:
            path=Path(data[0])
            setname.add(os.path.dirname(path))
        # print(setAll)
        return setname


def generateDataMap(setname,name1,name2):
    with open(name2,'w',encoding='utf-8',newline='') as wt:
        writer=csv.writer(wt)
        with open(name1,'r',encoding='utf-8') as fp:
            reader=csv.reader(fp)
            for data in reader:
                path=Path(data[0])
                a=set()
                a.add(os.path.dirname(path))
                if(a.issubset(setname)):
                    writer.writerow(data)



setAll=generate(filename2,setAll)
lens=len(setAll)
listAll=list(setAll)
list_train=listAll[:round(0.7*lens)]
list_val=listAll[round(0.7*lens):round(0.9*lens)]
list_test=listAll[round(0.9*lens):lens-1]
set_train=set(list_train)
set_val=set(list_val)
set_test=set(list_test)

generateDataMap(set_train,filename2,mapName1)
generateDataMap(set_val,filename2,mapName2)
generateDataMap(set_test,filename2,mapName3)


#judge that we devide the dataset according to the people
set1=set()
set2=set()
set3=set()
set1=generate(mapName1,set1)
print(len(set1))
set2=generate(mapName2,set2)
print(len(set2))
set3=generate(mapName3,set3)
print(len(set3))


