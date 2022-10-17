import os
import csv
#==================train_set=====================
path0_tr= '/data/kehao/DataSet/trainSet/0/'
path1_tr= '/data/kehao/DataSet/trainSet/1/'
path2_tr= '/data/kehao/DataSet/trainSet/2/'
path3_tr= '/data/kehao/DataSet/trainSet/3/'
path4_tr= '/data/kehao/DataSet/trainSet/4/'
path5_tr= '/data/kehao/DataSet/trainSet/5/'
path6_tr= '/data/kehao/DataSet/trainSet/6/'

#===================val_set======================
path0_val= '/data/kehao/DataSet/valSet/0/'
path1_val= '/data/kehao/DataSet/valSet/1/'
path2_val= '/data/kehao/DataSet/valSet/2/'
path3_val= '/data/kehao/DataSet/valSet/3/'
path4_val= '/data/kehao/DataSet/valSet/4/'
path5_val= '/data/kehao/DataSet/valSet/5/'
path6_val= '/data/kehao/DataSet/valSet/6/'

#===================test_set======================
path0_tst= '/data/kehao/DataSet/testSet/0/'
path1_tst= '/data/kehao/DataSet/testSet/1/'
path2_tst= '/data/kehao/DataSet/testSet/2/'
path3_tst= '/data/kehao/DataSet/testSet/3/'
path4_tst= '/data/kehao/DataSet/testSet/4/'
path5_tst= '/data/kehao/DataSet/testSet/5/'
path6_tst= '/data/kehao/DataSet/testSet/6/'

#======================save_name===================
name1='/home/kehao/bone_detection_classification/csv/box_train_mhd.csv'
name2='/home/kehao/bone_detection_classification/csv/box_val_mhd.csv'
name3='/home/kehao/bone_detection_classification/csv/box_test_mhd.csv'
#=================================================

def savaPath(name,path0,path1,path2,path3,path4,path5,path6):
    with open(name,'w',encoding='utf-8',newline='') as wt:
        writer=csv.writer(wt)
        for i in os.listdir(path0):
            print("==============")
            data=list()
            data.append(path0+i)
            data.append(0)
            writer.writerow(data)
            print(path0+i)

        for i in os.listdir(path1):
            print("==============")
            data=list()
            data.append(path1+i)
            data.append(1)
            writer.writerow(data)
            print(path1+i)
        
        for i in os.listdir(path2):
            print("==============")
            data=list()
            data.append(path2+i)
            data.append(2)
            writer.writerow(data)
            print(path2+i)
        
        for i in os.listdir(path3):
            print("==============")
            data=list()
            data.append(path3+i)
            data.append(3)
            writer.writerow(data)
            print(path3+i)

        for i in os.listdir(path4):
            print("==============")
            data=list()
            data.append(path4+i)
            data.append(4)
            writer.writerow(data)
            print(path4+i)
        
        for i in os.listdir(path5):
            print("==============")
            data=list()
            data.append(path5+i)
            data.append(5)
            writer.writerow(data)
            print(path5+i)

        for i in os.listdir(path6):
            print("==============")
            data=list()
            data.append(path6+i)
            data.append(6)
            writer.writerow(data)
            print(path6+i)



if __name__=='__main__':
    savaPath(name1,path0_tr,path1_tr,path2_tr,path3_tr,path4_tr,path5_tr,path6_tr)
    savaPath(name2,path0_val,path1_val,path2_val,path3_val,path4_val,path5_val,path6_val)
    savaPath(name3,path0_tst,path1_tst,path2_tst,path3_tst,path4_tst,path5_tst,path6_tst)
