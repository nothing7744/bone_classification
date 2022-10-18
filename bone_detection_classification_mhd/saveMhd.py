import SimpleITK as sitk
import numpy as np
import pandas as pd



dirname1='/data/kehao/DataSet/trainSet/'
dirname2='/data/kehao/DataSet/valSet/'
dirname3='/data/kehao/DataSet/testSet/'
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

csvfile_val= pd.read_csv(mapName2)
csvfile_val=np.array(csvfile_val)

csvfile_test= pd.read_csv(mapName3)
csvfile_test=np.array(csvfile_test)





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
        # source=tio.ScalarImage.from_sitk(source)
        # source=np.array(source)
        source=sitk.GetArrayFromImage(source)
        edge=max(max(float(data[5]),float(data[6])),float(data[7]))
        # sub_box=source[round((grapCentCoord[0]-edge*1.6/2)/spacing[0]):round((grapCentCoord[0]+edge*1.6/2)/spacing[0]),round((grapCentCoord[1]-edge*1.6/2)/spacing[1]):round((grapCentCoord[1]+edge*1.6/2)/spacing[1]),round((grapCentCoord[2]-edge*1.6/2)/spacing[2]):round((grapCentCoord[2]+edge*1.6/2)/spacing[2])]
        sub_box=source[round((grapCentCoord[2]-edge*1.6/2)/spacing[2]):round((grapCentCoord[2]+edge*1.6/2)/spacing[2]),round((grapCentCoord[1]-edge*1.6/2)/spacing[1]):round((grapCentCoord[1]+edge*1.6/2)/spacing[1]),round((grapCentCoord[0]-edge*1.6/2)/spacing[0]):round((grapCentCoord[0]+edge*1.6/2)/spacing[0])]
        if(sub_box.shape[0]==0 or sub_box.shape[1]==0 or sub_box.shape[2]==0):
            pass
        else:
            img_t1=sitk.GetImageFromArray(sub_box)
            sitk.WriteImage(img_t1,dirname+str(data[20])+'/output'+str(i)+'.mhd')


if __name__=='__main__':
    PathtoImage(csvfile_train,dirname1)
    PathtoImage(csvfile_val,dirname2)
    PathtoImage(csvfile_test,dirname3)
    # PathtoImage(csvfile_label,dirname4)



   







 

