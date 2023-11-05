from torchvision.datasets.vision import VisionDataset
from PIL import Image
import os
import os.path
import json
import time
import numpy as np
import copy
import itertools
from collections import defaultdict
import sys

import json
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, DistributedSampler
PYTHON_VERSION = sys.version_info[0]
if PYTHON_VERSION == 2:
    from urllib import urlretrieve
elif PYTHON_VERSION == 3:
    from urllib.request import urlretrieve


#######################start####################################
def _isArrayLike(obj):
    return hasattr(obj, '__iter__') and hasattr(obj, '__len__')

class BoneDetection():
    def __init__(self, root, annFile, transform=None, target_transform=None, transforms=None):
        # super(BoneDetection, self).__init__(root, transforms, transform, target_transform)
        self.dataset =Bone_lessions(annFile)
        # print(type(self.dataset))
        # print((self.dataset).shape) 
        # self.ids = list(sorted(self.dataset.imgs.keys()))
        self.ids = list(sorted(self.dataset.imgs))
        # print(len(self.ids))


    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: Tuple (image, target). target is the object returned by ``coco.loadAnns``.
        """
        dataset = self.dataset
        img_id = self.ids[index]
        # print(type(img_id))
        # print(img_id)
        target = dataset.loadAnns(img_id)
        path = dataset.loadImgs(img_id)[0]['filename']
        #这里得到image_path,和target,最后和dataloader一起进行测试
        # img = Image.open(os.path.join(self.root, path)).convert('RGB')
        # #对得到的数据进行数据增强
        # if self.transforms is not None:
        #     img, target = self.transforms(img, target)
        return (path,target)
    

    def __len__(self):
        return len(self.ids)
    

class Bone_lessions:
    def __init__(self, annotation_file=None):
        """
        Constructor of Microsoft COCO helper class for reading and visualizing annotations.
        :param annotation_file (str): location of annotation file
        :param image_folder (str): location to the folder that hosts images.
        :return:
        """
        # load dataset
        self.dataset,self.anns,self.cats,self.imgs = dict(),dict(),dict(),dict()
        self.imgToAnns, self.catToImgs = defaultdict(list), defaultdict(list)
        if not annotation_file == None:
            print('loading annotations into memory...')
            tic = time.time()
            with open(annotation_file, 'r') as f:
                print("json file")
                dataset = json.load(f)
                print(type(dataset))
            assert type(dataset)==dict, 'annotation file format {} not supported'.format(type(dataset))
            print('Done (t={:0.2f}s)'.format(time.time()- tic))
            self.dataset = dataset
            self.createIndex()

    def createIndex(self):
        # create index
        print('creating index...')
        anns, imgs = {}, {}
        if 'annotations' in self.dataset:
            for ann in self.dataset['annotations']:
                anns[ann['id']] = ann
                
        if 'image' in self.dataset:
            for img in self.dataset['image']:
                imgs[img['id']] = img
        print('index created!')
        # create class members
        self.anns = anns
        self.imgs = imgs
    
    def getImgIds(self, imgIds=[], catIds=[]):
        '''
        Get img ids that satisfy given filter conditions.
        :param imgIds (int array) : get imgs for given ids
        :param catIds (int array) : get imgs with all given cats
        :return: ids (int array)  : integer array of img ids
        '''
        imgIds = imgIds if _isArrayLike(imgIds) else [imgIds]
        catIds = catIds if _isArrayLike(catIds) else [catIds]
        if len(imgIds) == len(catIds) == 0:
            ids = self.imgs.keys()
        else:
            ids = set(imgIds)
            for i, catId in enumerate(catIds):
                if i == 0 and len(ids) == 0:
                    ids = set(self.catToImgs[catId])
                else:
                    ids &= set(self.catToImgs[catId])
        return list(ids)

    def loadAnns(self, ids=[]):
        """
        Load anns with the specified ids.
        :param ids (int array)       : integer ids specifying anns
        :return: anns (object array) : loaded ann objects
        """
        # if _isArrayLike(ids):
        #     return [self.anns[id] for id in ids]
        # else:
        return [self.anns[ids]]
        
    def loadImgs(self, ids=[]):
        """
        Load anns with the specified ids.
        :param ids (int array)       : integer ids specifying img
        :return: imgs (object array) : loaded img objects
        """
        # if _isArrayLike(ids):
        #     return [self.imgs[id] for id in ids]
        # else:
        return [self.imgs[ids]]
                
###这里数据的加载有问题
root="data/kehao/"
dataset_train = BoneDetection(root,annFile="./data_json/train.json")
data_loader_train = DataLoader(dataset_train, batch_size=128, num_workers=16,shuffle=True)

##  check data和dataloader的部分:
##  这里我们验证dataloader的功能是否正常


if __name__=='__main__':
    print("hello,world")
    print(len(data_loader_train))
    for i,batch in enumerate(data_loader_train):
        print("batch dataloader")
        (path,anno)=batch
        for p in path:
            print(p)
        for b in anno:
            print(b)

            
        # for path,anno in batch:
        #     print(path)
        #     print(anno)

