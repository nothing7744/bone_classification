import os
from tqdm import tqdm
import numpy as np
import torch
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from tensorboardX import SummaryWriter
import time
import torch.nn as nn
import torch
import torchvision.transforms as transforms
from PIL import Image
from matplotlib import pyplot as plt
import torchvision.models as models
import torchsummary
import time
from torch.optim.lr_scheduler import ExponentialLR
from generateModel import generate_model
from MyDataset import weightGen
from MyDataset import MyDatasets
from torch.utils.data import WeightedRandomSampler
from torch.optim.lr_scheduler import StepLR
from loss import FocalLoss,CELoss,CB_loss
import warnings

print('device =',device)
print(torch.cuda.get_device_name(0))

#ingore warning,but the best way should be solve the warning
warnings.filterwarnings(action='ignore' ,message ='',category = Warning, module='',lineno = 0,append = False)
warnings.simplefilter(action='ignore' , category = Warning,lineno = 0,append = False)




model = generate_model(model_type='resnet', model_depth=50,
                   input_W=224, input_H=224, input_D=224, resnet_shortcut='B',
                   no_cuda=False, gpu_id=[0,1,2,3],
                   nb_class=7)
model.to(device)


print("===========model finish======================")
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3,weight_decay=0.001)


criterion = nn.CrossEntropyLoss()
# alpha=[27.76,10,3.9,30.4,5.7,1,1.33]
# criterion=FocalLoss(7, alpha=None, gamma=2, use_alpha=False, size_average=False)


# scheduler = ExponentialLR(optimizer, gamma=0.99)
scheduler = StepLR(optimizer, step_size=10, gamma=0.5)
num_epochs =50
dirname='/home/kehao/bone_detection_classification/'
csvname1='./csv/box_train.csv'
csvname2='./csv/box_val.csv'
csvname3='./csv/box_test.csv'
#######################################


weight1=torch.tensor(weightGen(csvname1))
weight2=torch.tensor(weightGen(csvname2))
weight3=torch.tensor(weightGen(csvname3))


######################################
train_set=MyDatasets(dirname,csvname1)
val_set=MyDatasets(dirname,csvname2)
test_set=MyDatasets(dirname,csvname3)
########################################
batch=8

sampler1=WeightedRandomSampler(weight1,len(weight1),replacement=True)
# sampler2=WeightedRandomSampler(weight2,len(weight2),replacement=True)


# sampler1=WeightedRandomSampler(weight1,50,replacement=True)
# sampler2=WeightedRandomSampler(weight2,50,replacement=True)
# sampler3=WeightedRandomSampler(weight3,len(weight3),replacement=True)


# sampler3=WeightedRandomSampler(weight3,100,replacement=True)


train_loader = DataLoader(train_set, batch_size=batch, sampler=sampler1,num_workers=16,shuffle=False)
# train_loader = DataLoader(train_set, batch_size=batch, num_workers=16,shuffle=True)
# val_loader=DataLoader(val_set, batch_size=batch, sampler=sampler2,num_workers=16,shuffle=False)
# test_loader=DataLoader(test_set, batch_size=batch, sampler=sampler3,shuffle=False)
val_loader=DataLoader(val_set, batch_size=batch, num_workers=16,shuffle=True)
test_loader=DataLoader(test_set, batch_size=batch,  num_workers=16,shuffle=True)
summaryWriter = SummaryWriter(log_dir='logs',comment='Linear')
##############################################################################


RESUME=False
start_epoch=0
if RESUME:
    path_checkpoint = "/data/kehao/models/checkpoint/ckpt_best_10.pth"  
    checkpoint = torch.load(path_checkpoint)  
    model.load_state_dict(checkpoint['net'])  
    optimizer.load_state_dict(checkpoint['optimizer'])  
    start_epoch = checkpoint['epoch']  



def train():
    total_step = len(train_loader)
    time_list = []
    print("===================train start=====================")
    fw = open("/home/kehao/bone_detection_classification/logs/trlog.txt", 'w')
    for epoch in range(num_epochs):
        matrix_train=np.zeros((7,7))
        matrix=np.zeros((7,7))
        precision_train=np.zeros(7)
        precision=np.zeros(7)
        recall_train=np.zeros(7)
        recall=np.zeros(7)
        epoch=epoch+start_epoch
        start = time.time()
        per_epoch_loss = 0
        num_correct= 0
        val_num_correct = 0 
   
        model.train()
        print("=============model train======================")
        with torch.enable_grad():
            for i,batch in tqdm(enumerate(train_loader)):
                x = batch['source']['data']
                label = batch['label']
                # x=x.type(torch.FloatTensor).cuda()
                x=x.type(torch.FloatTensor).to(device)
                # label = label.type(torch.LongTensor).cuda()
                label = label.type(torch.LongTensor).to(device)
                label = torch.squeeze(label)                                                                   
            # Forward pass
                logits = model(x)
                loss = criterion(logits, label)
                per_epoch_loss += loss.item()

            # Backward and optimize
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                pred = logits.argmax(dim=1)
                num_correct += torch.eq(pred, label).sum().float().item()
        #======================= print info ========================================  
                for p, t in zip(pred, label):
                    matrix_train[p, t] += 1
            sum1_train=np.sum(matrix_train,1)
            sum2_train=np.sum(matrix_train,0)
            for i in range(len(matrix)):
                if sum1_train[i]==0:
                    pass
                else:
                    recall_train[i]=matrix_train[i,i]/sum2_train[i]
                    precision_train[i]=matrix_train[i,i]/sum1_train[i]
            print("Train Epoch: {}\t Loss: {:.6f}\t Acc: {:.6f}".format(epoch,per_epoch_loss/total_step,num_correct/len(train_loader.dataset)))
            summaryWriter.add_scalars('loss', {"loss":(per_epoch_loss/total_step)}, epoch)
            summaryWriter.add_scalars('acc', {"acc":num_correct/len(train_loader.dataset)}, epoch)
            print(matrix_train)
            print(recall_train)
            print(precision_train)
            fw.write("=====matrix_train"+str(epoch)+"=========")
            fw.write("\n") 
            fw.write("Train Epoch:"+str(epoch)+"  Loss:"+str(per_epoch_loss/total_step)+"  Acc:"+str(num_correct/len(train_loader.dataset)))
            fw.write("\n")
            fw.write(str(matrix_train))  
            fw.write("\n")
            fw.write(str(recall_train))  
            fw.write("\n")
            fw.write(str(precision_train))  
            fw.write("\n")
        #================================================================================



        model.eval()
        with torch.no_grad():
            for i,batch in tqdm(enumerate(val_loader)):
                x = batch['source']['data']
                label = batch['label']          
                # x=x.type(torch.FloatTensor).cuda()      
                x=x.type(torch.FloatTensor).to(device) 
                # label = label.type(torch.LongTensor).cuda()        
                label = label.type(torch.LongTensor).to(device)
                label = torch.squeeze(label)               
                 # Forward pass
                logits = model(x)
                pred = logits.argmax(dim=1)
                val_num_correct += torch.eq(pred, label).sum().float().item()
        #===================print info==========================================================
                for p, t in zip(pred, label):
                    matrix[p, t] += 1
            sum1=np.sum(matrix,1)
            sum2=np.sum(matrix,0)
            for i in range(len(matrix)):
                if sum1[i]==0:
                    pass
                else:
                    recall[i]=matrix[i,i]/sum2[i]
                    precision[i]=matrix[i,i]/sum1[i]
            print("val Epoch: {}\t Acc: {:.6f}".format(epoch,val_num_correct/len(val_loader.dataset)))
            summaryWriter.add_scalars('acc', {"val_acc":val_num_correct/len(val_loader.dataset)}, epoch)
            summaryWriter.add_scalars('time', {"time":(time.time() - start)}, epoch)
        print(matrix)
        print(recall)
        print(precision)
        fw.write("=====matrix_val"+str(epoch)+"=========")
        fw.write("\n") 
        fw.write("Val Epoch:"+str(epoch)+"  Acc:"+str(val_num_correct/len(val_loader.dataset)))
        fw.write("\n")
        fw.write(str(matrix)) 
        fw.write("\n")
        fw.write(str(recall))  
        fw.write("\n")
        fw.write(str(precision))  
        fw.write("\n")
        #=============================================================================================
        scheduler.step()
    



        if epoch%2==0:
            checkpoint = {
                "net": model.state_dict(),
                'optimizer':optimizer.state_dict(),
                "epoch": epoch
                }  
            if not os.path.isdir("/data/kehao/models/checkpoint"):
                os.mkdir("/data/kehao/models/checkpoint")
            torch.save(checkpoint, '/data/kehao/models/checkpoint/ckpt_best_%s.pth' %(str(epoch)))
    fw.close()

        

if __name__=='__main__':
    print("hello,world")
    train()








