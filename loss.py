import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class CELoss(nn.Module):
    def __init__(self, class_num, alpha=None, use_alpha=False, size_average=False):
        super(CELoss, self).__init__()
        self.class_num = class_num
        self.alpha = alpha
        if use_alpha:
            self.alpha = torch.tensor(alpha).cuda()
        self.softmax = nn.Softmax(dim=1)
        self.use_alpha = use_alpha
        self.size_average = size_average

    def forward(self, pred, target):
        prob = self.softmax(pred.view(-1,self.class_num))
        prob = prob.clamp(min=0.0001,max=1.0)

        target_ = torch.zeros(target.size(0),self.class_num).cuda()


        #scatter_(dim,index,src)->tensor
        target_.scatter_(1, target.view(-1, 1).long(), 1.)


        #alpha is like
        #alpha=[[0.1,0.2,0.3]
        #       [0.1,0.2,0.3]
        #       [0.1,0.2,0.3]]

        if self.use_alpha:
            batch_loss = - self.alpha.double() * prob.log().double() * target_.double()
        else:
            batch_loss = - prob.log().double() * target_.double()
        batch_loss = batch_loss.sum(dim=1)

 
        if self.size_average:
            loss = batch_loss.mean()
        else:
            loss = batch_loss.sum()
        return loss

# batchsize=8
# alpha=[27.76,10,3.9,30.4,5.7,1,1.33]
# alpha = torch.tensor(alpha).float()
# alpha=alpha.unsqueeze(0)
# alpha=alpha.repeat(batchsize,1)

class FocalLoss(nn.Module):
    def __init__(self, class_num, alpha=None, gamma=2, use_alpha=False, size_average=False):
        super(FocalLoss, self).__init__()
        self.class_num = class_num
        self.alpha = alpha
        self.gamma = gamma
        if use_alpha:
            self.alpha = torch.tensor(alpha).cuda()
        self.softmax = nn.Softmax(dim=1)
        self.use_alpha = use_alpha
        self.size_average = size_average


    def forward(self, pred, target):
        prob = self.softmax(pred.view(-1,self.class_num))
        # prob = self.softmax(pred)
        prob = prob.clamp(min=0.0001,max=1.0)
        target_ = torch.zeros(target.size(0),self.class_num).cuda()
        target_.scatter_(1, target.view(-1, 1).long(), 1.)
        if self.use_alpha:
            batch_loss = - self.alpha.double() * torch.pow(1-prob,self.gamma).double() * prob.log().double() * target_.double()
        else:
            batch_loss = - torch.pow(1-prob,self.gamma).double() * prob.log().double() * target_.double()
        batch_loss = batch_loss.sum(dim=1)

        if self.size_average:
            loss = batch_loss.mean()
        else:
            loss = batch_loss.sum()
        return loss




        """Compute the Class Balanced Loss between `logits` and the ground truth `labels`.
    Class Balanced Loss: ((1-beta)/(1-beta^n))*Loss(labels, logits)
    where Loss is one of the standard losses used for Neural Networks.
    Args:
      labels: A int tensor of size [batch].
      logits: A float tensor of size [batch, no_of_classes].
      samples_per_cls: A python list of size [no_of_classes].
      no_of_classes: total number of classes. int
      loss_type: string. One of "sigmoid", "focal", "softmax".
      beta: float. Hyperparameter for Class balanced loss.
      gamma: float. Hyperparameter for Focal loss.
    Returns:
      cb_loss: A float tensor representing class balanced loss
    """


class CB_loss(nn.Module):
    def __init__(self,labels, logits, samples_per_cls, no_of_classes, beta):
        super(CB_loss,self).__init__()
        effective_num = 1.0 - np.power(beta, samples_per_cls)
        weights = (1.0 - beta) / np.array(effective_num)
        weights = weights / np.sum(weights) * no_of_classes

        labels_one_hot = F.one_hot(labels, no_of_classes).float()
        weights = torch.tensor(weights).float()
        weights = weights.unsqueeze(0)
        # weights = weights.repeat(labels_one_hot.shape[0],1) * labels_one_hot
        # weights = weights.sum(1)
        # weights = weights.unsqueeze(1)
        # weights = weights.repeat(1,no_of_classes)
        weights = weights.repeat(labels_one_hot.shape[0],1)
        self.weight=weights 
    

    def forward(self,logits,labels):
        focal_loss=FocalLoss(7, alpha=self.weights, gamma=2, use_alpha=True, size_average=False)
        cb_loss = focal_loss(logits, labels)
        return cb_loss