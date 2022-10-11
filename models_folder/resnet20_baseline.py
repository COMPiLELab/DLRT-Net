#%%

'''
Properly implemented ResNet-s for CIFAR10 as described in paper [1].
The implementation and structure of this file is hugely influenced by [2]
which is implemented for ImageNet and doesn't have option A for identity.
Moreover, most of the implementations on the web is copy-paste from
torchvision's resnet and has wrong number of params.
Proper ResNet-s for CIFAR10 (for fair comparision and etc.) has following
number of layers and parameters:
name      | layers | params
ResNet20  |    20  | 0.27M
ResNet32  |    32  | 0.46M
ResNet44  |    44  | 0.66M
ResNet56  |    56  | 0.85M
ResNet110 |   110  |  1.7M
ResNet1202|  1202  | 19.4m
which this implementation indeed has.
Reference:
[1] Kaiming He, Xiangyu Zhang, Shaoqing Ren, Jian Sun
    Deep Residual Learning for Image Recognition. arXiv:1512.03385
[2] https://github.com/pytorch/vision/blob/master/torchvision/models/resnet.py
If you use this implementation in you work, please don't forget to mention the
author, Yerlan Idelbayev.
'''
import torch
import torch.nn as nn
import torch.nn.functional as F

__all__ = ['ResNet', 'resnet20', 'resnet32', 'resnet44', 'resnet56', 'resnet110', 'resnet1202']

import torch
import torch.nn as  nn
import torch.nn.functional as F
import os 
import sys 
sys.path.insert(1, os.path.join(sys.path[0], '..'))
from optimizer_KLS.my_conv import Conv2d_lr
from optimizer_KLS.Linear_layer_lr_new import Linear
import torch

full_rank_reconstruct = False   # True when loading the weights, false to train full rank

class Flatten(nn.Module):
    def forward(self, input):
        '''
        Note that input.size(0) is usually the batch size.
        So what it does is that given any input with input.size(0) # of batches,
        will flatten to be 1 * nb_elements.
        '''
        batch_size = input.size(0)
        out = input.view(batch_size,-1)
        return out # (batch_size, *size)


def get_children(model: torch.nn.Module):
    # get children form model!
    children = list(model.children())
    flatt_children = []
    if children == []:
        # if model has no children; model is last child! :O
        return model
    else:
       # look for children from children... to the last child!
       for child in children:
            try:
                flatt_children.extend(get_children(child))
            except TypeError:
                flatt_children.append(get_children(child))
    return flatt_children


class Reference:
    def __init__(self, val):
        self._value = val # just refers to val, no copy

    def get(self):
        return self._value

    def set(self, val):
        self._value = val


class LambdaLayer(nn.Module):
    def __init__(self, lambd):
        super(LambdaLayer, self).__init__()
        self.lambd = lambd

    def forward(self, x):
        return self.lambd(x)



class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1, option='A',device = 'cpu'):
        super(BasicBlock, self).__init__()
        self.device = device
        self.conv1 = Conv2d_lr(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False,rank = None,#min([9*in_planes,planes]),\
                              device = self.device,full_rank_construct=full_rank_reconstruct)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = Conv2d_lr(planes, planes, kernel_size=3, stride=1, padding=1, bias=False,rank = None,#min([9*in_planes,planes]),\
                                device  = self.device,full_rank_construct=full_rank_reconstruct)
        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != planes:
            if option == 'A':
                """
                For CIFAR10 ResNet paper uses option A.
                """
                self.shortcut = LambdaLayer(lambda x:
                                            F.pad(x[:, :, ::2, ::2], (0, 0, 0, 0, planes//4, planes//4), "constant", 0))
            elif option == 'B':
                self.shortcut = nn.Sequential(
                     Conv2d_lr(in_planes, self.expansion * planes, kernel_size=1, stride=stride, bias=False,\
                               rank = None,#min([in_planes,self.expansion*planes]),
                               device = self.device,full_rank_construct=full_rank_reconstruct),
                     nn.BatchNorm2d(self.expansion * planes)
                )

        self.layer = [self.conv1,self.conv2,self.bn1,self.bn2]

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class LambdaLayer(nn.Module):
    def __init__(self, lambd):
        super(LambdaLayer, self).__init__()
        self.lambd = lambd

    def forward(self, x):
        return self.lambd(x)



class ResNet(nn.Module):
    def __init__(self, block, num_blocks, num_classes=10,device = 'cpu'):
        super(ResNet, self).__init__()
        self.in_planes = 16
        
        self.device = device
        self.conv1 = Conv2d_lr(3, 16, kernel_size=3, stride=1, padding=1, bias=False,rank = None
                               ,device = self.device,full_rank_construct=full_rank_reconstruct)
        self.bn1 = nn.BatchNorm2d(16)
        self.layer1,self.layer1_list  = self._make_layer(block, 16, num_blocks[0], stride=1)
        self.layer2,self.layer2_list  = self._make_layer(block, 32, num_blocks[1], stride=2)
        self.layer3,self.layer3_list  = self._make_layer(block, 64, num_blocks[2], stride=2)
        self.linear0 = Linear(64,64,rank = None,
                              device = self.device,full_rank_construct=full_rank_reconstruct)
        self.linear1 = Linear(64,64,rank = None,
                              device = self.device,full_rank_construct=full_rank_reconstruct)
        self.linear = Linear(64, num_classes,device = self.device)

        self.dp1 = torch.nn.Dropout(p = 0.3)
        self.dp2 = torch.nn.Dropout(p = 0.3)
        self.layer = get_children(self)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride,device = self.device))
            self.in_planes = planes * block.expansion

        return nn.Sequential(*layers),layers

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = F.avg_pool2d(out, out.size()[3])
        out = out.view(out.size(0), -1)
        out = self.dp1(out)
        out = self.linear0(out)
        out = self.linear1(out)
        out = self.linear(out)
        return out


    def update_step_(self,l,new_step = 'K'):

        if isinstance(l,torch.nn.Module) and len(list(l.modules()))==1:

            if hasattr(l,'lr') and l.lr:

                l.step = new_step
        elif isinstance(l,list):
            for layer in l:
                self.update_step_(layer,new_step)
        elif isinstance(l,torch.nn.Module) and len(list(l.modules()))>1:
            for layer in l.modules():
                self.update_step_(layer,new_step)

    def update_step(self,new_step = 'K'):

        self.update_step_(self.layer,new_step)

    def populate_gradients(self,x,y,criterion,step = 'all'):

        if step == 'all':
        
            self.update_step(new_step = 'K')
            output = self.forward(x)
            loss = criterion(output,y)
            loss.backward()
            self.update_step(new_step = 'L')
            output = self.forward(x)
            loss = criterion(output,y)
            loss.backward()
            return loss,output.detach()

        else:
            
            self.update_step(new_step = step)
            loss = criterion(self.forward(x),y)
            return loss

    def to_low_rank(self):

        for l in self.layer:

            if isinstance(l,Conv2d_lr) or isinstance(l,Linear):

                l.switch_lowrank()
        
        
def ResNet20(device = 'cpu'):
    return ResNet(BasicBlock, [3, 3, 3],device = device)

def ResNet20_cifar100(device = 'cpu'):
    return ResNet(BasicBlock, [3, 3, 3],num_classes = 100,device = device)