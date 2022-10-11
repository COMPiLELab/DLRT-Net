#%%
import torch
import torch.nn as  nn
import torch.nn.functional as F
import os 
import sys 
sys.path.insert(1, os.path.join(sys.path[0], '..'))
from optimizer_KLS.my_conv import Conv2d_lr
from optimizer_KLS.Linear_layer_lr_new import Linear
import torch


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


class Bottleneck(nn.Module):
    expansion = 4
    def __init__(self, in_channels, out_channels, i_downsample=None, stride=1,device = 'cpu'):
        super(Bottleneck, self).__init__()
        self.device = device
        
        self.conv1 = Conv2d_lr(in_channels, out_channels, kernel_size=1, stride=1, padding=0,
                                                rank = None)#min([in_channels,out_channels]),device = self.device)
        self.batch_norm1 = nn.BatchNorm2d(out_channels)
        
        self.conv2 = Conv2d_lr(out_channels, out_channels, kernel_size=3, stride=stride, padding=1,
                                  rank = None,#min([out_channels,9*in_channels]),\
                                  device = self.device)
        self.batch_norm2 = nn.BatchNorm2d(out_channels)
        
        self.conv3 = Conv2d_lr(out_channels, out_channels*self.expansion, kernel_size=1, stride=1, padding=0,\
                                rank = None,#min([out_channels,in_channels]),
                                device = self.device)
        self.batch_norm3 = nn.BatchNorm2d(out_channels*self.expansion)
        
        self.i_downsample = i_downsample
        self.stride = stride
        self.relu = nn.ReLU()
        self.layer = [self.conv1,self.batch_norm1,self.conv2,self.batch_norm2,self.conv3,self.batch_norm3]
        
    def forward(self, x):
        identity = x.clone()
        x = self.relu(self.batch_norm1(self.conv1(x)))
        
        x = self.relu(self.batch_norm2(self.conv2(x)))
        
        x = self.conv3(x)
        x = self.batch_norm3(x)
        
        #downsample if needed
        if self.i_downsample is not None:
            identity = self.i_downsample(identity)
        #add identity
        x+=identity
        x=self.relu(x)
        
        return x

class Block(nn.Module):
    expansion = 1
    def __init__(self, in_channels, out_channels, i_downsample=None, stride=1,device = 'cpu'):
        super(Block, self).__init__()
       
        self.device = device
        self.conv1 = Conv2d_lr(in_channels, out_channels, kernel_size=3, padding=1, stride=stride, bias=False,\
                            rank = min([out_channels,9*in_channels]),device = self.device )
        self.batch_norm1 = nn.BatchNorm2d(out_channels)
        self.conv2 = Conv2d_lr(out_channels, out_channels, kernel_size=3, padding=1, stride=stride, bias=False,\
                        rank = min([out_channels,9*in_channels]),device = self.device )
        self.batch_norm2 = nn.BatchNorm2d(out_channels)

        self.i_downsample = i_downsample
        self.stride = stride
        self.relu = nn.ReLU()
        self.layer  = [self.conv1,self.conv2]

    def forward(self, x):
      identity = x.clone()

      x = self.relu(self.batch_norm2(self.conv1(x)))
      x = self.batch_norm2(self.conv2(x))

      if self.i_downsample is not None:
          identity = self.i_downsample(identity)
      print(x.shape)
      print(identity.shape)
      x += identity
      x = self.relu(x)
      return x


        
        
class ResNet(nn.Module):
    def __init__(self, ResBlock, layer_list, num_classes, num_channels=3,device = 'cpu'):
        super(ResNet, self).__init__()
        self.in_channels = 64
        self.device = device

        self.conv1 = Conv2d_lr(num_channels, 64, kernel_size=7, stride=2, padding=3, bias=False,\
                          rank = None,#min([64,49*num_channels]),
                          device = self.device )
        self.batch_norm1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU()
        self.max_pool = nn.MaxPool2d(kernel_size = 3, stride=2, padding=1)
        
        self.layer1,self.layer1_list = self._make_layer(ResBlock, layer_list[0], planes=64)
        self.layer2,self.layer2_list = self._make_layer(ResBlock, layer_list[1], planes=128, stride=2)
        self.layer3,self.layer3_list = self._make_layer(ResBlock, layer_list[2], planes=256, stride=2)
        self.layer4,self.layer4_list = self._make_layer(ResBlock, layer_list[3], planes=512, stride=2)
        
        self.avgpool = nn.AdaptiveAvgPool2d((1,1))
        self.linear = Linear(512*ResBlock.expansion,512*ResBlock.expansion,rank = 512*ResBlock.expansion,device = self.device)# new
        #self.linear1 = Linear(512*ResBlock.expansion,512*ResBlock.expansion,rank = 512*ResBlock.expansion,device = self.device)# new
        self.fc = Linear(512*ResBlock.expansion, num_classes,device = self.device)

        # self.layer = [self.conv1,self.batch_norm1]
        # auxiliary = self.layer1_list+self.layer2_list+\
        #                 self.layer3_list+self.layer4_list
        # for l in auxiliary:
        #     if hasattr(l,'layer'):
        #         self.layer +=l.layer

        # self.layer.append(self.fc)
        self.layer = get_children(self)
        
    def forward(self, x):
        x = self.relu(self.batch_norm1(self.conv1(x)))
        x = self.max_pool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        
        x = self.avgpool(x)
        x = x.reshape(x.shape[0], -1)
        x  = self.linear(x)  # new
        x = self.relu(x)# new
        #x  = self.linear1(x)  # new
        #x = self.relu(x)# new
        x = self.fc(x)
        
        return x
        
    def _make_layer(self, ResBlock, blocks, planes, stride=1):
        ii_downsample = None
        layers = []
        
        if stride != 1 or self.in_channels != planes*ResBlock.expansion:
            ii_downsample = nn.Sequential(
                Conv2d_lr(self.in_channels, planes*ResBlock.expansion, kernel_size=1, stride=stride,\
                rank = None,device = self.device),
                nn.BatchNorm2d(planes*ResBlock.expansion)
            )
            
        layers.append(ResBlock(self.in_channels, planes, i_downsample=ii_downsample, stride=stride,device = self.device))
        self.in_channels = planes*ResBlock.expansion
        
        for i in range(blocks-1):
            layers.append(ResBlock(self.in_channels, planes,device = self.device))
            
        return nn.Sequential(*layers),layers


    def update_step(self,new_step = 'K'):
        for l in self.layer:
            if hasattr(l,'lr') and l.lr:
                l.step = new_step

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

        
        
def ResNet50(num_classes, channels=3,device = 'cpu'):
    return ResNet(Bottleneck, [3,4,6,3], num_classes, channels,device)
    
def ResNet101(num_classes, channels=3):
    return ResNet(Bottleneck, [3,4,23,3], num_classes, channels)

def ResNet152(num_classes, channels=3):
    return ResNet(Bottleneck, [3,8,36,3], num_classes, channels)

# f = ResNet50(10,3)
# # print(f.layer)
# from optimizer_KLS.train_custom_optimizer import * 

# total_params_full = full_count_params(f, False)
# total_params_full_grads = full_count_params(f, False, True)
# params_test = count_params_test(f, False)
# cr_test = round(params_test / total_params_full, 3)

# print(f' compression ratio {cr_test}')

# %%
