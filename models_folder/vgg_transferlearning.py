#%%
import torch
from torch import nn
import torch.nn.functional as F


from torch.autograd import Variable

import torch
import torch.nn as  nn
import torch.nn.functional as F
import os 
import sys 
sys.path.insert(1, os.path.join(sys.path[0], '..'))
from optimizer_KLS.my_conv import Conv2d_lr
from optimizer_KLS.Linear_layer_lr_new import Linear
import torch
device = 'cpu'

full_rank_reconstruct = False

def __init__():

    return 

VGG_types = {
    "VGG11": [64, "M", 128, "M", 256, 256, "M", 512, 512, "M", 512, 512, "M"],
    "VGG13": [
        64,
        64,
        "M",
        128,
        128,
        "M",
        256,
        256,
        "M",
        512,
        512,
        "M",
        512,
        512,
        "M",
    ],
    "VGG16": [
        64,
        64,
        "M",
        128,
        128,
        "M",
        256,
        256,
        256,
        "M",
        512,
        512,
        512,
        "M",
        512,
        512,
        512,
        "M",
    ],
    "VGG19": [
        64,
        64,
        "M",
        128,
        128,
        "M",
        256,
        256,
        256,
        256,
        "M",
        512,
        512,
        512,
        512,
        "M",
        512,
        512,
        512,
        512,
        "M",
    ],
}

class Flatten(nn.Module):
    def forward(self, input):
        '''
        Note that input.size(0) is usually the batch size.
        So what it does is that given any input with input.size(0) # of batches,
        will flatten to be 1 * nb_elements.
        '''
        batch_size = input.size(0)
        out = input.view(batch_size,-1)
        return out 



class VGG(nn.Module):
    def __init__(
        self,
        architecture,
        in_channels=3, 
        in_height=224, 
        in_width=224, 
        num_hidden=4096,
        num_classes=1000,
        device = 'cpu',
        full_rank_reconstruct = False
    ):
        super(VGG, self).__init__()
        self.device = device
        self.in_channels = in_channels
        self.in_width = in_width
        self.in_height = in_height
        self.num_hidden = num_hidden
        self.num_classes = num_classes
        self.layer = torch.nn.Sequential()

        in_channels = self.in_channels
        j = 0
        for x in architecture:
            if type(x) == int:
                out_channels = x
                    
                self.layer.add_module('conv_'+str(j),Conv2d_lr(
                            in_channels=in_channels,
                            out_channels=out_channels,
                            kernel_size=3,
                            stride=(1,1),
                            padding=(1, 1),
                            rank = None,#min([in_channels*9,out_channels]),
                            device = self.device,full_rank_construct=full_rank_reconstruct
                        ))
                self.layer.add_module('bn_'+str(j),nn.BatchNorm2d(out_channels))
                self.layer.add_module('relu_'+str(j),nn.ReLU())  
                in_channels = x
            else:
                self.layer.add_module('maxpool_'+str(j),
                    nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2))
                )
            j+=1

        pool_count = architecture.count("M")
        factor = (2 ** pool_count)
        if (self.in_height % factor) + (self.in_width % factor) != 0:
            raise ValueError(f"`in_height` and `in_width` must be multiples of {factor}")
        out_height = self.in_height // factor
        out_width = self.in_width // factor
        last_out_channels = next(
            x for x in architecture[::-1] if type(x) == int
        )

        self.layer.add_module('flat',Flatten())
        self.layer.add_module('linear_'+str(1),Linear(
                last_out_channels * out_height * out_width, 
                self.num_hidden,device = self.device,rank = min([self.num_hidden,last_out_channels * out_height * out_width]) ,
                full_rank_construct=full_rank_reconstruct))
        self.layer.add_module('relu_'+str(j+1),nn.ReLU())
        self.layer.add_module('drop_'+str(1),nn.Dropout(p=0.5))
        self.layer.add_module('linear_'+str(2),Linear(self.num_hidden, self.num_hidden,
                                device = self.device,rank = self.num_hidden,
                                full_rank_construct=full_rank_reconstruct))
        self.layer.add_module('relu_'+str(j+2),nn.ReLU())
        self.layer.add_module('drop_2',nn.Dropout(p=0.5))
        self.layer.add_module('classifier',Linear(self.num_hidden, self.num_classes,device = self.device))


        
    def forward(self, x):
        for l in self.layer:
          x = l(x)
        return x


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

# load the weights from full_rank

# device = 'cpu'
# from torch import float16
# import numpy as np
# from optimizer_KLS.dlrt_optimizer import dlr_opt

# f = ResNet20()
# f.load_state_dict(torch.load('./results_cifar10/_running_data_0.1_best_weights.pt'))
# f.to_low_rank()
# optimizer = dlr_opt(f,tau = 0.1,theta = 0.01,KLS_optim=torch.optim.SGD)
# optimizer.postprocess_step()

# NN =  VGG(
#     in_channels=3, 
#     in_height=32, 
#     in_width=32,num_hidden = 10, 
#     architecture=VGG_types["VGG19"],device = device
# ).to(device)
# %%

