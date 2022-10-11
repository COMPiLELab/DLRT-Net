#%%
import torch
import torch.nn as nn
from optimizer_KLS.my_conv import *
from optimizer_KLS.Linear_layer_lr_new import *

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
    """
    Standard PyTorch implementation of VGG. Pretrained imagenet model is used.
    """
    def __init__(self,num_hidden = 4096,num_classes = 10,device = 'cpu'):
        super(VGG).__init__()
    
        self.features = nn.Sequential(
            # conv1
            Conv2d_lr(3, 64, 3, padding=1,rank = min),
            nn.ReLU(),
            nn.Conv2d(64, 64, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, stride=2, return_indices=True),
            
            # conv2
            nn.Conv2d(64, 128, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(128, 128, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, stride=2, return_indices=True),

            # conv3
            nn.Conv2d(128, 256, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(256, 256, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(256, 256, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, stride=2, return_indices=True),

            # conv4
            nn.Conv2d(256, 512, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(512, 512, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(512, 512, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, stride=2, return_indices=True),

            # conv5
            nn.Conv2d(512, 512, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(512, 512, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(512, 512, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, stride=2, return_indices=True),
            # classifier
            Flatten(),
            nn.Linear(512 * 7 * 7, num_hidden),
            nn.ReLU(),
            nn.Dropout(),
            nn.Linear(num_hidden, num_hidden),
            nn.ReLU(),
            nn.Dropout(),
            nn.Linear(num_hidden, num_classes)
        )

        # We need these for MaxUnpool operation
        self.conv_layer_indices = [0, 2, 5, 7, 10, 12, 14, 17, 19, 21, 24, 26, 28]
        self.layer = get_children(self)
        # self.feature_maps = OrderedDict()
        # self.pool_locs = OrderedDict()
        
    def forward(self, x):
        # for layer in self.features:
        #     if isinstance(layer, nn.MaxPool2d):
        #         x, location = layer(x)
        #     else:
        #         x = layer(x)
        out = x
        for l in self.layer:
            out = l(out)
        return out
        
        # x = x.view(x.size()[0], -1)
        # x = self.classifier(x)
        # return x