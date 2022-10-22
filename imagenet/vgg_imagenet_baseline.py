# %%
import os
import sys
import argparse
sys.path.insert(1, os.path.join(sys.path[0], '..'))
from models_folder.vgg_baseline import VGG,VGG_types
import torch
from optimizer_KLS.dlrt_optimizer import dlr_opt
from optimizer_KLS.train_custom_optimizer import *
from optimizer_KLS.train_experiments import *

# from torchvision.datasets import ImageNet

import torchvision.datasets as datasets
import torchvision.transforms as T
import torch.utils.data as data

device = "cuda" if torch.cuda.is_available() else "cpu"

##############################################

############################################## parser creation
parser = argparse.ArgumentParser(description='Pytorch dlrt training for vgg of imagenet')
parser.add_argument('--batch_size', type=int, default=128, metavar='N',
                    help='input batch size for training (default: 128)')
parser.add_argument('--epochs', type=int, default=100, metavar='N',
                    help='number of epochs to train (default: 100)')
parser.add_argument('--lr', type=float, default=0.05, metavar='LR',
                    help='learning rate for dlrt optimizer (default: 0.05)')  
parser.add_argument('--theta', type=float, default=0.0, metavar='THETA',
                    help='threshold for the rank adaption step (default: 0.08)')  
parser.add_argument('--momentum', type=float, default=0.1, metavar='MOMENTUM',
                    help='momentum (default: 0.1)')  
parser.add_argument('--workers', type=int, default=1, metavar='WORKERS',
                    help='number of workers for the dataloaders (default: 1)')      
parser.add_argument('--net_name', type=str, default='vgg_imagenet_baseline_', metavar='NET_NAME',
                    help='network name for the saved results (default: 1)')      

args = parser.parse_args()
############################################## Net creation
num_classes = 1000
criterion = torch.nn.CrossEntropyLoss()
NN = VGG(num_classes = num_classes,
    architecture=VGG_types["VGG16"],device = device
).to(device)
##############################################


############################################## optimizer creation
optimizer = torch.optim.SGD(NN.parameters(),lr = args.lr,momentum = args.momentum)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer)
path = './results/vgg/'
############################################## 


# Create ImageNet Dataloader
# Create dataset

import pathlib
print(pathlib.Path().resolve())
print("________")

imageNet_location = "imageNet"

# === data transformation === # 
normalize = T.Normalize(mean=[0.485, 0.456, 0.406],
                        std=[0.229, 0.224, 0.225])


train_T = T.Compose([   T.RandomResizedCrop(224),
                        T.RandomHorizontalFlip(), 
                        T.ToTensor(), 
                        normalize,
                    ])

test_T = T.Compose([T.Resize(256),
                    T.CenterCrop(224),
                    T.ToTensor(),
                    normalize,
                    ])                  


# === dataset object === # 
train_dataset = datasets.ImageFolder(   root= './data/imagenet/train',
                                        transform= train_T, 
                                    )

validation_dataset = datasets.ImageFolder(    root= "./data/imagenet/val", 
                                        transform= test_T,
                                    ) 

# Create dataloader
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size,
                                            shuffle=True, num_workers=args.workers, pin_memory=True,
                                            sampler=None)

validation_loader = torch.utils.data.DataLoader(validation_dataset, batch_size=args.batch_size,
                                            shuffle=True, num_workers=args.workers, pin_memory=True,
                                            sampler=None)

train_baseline(NN = NN,epochs = args.epochs,criterion = criterion,optimizer = optimizer,scheduler = scheduler,
                    train_loader=train_loader,validation_loader=validation_loader,path = path,device = device,
                    net_name = args.net_name,save_weights=args.save_weights)
