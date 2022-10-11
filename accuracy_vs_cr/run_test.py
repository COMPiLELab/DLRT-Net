import torch
from test_network import Lenet5, Lenet5_cifar10
import argparse
import numpy as np
import tensorflow as tf
import pandas as pd
from sklearn.model_selection import train_test_split
from optimizer_KLS.train_custom_optimizer import * 
from train_accuracy_vs_cr import * 
from optimizer_KLS.dlrt_optimizer import dlr_opt

# for cifar10
import torchvision.datasets as datasets
import torchvision.transforms as T
import torch.utils.data as data

parser = argparse.ArgumentParser(description='Pytorch dlrt accuracy vs compression ratio')  
parser.add_argument('--epochs', type=int, default=100, metavar='EPOCHS',
                    help='number of epochs for training (default: 100)')  
parser.add_argument('--batch_size', type=int, default=128, metavar='BATCH_SIZE',
                    help='batch size for training (default: 128)')  
parser.add_argument('--cv_runs', type=int, default=5, metavar='CV_RUNS',
                    help='number of runs for c.i. (default: 10)')  
parser.add_argument('--step', type=float, default=0.1, metavar='STEP',
                    help='step for the timing grid of the experiment (default: 0.1)')
parser.add_argument('--lr', type=float, default=0.05, metavar='LR',
                    help='learning rate for the training (default: 0.05)')                    
parser.add_argument('--device', type=str, default='cuda', metavar='device',
                    help='device to use for the experiment (default: cuda)')         
parser.add_argument('--workers', type=int, default=1, metavar='workers',
                    help='workers for the dataloader (default: 1)')                        
args = parser.parse_args()
device = 'cuda' if torch.cuda.is_available() and args.device =='cuda' else 'cpu'


path = './results/'


criterion = torch.nn.CrossEntropyLoss()
compression_ratios = np.concatenate([np.arange(0,0.9,0.1),np.arange(0.9,1,args.step)])
starting_ranks = [20,50,500]


# === data transformation === #
# transform_train = T.Compose([
#     T.RandomCrop(32, padding=4),
#     T.RandomHorizontalFlip(),
#     T.ToTensor(),
#     T.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
# ])

# transform_test = T.Compose([
#     T.ToTensor(),
#     T.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
# ])

# dataloader = datasets.CIFAR10

# trainset = dataloader(root='./data', train=True, download=True, transform=transform_train)
# train_loader = data.DataLoader(trainset, batch_size=args.batch_size, shuffle=True, num_workers=args.workers)

# testset = dataloader(root='./data', train=False, download=False, transform=transform_test)
# validation_loader = data.DataLoader(testset, batch_size=args.batch_size, shuffle=False, num_workers=args.workers)

sizes = [(25,20),(20*25,50),(800,500),(500,100)]

for cv in range(args.cv_runs):

    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data(path="mnist.npz")

    x = np.vstack([x_train,x_test])
    y = np.hstack([y_train,y_test])

    x_train,x_test,y_train,y_test = train_test_split(x,y,train_size=60000,stratify = y)

    ## for cifar
    x_train,x_test = x_train.reshape(x_train.shape[0],1,x_train.shape[1],x_train.shape[2]),x_test.reshape(x_test.shape[0],1,x_test.shape[1],x_test.shape[2])  
    y_train,y_test = y_train.reshape(y_train.shape[0]),y_test.reshape(y_test.shape[0])
    ##

    x_train,x_test,y_train,y_test = torch.tensor(x_train).float()/255,torch.tensor(x_test).float()/255,torch.tensor(y_train),torch.tensor(y_test)


    #x_train,x_val,y_train,y_val = train_test_split(x_train,y_train,train_size = 50000,stratify = y_train)

    ##
    print(f'train shape {x_train.shape}')
    print(f'val shape {x_test.shape}')
    # print(f'val shape {x_val.shape}')
    # print(f'test shape {x_test.shape}')


    batch_size_train,batch_size_test = args.batch_size,args.batch_size

    train_loader = torch.utils.data.DataLoader(
        [(x_train[i],y_train[i]) for i in range(x_train.shape[0])],
        batch_size=batch_size_train, shuffle=True)

    validation_loader = torch.utils.data.DataLoader(
        [(x_test[i],y_test[i]) for i in range(x_test.shape[0])],
        batch_size=batch_size_test, shuffle=True)

    # baseline

    ranks = [None]*4
    NN = Lenet5(device,ranks = ranks,fixed = False)
    optimizer = torch.optim.SGD(NN.parameters(),lr = args.lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer,patience = args.epochs)

    
    train_baseline(NN,args.epochs,criterion,
                        optimizer,scheduler,'Baseline',cv,train_loader,
                        validation_loader,path,device,'Lenet5_baseline_')

    for cr in compression_ratios:

        print(f'cr:{cr}')
        print('='*50)


        # (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data(path="mnist.npz")


        # x = np.vstack([x_train,x_test])
        # y = np.hstack([y_train,y_test])

        # x_train,x_test,y_train,y_test = train_test_split(x,y,train_size=60000,stratify = y)

        # ## for cifar
        # x_train,x_test = x_train.reshape(x_train.shape[0],1,x_train.shape[1],x_train.shape[2]),x_test.reshape(x_test.shape[0],1,x_test.shape[1],x_test.shape[2])  
        # y_train,y_test = y_train.reshape(y_train.shape[0]),y_test.reshape(y_test.shape[0])
        # ##

        # x_train,x_test,y_train,y_test = torch.tensor(x_train).float()/255,torch.tensor(x_test).float()/255,torch.tensor(y_train),torch.tensor(y_test)


        # #x_train,x_val,y_train,y_val = train_test_split(x_train,y_train,train_size = 50000,stratify = y_train)

        # ##
        # print(f'train shape {x_train.shape}')
        # print(f'val shape {x_test.shape}')
        # # print(f'val shape {x_val.shape}')
        # # print(f'test shape {x_test.shape}')


        # batch_size_train,batch_size_test = args.batch_size,args.batch_size

        # train_loader = torch.utils.data.DataLoader(
        #     [(x_train[i],y_train[i]) for i in range(x_train.shape[0])],
        #     batch_size=batch_size_train, shuffle=True)

        # validation_loader = torch.utils.data.DataLoader(
        #     [(x_test[i],y_test[i]) for i in range(x_test.shape[0])],
        #     batch_size=batch_size_test, shuffle=True)

        ############ NET

        #ranks = [max([int(r*(1-cr)),4]) for r in starting_ranks]
        ranks = [max([int((1-cr)*size[0]*size[1]/(size[0]+size[1]+1)),2]) for size in sizes]
        NN = Lenet5(device,ranks = ranks,fixed = True)
        optimizer = dlr_opt(NN,tau = args.lr,theta = 0.0,KLS_optim=torch.optim.SGD)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer.integrator,patience = args.epochs)


        train_acc_vs_cr(NN,args.epochs,criterion,
                            optimizer,scheduler,1-cr,cv,train_loader,
                            validation_loader,path,device)