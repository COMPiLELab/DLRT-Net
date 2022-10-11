#%%
import torch
import numpy as np
import os 
import sys 
import argparse
sys.path.insert(1, os.path.join(sys.path[0], '..'))
from optimizer_KLS.dlrt_optimizer import dlr_opt
from optimizer_KLS.train_custom_optimizer import *
import tensorflow as tf
from Lenet5_vanilla import Lenet5
from sklearn.model_selection import train_test_split
import torch
from torch import float16
from torchmetrics import Accuracy
from tqdm import tqdm
import torch.utils.data as data

device = "cuda" if torch.cuda.is_available() else "cpu"


def test_phase_metrics(NN, criterion, dataloader):  # pseudo code
    
    NN.eval()
    top1_metric = Accuracy().to(device)
    top5_metric = Accuracy(top_k=5).to(device)
    # top-1 accuracy
    top_1_accuracy = 0.0
    top_5_accuracy = 0.0
    with torch.no_grad():
        k = len(dataloader)
        for data in dataloader:
            inputs, labels = data
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = NN(inputs).detach()
            loss = criterion(outputs, labels)
            top_1_accuracy += top1_metric(outputs, labels) / k
            top_5_accuracy += top5_metric(outputs, labels) / k
    return float(top_1_accuracy), float(top_5_accuracy), float(loss)


device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using {device} device")


############################################## parser creation
parser = argparse.ArgumentParser(description='Pytorch vanilla layer decomposition training for Lenet5 on Mnist')
parser.add_argument('--batch_size', type=int, default=128, metavar='N',
                    help='input batch size for training (default: 128)')
parser.add_argument('--epochs', type=int, default=20, metavar='N',
                    help='number of epochs to train (default: 100)')
parser.add_argument('--lr', type=float, default=0.01, metavar='LR',
                    help='learning rate for dlrt optimizer (default: 0.05)')    
parser.add_argument('--cv_runs', type=int, default=10, metavar='CV_RUNS',
                    help='number of runs for c.i. (default: 10)')      
parser.add_argument('--decay', type=float, default=1, metavar='decay',
                    help='decay for the starting singular values (default: 1)')      
args = parser.parse_args()


d = int(args.decay)

print(f'd:{d},args.decay{args.decay}')

MAX_EPOCHS = args.epochs

def accuracy(outputs,labels):

    return torch.mean(torch.tensor(torch.argmax(outputs.detach(),axis = 1) == labels,dtype = float16))




metric  = accuracy
criterion = torch.nn.CrossEntropyLoss(reduction='mean') 
metric_name = 'accuracy'

theta = 'vanilla_UV'
index = 0

for cv_run in range(args.cv_runs):

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


    # validation_loader = torch.utils.data.DataLoader(
    #     [(x_val[i],y_val[i]) for i in range(x_val.shape[0])],
    #     batch_size=batch_size_test, shuffle=True)

    # test_loader = torch.utils.data.DataLoader(
    # [(x_test[i],y_test[i]) for i in range(x_test.shape[0])],
    # batch_size=batch_size_test, shuffle=True)


    f = Lenet5(device = device,decay = args.decay)
    f = f.to(device)
    optimizer = torch.optim.SGD(f.parameters(),lr = args.lr)
    setattr(optimizer,'theta','vanilla')
    setattr(optimizer,'tau',0.01)
    if d == 10:
        path = './results_vanilla/cv_vanilla/_running_data_vanilla'+str(cv_run)
    elif d == 1:
        path = './results_vanilla/cv_vanilla_baseline/_running_data_vanilla'+str(cv_run)
    elif d == 2:
        path = './results_vanilla/cv_vanilla_exp2/_running_data_vanilla'+str(cv_run)

    print('='*100)
    print(f'run number {index} \n theta = {theta}')
    index+=1


    running_data = pd.DataFrame(data=None, columns=['epoch', 'theta', 'learning_rate', 'train_loss', 'train_accuracy(%)',
                                                    'top_5_accuracy(%)', \
                                                    'test_accuracy(%)', 'top_5_test_accuracy(%)', \
                                                    'ranks', '# effective parameters', 'cr_test (%)',
                                                    '# effective parameters train', 'cr_train (%)', \
                                                    '# effective parameters train with grads', 'cr_train_grads (%)'])
    file_name = path


    count_bias = False
    total_params_full = full_count_params(f, count_bias)
    total_params_full_grads = full_count_params(f, count_bias, True)


    top1_metric = Accuracy().to(device)
    top5_metric = Accuracy(top_k = 5).to(device)

    acc_hist_test, top_5_acc_hist_test, loss_hist_test = test_phase_metrics(f, criterion, validation_loader)
    params_test = count_params_test(f, count_bias)
    cr_test = round(params_test / total_params_full, 3)
    params_train = count_params_train(f, count_bias)
    cr_train = round(params_train / total_params_full, 3)
    params_train_grads = count_params_train(f, count_bias, True)
    cr_train_grads = round(params_train_grads / total_params_full_grads, 3)
    ranks = []
    for i, l in enumerate(f.layer):
        if hasattr(l, 'lr') and l.lr:
            ranks.append(l.dynamic_rank)
    epoch_data = [0, optimizer.theta, round(optimizer.tau, 5), None, None,
                None, \
                round(acc_hist_test * 100, 4), round(top_5_acc_hist_test*100,4), ranks, params_test,
                round(100 * (1 - cr_test), 4), \
                params_train, round(100 * (1 - cr_train), 4), params_train_grads,
                round(100 * (1 - cr_train_grads), 4)]

    running_data.loc[-1] = epoch_data

    for epoch in tqdm(range(MAX_EPOCHS)):

        print(f'epoch {epoch}---------------------------------------------')
        loss_hist = 0
        acc_hist = 0
        top_5_acc_hist = 0
        k = len(train_loader)

        f.train()
        total = 0.0
        correct = 0.0
        for i, data in enumerate(train_loader):  # train
            f.zero_grad()
            optimizer.zero_grad()
            inputs, labels = data
            inputs, labels = inputs.to(device), labels.to(device)
            loss = criterion(f(inputs),labels)
            loss.backward()
            loss_hist += float(loss.item()) / k
            outputs = f(inputs).detach().to(device)
            top_5_acc_hist +=  float(top5_metric(outputs, labels)) / k
            predicted = torch.argmax(outputs.data, 1)
            total += labels.size(0)
            correct += torch.tensor(predicted == labels,dtype = float16).sum().item()
            optimizer.step()
        acc_hist,top_5_acc_hist = correct/total,top_5_acc_hist

        acc_hist_test, top_5_acc_hist_test, loss_hist_test = test_phase_metrics(f, criterion, validation_loader)

        print(f'epoch[{epoch}]: loss: {loss_hist:9.4f} | top_1,top_5_acc: {acc_hist:9.4f},{top_5_acc_hist:9.4f}\n')
        print('=' * 100)
        ranks = []
        for i, l in enumerate(f.layer):
            if hasattr(l, 'lr') and l.lr:
                print(f'rank layer {i} {l.dynamic_rank}')
                ranks.append(l.dynamic_rank)
        print('\n')

        params_test = count_params_test(f, count_bias)
        cr_test = round(params_test / total_params_full, 3)
        params_train = count_params_train(f, count_bias)
        cr_train = round(params_train / total_params_full, 3)
        params_train_grads = count_params_train(f, count_bias, True)
        cr_train_grads = round(params_train_grads / total_params_full_grads, 3)
        epoch_data = [epoch, optimizer.theta, round(optimizer.tau, 5), round(loss_hist, 3), round(acc_hist * 100, 4),
                    round(top_5_acc_hist * 100, 4), \
                    round(acc_hist_test * 100, 4), round(top_5_acc_hist_test*100,4), ranks, params_test,
                    round(100 * (1 - cr_test), 4), \
                    params_train, round(100 * (1 - cr_train), 4), params_train_grads,
                    round(100 * (1 - cr_train_grads), 4)]

        running_data.loc[epoch] = epoch_data

        if file_name is not None:
            if d == 10:
                running_data.to_csv(file_name +'vanilla_lenet_exp10' + '.csv')
            elif d==1:
                running_data.to_csv(file_name +'vanilla_lenet_baseline' + '.csv')
            elif d == 2:
                running_data.to_csv(file_name +'vanilla_lenet_exp2' + '.csv')

        if epoch == 0:
            best_val_loss = loss_hist_test

        if loss_hist_test < best_val_loss:
            pass#torch.save(f.state_dict(), path +'vanilla_lenet_' + '_best_weights.pt')






