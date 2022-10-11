#%%
from tqdm import tqdm
import torch
from torchmetrics import Accuracy
from torch import float16
import pandas as pd
import os
import sys

sys.path.insert(1, os.path.join(sys.path[0], '..'))
from optimizer_KLS.train_custom_optimizer import * 

def test_phase_metrics(NN, criterion, dataloader,device = 'cpu'):  # pseudo code
    
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


def train_and_finetune(NN,epochs,criterion,optimizer,scheduler,train_loader,validation_loader,path,device = 'cpu',net_name = 'vgg',save_weights = False):

    top1_metric = Accuracy().to(device)
    top5_metric = Accuracy(top_k = 5).to(device)

    running_data = pd.DataFrame(data=None, columns=['epoch', 'theta', 'learning_rate', 'train_loss', 'train_accuracy(%)',
                                                    'top_5_accuracy(%)', \
                                                    'test_accuracy(%)', 'top_5_test_accuracy(%)', \
                                                    'ranks', '# effective parameters', 'cr_test (%)',
                                                    '# effective parameters train', 'cr_train (%)', \
                                                    '# effective parameters train with grads', 'cr_train_grads (%)'])

    count_bias = False
    total_params_full = full_count_params(NN, count_bias)
    total_params_full_grads = full_count_params(NN, count_bias, True)
    file_name = path


    for epoch in tqdm(range(epochs)):

        print(f'epoch {epoch}---------------------------------------------')
        loss_hist = 0
        acc_hist = 0
        top_5_acc_hist = 0
        k = len(train_loader)

        NN.train()
        total = 0.0
        correct = 0.0
        for i, data in enumerate(train_loader):  # train
            NN.zero_grad()
            optimizer.zero_grad()
            inputs, labels = data
            inputs, labels = inputs.to(device), labels.to(device)

            def closure():
                loss = NN.populate_gradients(inputs, labels, criterion, step='S')
                return loss


            optimizer.preprocess_step()
            loss,outputs = NN.populate_gradients(inputs, labels, criterion)

            loss_hist += float(loss.item()) / k
            outputs = outputs.to(device)#NN(inputs).detach().to(device)
            top_5_acc_hist +=  float(top5_metric(outputs, labels)) / k
            predicted = torch.argmax(outputs.data, 1)
            total += labels.size(0)
            correct += torch.tensor(predicted == labels,dtype = float16).sum().item()
            optimizer.step(closure=closure)
        acc_hist,top_5_acc_hist = correct/total,top_5_acc_hist
        optimizer.preprocess_step()  # last update after training
        NN.update_step()

        acc_hist_test, top_5_acc_hist_test, loss_hist_test = test_phase_metrics(NN, criterion, validation_loader,device)

        print(f'epoch[{epoch}]: loss: {loss_hist:9.4f} | top_1,top_5_acc: {acc_hist:9.4f},{top_5_acc_hist:9.4f}')
        print('=' * 100)
        ranks = []
        for i, l in enumerate(NN.layer):
            if hasattr(l, 'lr') and l.lr:
                print(f'rank layer {i} {l.dynamic_rank}')
                ranks.append(l.dynamic_rank)
        print('\n')

        params_test = count_params_test(NN, count_bias)
        cr_test = round(params_test / total_params_full, 3)
        params_train = count_params_train(NN, count_bias)
        cr_train = round(params_train / total_params_full, 3)
        params_train_grads = count_params_train(NN, count_bias, True)
        cr_train_grads = round(params_train_grads / total_params_full_grads, 3)
        lr = float(optimizer.integrator.param_groups[0]['lr'])
        epoch_data = [epoch, optimizer.theta, round(lr, 5), round(loss_hist, 3), round(acc_hist * 100, 4),
                    round(top_5_acc_hist * 100, 4), \
                    round(acc_hist_test * 100, 4), round(top_5_acc_hist_test*100,4), ranks, params_test,
                    round(100 * (1 - cr_test), 4), \
                    params_train, round(100 * (1 - cr_train), 4), params_train_grads,
                    round(100 * (1 - cr_train_grads), 4)]

        running_data.loc[epoch] = epoch_data
        scheduler.step(loss_hist)

        if file_name is not None:
            running_data.to_csv(file_name +'_running_data_'+net_name + str(optimizer.theta)+ '.csv')

        if epoch == 0:
            best_val_loss = loss_hist_test

        if (loss_hist_test < best_val_loss) and save_weights:
            torch.save(NN.state_dict(), path +net_name + str(optimizer.theta)+'_best_weights.pt')


    optimizer.activate_S_fine_tuning()
    print("START FINETUNING")

    for epoch in tqdm(range(epochs)):

        print(f'epoch {epoch}---------------------------------------------')
        loss_hist = 0
        acc_hist = 0
        top_5_acc_hist = 0
        k = len(train_loader)
        NN.train()
        for i, data in enumerate(train_loader):  # train
            NN.zero_grad()
            optimizer.zero_grad()
            inputs, labels = data
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = NN(inputs).to(device)
            loss = criterion(outputs,labels)
            loss.backward()
            outputs = outputs.detach()
            loss_hist += float(loss.item()) / k

            acc_hist += float(top1_metric(outputs, labels)) / k
            top_5_acc_hist += float(top5_metric(outputs, labels)) / k
            optimizer.S_finetune_step()


        acc_hist_test, top_5_acc_hist_test, loss_hist_test = test_phase_metrics(NN, criterion, validation_loader,device)

        print(f'epoch[{epoch}]: loss: {loss_hist:9.4f} | top_1,top_5_acc: {acc_hist:9.4f},{top_5_acc_hist:9.4f}')
        print('=' * 100)
        ranks = []
        for i, l in enumerate(NN.layer):
            if hasattr(l, 'lr') and l.lr:
                print(f'rank layer {i} {l.dynamic_rank}')
                ranks.append(l.dynamic_rank)
        print('\n')
        scheduler.step(loss_hist)

        params_test = count_params_test(NN, count_bias)
        cr_test = round(params_test / total_params_full, 3)
        params_train = count_params_train(NN, count_bias)
        cr_train = round(params_train / total_params_full, 3)
        params_train_grads = count_params_train(NN, count_bias, True)
        cr_train_grads = round(params_train_grads / total_params_full_grads, 3)
        lr = float(optimizer.integrator.param_groups[0]['lr'])
        epoch_data = [epoch, optimizer.theta, round(lr, 5), round(loss_hist, 3), round(acc_hist * 100, 4),
                    round(top_5_acc_hist * 100, 4), \
                    round(acc_hist_test * 100, 4), round(top_5_acc_hist_test*100,4), ranks, params_test,
                    round(100 * (1 - cr_test), 4), \
                    params_train, round(100 * (1 - cr_train), 4), params_train_grads,
                    round(100 * (1 - cr_train_grads), 4)]

        running_data.loc[epoch+epochs] = epoch_data

        if file_name is not None:
            running_data.to_csv(file_name +'_running_data_'+net_name+ str(optimizer.theta)+'_ft.csv')

        if epoch == 0:
            best_val_loss = loss_hist_test

        if (loss_hist_test < best_val_loss) and save_weights:
            torch.save(NN.state_dict(), path +net_name+ str(optimizer.theta)+ '_ft_best_weights.pt')



def train_baseline(NN,epochs,criterion,optimizer,scheduler,train_loader,validation_loader,path,device = 'cpu',net_name = 'vgg',save_weights = False):

    top1_metric = Accuracy().to(device)
    top5_metric = Accuracy(top_k = 5).to(device)

    running_data = pd.DataFrame(data=None, columns=['epoch', 'theta', 'learning_rate', 'train_loss', 'train_accuracy(%)',
                                                    'top_5_accuracy(%)', \
                                                    'test_accuracy(%)', 'top_5_test_accuracy(%)', \
                                                    'ranks', '# effective parameters', 'cr_test (%)',
                                                    '# effective parameters train', 'cr_train (%)', \
                                                    '# effective parameters train with grads', 'cr_train_grads (%)'])

    count_bias = False
    total_params_full = full_count_params(NN, count_bias)
    total_params_full_grads = full_count_params(NN, count_bias, True)
    file_name = path


    for epoch in tqdm(range(epochs)):

        print(f'epoch {epoch}---------------------------------------------')
        loss_hist = 0
        acc_hist = 0
        top_5_acc_hist = 0
        k = len(train_loader)

        NN.train()
        total = 0.0
        correct = 0.0
        for i, data in enumerate(train_loader):  # train
            NN.zero_grad()
            optimizer.zero_grad()
            inputs, labels = data
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = NN(inputs).to(device)
            loss = criterion(outputs,labels)
            loss.backward()
            outputs = outputs.detach()
            loss_hist += float(loss.item()) / k
            top_5_acc_hist +=  float(top5_metric(outputs, labels)) / k
            predicted = torch.argmax(outputs.data, 1)
            total += labels.size(0)
            correct += torch.tensor(predicted == labels,dtype = float16).sum().item()
            optimizer.step()
        acc_hist,top_5_acc_hist = correct/total,top_5_acc_hist

        acc_hist_test, top_5_acc_hist_test, loss_hist_test = test_phase_metrics(NN, criterion, validation_loader,device)

        print(f'epoch[{epoch}]: loss: {loss_hist:9.4f} | top_1,top_5_acc: {acc_hist:9.4f},{top_5_acc_hist:9.4f}')
        print('=' * 100)
        ranks = []
        for i, l in enumerate(NN.layer):
            if hasattr(l, 'lr') and l.lr:
                print(f'rank layer {i} {l.dynamic_rank}')
                ranks.append(l.dynamic_rank)
        print('\n')

        params_test = count_params_test(NN, count_bias)
        cr_test = round(params_test / total_params_full, 3)
        params_train = count_params_train(NN, count_bias)
        cr_train = round(params_train / total_params_full, 3)
        params_train_grads = count_params_train(NN, count_bias, True)
        cr_train_grads = round(params_train_grads / total_params_full_grads, 3)
        lr = float(optimizer.param_groups[0]['lr'])
        epoch_data = [epoch, 0.0, round(lr, 5), round(loss_hist, 3), round(acc_hist * 100, 4),
                    round(top_5_acc_hist * 100, 4), \
                    round(acc_hist_test * 100, 4), round(top_5_acc_hist_test*100,4), ranks, params_test,
                    round(100 * (1 - cr_test), 4), \
                    params_train, round(100 * (1 - cr_train), 4), params_train_grads,
                    round(100 * (1 - cr_train_grads), 4)]

        running_data.loc[epoch] = epoch_data
        scheduler.step(loss_hist)

        if file_name is not None:
            running_data.to_csv(file_name +'_running_data_'+net_name + '_0.0_'+ '.csv')

        if epoch == 0:
            best_val_loss = loss_hist_test

        if (loss_hist_test < best_val_loss) and save_weights:
            torch.save(NN.state_dict(), path +net_name + '_0.0_'+'_best_weights.pt')