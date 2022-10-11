#%%
from tqdm import tqdm
import torch
from torch import float16
import pandas as pd

def full_count_params(NN,count_bias = False,with_grads = False):

    """ 
    Function that counts the total number of parameters needed for a full rank version of NN
    INPUTS:
    NN: neural network
    count_bias : flag variable, True if the biases are to be included in the total or not

    OUTPUTS:
    total_params : total number of parameters in the full rank version of NN
    """

    total_params = 0

    for l in NN.layer:

        n = str(l)

        if 'Linear' in n:

            total_params += 2*l.in_features*l.out_features if with_grads else l.in_features*l.out_features

            if count_bias and l.bias is not None:

                total_params += 2*len(l.bias) if with_grads else len(l.bias)

        if 'Conv' in n:

            total_params += 2*l.kernel_size_number*l.in_channels*l.out_channels if with_grads else l.kernel_size_number*l.in_channels*l.out_channels

            if count_bias and l.bias is not None:

                total_params += 2*len(l.bias) if with_grads else len(l.bias)

    return total_params




def count_params(T,with_grads = False):

    """ 
    function to count number of parameters inside a tensor
    INPUT:
    T : torch.tensor or None
    output:
    number of parameters contained in T
    """

    if len(T.shape)>1:

        if with_grads:

            return 2*int(torch.prod(torch.tensor(T.shape)))

        else:

            return int(torch.prod(torch.tensor(T.shape)))

    elif T == None:

        return 0

    else:

        if with_grads:

            return 2*T.shape[0]
        
        else:

            return T.shape[0]


def count_params_train(NN,count_bias = False,with_grads = False):

    """ 
    function to count the parameters in the train phase
    
    INPUTS:
    NN : neural network
    count_bias : flag variable, True if the biases are to be included in the total or not
    """

    total_params = 0

    for l in NN.layer:

        if hasattr(l,'lr') and l.lr:

            if not l.fixed:

                total_params += count_params(l.K[:,:l.dynamic_rank],with_grads)
                total_params += count_params(l.L[:,:l.dynamic_rank],with_grads)
                total_params += count_params(l.U[:,:l.dynamic_rank])
                total_params += count_params(l.V[:,:l.dynamic_rank])
                total_params += count_params(l.U_hat[:,:2*l.dynamic_rank])
                total_params += count_params(l.V_hat[:,:2*l.dynamic_rank])
                total_params += count_params(l.S_hat[:2*l.dynamic_rank,:2*l.dynamic_rank],with_grads)
                total_params += count_params(l.M_hat[:2*l.dynamic_rank,:l.dynamic_rank])
                total_params += count_params(l.N_hat[:2*l.dynamic_rank,:l.dynamic_rank])
                if count_bias:
                    total_params +=count_params(l.bias)

            else:

                total_params += count_params(l.K[:,:l.dynamic_rank],with_grads)
                total_params += count_params(l.L[:,:l.dynamic_rank],with_grads)
                total_params += count_params(l.U[:,:l.dynamic_rank])
                total_params += count_params(l.V[:,:l.dynamic_rank])
                total_params += count_params(l.S_hat[:2*l.dynamic_rank,:2*l.dynamic_rank],with_grads)
                total_params += count_params(l.M_hat[:2*l.dynamic_rank,:l.dynamic_rank])
                total_params += count_params(l.N_hat[:2*l.dynamic_rank,:l.dynamic_rank])
                if count_bias:
                    total_params +=count_params(l.bias)

        else:

            for n,p in l.named_parameters():

                if 'bias' not in n:

                    total_params += count_params(p,with_grads)   # add with grads

                elif 'bias' in n and count_bias:

                    total_params += count_params(p)

    return total_params


def count_params_test(NN,count_bias = False):

    """ 
    function to count the parameters in the test phase
    
    INPUTS:
    NN : neural network
    count_bias : flag variable, True if the biases are to be included in the total or not
    """

    total_params = 0

    for l in NN.layer:

        if hasattr(l,'lr') and l.lr:

            total_params += count_params(l.K[:,:l.dynamic_rank])
            total_params += count_params(l.L[:,:l.dynamic_rank])
            if count_bias:
                total_params +=count_params(l.bias)

        else:

            for n,p in l.named_parameters():

                if 'bias' not in n:

                    total_params += count_params(p)

                elif 'bias' in n and count_bias:

                    total_params +=count_params(p)

    return total_params




def accuracy(outputs,labels):

    return torch.mean(torch.tensor(torch.argmax(outputs.detach(),axis = 1) == labels,dtype = float16))

            


def train_dlrt(NN,optimizer,train_loader,validation_loader,test_loader,criterion,metric,epochs,
                metric_name = 'accuracy',device = 'cpu',count_bias = False,path = None,fine_tune = False,scheduler = None):

    """ 
    INPUTS:
    NN : neural network with custom layers and methods to optimize with dlra
    train/validation/test_loader : loader for datasets
    criterion : loss function
    metric : metric function
    epochs : number of epochs to train
    metric_name : name of the used metric
    count_bias : flag variable if to count biases in params_count or not
    path : path string for where to save the results

    OUTPUTS:
    running_data : Pandas dataframe with the results of the run
    """

    running_data = pd.DataFrame(data = None,columns = ['epoch','theta','learning_rate','train_loss','train_'+metric_name+'(%)','validation_loss',\
                                                        'validation_'+metric_name+'(%)','test_'+metric_name+'(%)',\
                                                     'ranks','# effective parameters','cr_test (%)','# effective parameters train','cr_train (%)',\
                                                     '# effective parameters train with grads','cr_train_grads (%)'])

    total_params_full = full_count_params(NN,count_bias)
    total_params_full_grads = full_count_params(NN,count_bias,True)
    #scheduler_rate = optimizer.scheduler_change_rate

    file_name = path

    if not fine_tune:

        if path is not None:
            file_name += '.csv'#'\_running_data_'+str(optimizer.theta)+'.csv'

        for epoch in tqdm(range(epochs)):

            print(f'epoch {epoch}---------------------------------------------')
            loss_hist = 0
            acc_hist = 0
            k = len(train_loader)

            for i,data in enumerate(train_loader):  # train
                NN.zero_grad()
                optimizer.zero_grad()
                inputs,labels = data
                inputs,labels = inputs.to(device),labels.to(device)
                def closure():
                    loss = NN.populate_gradients(inputs,labels,criterion,step = 'S')
                    return loss
                optimizer.preprocess_step()
                loss,outputs = NN.populate_gradients(inputs,labels,criterion)
                loss_hist+=float(loss.item())/k
                outputs = outputs.to(device)#NN(inputs).detach().to(device)
                acc_hist += float(metric(outputs,labels))/k
                optimizer.step(closure = closure)

            optimizer.preprocess_step()   # last update after training
            NN.update_step()

            with torch.no_grad():
                k = len(validation_loader)
                loss_hist_val = 0.0
                acc_hist_val = 0.0
                for i,data in enumerate(validation_loader):   # validation 
                    inputs,labels = data
                    inputs,labels = inputs.to(device),labels.to(device)
                    outputs = NN(inputs).detach().to(device)
                    loss_val = criterion(outputs,labels)
                    loss_hist_val+=float(loss_val.item())/k
                    acc_hist_val += float(metric(outputs,labels))/k


                k = len(test_loader)
                loss_hist_test = 0.0
                acc_hist_test= 0.0
                for i,data in enumerate(test_loader):   # validation 
                    inputs,labels = data
                    inputs,labels = inputs.to(device),labels.to(device)
                    outputs = NN(inputs).detach().to(device)
                    loss_test = criterion(outputs,labels)
                    loss_hist_test += float(loss_test.item())/k
                    acc_hist_test += float(metric(outputs,labels))/k

            print(f'epoch[{epoch}]: loss: {loss_hist:9.4f} | {metric_name}: {acc_hist:9.4f} | val loss: {loss_hist_val:9.4f} | val {metric_name}:{acc_hist_val:9.4f}')
            print('='*100)
            ranks = []
            for i,l in enumerate(NN.layer):
                if hasattr(l,'lr') and l.lr:
                    print(f'rank layer {i} {l.dynamic_rank}')
                    ranks.append(l.dynamic_rank)
            print('\n')

            params_test = count_params_test(NN,count_bias)
            cr_test = round(params_test/total_params_full,3)
            params_train = count_params_train(NN,count_bias)
            cr_train = round(params_train/total_params_full,3)
            params_train_grads = count_params_train(NN,count_bias,True)
            cr_train_grads = round(params_train_grads/total_params_full_grads,3)
            epoch_data = [epoch,optimizer.theta,round(optimizer.tau,5),round(loss_hist,3),round(acc_hist*100,4),round(loss_hist_val,3),\
                        round(acc_hist_val*100,4),round(acc_hist_test*100,4),ranks,params_test,round(100*(1-cr_test),4),\
                            params_train,round(100*(1-cr_train),4),params_train_grads,round(100*(1-cr_train_grads),4)]

            running_data.loc[epoch] = epoch_data

            if file_name is not None and (epoch%10 == 0 or epoch == epochs-1):

                running_data.to_csv(file_name)

            if scheduler is not None:

                scheduler.step(loss_hist)

            # if epoch%scheduler_rate == 0:

            #     optimizer.scheduler_step()

            if epoch == 0:

                best_val_loss = loss_hist_val

            if loss_hist_val<best_val_loss:

                torch.save(NN.state_dict(),path+'\_best_weights_'+str(optimizer.theta)+'.pt')

        return running_data

    else:

        if path is not None:
            file_name += '_finetune.csv'#'\_running_data_'+str(optimizer.theta)+'.csv'

        for epoch in tqdm(range(epochs)):

            print(f'epoch {epoch}---------------------------------------------')
            loss_hist = 0
            acc_hist = 0
            k = len(train_loader)

            for i,data in enumerate(train_loader):  # train
                NN.zero_grad()
                optimizer.zero_grad()
                inputs,labels = data
                inputs,labels = inputs.to(device),labels.to(device)
                outputs = NN(inputs).to(device)
                loss = criterion(outputs,labels)
                loss.backward()
                loss_hist+=float(loss.item())/k
                acc_hist += float(metric(outputs.detach(),labels))/k
                optimizer.S_finetune_step()


            with torch.no_grad():
                k = len(validation_loader)
                loss_hist_val = 0.0
                acc_hist_val = 0.0
                for i,data in enumerate(validation_loader):   # validation 
                    inputs,labels = data
                    inputs,labels = inputs.to(device),labels.to(device)
                    outputs = NN(inputs).detach().to(device)
                    loss_val = criterion(outputs,labels)
                    loss_hist_val+=float(loss_val.item())/k
                    acc_hist_val += float(metric(outputs,labels))/k

                k = len(test_loader)
                loss_hist_test = 0.0
                acc_hist_test= 0.0
                for i,data in enumerate(test_loader):   # validation 
                    inputs,labels = data
                    inputs,labels = inputs.to(device),labels.to(device)
                    outputs = NN(inputs).detach().to(device)
                    loss_test = criterion(outputs,labels)
                    loss_hist_test += float(loss_test.item())/k
                    acc_hist_test += float(metric(outputs,labels))/k

            print(f'epoch[{epoch}]: loss: {loss_hist:9.4f} | {metric_name}: {acc_hist:9.4f} | val loss: {loss_hist_val:9.4f} | val {metric_name}:{acc_hist_val:9.4f}')
            print('='*100)
            ranks = []
            for i,l in enumerate(NN.layer):
                if hasattr(l,'lr') and l.lr:
                    print(f'rank layer {i} {l.dynamic_rank}')
                    ranks.append(l.dynamic_rank)
            print('\n')

            params_test = count_params_test(NN,count_bias)
            cr_test = round(params_test/total_params_full,3)
            params_train = count_params_train(NN,count_bias)
            cr_train = round(params_train/total_params_full,3)
            params_train_grads = count_params_train(NN,count_bias,True)
            cr_train_grads = round(params_train_grads/total_params_full_grads,3)
            epoch_data = [epoch,optimizer.theta,round(optimizer.tau,5),round(loss_hist,3),round(acc_hist*100,4),round(loss_hist_val,3),\
                        round(acc_hist_val*100,4),round(acc_hist_test*100,4),ranks,params_test,round(100*(1-cr_test),4),\
                            params_train,round(100*(1-cr_train),4),params_train_grads,round(100*(1-cr_train_grads),4)]

            running_data.loc[epoch] = epoch_data

            if file_name is not None and (epoch%10 == 0 or epoch == epochs-1):

                running_data.to_csv(file_name)

            
            if scheduler is not None:

                scheduler.step(loss_hist)

            # if epoch%scheduler_rate == 0:

            #     optimizer.scheduler_step()

            if epoch == 0:

                best_val_loss = loss_hist_val

            if loss_hist_val<best_val_loss:

                torch.save(NN.state_dict(),path+'\_best_weights_finetune_'+str(optimizer.theta)+'.pt')

        return running_data