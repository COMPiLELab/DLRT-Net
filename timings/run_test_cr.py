import torch
from test_network import net
import argparse
import numpy as np
import tensorflow as tf
import pandas as pd
import time
from optimizer_KLS.train_custom_optimizer import * 

parser = argparse.ArgumentParser(description='Pytorch dlrt timings')  
parser.add_argument('--cv_runs', type=int, default=5, metavar='CV_RUNS',
                    help='number of runs for c.i. (default: 10)')  
parser.add_argument('--step', type=float, default=0.1, metavar='STEP',
                    help='step for the timing grid of the experiment (default: 0.1)')
parser.add_argument('--device', type=str, default='cpu', metavar='device',
                    help='device to use for the experiment (default: cuda)')                        
args = parser.parse_args()

device = 'cuda' if torch.cuda.is_available() and args.device =='cuda' else 'cpu'

# dataset creation

(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data(path="mnist.npz")
x = np.vstack([x_train,x_test])
y = np.hstack([y_train,y_test])
x,y = torch.tensor(x).float()/255,torch.tensor(y)
x,y = x.to(device),y.to(device)

compression_ratios = np.arange(0,1,args.step)
starting_ranks = [500,300,100]
sizes = [784,500,300,100,10]

timings_data = pd.DataFrame(data=None, columns=['cr_layerwise','ranks','device','time forward train','time backward train','time forward test',\
                                                     '# effective parameters', 'cr_test (%)',\
                                                    '# effective parameters train', 'cr_train (%)', \
                                                    '# effective parameters train with grads', 'cr_train_grads (%)'])

counter = 0
criterion = torch.nn.CrossEntropyLoss()

for _ in range(args.cv_runs):

    print(f'cv run = {_}')
    print('='*50)

    ranks = [None]*4
    NN = net(device =device,ranks = ranks,fixed = False).to(device)

    count_bias = False
    total_params_full = full_count_params(NN, count_bias)
    total_params_full_grads = full_count_params(NN, count_bias, True)

    time_forward_train = time.time()
    NN(x)
    time_forward_train = time.time()-time_forward_train

    time_backward_train = time.time()
    loss = criterion(NN(x),y)
    loss.backward()
    time_backward_train = time.time()-time_backward_train

    time_forward_test = time.time()
    NN(x)
    time_forward_test = time.time()-time_forward_test



    params_test = count_params_test(NN, count_bias)
    cr_test = round(params_test / total_params_full, 3)
    params_train = count_params_train(NN, count_bias)
    cr_train = round(params_train / total_params_full, 3)
    params_train_grads = count_params_train(NN, count_bias, True)
    cr_train_grads = round(params_train_grads / total_params_full_grads, 3)



    cr_data = ['baseline', [l.rank for l in NN.layer if hasattr(l,'rank')], device, round(time_forward_train, 4), round(time_backward_train, 4), \
                    round(time_forward_test, 4), params_test,\
                    round(100 * (1 - cr_test), 4), \
                    params_train, round(100 * (1 - cr_train), 4), params_train_grads,
                    round(100 * (1 - cr_train_grads), 4)]

    timings_data.loc[counter] = cr_data

    timings_data.to_csv('./results/timings_cr.csv')

    counter+=1

    for i,cr in enumerate(compression_ratios):

        #ranks = [max([int(r*(1-cr)),2]) for r in starting_ranks]
        ranks = [int((1-cr)*sizes[i]*sizes[i+1]/(sizes[i]+sizes[i+1]+1)) for i in range(len(sizes)-1)]
        NN = net(device =device,ranks = ranks).to(device)

        count_bias = False
        total_params_full = full_count_params(NN, count_bias)
        total_params_full_grads = full_count_params(NN, count_bias, True)

        time_forward_train = time.time()
        NN.forward_train(x)
        time_forward_train = time.time()-time_forward_train

        time_backward_train = time.time()
        NN.backward_train(x,y,criterion)
        time_backward_train = time.time()-time_backward_train

        time_forward_test = time.time()
        NN(x)
        time_forward_test = time.time()-time_forward_test



        params_test = count_params_test(NN, count_bias)
        cr_test = round(params_test / total_params_full, 3)
        params_train = count_params_train(NN, count_bias)
        cr_train = round(params_train / total_params_full, 3)
        params_train_grads = count_params_train(NN, count_bias, True)
        cr_train_grads = round(params_train_grads / total_params_full_grads, 3)



        cr_data = [round(cr,3), [l.rank for l in NN.layer if hasattr(l,'rank')], device, round(time_forward_train, 4), round(time_backward_train, 4), \
                        round(time_forward_test, 4), params_test,\
                        round(100 * (1 - cr_test), 4), \
                        params_train, round(100 * (1 - cr_train), 4), params_train_grads,
                        round(100 * (1 - cr_train_grads), 4)]

        timings_data.loc[counter] = cr_data

        timings_data.to_csv('./results/timings_cr.csv')

        counter+=1