#%%
from distutils.log import warn
from email.mime import base
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
from os import walk
path = './results/'
warnings.filterwarnings('ignore')

def extract_best_val(run,metric = 'acc'):    
  if metric == 'acc':
    i = np.argmax(run['test_accuracy(%)'])
  else:
    i = np.argmin(run['validation_loss'])
  return run.loc[i].to_frame()


grid = ['baseline',0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,0.02,0.04,0.06,0.08]
runs = []

file_names = list()
for root, dirc, files in walk(path):
    for FileName in files:
      if 'running' in FileName and 'old.csv' not in FileName:
        el = pd.read_csv(path+FileName)
        el.drop(columns = ['Unnamed: 0'],inplace = True)
        runs.append(el)
        file_names.append(FileName)

#print(runs[0])
#print(file_names)

# group_runs = []

# for i,suffix in enumerate(grid):

#     group_runs.append([])

#     for j,name in enumerate(file_names):

#         if str(suffix) in name:

#             group_runs[i].append(runs[j])


#print(group_runs[3])


cv_results = []
for i,el in enumerate(runs):
  cv_results.append(extract_best_val(el,'acc'))
cv_results = pd.concat(cv_results,axis = 1).transpose()
#cv_results.loc[cv_results['cr layerwise'] == 'baseline']['cr layerwise'] = -1
cv_results_baseline = cv_results.loc[cv_results['cr layerwise'] == 'Baseline']
cv_results_dlrt = cv_results.loc[cv_results['cr layerwise'] != 'Baseline']
#print(cv_results_dlrt['cr layerwise'].values)
cv_results_dlrt.sort_values(by = 'cr layerwise',inplace = True)
mean_acc_dlrt = cv_results_dlrt[['cr layerwise','test_accuracy(%)','cr_train_grads (%)']]

plt.clf()
sns.set_theme()
sns.set_style("white")
colors = ['k', 'r', 'g', 'b']
symbol_size = 0.7
markersize = 2.5
markerwidth = 0.5
upper_bounds_dlrt = []
lower_bounds_dlrt = []
mean_dlrt = []
for cr in mean_acc_dlrt['cr layerwise'].values:

    indexes = mean_acc_dlrt['cr layerwise'] == cr
    #plt.plot(cr,mean_acc_dlrt.loc[indexes][["test_accuracy(%)"]].mean(), '-ok')
    #plt.plot(mean_vanilla[["test_accuracy(%)"]], '-or')
    upper_bound_dlrt = mean_acc_dlrt.loc[indexes][["test_accuracy(%)",'cr_train_grads (%)']].mean(axis = 0)+mean_acc_dlrt.loc[indexes][["test_accuracy(%)",'cr_train_grads (%)']].std(axis = 0)
    lower_bound_dlrt = mean_acc_dlrt.loc[indexes][["test_accuracy(%)",'cr_train_grads (%)']].mean(axis = 0)-mean_acc_dlrt.loc[indexes][["test_accuracy(%)",'cr_train_grads (%)']].std(axis = 0)
    #bounds_dlrt.append((lower_bound_dlrt,upper_bound_dlrt))
    upper_bounds_dlrt.append(upper_bound_dlrt)
    lower_bounds_dlrt.append(lower_bound_dlrt)
    mean_dlrt.append(mean_acc_dlrt.loc[indexes][["test_accuracy(%)",'cr_train_grads (%)']].mean(axis = 0))
    #plt.fill_between(range(10),upper_bound_dlrt,lower_bound_dlrt, color='black', alpha=.2)
    #plt.fill_between(range(10),upper_bound_vanilla,lower_bound_vanilla, color='red', alpha=.2)

n = len(mean_acc_dlrt['cr layerwise'].values)
baseline_mean = [cv_results_baseline[["test_accuracy(%)"]].mean()]*n
upper_bound_baseline = [cv_results_baseline[["test_accuracy(%)"]].mean()+cv_results_baseline[["test_accuracy(%)"]].std()]*n
lower_bound_baseline = [cv_results_baseline[["test_accuracy(%)"]].mean()-cv_results_baseline[["test_accuracy(%)"]].std()]*n

#print(np.array(upper_bounds_dlrt,dtype = float).reshape(-1,))
plt.plot(mean_acc_dlrt['cr layerwise'].values,[el['test_accuracy(%)'] for el in mean_dlrt], '-ok')
plt.fill_between(np.array(mean_acc_dlrt['cr layerwise'].values,dtype = float),
                np.array([el['test_accuracy(%)'] for el in upper_bounds_dlrt],dtype = float).reshape(-1,),
                np.array([el['test_accuracy(%)'] for el in lower_bounds_dlrt],dtype = float).reshape(-1,), color='black', alpha=.2,label = '_nolegend_')

########## for cr grads
# plt.plot(mean_acc_dlrt['cr layerwise'].values,[el['cr_train_grads (%)'] for el in mean_dlrt], '-',color = 'blue')
# plt.fill_between(np.array(mean_acc_dlrt['cr layerwise'].values,dtype = float),
#                 np.array([el['cr_train_grads (%)'] for el in upper_bounds_dlrt],dtype = float).reshape(-1,),
#                 np.array([el['cr_train_grads (%)'] for el in lower_bounds_dlrt],dtype = float).reshape(-1,), color='blue', alpha=.2,label = '_nolegend_')

plt.plot(mean_acc_dlrt['cr layerwise'].values,baseline_mean,'--',color = 'red')
plt.fill_between(np.array(mean_acc_dlrt['cr layerwise'].values,dtype = float),
                np.array(upper_bound_baseline,dtype = float).reshape(-1,),
                np.array(lower_bound_baseline,dtype = float).reshape(-1,), color='red', alpha=.2)
# # plt.plot(dlra_3layer[["acc_test"]], '--g')
plt.xticks(np.arange(0,1.1,0.1),np.round(np.arange(0,1.1,0.1),2))
# plt.legend(["DLRT test accuracy(%)",'overall compression ratio gradients(%)','baseline test accuracy(%)'])
plt.legend(["DLRT",'baseline'])
plt.ylim([89, 100])

plt.ylabel("test accuracy[%]")
plt.xlabel("compression ratio layer-wise")

ax = plt.gca()  # you first need to get the axis handle
x_left, x_right = ax.get_xlim()
y_low, y_high = ax.get_ylim()
ratio = 0.5
plt.savefig("acc_vs_cr.pdf", dpi=500,format = 'pdf')
plt.clf()
plt.show()




# data_dlrt = pd.read_csv('_running_data_dlrt0dlrt_lenet.csv')[['epoch','test_accuracy(%)']]
# data_vanilla = pd.read_csv('_running_data_vanilla0vanilla_lenet.csv')[['epoch','test_accuracy(%)']]
# mean = data_dlrt.copy()
# mean_vanilla = data_vanilla.copy()
# for cv in range(1,10):
#     data = pd.read_csv('_running_data_dlrt'+str(cv)+'dlrt_lenet.csv')[['epoch','test_accuracy(%)']]
#     data_dlrt = pd.concat([data_dlrt,data])
#     data_UV = pd.read_csv('_running_data_vanilla'+str(cv)+'vanilla_lenet.csv')[['epoch','test_accuracy(%)']]
#     mean+=data
#     mean_vanilla+=data_UV

# mean = mean/10
# mean_vanilla = mean_vanilla/10

# data_dlrt = pd.read_csv('_running_data_dlrt0dlrt_lenet.csv')[['epoch','test_accuracy(%)']]
# data_vanilla = pd.read_csv('_running_data_vanilla0vanilla_lenet.csv')[['epoch','test_accuracy(%)']]
# std = (data_dlrt.copy()-mean)**2
# std_vanilla = (data_vanilla.copy()-mean_vanilla)**2
# for cv in range(1,10):
#     data = pd.read_csv('_running_data_dlrt'+str(cv)+'dlrt_lenet.csv')[['epoch','test_accuracy(%)']]
#     data_dlrt = pd.concat([data_dlrt,data])
#     data_UV = pd.read_csv('_running_data_vanilla'+str(cv)+'vanilla_lenet.csv')[['epoch','test_accuracy(%)']]
#     std +=(data-mean)**2
#     std_vanilla +=(data_UV-mean_vanilla)**2
# std = (np.sqrt(std/10))[['test_accuracy(%)']]
# std_vanilla = (np.sqrt(std_vanilla/10))[['test_accuracy(%)']]


# upper_bound_dlrt = [(mean.loc[i]+std.loc[i])['test_accuracy(%)'] for i in range(len(std))]
# lower_bound_dlrt = [(mean.loc[i]-std.loc[i])['test_accuracy(%)'] for i in range(len(std))]

# upper_bound_vanilla = [(mean_vanilla.loc[i]+std_vanilla.loc[i])['test_accuracy(%)'] for i in range(len(std_vanilla))]
# lower_bound_vanilla = [(mean_vanilla.loc[i]-std_vanilla.loc[i])['test_accuracy(%)'] for i in range(len(std_vanilla))]


# plt.clf()
# sns.set_theme()
# sns.set_style("white")
# colors = ['k', 'r', 'g', 'b']
# symbol_size = 0.7
# markersize = 2.5
# markerwidth = 0.5
# plt.plot(mean[["test_accuracy(%)"]], '-ok')
# plt.plot(mean_vanilla[["test_accuracy(%)"]], '-or')
# plt.fill_between(range(10),upper_bound_dlrt,lower_bound_dlrt, color='black', alpha=.2)
# plt.fill_between(range(10),upper_bound_vanilla,lower_bound_vanilla, color='red', alpha=.2)
# # plt.plot(dlra_3layer[["acc_test"]], '--g')
# plt.legend(["DLRT", r"$UV^T$ factorization"])
# plt.ylim([10, 100])
# plt.ylabel("test accuracy [%]")
# plt.xlabel("epoch")

# ax = plt.gca()  # you first need to get the axis handle
# x_left, x_right = ax.get_xlim()
# y_low, y_high = ax.get_ylim()
# ratio = 0.5
# plt.savefig("crossval_acc_dlrt_vs_UV.png", dpi=500)
# plt.clf()

