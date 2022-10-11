#%%
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np


data_dlrt = pd.read_csv('_running_data_dlrt0dlrt_lenet_exp2.csv')[['epoch','test_accuracy(%)']]
data_vanilla = pd.read_csv('_running_data_vanilla0vanilla_lenet_exp2.csv')[['epoch','test_accuracy(%)']]
mean = data_dlrt.copy()
mean_vanilla = data_vanilla.copy()
for cv in range(1,10):
    data = pd.read_csv('_running_data_dlrt'+str(cv)+'dlrt_lenet_exp2.csv')[['epoch','test_accuracy(%)']]
    data_dlrt = pd.concat([data_dlrt,data])
    data_UV = pd.read_csv('_running_data_vanilla'+str(cv)+'vanilla_lenet_exp2.csv')[['epoch','test_accuracy(%)']]
    mean+=data
    mean_vanilla+=data_UV

mean = mean/10
mean_vanilla = mean_vanilla/10

data_dlrt = pd.read_csv('_running_data_dlrt0dlrt_lenet_exp2.csv')[['epoch','test_accuracy(%)']]
data_vanilla = pd.read_csv('_running_data_vanilla0vanilla_lenet_exp2.csv')[['epoch','test_accuracy(%)']]
std = (data_dlrt.copy()-mean)**2
std_vanilla = (data_vanilla.copy()-mean_vanilla)**2
for cv in range(1,10):
    data = pd.read_csv('_running_data_dlrt'+str(cv)+'dlrt_lenet_exp2.csv')[['epoch','test_accuracy(%)']]
    data_dlrt = pd.concat([data_dlrt,data])
    data_UV = pd.read_csv('_running_data_vanilla'+str(cv)+'vanilla_lenet_exp2.csv')[['epoch','test_accuracy(%)']]
    std +=(data-mean)**2
    std_vanilla +=(data_UV-mean_vanilla)**2
std = (np.sqrt(std/10))[['test_accuracy(%)']]
std_vanilla = (np.sqrt(std_vanilla/10))[['test_accuracy(%)']]


upper_bound_dlrt = [(mean.loc[i]+std.loc[i])['test_accuracy(%)'] for i in range(len(std))]
lower_bound_dlrt = [(mean.loc[i]-std.loc[i])['test_accuracy(%)'] for i in range(len(std))]

upper_bound_vanilla = [(mean_vanilla.loc[i]+std_vanilla.loc[i])['test_accuracy(%)'] for i in range(len(std_vanilla))]
lower_bound_vanilla = [(mean_vanilla.loc[i]-std_vanilla.loc[i])['test_accuracy(%)'] for i in range(len(std_vanilla))]


plt.clf()
sns.set_theme()
sns.set_style("white")
colors = ['k', 'r', 'g', 'b']
symbol_size = 0.7
markersize = 2.5
markerwidth = 0.5
plt.plot(mean[["test_accuracy(%)"]][0:11], '-ok')
plt.plot(mean_vanilla[["test_accuracy(%)"]][0:11], '-or')
plt.fill_between(range(11),upper_bound_dlrt[0:11],lower_bound_dlrt[0:11], color='black', alpha=.2)
plt.fill_between(range(11),upper_bound_vanilla[0:11],lower_bound_vanilla[0:11], color='red', alpha=.2)
# plt.plot(dlra_3layer[["acc_test"]], '--g')
plt.xticks(range(0,11,2),range(0,11,2))
plt.legend(["DLRT", r"$UV^T$ factorization"])
plt.ylim([10, 100])
plt.ylabel("test accuracy [%]")
plt.xlabel("epoch")

ax = plt.gca()  # you first need to get the axis handle
x_left, x_right = ax.get_xlim()
y_low, y_high = ax.get_ylim()
ratio = 0.5
plt.savefig("crossval_acc_dlrt_vs_UV_exp2.pdf", dpi=500,format = 'pdf')
plt.clf()



########################################################## loss


data_dlrt = pd.read_csv('_running_data_dlrt0dlrt_lenet_exp2.csv')[['epoch','train_loss']]
data_vanilla = pd.read_csv('_running_data_vanilla0vanilla_lenet_exp2.csv')[['epoch','train_loss']]
mean = data_dlrt.copy()
mean_vanilla = data_vanilla.copy()
for cv in range(1,10):
    data = pd.read_csv('_running_data_dlrt'+str(cv)+'dlrt_lenet_exp2.csv')[['epoch','train_loss']]
    data_dlrt = pd.concat([data_dlrt,data])
    data_UV = pd.read_csv('_running_data_vanilla'+str(cv)+'vanilla_lenet_exp2.csv')[['epoch','train_loss']]
    mean+=data
    mean_vanilla+=data_UV

mean = mean/10
mean_vanilla = mean_vanilla/10

data_dlrt = pd.read_csv('_running_data_dlrt0dlrt_lenet_exp2.csv')[['epoch','train_loss']]
data_vanilla = pd.read_csv('_running_data_vanilla0vanilla_lenet_exp2.csv')[['epoch','train_loss']]
std = (data_dlrt.copy()-mean)**2
std_vanilla = (data_vanilla.copy()-mean_vanilla)**2
for cv in range(1,10):
    data = pd.read_csv('_running_data_dlrt'+str(cv)+'dlrt_lenet_exp2.csv')[['epoch','train_loss']]
    data_dlrt = pd.concat([data_dlrt,data])
    data_UV = pd.read_csv('_running_data_vanilla'+str(cv)+'vanilla_lenet_exp2.csv')[['epoch','train_loss']]
    std +=(data-mean)**2
    std_vanilla +=(data_UV-mean_vanilla)**2
std = (np.sqrt(std/10))[['train_loss']]
std_vanilla = (np.sqrt(std_vanilla/10))[['train_loss']]


upper_bound_dlrt = [(mean.loc[i]+std.loc[i])['train_loss'] for i in range(len(std))]
lower_bound_dlrt = [(mean.loc[i]-std.loc[i])['train_loss'] for i in range(len(std))]

upper_bound_vanilla = [(mean_vanilla.loc[i]+std_vanilla.loc[i])['train_loss'] for i in range(len(std_vanilla))]
lower_bound_vanilla = [(mean_vanilla.loc[i]-std_vanilla.loc[i])['train_loss'] for i in range(len(std_vanilla))]


plt.clf()
sns.set_theme()
sns.set_style("white")
colors = ['k', 'r', 'g', 'b']
symbol_size = 0.7
markersize = 2.5
markerwidth = 0.5
plt.plot(mean[['train_loss']][0:11], '-ok')
plt.plot(mean_vanilla[['train_loss']][0:11], '-or')
plt.fill_between(range(11),upper_bound_dlrt[0:11],lower_bound_dlrt[0:11], color='black', alpha=.2)
plt.fill_between(range(11),upper_bound_vanilla[0:11],lower_bound_vanilla[0:11], color='red', alpha=.2)
# plt.plot(dlra_3layer[["acc_test"]], '--g')
plt.xticks(range(1,11,1),range(1,11,1))
plt.legend(["DLRT", r"$UV^T$ factorization"])
plt.ylim([0, 2.5])
plt.ylabel("train loss")
plt.xlabel("epoch")

ax = plt.gca()  # you first need to get the axis handle
x_left, x_right = ax.get_xlim()
y_low, y_high = ax.get_ylim()
ratio = 0.5
plt.savefig("crossval_loss_dlrt_vs_UV_exp2.pdf", dpi=500,format = 'pdf')
plt.clf()


# %%
