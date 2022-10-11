#%%
from distutils.log import warn
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
path = './results/'
warnings.filterwarnings('ignore')

cv = 5
step = 0.05
grid = np.arange(0,1,step)

timings = pd.read_csv(path+'timings_cr.csv')
mean_vanilla = timings[timings['cr_layerwise'] == 'baseline'][['time forward train','time backward train','time forward test']].mean()
std_vanilla = timings[timings['cr_layerwise'] == 'baseline'][['time forward train','time backward train','time forward test']].std()
mean = pd.DataFrame(data=None, columns=['cr_layerwise','time forward train','time backward train','time forward test'])
stds = pd.DataFrame(data=None, columns=['cr_layerwise','time forward train','time backward train','time forward test'])
for i,cr in enumerate(grid):
    boolean_index = abs(timings[timings['cr_layerwise']!='baseline']['cr_layerwise'].astype(float) - cr)<0.01
    filtered_timings = timings[timings['cr_layerwise']!='baseline'].loc[boolean_index][['cr_layerwise','time forward train','time backward train','time forward test']]
    # if i == 3:
    #     print(filtered_timings)
    mean.loc[i] = filtered_timings.mean()
    stds.loc[i] = filtered_timings.std()

# print(mean)

upper_bound_dlrt_forw_train =[]
lower_bound_dlrt_forw_train =[]
upper_bound_dlrt_back_train =[]
lower_bound_dlrt_back_train =[]
upper_bound_dlrt_forw_test = []
lower_bound_dlrt_forw_test = []

#print(mean)

for i in range(len(mean)):

    upper_bound_dlrt_forw_train.append( [(mean.loc[i]+stds.loc[i])['time forward train'] for i in range(len(stds)) ])
    lower_bound_dlrt_forw_train.append( [(mean.loc[i]-stds.loc[i])['time forward train'] for i in range(len(stds)) ])

    upper_bound_dlrt_back_train.append( [(mean.loc[i]+stds.loc[i])['time backward train'] for i in range(len(stds))])
    lower_bound_dlrt_back_train.append( [(mean.loc[i]-stds.loc[i])['time backward train'] for i in range(len(stds))])

    upper_bound_dlrt_forw_test.append( [(mean.loc[i]+stds.loc[i])['time forward test'] for i in range(len(stds))  ])
    lower_bound_dlrt_forw_test.append( [(mean.loc[i]-stds.loc[i])['time forward test'] for i in range(len(stds))  ])


upper_bound_vanilla_forw_train = [(mean_vanilla+std_vanilla)['time forward train'] for i in range(len(grid))]
lower_bound_vanilla_forw_train = [(mean_vanilla-std_vanilla)['time forward train'] for i in range(len(grid))]


upper_bound_vanilla_back_train = [(mean_vanilla+std_vanilla)['time backward train'] for i in range(len(grid))]
lower_bound_vanilla_back_train = [(mean_vanilla-std_vanilla)['time backward train'] for i in range(len(grid))]

upper_bound_vanilla_forw_test = [(mean_vanilla+std_vanilla)['time forward test'] for i in range(len(grid))]
lower_bound_vanilla_forw_test = [(mean_vanilla-std_vanilla)['time forward test'] for i in range(len(grid))]


plt.clf()
sns.set_theme()
sns.set_style("white")
colors = ['k', 'r', 'g', 'b']
symbol_size = 0.7
markersize = 2.5
markerwidth = 0.5
#print(mean['time forward train'])
plt.plot(grid,mean['time forward train'], '-ok',color = colors[0])
plt.fill_between(grid,upper_bound_dlrt_forw_train[i],lower_bound_dlrt_forw_train[i],color = colors[0], alpha=.2,label = '_nolegend_')

plt.plot(grid,mean['time backward train'], '-ok',color = colors[1])
plt.fill_between(grid,upper_bound_dlrt_back_train[i],lower_bound_dlrt_back_train[i],color = colors[1], alpha=.2,label = '_nolegend_')

plt.plot(grid,mean['time forward test'], '-ok',color = colors[2])
plt.fill_between(grid,upper_bound_dlrt_forw_test[i],lower_bound_dlrt_forw_test[i],color = colors[2], alpha=.2,label = '_nolegend_')

# baselines

plt.plot(grid,[mean_vanilla['time backward train']]*len(grid), '--',color = colors[1])
plt.fill_between(grid,upper_bound_vanilla_back_train[i],lower_bound_vanilla_back_train[i],color = colors[1], alpha=.2,label = '_nolegend_')

plt.plot(grid,[mean_vanilla['time forward test']]*len(grid), '--',color = colors[2])
plt.fill_between(grid,upper_bound_vanilla_forw_test[i],lower_bound_vanilla_forw_test[i],color = colors[2], alpha=.2,label = '_nolegend_')

plt.xticks(np.arange(0,1,0.1),np.round(np.arange(0,1,0.1),2))
plt.legend(["forward train", 'backward train','forward test', 'baseline backward train','baseline forward'])
plt.ylim([0, 12])
plt.ylabel("timing [s]")
plt.xlabel("layerwise compression ratio")




ax = plt.gca()  # you first need to get the axis handle
x_left, x_right = ax.get_xlim()
y_low, y_high = ax.get_ylim()
ratio = 0.5
plt.savefig("timings_plot_cr.pdf", dpi=500,format = 'pdf')
plt.clf()
plt.show()


# crlayerwise vs space


mean_vanilla = timings[timings['cr_layerwise'] == 'baseline'][['cr_test (%)','cr_train (%)','cr_train_grads (%)']].mean()
std_vanilla = timings[timings['cr_layerwise'] == 'baseline'][['cr_test (%)','cr_train (%)','cr_train_grads (%)']].std()
mean = pd.DataFrame(data=None, columns=['cr_layerwise','cr_test (%)','cr_train (%)','cr_train_grads (%)'])
stds = pd.DataFrame(data=None, columns=['cr_layerwise','cr_test (%)','cr_train (%)','cr_train_grads (%)'])
for i,cr in enumerate(grid):
    boolean_index = abs(timings[timings['cr_layerwise']!='baseline']['cr_layerwise'].astype(float) - cr)<0.01
    filtered_timings = timings[timings['cr_layerwise']!='baseline'].loc[boolean_index][['cr_layerwise','cr_test (%)','cr_train (%)','cr_train_grads (%)']]
    # if i == 3:
    #     print(filtered_timings)
    mean.loc[i] = filtered_timings.mean()
    stds.loc[i] = filtered_timings.std()

# print(mean)

upper_bound_dlrt_forw_train =[]
lower_bound_dlrt_forw_train =[]
upper_bound_dlrt_back_train =[]
lower_bound_dlrt_back_train =[]
upper_bound_dlrt_forw_test = []
lower_bound_dlrt_forw_test = []

#print(mean)

for i in range(len(mean)):

    upper_bound_dlrt_forw_train.append( [(mean.loc[i]+stds.loc[i])['cr_test (%)'] for i in range(len(stds)) ])
    lower_bound_dlrt_forw_train.append( [(mean.loc[i]-stds.loc[i])['cr_test (%)'] for i in range(len(stds)) ])

    upper_bound_dlrt_back_train.append( [(mean.loc[i]+stds.loc[i])['cr_train (%)'] for i in range(len(stds))])
    lower_bound_dlrt_back_train.append( [(mean.loc[i]-stds.loc[i])['cr_train (%)'] for i in range(len(stds))])

    upper_bound_dlrt_forw_test.append( [(mean.loc[i]+stds.loc[i])['cr_train_grads (%)'] for i in range(len(stds))  ])
    lower_bound_dlrt_forw_test.append( [(mean.loc[i]-stds.loc[i])['cr_train_grads (%)'] for i in range(len(stds))  ])


upper_bound_vanilla_forw_train = [(mean_vanilla+std_vanilla)['cr_test (%)'] for i in range(len(grid))]
lower_bound_vanilla_forw_train = [(mean_vanilla-std_vanilla)['cr_test (%)'] for i in range(len(grid))]


upper_bound_vanilla_back_train = [(mean_vanilla+std_vanilla)['cr_train (%)'] for i in range(len(grid))]
lower_bound_vanilla_back_train = [(mean_vanilla-std_vanilla)['cr_train (%)'] for i in range(len(grid))]

upper_bound_vanilla_forw_test = [(mean_vanilla+std_vanilla)['cr_train_grads (%)'] for i in range(len(grid))]
lower_bound_vanilla_forw_test = [(mean_vanilla-std_vanilla)['cr_train_grads (%)'] for i in range(len(grid))]


plt.clf()
sns.set_theme()
sns.set_style("white")
colors = ['k', 'r', 'g', 'b']
symbol_size = 0.7
markersize = 2.5
markerwidth = 0.5
#print(mean['time forward train'])
plt.plot(grid,mean['cr_test (%)'], '-ok',color = colors[2])
plt.fill_between(grid,upper_bound_dlrt_forw_train[i],lower_bound_dlrt_forw_train[i],color = colors[0], alpha=.2,label = '_nolegend_')

plt.plot(grid,mean['cr_train (%)'], '-ok',color = colors[0])
plt.fill_between(grid,upper_bound_dlrt_back_train[i],lower_bound_dlrt_back_train[i],color = colors[1], alpha=.2,label = '_nolegend_')

plt.plot(grid,mean['cr_train_grads (%)'], '-ok',color = colors[1])
plt.fill_between(grid,upper_bound_dlrt_forw_test[i],lower_bound_dlrt_forw_test[i],color = colors[2], alpha=.2,label = '_nolegend_')

# baselines

plt.plot(grid,[mean_vanilla['cr_test (%)']]*len(grid), '--',color = colors[0])
plt.fill_between(grid,upper_bound_vanilla_forw_train[i],lower_bound_vanilla_forw_train[i],color = colors[1], alpha=.2,label = '_nolegend_')


plt.xticks(np.arange(0,1,0.1),np.round(np.arange(0,1,0.1),2))
plt.legend(["cr test", 'cr train','cr train grads', 'baseline full-rank'])
plt.ylim([-200, 100])
plt.ylabel("overall cr [%]")
plt.xlabel("layerwise compression ratio")




ax = plt.gca()  # you first need to get the axis handle
x_left, x_right = ax.get_xlim()
y_low, y_high = ax.get_ylim()
ratio = 0.5
plt.savefig("cr_ratio_plot_crexp.pdf", dpi=500,format = 'pdf')
plt.clf()
plt.show()