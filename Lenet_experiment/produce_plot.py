#%%
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

cross_val_results = pd.read_csv('cross_val_tab_paper.csv')

cross_val_dlrt = cross_val_results.loc[cross_val_results['theta']!='Baseline']
cross_val_baseline = cross_val_results.loc[cross_val_results['theta']=='Baseline']

grid = [0.09,0.11,0.13,0.15,0.2,0.3,0.4,0.45]

plt.clf()
sns.set_theme()
sns.set_style("white")
colors = ['k', 'r', 'g', 'b']
symbol_size = 0.7
markersize = 2.5
markerwidth = 0.5
lower_bound = cross_val_dlrt['test accuracy mean'] - cross_val_dlrt['test accuracy std']
upper_bound = cross_val_dlrt['test accuracy mean'] + cross_val_dlrt['test accuracy std']
plt.plot(cross_val_dlrt['theta'],cross_val_dlrt['test accuracy mean'], '-ok',color = colors[0])
plt.fill_between(range(len(grid)),upper_bound,lower_bound,color = colors[0], alpha=.2,label = '_nolegend_')

lower_bound_baseline = cross_val_baseline['test accuracy mean'] - cross_val_baseline['test accuracy std']
upper_bound_baseline = cross_val_baseline['test accuracy mean'] + cross_val_baseline['test accuracy std']
plt.plot(cross_val_dlrt['theta'],[cross_val_baseline['test accuracy mean']]*len(grid), '--',color = colors[1])
plt.fill_between(range(len(grid)),upper_bound_baseline,lower_bound_baseline,color = colors[1], alpha=.2,label = '_nolegend_')


plt.plot(cross_val_dlrt['theta'],cross_val_dlrt['cr test'], '-.',color = colors[2])
plt.plot(cross_val_dlrt['theta'],cross_val_dlrt['cr train'], '-.',color = colors[3])


plt.xticks(range(len(grid)),grid)
plt.legend(['test accuracy dlrt (%)','test accuracy baseline (%)','compression ratio test(%)','compression ratio train(%)'])
plt.ylim([87, 100])
plt.ylabel("[%]")
plt.xlabel(r"relative threshold $\tau$")




ax = plt.gca()  # you first need to get the axis handle
x_left, x_right = ax.get_xlim()
y_low, y_high = ax.get_ylim()
ratio = 0.5
plt.savefig("Lenet5_mnist.pdf", dpi=500,format = 'pdf')
plt.clf()
plt.show()