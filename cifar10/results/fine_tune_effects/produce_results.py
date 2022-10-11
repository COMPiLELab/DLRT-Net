#%%
import pandas as pd

final_results = pd.DataFrame(data = None,columns = ['name','test_acc','test_acc_ft','delta','cr_test'
                            ])
results  = pd.DataFrame(data = None,columns = ['test_acc','test_acc_ft','delta','cr_test'
                            ])

for cv in range(1,6):
    data = pd.read_csv('_running_data_vgg_cifar10_cv'+str(cv)+'_0.1_ft.csv')
    results.loc[cv] = [data['test_accuracy(%)'].loc[19],data['test_accuracy(%)'].loc[39],
                        data['test_accuracy(%)'].loc[39]-data['test_accuracy(%)'].loc[19],data['cr_test (%)'].loc[39]]

final_results.loc[1] = ['vgg_0.1',str(results['test_acc'].mean())+'+-'+str(results['test_acc'].std()),
                    str(results['test_acc_ft'].mean())+'+-'+str(results['test_acc_ft'].std()),
                    str(results['delta'].mean())+'+-'+str(results['delta'].std()),
                    str(results['cr_test'].mean())+'+-'+str(results['cr_test'].std()),
                    ]

results  = pd.DataFrame(data = None,columns = ['test_acc','test_acc_ft','delta','cr_test'
                            ])


for cv in range(2,7):#range(0,2):
    data = pd.read_csv('_running_data_alexnet_cifar10_cv'+str(cv)+'_0.1_ft.csv')
    results.loc[cv] = [data['test_accuracy(%)'].loc[19],data['test_accuracy(%)'].loc[39],
                        data['test_accuracy(%)'].loc[39]-data['test_accuracy(%)'].loc[19],data['cr_test (%)'].loc[39]]

final_results.loc[2] = ['alexnet_0.1',str(results['test_acc'].mean())+'+-'+str(results['test_acc'].std()),
                    str(results['test_acc_ft'].mean())+'+-'+str(results['test_acc_ft'].std()),
                    str(results['delta'].mean())+'+-'+str(results['delta'].std()),
                    str(results['cr_test'].mean())+'+-'+str(results['cr_test'].std()),
                    ]

results  = pd.DataFrame(data = None,columns = ['test_acc','test_acc_ft','delta','cr_test'
                            ])


for cv in range(0,5):
    data = pd.read_csv('_running_data_lenet_cifar10_cv'+str(cv)+'_0.1_ft.csv')
    results.loc[cv] = [data['test_accuracy(%)'].loc[19],data['test_accuracy(%)'].loc[39],
                        data['test_accuracy(%)'].loc[39]-data['test_accuracy(%)'].loc[19],data['cr_test (%)'].loc[39]]

final_results.loc[3] = ['lenet_0.1',str(results['test_acc'].mean())+'+-'+str(results['test_acc'].std()),
                    str(results['test_acc_ft'].mean())+'+-'+str(results['test_acc_ft'].std()),
                    str(results['delta'].mean())+'+-'+str(results['delta'].std()),
                    str(results['cr_test'].mean())+'+-'+str(results['cr_test'].std()),
                    ]


results  = pd.DataFrame(data = None,columns = ['test_acc','test_acc_ft','delta','cr_test'
                            ])


for cv in range(0,5):
    data = pd.read_csv('_running_data_resnet20_cifar10_cv'+str(cv)+'_0.1_ft.csv')
    results.loc[cv] = [data['test_accuracy(%)'].loc[19],data['test_accuracy(%)'].loc[39],
                        data['test_accuracy(%)'].loc[39]-data['test_accuracy(%)'].loc[19],data['cr_test (%)'].loc[39]]

final_results.loc[4] = ['resnet20_0.1',str(results['test_acc'].mean())+'+-'+str(results['test_acc'].std()),
                    str(results['test_acc_ft'].mean())+'+-'+str(results['test_acc_ft'].std()),
                    str(results['delta'].mean())+'+-'+str(results['delta'].std()),
                    str(results['cr_test'].mean())+'+-'+str(results['cr_test'].std()),
                    ]

final_results.to_csv('fine_tune_effects.csv')

# %%
