#%%
from torchviz import make_dot
import torch
import sys,os
sys.path.insert(1, os.path.join(sys.path[0], '..'))

from Lenet5 import Lenet5

NN = Lenet5()

x = torch.randn((1,1,28,28))
# y_hat = NN(x)
# NN.update_step('L')
# y_hat= NN(x)
NN.update_step('S')
y_hat= NN(x)

make_dot(y_hat, params=dict(list(NN.named_parameters())),show_attrs=False).render("./computational_graphs/rnn_torchviz_S", format="pdf")

