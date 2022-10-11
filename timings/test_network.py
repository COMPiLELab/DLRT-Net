#%%
import sys,os
sys.path.insert(1, os.path.join(sys.path[0], '..'))
from optimizer_KLS.my_conv import Conv2d_lr
from optimizer_KLS.Linear_layer_lr_new import Linear
import torch

class net(torch.nn.Module):
    def __init__(self,device = 'cpu',ranks = [500,300,100],fixed = True):
     
        super(net, self).__init__()
        self.device = device
        self.ranks = ranks
        self.fixed = fixed
        self.layer = torch.nn.Sequential(
            torch.nn.Flatten(),
            Linear(28*28,500,rank = self.ranks[0],device = self.device,fixed = self.fixed),  
            torch.nn.ReLU(),
            Linear(500,300,rank = self.ranks[1],device = self.device,fixed = self.fixed),  
            torch.nn.ReLU(),
            Linear(300,out_features = 100,rank = self.ranks[2],device = self.device,fixed = self.fixed),  
            torch.nn.ReLU(),
            Linear(100,out_features = 10,device = self.device)
        )

    def forward(self, x):
        for layer in self.layer:
            x = layer(x)
        return x

    def update_step(self,new_step = 'K'):
        for l in self.layer:
            if hasattr(l,'lr') and l.lr:
                l.step = new_step

    def populate_gradients(self,x,y,criterion,step = 'all'):

        if step == 'all':
        
            self.update_step(new_step = 'K')
            loss = criterion(self.forward(x),y)
            loss.backward()
            self.update_step(new_step = 'L')
            loss = criterion(self.forward(x),y)
            loss.backward()

        else:
            
            self.update_step(new_step = step)
            loss = criterion(self.forward(x),y)

        return loss

    def forward_train(self,x):

        self.update_step(new_step='K')
        self.forward(x)
        self.update_step(new_step='L')
        self.forward(x)
        self.update_step(new_step='S')
        self.forward(x)


    def backward_train_not_eff(self,x,y,criterion):

            self.update_step(new_step = 'K')
            loss = criterion(self.forward(x),y)
            loss.backward()
            self.update_step(new_step = 'L')
            loss = criterion(self.forward(x),y)
            loss.backward()
            self.update_step(new_step = 'S')
            loss = criterion(self.forward(x),y)
            loss.backward()
            return loss

    
    def backward_train(self,x,y,criterion):

            self.update_step(new_step = 'K')
            loss = criterion(self.forward(x),y)
            total_loss = loss
            #loss.backward()
            self.update_step(new_step = 'L')
            loss = criterion(self.forward(x),y)
            #loss.backward()
            total_loss+=loss
            self.update_step(new_step = 'S')
            loss = criterion(self.forward(x),y)
            total_loss+=loss
            total_loss.backward()
            return total_loss


# check new backprop approach
# import numpy as np  
        
# x = torch.randn((10,28,28))
# y = torch.tensor(np.random.choice(range(10),10))
# criterion = torch.nn.CrossEntropyLoss()

# f = net(ranks = [100,100,50])
# state_d = f.state_dict()
# g = net(ranks = [100,100,50])
# g.load_state_dict(state_d)

# f.backward_train(x,y,criterion)
# print([(n,p.grad is not None) for n,p in f.named_parameters()])
# g.backward_train_not_eff(x,y,criterion)

# print([torch.allclose(p[0].grad , p[1].grad) for p in zip(f.parameters(),g.parameters()) if p[0].grad is not None])