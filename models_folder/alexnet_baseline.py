import torch
from optimizer_KLS.my_conv import Conv2d_lr
from optimizer_KLS.Linear_layer_lr_new import Linear

class Flatten(torch.nn.Module):
    def forward(self, input):
        '''
        Note that input.size(0) is usually the batch size.
        So what it does is that given any input with input.size(0) # of batches,
        will flatten to be 1 * nb_elements.
        '''
        batch_size = input.size(0)
        out = input.view(batch_size,-1)
        return out # (batch_size, *size)

class AlexNet(torch.nn.Module):
    def __init__(self, output_dim,device = 'cpu'):
        super().__init__()
        self.device = device
        self.layer = torch.nn.Sequential(
            Conv2d_lr(in_channels = 3,out_channels = 64,kernel_size= 3,stride =  2, padding = 1,device = self.device,rank = None),  # in_channels, out_channels, kernel_size, stride, padding
            torch.nn.BatchNorm2d(64),
            torch.nn.MaxPool2d(2),  # kernel_size
            torch.nn.ReLU(),
            Conv2d_lr(64, 192, 3, padding=1,device = self.device,rank = None),
            torch.nn.BatchNorm2d(192),
            torch.nn.MaxPool2d(2),
            torch.nn.ReLU(),
            Conv2d_lr(192, 384, 3, padding=1,device = self.device),
            torch.nn.BatchNorm2d(384),
            torch.nn.ReLU(),
            Conv2d_lr(384, 256, 3, padding=1,device = self.device,rank = None),
            torch.nn.BatchNorm2d(256),
            torch.nn.ReLU(),
            Conv2d_lr(256, 256, 3, padding=1,device = self.device,rank = None),
            torch.nn.BatchNorm2d(256),
            torch.nn.MaxPool2d(2),
            torch.nn.ReLU(),
            Flatten(),
            torch.nn.Dropout(0.2),
            Linear(256 * 2 * 2, 4096,rank = None,device = self.device),
            torch.nn.ReLU(),
            torch.nn.Dropout(0.2),
            Linear(4096, 4096,rank = None,device = self.device),
            torch.nn.ReLU(),
            Linear(4096, output_dim,device = self.device),
        )

    def forward(self, x):
        
        for l in self.layer:
            x = l(x)
        return x

    def update_step(self,new_step = 'K'):
        for l in self.layer:
            if hasattr(l,'lr') and l.lr:
                l.step = new_step

    def populate_gradients(self,x,y,criterion,step = 'all'):

        if step == 'all':
        
            self.update_step(new_step = 'K')
            output = self.forward(x)
            loss = criterion(output,y)
            loss.backward()
            self.update_step(new_step = 'L')
            output = self.forward(x)
            loss = criterion(output,y)
            loss.backward()
            return loss,output.detach()

        else:
            
            self.update_step(new_step = step)
            loss = criterion(self.forward(x),y)
            return loss
