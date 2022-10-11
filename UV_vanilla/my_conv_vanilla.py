#%%
# imports 
import math
import warnings
import numpy as np
import torch
import torch.nn.functional as F
from torch.nn import init
import warnings
warnings.filterwarnings("ignore", category=Warning)

# low rank convolution class 

class Conv2d_lr(torch.nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, dilation=1, padding=0, stride=1,bias = True,step = 'S',rank = None,
    fixed = False,dtype = None,device = None,load_weights = None,full_rank_construct = False,decay = 1)->None:

        """  
        Initializer for the convolutional low rank layer (filterwise), extention of the classical Pytorch's convolutional layer.
        INPUTS:
        in_channels: number of input channels (Pytorch's standard)
        out_channels: number of output channels (Pytorch's standard)
        kernel_size : kernel_size for the convolutional filter (Pytorch's standard)
        dilation : dilation of the convolution (Pytorch's standard)
        padding : padding of the convolution (Pytorch's standard)
        stride : stride of the filter (Pytorch's standard)
        bias  : flag variable for the bias to be included (Pytorch's standard)
        step : string variable ('K','L' or 'S') for which forward phase to use
        rank : rank variable, None if the layer has to be treated as a classical Pytorch Linear layer (with weight and bias). If
                it is an int then it's either the starting rank for adaptive or the fixed rank for the layer.
        fixed : flag variable, True if the rank has to be fixed (KLS training on this layer)
        load_weights : variables to load (Pytorch standard, to finish)
        dtype : Type of the tensors (Pytorch standard, to finish)
        """
            
        factory_kwargs = {'device': device, 'dtype': dtype}
        super(Conv2d_lr, self).__init__()

        self.kernel_size = [kernel_size, kernel_size] if isinstance(kernel_size,int) else kernel_size
        self.kernel_size_number = kernel_size * kernel_size
        self.out_channels = out_channels
        self.dilation = dilation if type(dilation)==tuple else (dilation, dilation)
        self.padding = padding if type(padding) == tuple else(padding, padding)
        self.stride = (stride if type(stride)==tuple else (stride, stride))
        self.in_channels = in_channels
        self.rank = rank
        self.device = device
        self.dtype = dtype
        self.fixed = fixed
        self.decay = decay
        self.load_weights = load_weights
        self.weight = torch.nn.Parameter(torch.empty(tuple([self.out_channels, self.in_channels] +self.kernel_size),**factory_kwargs),requires_grad = True)
        self.lr = True if self.rank!=None else False
        self.rmax = int(min([self.out_channels, self.in_channels*self.kernel_size_number]) / 2)
        self.full_rank_construct = full_rank_construct
        if not self.fixed:
            self.rank = None if rank == None else min([rank,self.rmax])
        else:
            self.rank = min([rank,self.out_channels,self.in_channels*self.kernel_size_number])
        self.dynamic_rank = self.rank
        self.step = step

        if bias:
            self.bias = torch.nn.Parameter(torch.empty(self.out_channels,**factory_kwargs))
        else:
            self.bias = torch.nn.Parameter(torch.zeros(self.out_channels,**factory_kwargs))
    
        # Weights and Bias initialization
        if self.load_weights == None:
            self.reset_parameters()
        else:
            param,b = self.load_weights
            self.bias = torch.nn.Parameter(b)
            self.weight = torch.nn.Parameter(param,requires_grad = True)

        if self.lr and not self.full_rank_construct:

            if not self.fixed:

                n,m = self.out_channels,self.in_channels*self.kernel_size_number

                _,s_ordered,_ = torch.linalg.svd(torch.diag(torch.abs(torch.randn(2*self.rmax))))
                U = torch.randn(n,self.rmax)
                V = torch.randn(m,self.rmax)
                U,_,_ = torch.linalg.svd(U)
                V,_,_ = torch.linalg.svd(V)
                self.U = torch.nn.Parameter(U.to(device) ,requires_grad=False)             
                self.S_hat = torch.nn.Parameter(torch.diag(s_ordered).to(device))                                       
                self.V = torch.nn.Parameter(V.to(device),requires_grad=False)
                self.U_hat = torch.nn.Parameter( torch.randn(n,2*self.rmax).to(device) ,requires_grad = False)
                self.V_hat = torch.nn.Parameter(torch.randn(m,2*self.rmax).to(device) ,requires_grad = False)
                self.K = torch.nn.Parameter(torch.randn(n,self.rmax).to(device))
                self.L = torch.nn.Parameter(torch.randn(m,self.rmax).to(device))
                self.N_hat = torch.nn.Parameter(torch.randn(2*self.rmax,self.rmax).to(device) ,requires_grad = False)
                self.M_hat = torch.nn.Parameter( torch.randn(2*self.rmax,self.rmax).to(device) ,requires_grad = False)
                self.S = torch.nn.Parameter(torch.randn(self.rmax,self.rmax).to(device),requires_grad = False)
                self.weight = None
                self.id = id(self.K)

            else:

                n,m = self.out_channels,self.in_channels*self.kernel_size_number

                _,s_ordered,_ = torch.linalg.svd(torch.diag(torch.abs(torch.randn(self.rank))))
                U = torch.randn(n,self.rank)
                V = torch.randn(m,self.rank)
                U,_,_ = torch.linalg.svd(U)
                V,_,_ = torch.linalg.svd(V)
                U = U.to(device)
                V = V.to(device)
                self.S_hat = torch.nn.Parameter(torch.diag(torch.sqrt(s_ordered)).to(device))
                exp_decay = torch.tensor([1/(self.decay)**k for k in range(len(s_ordered))])  
                s_ordered = s_ordered*exp_decay
                self.U = torch.nn.Parameter((U[:,:self.rank]@self.S_hat).to(device) ,requires_grad=False)                                                
                self.V = torch.nn.Parameter((V[:,:self.rank]@self.S_hat).to(device),requires_grad=False)
                self.K = torch.nn.Parameter(torch.randn(n,self.rank).to(device))
                self.L = torch.nn.Parameter(torch.randn(m,self.rank).to(device))
                self.N_hat = torch.nn.Parameter(torch.randn(self.rank,self.rank).to(device) ,requires_grad = False)
                self.M_hat = torch.nn.Parameter( torch.randn(self.rank,self.rank).to(device) ,requires_grad = False)
                self.weight = None

    def switch_lowrank(self):
        
        w,b = self.weight,self.bias
        device = self.device
        if not self.fixed:
            self.rank = None if self.rank == None else min([self.rank,self.rmax])
        else:
            self.rank = min([self.rank,self.out_channels,self.in_channels*self.kernel_size_number])
        self.dynamic_rank = self.rank

        self.bias = b

        if self.lr:

            if not self.fixed:

                n,m = self.out_channels,self.in_channels*self.kernel_size_number

                U_load,S_load,V_load = torch.linalg.svd(w.view(n,m))
                V_load = V_load.T
                r = len(S_load)
                _,s_ordered,_ = torch.linalg.svd(torch.diag(torch.abs(torch.randn(2*self.rmax-r))))
                s_ordered = torch.tensor(torch.concat([S_load,s_ordered.to(device)])).to(device)
                self.U = torch.nn.Parameter(U_load.to(device) ,requires_grad=False)             
                self.S_hat = torch.nn.Parameter(torch.diag(s_ordered).to(device))                                       
                self.V = torch.nn.Parameter(V_load.to(device),requires_grad=False)
                self.U_hat = torch.nn.Parameter( torch.randn(n,2*self.rmax).to(device) ,requires_grad = False)
                self.V_hat = torch.nn.Parameter(torch.randn(m,2*self.rmax).to(device) ,requires_grad = False)
                self.K = torch.nn.Parameter(torch.randn(n,self.rmax).to(device))
                self.L = torch.nn.Parameter(torch.randn(m,self.rmax).to(device))
                self.N_hat = torch.nn.Parameter(torch.randn(r,r).to(device) ,requires_grad = False)
                self.M_hat = torch.nn.Parameter( torch.randn(r,r).to(device) ,requires_grad = False)
                self.weight = None
                self.id = id(self.K)
            else:
                n,m = self.out_channels,self.in_channels*self.kernel_size_number

                U_load,S_load,V_load = torch.linalg.svd(w.view(n,m))
                V_load = V_load.T
                r = self.rank
                self.U = torch.nn.Parameter(U_load[:,:r].to(device) ,requires_grad=True)             
                self.S_hat = torch.nn.Parameter(torch.diag(S_load[:r]).to(device),requires_grad = False)                                       
                self.V = torch.nn.Parameter(V_load[:,:r].to(device),requires_grad=True)
                self.K = torch.nn.Parameter(torch.randn(n,r).to(device),requires_grad = False)
                self.L = torch.nn.Parameter(torch.randn(m,r).to(device),requires_grad = False)
                self.N_hat = torch.nn.Parameter(torch.randn(2*self.rmax,self.rmax).to(device) ,requires_grad = False)
                self.M_hat = torch.nn.Parameter( torch.randn(2*self.rmax,self.rmax).to(device) ,requires_grad = False)
                self.weight = None
                self.id = id(self.K)



    def reset_parameters(self) -> None:
        # Setting a=sqrt(5) in kaiming_uniform is the same as initializing with
        # uniform(-1/sqrt(k), 1/sqrt(k)), where k = weight.size(1) * prod(*kernel_size)
        # For more details see: https://github.com/pytorch/pytorch/issues/15314#issuecomment-477448573
        init.kaiming_uniform_(self.weight, a=math.sqrt(5))
         # for testing
        # self.original_weight = Parameter(self.weight.reshape(self.original_shape))
        if self.bias is not None:
            fan_in, _ = init._calculate_fan_in_and_fan_out(self.weight)
            if fan_in != 0:
                bound = 1 / math.sqrt(fan_in)
                init.uniform_(self.bias, -bound, bound)  


    def forward(self, input):

        """  
        forward phase for the convolutional layer. It has to contain the three different
        phases for the steps 'K','L' and 'S' in order to be optimizable using dlrt.

        """
        
        batch_size,_,_,_ = input.shape

        if not self.lr:

            return F.conv2d(input = input,weight = self.weight,bias = self.bias,stride = self.stride,
                padding = self.padding,dilation = self.dilation
            )

        else:

            if self.step == 'K':

                if not self.fixed:

                    K,V = self.K[:,:self.dynamic_rank],self.V[:,:self.dynamic_rank]

                else:

                    K,V = self.K,self.V

                inp_unf = F.unfold(input,self.kernel_size,padding = self.padding,stride = self.stride).to(self.device)
    
                if self.bias is None:
                    out_unf = (inp_unf.transpose(1, 2).matmul(V) )
                    out_unf = (out_unf.matmul(K.t()) + self.bias).transpose(1, 2)
                else:
                    out_h = int(np.floor(((input.shape[2]+2*self.padding[0]-self.dilation[0]*(self.kernel_size[0]-1)-1)/self.stride[0])+1))
                    out_w = int(np.floor(((input.shape[3]+2*self.padding[1]-self.dilation[1]*(self.kernel_size[1]-1)-1)/self.stride[1])+1))

                    out_unf = (inp_unf.transpose(1, 2).matmul(V) )
                    out_unf = (out_unf.matmul(K.t()) + self.bias).transpose(1, 2)
    
                return out_unf.view(batch_size, self.out_channels, out_h, out_w)

            elif self.step =='L':

                if not self.fixed:

                    U,L = self.U[:,:self.dynamic_rank],self.L[:,:self.dynamic_rank]
                
                else:

                    U,L = self.U,self.L

                inp_unf = F.unfold(input,self.kernel_size,padding = self.padding,stride = self.stride).to(self.device)
    
                if self.bias is None:
                    out_unf = (inp_unf.transpose(1, 2).matmul(L) )
                    out_unf = (out_unf.matmul(U.t()) + self.bias).transpose(1, 2)
                else:
                    out_h = int(np.floor(((input.shape[2]+2*self.padding[0]-self.dilation[0]*(self.kernel_size[0]-1)-1)/self.stride[0])+1))
                    out_w = int(np.floor(((input.shape[3]+2*self.padding[1]-self.dilation[1]*(self.kernel_size[1]-1)-1)/self.stride[1])+1))

                    out_unf = (inp_unf.transpose(1, 2).matmul(L) )
                    out_unf = (out_unf.matmul(U.t()) + self.bias).transpose(1, 2)
    
                return out_unf.view(batch_size, self.out_channels, out_h, out_w)
            
            elif self.step == 'S':

                if not self.fixed:

                    U_hat,S_hat,V_hat = self.U_hat[:,:2*self.dynamic_rank],self.S_hat[:2*self.dynamic_rank,:2*self.dynamic_rank],self.V_hat[:,:2*self.dynamic_rank]

                else:

                    U_hat,S_hat,V_hat = self.U,self.S_hat,self.V
    
                inp_unf = F.unfold(input,self.kernel_size,padding = self.padding,stride = self.stride).to(self.device)

                if self.bias is None:
                    out_unf = (inp_unf.transpose(1, 2).matmul(V_hat) )
                    #out_unf = (out_unf.matmul(S_hat.t()))
                    out_unf = (out_unf.matmul(U_hat.t()) + self.bias).transpose(1, 2)
                else:
                    out_h = int(np.floor(((input.shape[2]+2*self.padding[0]-self.dilation[0]*(self.kernel_size[0]-1)-1)/self.stride[0])+1))
                    out_w = int(np.floor(((input.shape[3]+2*self.padding[1]-self.dilation[1]*(self.kernel_size[1]-1)-1)/self.stride[1])+1))

                    out_unf = (inp_unf.transpose(1, 2).matmul(V_hat) )
                    #out_unf = (out_unf.matmul(S_hat.t()))
                    out_unf = (out_unf.matmul(U_hat.t()) + self.bias).transpose(1, 2)
    
                return out_unf.view(batch_size, self.out_channels, out_h, out_w)

            else:

                raise ValueError(f'incorrect step value {self.step}')
