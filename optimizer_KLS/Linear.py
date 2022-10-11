#%%
# redefinition of the linear layer, adding decomposition attributes
# to the weight object

import math       
import torch
from torch import Tensor
import torch.nn.init as init
import torch.nn.functional as F


class Linear(torch.nn.Module):
    

    __constants__ = ['in_features', 'out_features']
    in_features: int
    out_features: int
    weight: Tensor

    def __init__(self, in_features: int, out_features: int, bias: bool = True,
                 device=None, dtype=None,rank = None,fixed = False,load_weights = None,step = 'K',full_rank_construct = False) -> None:

        """  
        initializer for the low rank linear layer, extention of the classical Pytorch's Linear, 
        implementation to for general size inputs
        INPUTS:
        in_features : number of inputs features (Pytorch standard)
        out_features : number of output features (Pytorch standard)
        bias : flag for the presence of bias (Pytorch standard)
        device : device in where to put parameters
        dtype : type of the tensors (Pytorch standard)
        rank : rank variable, None if the layer has to be treated as a classical Pytorch Linear layer (with weight and bias). If
                it is an int then it's either the starting rank for adaptive or the fixed rank for the layer.
        fixed : flag variable, True if the rank has to be fixed (KLS training on this layer)
        load_weights : variables to load (Pytorch standard, to finish)
        step : flag variable ('K','L' or 'S') for which forward phase to use
        """
        factory_kwargs = {'device': device, 'dtype': dtype}
        super(Linear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = torch.nn.Parameter(torch.empty((out_features, in_features), **factory_kwargs))
        self.rank = rank
        self.device = device
        self.load_weights = load_weights
        self.fixed = fixed
        self.lr = True if self.rank!=None else False
        self.full_rank_construct = full_rank_construct
        self.rmax = int(min([self.in_features, self.out_features]) / 2)
        if not self.fixed:
            self.rank = None if rank == None else min([rank,self.rmax])
        else:
            self.rank = min([rank,self.in_features,self.out_features])
        self.dynamic_rank = self.rank
        self.step = step

        if bias:
                self.bias = torch.nn.Parameter(torch.empty(out_features, **factory_kwargs))
        else:
                self.register_parameter('bias', None)

        self.reset_parameters()

            
        if self.lr and not full_rank_construct:

            if not self.fixed:   # initialization for dlrt adaptive
                _,s_ordered,_ = torch.linalg.svd(torch.diag(torch.abs(torch.randn(2*self.rmax))))
                U = torch.randn(self.out_features,self.rmax)
                V = torch.randn(self.in_features,self.rmax)
                U,_,_ = torch.linalg.svd(U)
                V,_,_ = torch.linalg.svd(V)
                self.U = torch.nn.Parameter(U.to(device) ,requires_grad=False)             
                self.S_hat = torch.nn.Parameter(torch.diag(s_ordered).to(device))                                          
                self.V = torch.nn.Parameter(V.to(device),requires_grad=False)
                self.U_hat = torch.nn.Parameter( torch.randn(self.out_features,2*self.rmax).to(device) ,requires_grad = False)
                self.V_hat = torch.nn.Parameter(torch.randn(self.in_features,2*self.rmax).to(device) ,requires_grad = False)
                self.K = torch.nn.Parameter(torch.randn(self.out_features,self.rmax).to(device))
                self.L = torch.nn.Parameter(torch.randn(self.in_features,self.rmax).to(device))
                self.N_hat = torch.nn.Parameter(torch.randn(2*self.rmax,self.rmax).to(device) ,requires_grad = False)
                self.M_hat = torch.nn.Parameter( torch.randn(2*self.rmax,self.rmax).to(device) ,requires_grad = False)
                self.weight = None
            else:     # initialization for dlrt fixed rank
                _,s_ordered,_ = torch.linalg.svd(torch.diag(torch.abs(torch.randn(self.rank))))
                U = torch.randn(self.out_features,self.rank)
                V = torch.randn(self.in_features,self.rank)
                U,_,_ = torch.linalg.svd(U)
                V,_,_ = torch.linalg.svd(V)
                self.U = torch.nn.Parameter(U[:,:self.rank].to(device) ,requires_grad=False)             
                self.S_hat = torch.nn.Parameter(torch.diag(s_ordered).to(device))                                          
                self.V = torch.nn.Parameter(V[:,:self.rank].to(device),requires_grad=False)
                self.K = torch.nn.Parameter(torch.randn(self.out_features,self.rank).to(device))
                self.L = torch.nn.Parameter(torch.randn(self.in_features,self.rank).to(device))
                self.N_hat = torch.nn.Parameter(torch.randn(self.rank,self.rank).to(device) ,requires_grad = False)
                self.M_hat = torch.nn.Parameter( torch.randn(self.rank,self.rank).to(device) ,requires_grad = False)
                self.weight = None

    def switch_lowrank(self):
        
        w,b = self.weight,self.bias
        device = self.device
        if not self.fixed:
            self.rank = None if self.rank == None else min([self.rank,self.rmax])
        else:
            self.rank = min([self.rank,self.out_features,self.in_features])

        self.bias = b

        if self.lr:

            if not self.fixed:

                n,m = self.out_features,self.in_features

                U_load,S_load,V_load = torch.linalg.svd(w.view(n,m))
                V_load = V_load.T
                r = len(S_load)
                _,s_ordered,_ = torch.linalg.svd(torch.diag(torch.abs(torch.randn(2*self.rmax-r))))
                s_ordered = torch.tensor(torch.cat([S_load,s_ordered.to(device)])).to(device)
                self.U = torch.nn.Parameter(U_load.to(device) ,requires_grad=False)             
                self.S_hat = torch.nn.Parameter(torch.diag(s_ordered).to(device))                                       
                self.V = torch.nn.Parameter(V_load.to(device),requires_grad=False)
                self.U_hat = torch.nn.Parameter( torch.randn(n,2*self.rmax).to(device) ,requires_grad = False)
                self.V_hat = torch.nn.Parameter(torch.randn(m,2*self.rmax).to(device) ,requires_grad = False)
                self.K = torch.nn.Parameter(torch.randn(n,self.rmax).to(device))
                self.L = torch.nn.Parameter(torch.randn(m,self.rmax).to(device))
                self.N_hat = torch.nn.Parameter(torch.randn(2*self.rmax,self.rmax).to(device) ,requires_grad = False)
                self.M_hat = torch.nn.Parameter( torch.randn(2*self.rmax,self.rmax).to(device) ,requires_grad = False)
                self.weight = None
                self.id = id(self.K)
            else:

                n,m = self.out_features,self.in_features

                U_load,S_load,V_load = torch.linalg.svd(w.view(n,m))
                V_load = V_load.T
                r = self.rank
                self.U = torch.nn.Parameter(U_load[:,:r].to(device) ,requires_grad=False)             
                self.S_hat = torch.nn.Parameter(torch.diag(S_load[:r]).to(device))                                       
                self.V = torch.nn.Parameter(V_load[:,:r].to(device),requires_grad=False)
                self.K = torch.nn.Parameter(torch.randn(n,r).to(device))
                self.L = torch.nn.Parameter(torch.randn(m,r).to(device))
                self.N_hat = torch.nn.Parameter(torch.randn(2*self.rmax,self.rmax).to(device) ,requires_grad = False)
                self.M_hat = torch.nn.Parameter( torch.randn(2*self.rmax,self.rmax).to(device) ,requires_grad = False)
                self.weight = None
                self.id = id(self.K)
        


    def reset_parameters(self) -> None:
        # Setting a=sqrt(5) in kaiming_uniform is the same as initializing with
        # uniform(-1/sqrt(in_features), 1/sqrt(in_features)). For details, see
        # https://github.com/pytorch/pytorch/issues/57109
        init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
            init.uniform_(self.bias, -bound, bound)

    def forward(self, input: Tensor) -> Tensor:
        """  
        forward phase for the layer (the backward is automatically created by Pytorch since only standard functions are used). To use dlrt
        training the three kind of forward phases has to be included
        INPUTS:
        input: input tensor
        """
        if not self.lr:

            x = F.linear(input,self.weight,self.bias)

        else : 

            if self.step == 'K':
                if not self.fixed:
                    K,V = self.K[:,:self.dynamic_rank],self.V[:,:self.dynamic_rank]
                else:
                    K,V = self.K,self.V
                x = F.linear(input,V.T)#input.mm(V)
                x = F.linear(x,K)#x.mm(K.T)
                
                if self.bias is not None:

                    x = x+self.bias

            elif self.step == 'L':
                if not self.fixed:
                    L,U = self.L[:,:self.dynamic_rank],self.U[:,:self.dynamic_rank]
                else:
                    L,U = self.L,self.U
                x = F.linear(input,L.T)#input.mm(L)
                x = F.linear(x,U)#x.mm(U.T)
                if self.bias is not None:
                    x = x+self.bias
            
            elif self.step == 'S':

                if not self.fixed:

                    S_hat,U_hat,V_hat = self.S_hat[:2*self.dynamic_rank,:2*self.dynamic_rank],self.U_hat[:,:2*self.dynamic_rank],self.V_hat[:,:2*self.dynamic_rank]

                else:

                    S_hat,U_hat,V_hat = self.S_hat,self.U,self.V

                x = F.linear(input,V_hat.T)#input.mm(V_hat)
                x = F.linear(x,S_hat)#x.mm(S_hat.T)
                x = F.linear(x,U_hat)#x.mm(U_hat.T)
                if self.bias is not None:
                    x = x+self.bias
                    
            else:

                raise ValueError(f' incorrect step type {self.step}')
            
        return x 
