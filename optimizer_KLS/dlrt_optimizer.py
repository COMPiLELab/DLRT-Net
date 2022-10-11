#%%
import torch
import numpy as np
from tqdm import tqdm
from torch import float16   

class dlr_opt:

    def __init__(self,NN,tau = 0.01,theta = 0.1,absolute = False,
                KLS_optim = None,**kwargs):

        """
        initializer for the dlr_opt (dynamical low rank optimizer) class.
        INPUTS:
        NN: neural network with custom layers, methods and attributes needed (look at Lenet5 for an example) 
        tau : learning rate (integration step)
        theta : tolerance for singular values
        absolute : flag variable, True if theta has to be interpreted as an absolute tolerance  
        KLS_optim : Pytorch integrator to perform the integration step
        """

        self.NN = NN
        self.tau = tau
        self.theta = theta
        self.absolute = absolute
        self.kw = dict(kwargs)
        self.KLS_optim = KLS_optim

        if self.KLS_optim is not None:

            self.integrator = self.KLS_optim(self.NN.parameters(),lr = self.tau,**kwargs)

        else:

            self.integrator = torch.optim.SGD(self.NN.parameters(),lr = self.tau,**kwargs)


    @torch.no_grad()
    def K_postprocess_step(self):

        for l in self.NN.layer:

            if hasattr(l,'lr') and l.lr:

                if not l.fixed:
                    
                    U_hat = torch.hstack((l.K[:,:l.dynamic_rank],l.U[:,:l.dynamic_rank]))

                    try:
                        U_hat,_ = torch.linalg.qr(U_hat)
                    except:
                        U_hat,_ = np.linalg.qr(U_hat)
                        U_hat = torch.tensor(U_hat)
                    l.U_hat[:,:2*l.dynamic_rank] = U_hat
                    l.M_hat[:2*l.dynamic_rank,:l.dynamic_rank] = l.U_hat[:,:2*l.dynamic_rank].T@l.U[:,:l.dynamic_rank]
                
                else:

                    try:
                        U_hat,_ = torch.linalg.qr(l.K)

                    except:
                        U_hat,_ = np.linalg.qr(U_hat)
                        U_hat = torch.tensor(U_hat)
                    l.M_hat.data = U_hat.T@l.U.data
                    l.U.data = U_hat

    @torch.no_grad()
    def postprocess_step(self):
        
        self.K_postprocess_step()
        self.L_postprocess_step()

    @torch.no_grad()
    def K_integration_step(self):
        
        self.zero_bias_grad()
        self.integrator.step()

    @torch.no_grad()
    def zero_bias_grad(self):

        for l in self.NN.layer:

            if hasattr(l,'bias') and l.bias is not None:

                l.bias.grad = None

            if hasattr(l,'weight') and l.weight is not None:

                l.weight.grad = None

    @torch.no_grad()
    def L_postprocess_step(self):

        for l in self.NN.layer:

            if hasattr(l,'lr') and l.lr:

                if not l.fixed:

                    V_hat = torch.hstack((l.L[:,:l.dynamic_rank],l.V[:,:l.dynamic_rank]))
                    try :
                        V_hat,_ = torch.linalg.qr(V_hat)
                    except:
                        V_hat,_ = np.linalg.qr(V_hat.detach().numpy())
                        V_hat= torch.tensor(V_hat)
                    l.V_hat[:,:2*l.dynamic_rank] = V_hat
                    l.N_hat[:2*l.dynamic_rank,:l.dynamic_rank] = l.V_hat[:,:2*l.dynamic_rank].T@l.V[:,:l.dynamic_rank]

                else:

                    try :
                        V_hat,_ = torch.linalg.qr(l.L)
                    except:
                        V_hat,_ = np.linalg.qr(V_hat.detach().numpy())
                        V_hat= torch.tensor(V_hat)
                    l.N_hat.data = V_hat.T@l.V.data
                    l.V.data = V_hat


    
    @torch.no_grad()
    def L_integration_step(self):


        self.integrator.step()
        self.integrator.zero_grad()

    @torch.no_grad()
    def K_and_L_integration_step(self):
        
        self.zero_bias_grad()
        self.integrator.step()

    @torch.no_grad()
    def S_preprocess_step(self):

        for l in self.NN.layer:

            if hasattr(l,'lr') and l.lr:

                if not l.fixed:

                    s = l.M_hat[:2 * l.dynamic_rank, :l.dynamic_rank]@l.S_hat[: l.dynamic_rank, :l.dynamic_rank]@l.N_hat[:2 * l.dynamic_rank, :l.dynamic_rank].T
                    l.S_hat[:2*l.dynamic_rank,:2*l.dynamic_rank] = s

                else:

                    s = l.M_hat@l.S_hat@l.N_hat.T
                    l.S_hat.data = s



    @torch.no_grad()
    def K_preprocess_step(self):

        for l in self.NN.layer:

            if hasattr(l,'lr') and l.lr:

                if not l.fixed:
                
                    K = l.U[:,:l.dynamic_rank]@l.S_hat[:l.dynamic_rank,:l.dynamic_rank]
                    l.K[:,:l.dynamic_rank] = K

                else:

                    K = l.U.data@l.S_hat
                    l.K.data = K



    @torch.no_grad()
    def L_preprocess_step(self):

        for l in self.NN.layer:

            if hasattr(l,'lr') and l.lr:

                if not l.fixed:

                    L = l.V[:,:l.dynamic_rank]@l.S_hat[:l.dynamic_rank,:l.dynamic_rank].T
                    l.L[:,:l.dynamic_rank] = L

                else:

                    L = l.V.data@l.S_hat.T
                    l.L.data = L


    @torch.no_grad()
    def S_postprocess_step(self):

        for l in self.NN.layer:

            if hasattr(l,'lr') and l.lr:

                if not l.fixed:

                    # rank adaption

                    s_small = torch.clone(l.S_hat[:2 * l.dynamic_rank, :2 * l.dynamic_rank])
                    try:
                        u2, d, v2 = torch.linalg.svd(s_small)
                    except Exception as e:
                        print(e)
                        print(s_small)
                        u2, d, v2 = np.linalg.svd(s_small)

                    tmp = 0.0
                    tol = self.theta * torch.linalg.norm(d) if not self.absolute else self.theta 
                    rmax = int(np.floor(d.shape[0] / 2))
                    for j in range(0, 2 * rmax - 1):
                        tmp = torch.linalg.norm(d[j:2 * rmax - 1])
                        if tmp < tol:
                            rmax = j
                            break

                    rmax = min([rmax, l.rmax])
                    rmax = max([rmax, 2])

                    l.S_hat[:rmax,:rmax] = torch.diag(d[:rmax])
                    l.U[:, :rmax] = l.U_hat[:, :2 * l.dynamic_rank]@u2[:, :rmax]
                    l.V[:,:rmax] =  l.V_hat[:,:2 * l.dynamic_rank]@(v2[:, :rmax])
                    l.dynamic_rank = int(rmax)

    
    @torch.no_grad()
    def S_integration_step(self):

        self.integrator.step()
        self.integrator.zero_grad()
    

    @torch.no_grad()
    def preprocess_step(self):

        self.K_preprocess_step()
        self.L_preprocess_step()

    @torch.no_grad()
    def step(self,closure = None):

        """
        optimizer step for the dlrt.
        INPUTS:
        closure : function to compute the loss and backpropagate a second time (Pytorch standard)
        """

        # self.K_integration_step()
        # self.L_integration_step()
        self.K_and_L_integration_step()
        self.K_postprocess_step()
        self.L_postprocess_step()
        self.S_preprocess_step()
        self.zero_grad()
        if closure is not None:
            with torch.set_grad_enabled(True):
                loss = closure()
                loss.backward()
        self.S_integration_step()
        self.S_postprocess_step()
    
    @torch.no_grad()
    def zero_grad(self):
        for p in self.NN.parameters():
            if p.requires_grad:
                p.grad = None


    @torch.no_grad()
    def activate_S_fine_tuning(self):

        params = []

        for l in self.NN.layer:

            if hasattr(l,'lr') and l.lr:

                l.K.requires_grad = False
                l.L.requires_grad = False
                l.S_hat = torch.nn.Parameter(l.S_hat[:l.dynamic_rank,:l.dynamic_rank])
                l.fixed = True
                l.U = torch.nn.Parameter(l.U[:,:l.dynamic_rank],requires_grad = False)
                l.V = torch.nn.Parameter(l.V[:,:l.dynamic_rank],requires_grad = False)
                l.step = 'S'
                params.append(l.S_hat)
        params = torch.nn.ParameterList(params)
        self.integrator = self.KLS_optim(params,lr = self.tau,**self.kw)


    @torch.no_grad()
    def S_finetune_step(self):

        self.integrator.step()