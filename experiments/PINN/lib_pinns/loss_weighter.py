import torch
import numpy as np
import scipy
from typing import Sequence

class LossWeighter():
    
    def get_weights(self,losses:torch.Tensor=None,grads:torch.Tensor=None):
        pass
    
class LRAWeighter(LossWeighter):
    
    def __init__(self,beta=0.9) -> None:
        super().__init__()
        self.beta = beta
        self.weights=None
    
    def get_weights(self,losses:torch.Tensor=None,grads:torch.Tensor=None):
        if grads is None:
            raise ValueError("LRAWeighter needs grads to calculate weights")
        with torch.no_grad():
            if self.weights is None:
                self.weights = torch.ones(len(grads),device=grads.device)
            max_weight = torch.max(torch.abs(grads[0]))
            weight_now=[torch.tensor(1.0,device=grads.device)]
            for i,grad in enumerate(grads[1:]):
                weight_now.append((max_weight/(torch.mean(torch.abs(grad)))))
            weight_now=torch.stack(weight_now)
            self.weights = self.beta*self.weights+(1-self.beta)*weight_now
        return self.weights

class ReLoWeighter(LossWeighter):
    
    def __init__(self,beta=0.999, tau=0.1, rou=0.999) -> None:
        super().__init__()
        self.beta = beta
        self.tau = tau
        self.rou = rou
        self.weights=None
        self.loss_pre=None
        self.loss_initial=None

    def get_weights(self, losses: torch.Tensor=None, grads: torch.tensor=None):
        if losses is None:
            raise ValueError("ReLoWeighter needs losses to calculate weights")
        losses = losses.detach()
        with torch.no_grad():
            if self.weights is None:
                self.weights = torch.ones(len(losses),device=losses.device)
                self.loss_initial = losses
            else:
                weight_local=torch.nn.functional.softmax(losses/(self.tau*self.loss_pre),dim=0)*len(losses)
                rou=np.random.binomial(1,self.rou,1)
                if rou[0]>0.5:
                    self.weights=self.beta*self.weights+(1-self.beta)*weight_local
                else:
                    initial_soft=torch.nn.functional.softmax(losses/(self.tau*self.loss_initial),dim=0)*len(losses)
                    self.weights=self.beta*initial_soft+(1-self.beta)*weight_local
            self.loss_pre = losses
        return self.weights

class MinMaxWeighter(LossWeighter):
    
    def __init__(self,lr=0.001,use_adam=True) -> None:
        super().__init__()
        self.lr = lr
        self.use_adam = use_adam
        self.optimizer = None
        self.weights=None
    
    def get_weights(self,losses:torch.Tensor=None,grads:torch.Tensor=None):
        if losses is None:
            raise ValueError("MinMaxWeighter needs losses to calculate weights")
        losses = losses.detach()
        if self.weights is None:
            self.weights = torch.tensor([1.0]*len(losses),requires_grad=True,device=losses.device)
            if self.use_adam:
                self.optimizer = torch.optim.Adam([self.weights],lr=self.lr)
            else:
                self.optimizer = torch.optim.SGD([self.weights],lr=self.lr)
        else:
            self.optimizer.zero_grad()
            raw_weights=torch.nn.functional.softmax(self.weights,dim=0)*len(self.weights)
            loss = -1*torch.sum(raw_weights*losses)
            loss.backward()
            self.optimizer.step()
        return self.weights
