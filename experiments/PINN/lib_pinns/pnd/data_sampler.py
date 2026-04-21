#usr/bin/python3
# -*- coding: UTF-8 -*-
from .simulation_paras import *
import numpy as np
import torch
from typing import *
from scipy.stats.qmc import LatinHypercube

def value_func(x):
    return torch.sin(np.pi/2*x).sum(dim=-1,keepdim=True)

class PNdSamplerBase():
    
    def __init__(self,n_internal:int,
                 n_boundary:int,device:Union[str,torch.device],
                 update_data:bool=False,
                 seed:int=21339,
                 x_start=X_START,
                 x_end=X_END,
                 n_dim=N_DIM) -> None:
        self.fake_data=[0]
        self.x_start=x_start
        self.x_end=x_end
        self.n_dim=n_dim
        self.n_internal=n_internal
        self.n_boundary=n_boundary
        self.device=device
        self.update_data=update_data
        self.seed=seed
        if self.update_data:
            self.sample_boundary=self._sample_boundary
            self.sample_internal=self._sample_internal
        else:
            x_b,u_b=self._sample_boundary()
            self.sample_boundary=lambda: (x_b,u_b)
            x_i =self._sample_internal()
            self.sample_internal=lambda: x_i
        
    def _sample_boundary(self):
        raise NotImplementedError("This method should be implemented by the derived class")
    
    def _sample_internal(self):
        raise NotImplementedError("This method should be implemented by the derived class")
    
    def __len__(self):
        return 1
    
    def __getitem__(self,idx):
        return self.fake_data[idx]
    
class PNdSamplerLH(PNdSamplerBase):
    
    def __init__(self, n_internal: int, 
                 n_boundary: int, 
                 device: str | torch.device, 
                 update_data: bool = False, 
                 seed: int = 21339, 
                 x_start=X_START, 
                 x_end=X_END, 
                 n_dim=N_DIM) -> None:
        self.sampler=LatinHypercube(d=n_dim,seed=seed)
        super().__init__(n_internal, n_boundary, device, update_data, seed, x_start, x_end, n_dim)
        
            
    def _sample_boundary(self):
        with torch.no_grad():
            sample = self.sampler.random(n=self.n_boundary)*(self.x_end-self.x_start)+self.x_start
            boundary_sample = torch.tensor(sample, dtype=torch.float32,device=self.device)
            dims = torch.randint(0, self.n_dim, (self.n_boundary,),device=self.device)
            values = torch.randint(0, 2, (self.n_boundary,), dtype=torch.float32,device=self.device)*(self.x_end-self.x_start)+self.x_start
            boundary_sample[torch.arange(self.n_boundary), dims] = values
            return boundary_sample, value_func(boundary_sample)
    
    def _sample_internal(self):
        with torch.no_grad():
            sample = self.sampler.random(n=self.n_internal)*(self.x_end-self.x_start)+self.x_start
            internal_sample = torch.tensor(sample, dtype=torch.float32,device=self.device)
        internal_sample.requires_grad=True
        return internal_sample
    
class PNdSamplerMC(PNdSamplerBase):

    def __init__(self, n_internal: int, 
                 n_boundary: int, 
                 device: str | torch.device, 
                 update_data: bool = False, seed: int = 21339, 
                 x_start=X_START, x_end=X_END, n_dim=N_DIM) -> None:
        super().__init__(n_internal, n_boundary, device, update_data, seed, x_start, x_end, n_dim)

    def _sample_boundary(self):
        with torch.no_grad():
            sample = torch.rand(self.n_boundary,device=self.device)*(self.x_end-self.x_start)+self.x_start
            boundary_sample = torch.tensor(sample, dtype=torch.float32)
            dims = torch.randint(0, self.n_dim, (self.n_boundary,),device=self.device)
            values = torch.randint(0, 2, (self.n_boundary,), dtype=torch.float32,device=self.device)
            boundary_sample[torch.arange(self.n_boundary), dims] = values
            return boundary_sample, value_func(boundary_sample)
    
    def _sample_internal(self):
        with torch.no_grad():
            internal_sample= torch.rand(self.n_internal,device=self.device)*(self.x_end-self.x_start)+self.x_start
        internal_sample.requires_grad=True
        return internal_sample

def PNdSampler(n_internal:int,
               n_boundary:int,
               device:Union[str,torch.device]="cuda:0",
               update_data:bool=False,seed:int=21339,
               x_start:float=X_START,x_end:float=X_END,
               data_sampler:Literal["latin_hypercube","monte_carlo"]="latin_hypercube",
               n_dim=N_DIM)->PNdSamplerBase:
    if data_sampler=="latin_hypercube":
        return PNdSamplerLH(n_internal,n_boundary,device,update_data,seed,x_start,x_end,n_dim)
    elif data_sampler=="monte_carlo":
        return PNdSamplerMC(n_internal,n_boundary,device,update_data,seed,x_start,x_end,n_dim)
    else:
        raise ValueError(f"Invalid sampler type:{data_sampler}")
    
class PNdValidationDataSet():
    
    def __init__(self,x_start:float=X_START,x_end:float=X_END,
                 n_point=3125,
                 n_dim=N_DIM) -> None:
        num_samples_per_dim = int(np.ceil(n_point**(1/n_dim)))
        grid_points = [(torch.linspace(0, 1, num_samples_per_dim+2)[1:-1])*(x_end-x_start)+x_start for _ in range(n_dim)]
        meshgrid = torch.meshgrid(*grid_points)
        self.x = torch.stack(meshgrid, dim=-1).reshape(-1, n_dim)
        self.u=value_func(self.x)
        
        
class PNdValidationDataLoader():
    
    def __init__(self,validation_dataset:PNdValidationDataSet,device:Union[str,torch.device]="cuda:0") -> None:
        self.fake_data=[0]
        self.x=validation_dataset.x.to(device)
        self.u=validation_dataset.u.to(device)
        
    def __len__(self):
        return 1
    
    def __getitem__(self,idx):
        return self.fake_data[idx]
