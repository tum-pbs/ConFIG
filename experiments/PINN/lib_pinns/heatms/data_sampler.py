#usr/bin/python3
# -*- coding: UTF-8 -*-
from .simulation_paras import *
import numpy as np
import torch
from typing import *
from scipy.stats.qmc import LatinHypercube

def ini_func(x):
    return torch.sin(20*np.pi*x[:,0:1])*torch.sin(np.pi*x[:,1:2])

class HeatMSSamplerBase():
    
    def __init__(self,
                 n_internal:int,
                 n_boundary:int,
                 n_initial:int,
                 device:Union[str,torch.device],
                 update_data:bool=False,
                 seed:int=21339,
                 x_start=X_START,
                 x_end=X_END,
                 simulation_time=SIMULATION_TIME) -> None:
        self.fake_data=[0]
        self.x_start=x_start
        self.x_end=x_end
        self.n_dim=2
        self.n_internal=n_internal
        self.n_boundary=n_boundary
        self.n_initial=n_initial
        self.device=device
        self.update_data=update_data
        self.seed=seed
        self.simulation_time=simulation_time
        if self.update_data:
            self.sample_boundary=self._sample_boundary
            self.sample_internal=self._sample_internal
            self.sample_initial=self._sample_initial
        else:
            x_b,t_b,u_b=self._sample_boundary()
            self.sample_boundary=lambda: (x_b,t_b,u_b)
            x_i,t_i =self._sample_internal()
            self.sample_internal=lambda: (x_i,t_i)
            x_ini,t_ini,u_ini =self._sample_initial()
            self.sample_initial=lambda: (x_ini,t_ini,u_ini)
        
    def _sample_boundary(self):
        raise NotImplementedError("This method should be implemented by the derived class")
    
    def _sample_internal(self):
        raise NotImplementedError("This method should be implemented by the derived class")
    
    def _sample_initial(self):
        raise NotImplementedError("This method should be implemented by the derived class")
    
    def __len__(self):
        return 1
    
    def __getitem__(self,idx):
        return self.fake_data[idx]
    
class HeatMSSamplerLH(HeatMSSamplerBase):
    
    def __init__(self,
                 n_internal:int,
                 n_boundary:int,
                 n_initial:int,
                 device:Union[str,torch.device],
                 update_data:bool=False,
                 seed:int=21339,
                 x_start=X_START,
                 x_end=X_END,
                 simulation_time=SIMULATION_TIME) -> None:
        self.sampler=LatinHypercube(d=2,seed=seed)
        self.t_sampler=LatinHypercube(d=1,seed=seed)
        super().__init__(n_internal,n_boundary,n_initial,device,update_data,seed,x_start,x_end,simulation_time)
        
            
    def _sample_boundary(self):
        with torch.no_grad():
            sample = self.sampler.random(n=self.n_boundary)*(self.x_end-self.x_start)+self.x_start
            boundary_sample = torch.tensor(sample, dtype=torch.float32,device=self.device)
            dims = torch.randint(0, self.n_dim, (self.n_boundary,),device=self.device)
            values = torch.randint(0, 2, (self.n_boundary,), dtype=torch.float32,device=self.device)
            boundary_sample[torch.arange(self.n_boundary), dims] = values
            t=torch.tensor(self.t_sampler.random(n=self.n_boundary)*self.simulation_time, dtype=torch.float32,device=self.device)
            return boundary_sample, t ,torch.zeros([self.n_boundary,1],dtype=torch.float32,device=self.device)
    
    def _sample_internal(self):
        with torch.no_grad():
            sample = self.sampler.random(n=self.n_internal)*(self.x_end-self.x_start)+self.x_start
            internal_sample = torch.tensor(sample, dtype=torch.float32,device=self.device)
            t=torch.tensor(self.t_sampler.random(n=self.n_internal)*self.simulation_time, dtype=torch.float32,device=self.device)
        internal_sample.requires_grad=True
        t.requires_grad=True
        return internal_sample,t
    
    def _sample_initial(self):
        with torch.no_grad():
            sample = self.sampler.random(n=self.n_initial)*(self.x_end-self.x_start)+self.x_start
            ini_sample = torch.tensor(sample, dtype=torch.float32,device=self.device)
            t=torch.zeros([self.n_initial,1],dtype=torch.float32,device=self.device)
        return ini_sample,t,ini_func(ini_sample)

    
class HeatMSSamplerMC(HeatMSSamplerBase):

    def __init__(self,
                 n_internal:int,
                 n_boundary:int,
                 n_initial:int,
                 device:Union[str,torch.device],
                 update_data:bool=False,
                 seed:int=21339,
                 x_start=X_START,
                 x_end=X_END,
                 simulation_time=SIMULATION_TIME) -> None:
        super().__init__(n_internal,n_boundary,n_initial,device,update_data,seed,x_start,x_end,simulation_time)

    def _sample_boundary(self):
        with torch.no_grad():
            boundary_sample = torch.rand((self.n_boundary,2),device=self.device)*(self.x_end-self.x_start)+self.x_start
            dims = torch.randint(0, self.n_dim, (self.n_boundary,),device=self.device)
            values = torch.randint(0, 2, (self.n_boundary,), dtype=torch.float32,device=self.device)
            boundary_sample[torch.arange(self.n_boundary), dims] = values
            t=torch.rand((self.n_boundary,1),dtype=torch.float32,device=self.device)*self.simulation_time
            return boundary_sample, t ,torch.zeros([self.n_boundary,1],dtype=torch.float32,device=self.device)
    
    def _sample_internal(self):
        with torch.no_grad():
            internal_sample = torch.rand((self.n_internal,2),device=self.device)*(self.x_end-self.x_start)+self.x_start
            t=torch.rand((self.n_internal,1),dtype=torch.float32,device=self.device)*self.simulation_time
        internal_sample.requires_grad=True
        t.requires_grad=True
        return internal_sample,t
    
    def _sample_initial(self):
        with torch.no_grad():
            sample = self.sampler.random(n=self.n_initial)*(self.x_end-self.x_start)+self.x_start
            initial_sample = torch.tensor(sample, dtype=torch.float32,device=self.device)
            t=torch.zeros([self.n_initial,1],dtype=torch.float32,device=self.device)
        return initial_sample,t,ini_func(initial_sample)


def HeatMSSampler(n_internal:int,
                 n_boundary:int,
                 n_initial:int,
                 device:Union[str,torch.device],
                 update_data:bool=False,
                 seed:int=21339,
                 x_start=X_START,
                 x_end=X_END,
                 simulation_time=SIMULATION_TIME,
               data_sampler:Literal["latin_hypercube","monte_carlo"]="latin_hypercube",)->HeatMSSamplerBase:
    if data_sampler=="latin_hypercube":
        return HeatMSSamplerLH(n_internal,n_boundary,n_initial,device,update_data,seed,x_start,x_end,simulation_time)
    elif data_sampler=="monte_carlo":
        return HeatMSSamplerMC(n_internal,n_boundary,n_initial,device,update_data,seed,x_start,x_end,simulation_time)
    else:
        raise ValueError(f"Invalid sampler type:{data_sampler}")
    
class HeatMSValidationDataSet():
    
    def __init__(self,
                 data_path:str,
                 n_point=3125,) -> None:
        data=np.load(data_path)
        self.x = torch.from_numpy(data["x"]).to(torch.float32)  
        self.t = torch.from_numpy(data["t"]).to(torch.float32)  
        self.u = torch.from_numpy(data["u"]).to(torch.float32)   
        
        
class HeatMSValidationDataLoader():
    
    def __init__(self,validation_dataset:HeatMSValidationDataSet,device:Union[str,torch.device]="cuda:0") -> None:
        self.fake_data=[0]
        self.x=validation_dataset.x.to(device)
        self.t=validation_dataset.t.to(device)
        self.u=validation_dataset.u.to(device)
        
    def __len__(self):
        return 1
    
    def __getitem__(self,idx):
        return self.fake_data[idx]
