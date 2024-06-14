#usr/bin/python3
# -*- coding: UTF-8 -*-
from .simulation_paras import *
import numpy as np
import torch
from typing import *
from scipy.stats.qmc import LatinHypercube

def p_func_np(x:Union[float,Sequence],y:Optional[Union[float,Sequence]]=None,wave_length:float=LAMBDA):
    return 0.5*(1-np.exp(2*wave_length*x))

def v_func_np(x:Union[float,Sequence],y:Union[float,Sequence],wave_length:float=LAMBDA):
    return wave_length/(2*np.pi)*np.exp(wave_length*x)*np.sin(2*np.pi*y)

def u_func_np(x:Union[float,Sequence],y:Union[float,Sequence],wave_length:float=LAMBDA):
    return 1-np.exp(wave_length*x)*np.cos(2*np.pi*y)

def p_func_torch(x:torch.Tensor,y:Optional[torch.Tensor]=None,wave_length:float=LAMBDA):
    return 0.5*(1-torch.exp(2*wave_length*x))

def v_func_torch(x:torch.Tensor,y:torch.Tensor,wave_length:float=LAMBDA):
    return wave_length/(2*np.pi)*torch.exp(wave_length*x)*torch.sin(2*np.pi*y)

def u_func_torch(x:torch.Tensor,y:torch.Tensor,wave_length:float=LAMBDA):
    return 1-torch.exp(wave_length*x)*torch.cos(2*np.pi*y)

class KovasznaySamplerBase():
    
    def __init__(self,n_internal:int,n_boundary:int,device:Union[str,torch.device],
                 update_data:bool=False,seed:int=21339,
                 x_start:float=X_START,x_end:float=X_END,
                 y_start:float=Y_START,y_end:float=Y_END) -> None:
        self.fake_data=[0]
        self.x_start=x_start
        self.x_end=x_end
        self.y_start=y_start
        self.y_end=y_end
        self.n_internal=n_internal
        self.n_boundary=n_boundary
        self.n_boundary_top_bottom=int(n_boundary/(x_end-x_start+y_end-y_start)*(x_end-x_start))
        self.n_boundary_left_right=self.n_boundary-self.n_boundary_top_bottom
        self.device=device
        self.update_data=update_data
        self.seed=seed
        if self.update_data:
            self.sample_boundary=self._sample_boundary
            self.sample_internal=self._sample_internal
        else:
            x_b,y_b,uvp_b=self._sample_boundary()
            #torch.save((x_b,y_b,uvp_b),f"./Kovasznay_boundary.pt")
            #x_b,y_b,uvp_b=torch.load(f"./Kovasznay_boundary.pt")
            self.sample_boundary=lambda: (x_b,y_b,uvp_b)
            x_i,y_i=self._sample_internal()
            #torch.save((x_i,y_i),f"./Kovasznay_internal.pt")
            #x_i,y_i=torch.load(f"./Kovasznay_internal.pt")
            self.sample_internal=lambda: (x_i,y_i)
        
    def _sample_boundary(self):
        raise NotImplementedError("This method should be implemented by the derived class")
    
    def _sample_internal(self):
        raise NotImplementedError("This method should be implemented by the derived class")
    
    def __len__(self):
        return 1
    
    def __getitem__(self,idx):
        return self.fake_data[idx]
    
class KovasznaySamplerLH(KovasznaySamplerBase):
    
    def __init__(self, n_internal: int, n_boundary: int, device: Union[str,torch.device], update_data: bool = False, seed: int = 21339, x_start: float = X_START, x_end: float = X_END, y_start: float = X_START, y_end: float = X_END) -> None:
        self.random_engine_boundary=LatinHypercube(d=1,seed=seed)
        self.random_engine_internal=LatinHypercube(d=2,seed=seed)
        super().__init__(n_internal, n_boundary, device, update_data, seed, x_start, x_end, y_start, y_end)
            
    def _sample_boundary(self):
        with torch.no_grad():
            x_top_bottom=torch.tensor(
                self.random_engine_boundary.random(n=self.n_boundary_top_bottom)[:,0],
                device=self.device,dtype=torch.float32)*(self.x_end-self.x_start)+self.x_start
            y_top_bottom=torch.cat(
                [self.y_start*torch.ones(self.n_boundary_top_bottom//2,device=self.device),
                 self.y_end*torch.ones(self.n_boundary_top_bottom-self.n_boundary_top_bottom//2,device=self.device)
                 ])
            y_left_right=torch.tensor(
                self.random_engine_boundary.random(n=self.n_boundary_left_right)[:,0],
                device=self.device,dtype=torch.float32)*(self.y_end-self.y_start)+self.y_start
            x_left_right=torch.cat(
                [self.x_start*torch.ones(self.n_boundary_left_right//2,device=self.device),
                 self.x_end*torch.ones(self.n_boundary_left_right-self.n_boundary_left_right//2,device=self.device)
                 ])
            x_b=torch.cat([x_top_bottom,x_left_right])
            y_b=torch.cat([y_top_bottom,y_left_right])
            re = torch.stack([u_func_torch(x_b,y_b),v_func_torch(x_b,y_b),p_func_torch(x_b,y_b)], dim=-1)
            return x_b,y_b,re
    
    def _sample_internal(self):
        with torch.no_grad():
            sample = self.random_engine_internal.random(n=self.n_internal)
            x=torch.tensor(sample[:,0],device=self.device,dtype=torch.float32)*(self.x_end-self.x_start)+self.x_start
            y=torch.tensor(sample[:,1],device=self.device,dtype=torch.float32)*(self.y_end-self.y_start)+self.y_start
        x.requires_grad=True
        y.requires_grad=True
        return x,y
    
class KovasznaySamplerMC(KovasznaySamplerBase):
    
    def __init__(self, n_internal: int, n_boundary: int, device: Union[str,torch.device], update_data: bool = False, seed: int = 21339, x_start: float = X_START, x_end: float = X_END, y_start: float = X_START, y_end: float = X_END) -> None:
        super().__init__(n_internal, n_boundary, device, update_data, seed, x_start, x_end, y_start, y_end)
            
    def _sample_boundary(self):
        with torch.no_grad():
            x_top_bottom=torch.rand(self.n_boundary_top_bottom,device=self.device)*(self.x_end-self.x_start)+self.x_start
            y_top_bottom=torch.cat(
                [self.y_start*torch.ones(self.n_boundary_top_bottom//2,device=self.device),
                 self.y_end*torch.ones(self.n_boundary_top_bottom-self.n_boundary_top_bottom//2,device=self.device)
                 ])
            y_left_right=torch.rand(self.n_boundary_left_right,device=self.device)*(self.y_end-self.y_start)+self.y_start
            x_left_right=torch.cat(
                [self.x_start*torch.ones(self.n_boundary_left_right//2,device=self.device),
                 self.x_end*torch.ones(self.n_boundary_left_right-self.n_boundary_left_right//2,device=self.device)
                 ])
            x_b=torch.cat([x_top_bottom,x_left_right])
            y_b=torch.cat([y_top_bottom,y_left_right])
            uvp = torch.stack([u_func_torch(x_b,y_b),v_func_torch(x_b,y_b),p_func_torch(x_b,y_b)], dim=-1)
            return x_b,y_b,uvp
    
    def _sample_internal(self):
        with torch.no_grad():
            sample=torch.rand(2,self.n_internal,device=self.device)
            x=sample[0]*(self.x_end-self.x_start)+self.x_start
            y=sample[1]*(self.y_end-self.y_start)+self.y_start
        x.requires_grad=True
        y.requires_grad=True
        return x,y

def KovasznaySampler(n_internal:int=N_INTERNAL,n_boundary:int=N_BOUNDARY,
                     device:Union[str,torch.device]="cuda:0",
                     update_data:bool=False,seed:int=21339,
                     x_start:float=X_START,x_end:float=X_END,
                     y_start:float=Y_START,y_end:float=Y_END,
                     data_sampler:Literal["latin_hypercube","monte_carlo"]="latin_hypercube",)->KovasznaySamplerBase:
    if data_sampler=="latin_hypercube":
        return KovasznaySamplerLH(n_internal,n_boundary,device,update_data,seed,x_start,x_end,y_start,y_end)
    elif data_sampler=="monte_carlo":
        return KovasznaySamplerMC(n_internal,n_boundary,device,update_data,seed,x_start,x_end,y_start,y_end)
    else:
        raise ValueError(f"Invalid sampler type:{data_sampler}")
    
class KovasznayValidationDataSet():
    
    def __init__(self,x_start:float=X_START,x_end:float=X_END,
                 y_start:float=Y_START,y_end:float=Y_END,
                 n_point=1001) -> None:
        x_length=x_end-x_start
        y_length=y_end-y_start
        n_point_x=int(np.sqrt(n_point*x_length/y_length))
        n_point_y=int(n_point/n_point_x)
        self.x,self.y=np.meshgrid(np.linspace(x_start,x_end,n_point_x),np.linspace(y_start,y_end,n_point_y))
        self.x=self.x.reshape(-1)
        self.y=self.y.reshape(-1)
        self.uvp=np.stack([u_func_np(self.x,self.y),v_func_np(self.x,self.y),p_func_np(self.x,self.y)],axis=-1)
        
        
class KovasznayValidationDataLoader():
    
    def __init__(self,validation_dataset:KovasznayValidationDataSet,device:Union[str,torch.device]="cuda:0") -> None:
        self.fake_data=[0]
        self.xs=torch.from_numpy(validation_dataset.x).float().to(device)
        self.ys=torch.from_numpy(validation_dataset.y).float().to(device)
        self.uvps=torch.from_numpy(validation_dataset.uvp).float().to(device)

    def __len__(self):
        return 1
    
    def __getitem__(self,idx):
        return self.fake_data[idx]
