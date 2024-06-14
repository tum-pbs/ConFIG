#usr/bin/python3
# -*- coding: UTF-8 -*-
from .simulation_paras import *
import numpy as np
import torch
from typing import *
from scipy.stats.qmc import LatinHypercube

def u_func_np(x:Union[float,Sequence],
              y:Union[float,Sequence]=None,
              z:Union[float,Sequence]=None,
              t:Union[float,Sequence]=None,
              a:float=A,
              d:float=D):
    return -a*(
        np.exp(a*x)*np.sin(a*y+d*z)+
        np.exp(a*z)*np.cos(a*x+d*y)
        )*np.exp(-d**2*t)

def v_func_np(x:Union[float,Sequence],
              y:Union[float,Sequence]=None,
              z:Union[float,Sequence]=None,
              t:Union[float,Sequence]=None,
              a:float=A,
              d:float=D):
    return -a*(
        np.exp(a*y)*np.sin(a*z+d*x)+
        np.exp(a*x)*np.cos(a*y+d*z)
        )*np.exp(-d**2*t)

def w_func_np(x:Union[float,Sequence],
              y:Union[float,Sequence]=None,
              z:Union[float,Sequence]=None,
              t:Union[float,Sequence]=None,
              a:float=A,
              d:float=D):
    return -a*(
        np.exp(a*z)*np.sin(a*x+d*y)+
        np.exp(a*y)*np.cos(a*z+d*x)
        )*np.exp(-d**2*t)

def p_func_np(x:Union[float,Sequence],
              y:Union[float,Sequence]=None,
              z:Union[float,Sequence]=None,
              t:Union[float,Sequence]=None,
              a:float=A,
              d:float=D):
    return -0.5*a**2*(
        np.exp(2*a*x)+np.exp(2*a*y)+np.exp(2*a*z)+
        2*np.sin(a*x+d*y)*np.cos(a*z+d*x)*np.exp(a*(y+z))+
        2*np.sin(a*y+d*z)*np.cos(a*x+d*y)*np.exp(a*(z+x))+
        2*np.sin(a*z+d*x)*np.cos(a*y+d*z)*np.exp(a*(x+y))
    )*np.exp(-2*d**2*t)

def u_func_torch(x:torch.Tensor,
                 y:torch.Tensor,
                 z:torch.Tensor,
                 t:torch.Tensor,
                 a:float=A,
                 d:float=D):
    return -a*(
        torch.exp(a*x)*torch.sin(a*y+d*z)+
        torch.exp(a*z)*torch.cos(a*x+d*y)
        )*torch.exp(-d**2*t)

def v_func_torch(x:torch.Tensor,
                    y:torch.Tensor,
                    z:torch.Tensor,
                    t:torch.Tensor,
                    a:float=A,
                    d:float=D):
        return -a*(
            torch.exp(a*y)*torch.sin(a*z+d*x)+
            torch.exp(a*x)*torch.cos(a*y+d*z)
            )*torch.exp(-d**2*t)

def w_func_torch(x:torch.Tensor,
                    y:torch.Tensor,
                    z:torch.Tensor,
                    t:torch.Tensor,
                    a:float=A,
                    d:float=D):
        return -a*(
            torch.exp(a*z)*torch.sin(a*x+d*y)+
            torch.exp(a*y)*torch.cos(a*z+d*x)
            )*torch.exp(-d**2*t)

def p_func_torch(x:torch.Tensor,
                    y:torch.Tensor,
                    z:torch.Tensor,
                    t:torch.Tensor,
                    a:float=A,
                    d:float=D):
        return -0.5*a**2*(
            torch.exp(2*a*x)+torch.exp(2*a*y)+torch.exp(2*a*z)+
            2*torch.sin(a*x+d*y)*torch.cos(a*z+d*x)*torch.exp(a*(y+z))+
            2*torch.sin(a*y+d*z)*torch.cos(a*x+d*y)*torch.exp(a*(z+x))+
            2*torch.sin(a*z+d*x)*torch.cos(a*y+d*z)*torch.exp(a*(x+y))
        )*torch.exp(-2*d**2*t)
        
                 
class BeltramiSamplerBase():
    
    def __init__(self,n_internal:int=N_INTERNAL,
                 n_boundary:int=N_BOUNDARY,
                 n_initial:int=N_INITIAL,
                     device:Union[str,torch.device]="cuda:0",
                     update_data:bool=False,seed:int=21339,
                     x_start:float=X_START,x_end:float=X_END,
                     y_start:float=Y_START,y_end:float=Y_END,
                     z_start:float=Z_START,z_end:float=Z_END,
                     simulation_time:float=SIMULATION_TIME) -> None:
        self.fake_data=[0]
        self.x_start=x_start;self.x_end=x_end
        self.y_start=y_start;self.y_end=y_end
        self.z_start=z_start;self.z_end=z_end
        self.simulation_time=simulation_time
        self.n_internal=n_internal;self.n_boundary=n_boundary;self.n_initial=n_initial
        total_length=(x_end-x_start)+(y_end-y_start)+(z_end-z_start)
        self.n_length=int((n_boundary)/total_length*(x_end-x_start))
        self.n_width=int((n_boundary)/total_length*(y_end-y_start))
        self.n_height=n_boundary-self.n_length-self.n_width
        self.device=device
        self.update_data=update_data
        self.seed=seed
        if self.update_data:
            self.sample_initial_boundary=self._sample_initial_boundary
            self.sample_boundary=self._sample_boundary
            self.sample_initial=self._sample_initial
            self.sample_internal=self._sample_internal
        else:
            x_b,y_b,z_b,t_b,uvwp_b=self._sample_boundary()
            self.sample_boundary=lambda: (x_b,y_b,z_b,t_b,uvwp_b)
            x_i,y_i,z_i,t_i,uvwp_i=self._sample_initial()
            self.sample_initial=lambda: (x_i,y_i,z_i,t_i,uvwp_i)
            x_bi=torch.cat([x_b,x_i])
            y_bi=torch.cat([y_b,y_i])
            z_bi=torch.cat([z_b,z_i])
            t_bi=torch.cat([t_b,t_i])
            uvwp_bi=torch.stack(
            [u_func_torch(x_bi,y_bi,z_bi,t_bi),
             v_func_torch(x_bi,y_bi,z_bi,t_bi),
             w_func_torch(x_bi,y_bi,z_bi,t_bi),
             p_func_torch(x_bi,y_bi,z_bi,t_bi)], dim=-1)
            self.sample_initial_boundary=lambda: (x_bi,y_bi,z_bi,t_bi,uvwp_bi)
            x_int,y_int,z_int,t_int=self._sample_internal()
            self.sample_internal=lambda: (x_int,y_int,z_int,t_int)
        
    def _sample_initial_boundary(self):
        raise NotImplementedError("This method should be implemented by the derived class")
    
    def _sample_internal(self):
        raise NotImplementedError("This method should be implemented by the derived class")
    
    def __len__(self):
        return 1
    
    def __getitem__(self,idx):
        return self.fake_data[idx]
    
class BeltramiSamplerLH(BeltramiSamplerBase):
    
    def __init__(self, n_internal: int = N_INTERNAL, n_boundary: int = N_BOUNDARY, n_initial: int = N_INITIAL, device: str | torch.device = "cuda:0", update_data: bool = False, seed: int = 21339, x_start: float = X_START, x_end: float = X_END, y_start: float = Y_START, y_end: float = Y_END, z_start: float = Z_START, z_end: float = Z_END, simulation_time: float = SIMULATION_TIME) -> None:
        self.random_engine_boundary=LatinHypercube(d=1,seed=seed)
        self.random_engine_internal=LatinHypercube(d=4,seed=seed)
        self.random_engine_initial=LatinHypercube(d=3,seed=seed)
        super().__init__(n_internal, n_boundary, n_initial, device, update_data, seed, x_start, x_end, y_start, y_end, z_start, z_end, simulation_time)
    
    def _get_boundary_points(self):
        with torch.no_grad():
            x_length_boundary=torch.tensor(
                self.random_engine_boundary.random(n=self.n_length)[:,0],
                device=self.device,dtype=torch.float32)*(self.x_end-self.x_start)+self.x_start
            y_length_boundary=torch.cat(
                [self.y_start*torch.ones(self.n_length//4,device=self.device),
                    self.y_end*torch.ones(self.n_length//2,device=self.device),
                    self.y_start*torch.ones(self.n_length-self.n_length//4-self.n_length//2,device=self.device)
                 ])
            z_length_boundary=torch.cat(
                [self.z_start*torch.ones(self.n_length//2,device=self.device),
                    self.z_end*torch.ones(self.n_length-self.n_length//2,device=self.device)
                    ])
            
            x_width_boundary=torch.cat(
                [self.x_start*torch.ones(self.n_width//4,device=self.device),
                    self.x_end*torch.ones(self.n_width//2,device=self.device),
                    self.x_start*torch.ones(self.n_width-self.n_width//4-self.n_width//2,device=self.device)
                    ])
            y_width_boundary=torch.tensor(
                self.random_engine_boundary.random(n=self.n_width)[:,0],
                device=self.device,dtype=torch.float32)*(self.y_end-self.y_start)+self.y_start
            z_width_boundary=torch.cat(
                [self.z_start*torch.ones(self.n_width//2,device=self.device),
                    self.z_end*torch.ones(self.n_width-self.n_width//2,device=self.device)
                    ])

            x_height_boundary=torch.cat(
                [self.x_start*torch.ones(self.n_height//4,device=self.device),
                    self.x_end*torch.ones(self.n_height//2,device=self.device),
                    self.x_start*torch.ones(self.n_height-self.n_height//4-self.n_height//2,device=self.device)
                    ])
            y_height_boundary=torch.cat(
                [self.y_start*torch.ones(self.n_height//2,device=self.device),
                    self.y_end*torch.ones(self.n_height-self.n_height//2,device=self.device)
                    ])
            z_height_boundary=torch.tensor(
                self.random_engine_boundary.random(n=self.n_height)[:,0],
                device=self.device,dtype=torch.float32)*(self.z_end-self.z_start)+self.z_start
            x_b=torch.cat([x_length_boundary,x_width_boundary,x_height_boundary])
            y_b=torch.cat([y_length_boundary,y_width_boundary,y_height_boundary])
            z_b=torch.cat([z_length_boundary,z_width_boundary,z_height_boundary])
            return x_b,y_b,z_b
    
    def _sample_initial_boundary(self):
        with torch.no_grad():
            x_b,y_b,z_b,t_b=self._sample_boundary(return_uvwp=False)
            x_i,y_i,z_i,t_i=self._sample_initial(return_uvwp=False)
            x=torch.cat([x_b,x_i])
            y=torch.cat([y_b,y_i])
            z=torch.cat([z_b,z_i])
            t=torch.cat([t_b,t_i])
        return x,y,z,t,torch.stack(
            [u_func_torch(x,y,z,t),
             v_func_torch(x,y,z,t),
             w_func_torch(x,y,z,t),
             p_func_torch(x,y,z,t)], dim=-1)
    
    def _sample_initial(self,return_uvwp:bool=True):
        with torch.no_grad():
            sample = self.random_engine_initial.random(n=self.n_initial)
            x=torch.tensor(sample[:,0],device=self.device,dtype=torch.float32)*(self.x_end-self.x_start)+self.x_start      
            y=torch.tensor(sample[:,1],device=self.device,dtype=torch.float32)*(self.y_end-self.y_start)+self.y_start
            z=torch.tensor(sample[:,2],device=self.device,dtype=torch.float32)*(self.z_end-self.z_start)+self.z_start
            t=torch.zeros(self.n_initial,device=self.device,dtype=torch.float32)
        if return_uvwp:
            return x,y,z,t,torch.stack(
                [u_func_torch(x,y,z,t),
                    v_func_torch(x,y,z,t),
                    w_func_torch(x,y,z,t),
                    p_func_torch(x,y,z,t)], dim=-1)
        else:
            return x,y,z,t
    
    def _sample_boundary(self,return_uvwp:bool=True):
        x,y,z=self._get_boundary_points()
        t=torch.tensor(
                self.random_engine_boundary.random(n=self.n_boundary)[:,0],
                device=self.device,dtype=torch.float32)*self.simulation_time
        if return_uvwp:
            return x,y,z,t,torch.stack(
                [u_func_torch(x,y,z,t),
                    v_func_torch(x,y,z,t),
                    w_func_torch(x,y,z,t),
                    p_func_torch(x,y,z,t)], dim=-1)
        else:
            return x,y,z,t
        
    def _sample_internal(self):
        with torch.no_grad():
            sample = self.random_engine_internal.random(n=self.n_internal)
            x=torch.tensor(sample[:,0],device=self.device,dtype=torch.float32)*(self.x_end-self.x_start)+self.x_start      
            y=torch.tensor(sample[:,1],device=self.device,dtype=torch.float32)*(self.y_end-self.y_start)+self.y_start
            z=torch.tensor(sample[:,2],device=self.device,dtype=torch.float32)*(self.z_end-self.z_start)+self.z_start
            t=torch.tensor(sample[:,3],device=self.device,dtype=torch.float32)*self.simulation_time
        x.requires_grad=True
        y.requires_grad=True
        z.requires_grad=True
        t.requires_grad=True
        return x,y,z,t
    
class BeltramiSamplerMC(BeltramiSamplerBase):
    
    def __init__(self, n_internal:int = N_INTERNAL, n_boundary:int = N_BOUNDARY, n_initial:int = N_INITIAL, device: str | torch.device = "cuda:0", update_data: bool = False, seed:int = 21339, x_start: float = X_START, x_end: float = X_END, y_start: float = Y_START, y_end: float = Y_END, z_start: float = Z_START, z_end: float = Z_END, simulation_time: float = SIMULATION_TIME) -> None:
        super().__init__(n_internal, n_boundary, n_initial, device, update_data, seed, x_start, x_end, y_start, y_end, z_start, z_end, simulation_time)
            
    def _get_boundary_points(self):
        with torch.no_grad():
            x_length_boundary=torch.rand(self.n_length,
                device=self.device)*(self.x_end-self.x_start)+self.x_start
            y_length_boundary=torch.cat(
                [self.y_start*torch.ones(self.n_length//4,device=self.device),
                    self.y_end*torch.ones(self.n_length//2,device=self.device),
                    self.y_start*torch.ones(self.n_length-self.n_length//4-self.n_length//2,device=self.device)
                 ])
            z_length_boundary=torch.cat(
                [self.z_start*torch.ones(self.n_length//2,device=self.device),
                    self.z_end*torch.ones(self.n_length-self.n_length//2,device=self.device)
                    ])
            
            x_width_boundary=torch.cat(
                [self.x_start*torch.ones(self.n_width//4,device=self.device),
                    self.x_end*torch.ones(self.n_width//2,device=self.device),
                    self.x_start*torch.ones(self.n_width-self.n_width//4-self.n_width//2,device=self.device)
                    ])
            y_width_boundary=torch.rand(self.n_width,
                device=self.device)*(self.y_end-self.y_start)+self.y_start
            z_width_boundary=torch.cat(
                [self.z_start*torch.ones(self.n_width//2,device=self.device),
                    self.z_end*torch.ones(self.n_width-self.n_width//2,device=self.device)
                    ])

            x_height_boundary=torch.cat(
                [self.x_start*torch.ones(self.n_height//4,device=self.device),
                    self.x_end*torch.ones(self.n_height//2,device=self.device),
                    self.x_start*torch.ones(self.n_height-self.n_height//4-self.n_height//2,device=self.device)
                    ])
            y_height_boundary=torch.cat(
                [self.y_start*torch.ones(self.n_height//2,device=self.device),
                    self.y_end*torch.ones(self.n_height-self.n_height//2,device=self.device)
                    ])
            z_height_boundary=torch.rand(self.n_height,
                device=self.device)*(self.z_end-self.z_start)+self.z_start
            x_b=torch.cat([x_length_boundary,x_width_boundary,x_height_boundary])
            y_b=torch.cat([y_length_boundary,y_width_boundary,y_height_boundary])
            z_b=torch.cat([z_length_boundary,z_width_boundary,z_height_boundary])
            return x_b,y_b,z_b
    
    def _sample_initial_boundary(self):
        with torch.no_grad():
            x_b,y_b,z_b,t_b=self._sample_boundary(return_uvwp=False)
            x_i,y_i,z_i,t_i=self._sample_initial(return_uvwp=False)
            x=torch.cat([x_b,x_i])
            y=torch.cat([y_b,y_i])
            z=torch.cat([z_b,z_i])
            t=torch.cat([t_b,t_i])
        return x,y,z,t,torch.stack(
            [u_func_torch(x,y,z,t),
             v_func_torch(x,y,z,t),
             w_func_torch(x,y,z,t),
             p_func_torch(x,y,z,t)], dim=-1)

    def _sample_initial(self,return_uvwp:bool=True):
        with torch.no_grad():
            sample=torch.rand(3,self.n_initial,device=self.device)
            x=sample[0]*(self.x_end-self.x_start)+self.x_start
            y=sample[1]*(self.y_end-self.y_start)+self.y_start
            z=sample[2]*(self.z_end-self.z_start)+self.z_start
            t=torch.zeros(self.n_initial,device=self.device,dtype=torch.float32)
        if return_uvwp:
            return x,y,z,t,torch.stack(
                [u_func_torch(x,y,z,t),
                    v_func_torch(x,y,z,t),
                    w_func_torch(x,y,z,t),
                    p_func_torch(x,y,z,t)], dim=-1)
        else:
            return x,y,z,t
    
    def _sample_boundary(self,return_uvwp:bool=True):
        x,y,z=self._get_boundary_points()
        t=torch.rand(self.n_boundary,device=self.device)*self.simulation_time
        if return_uvwp:
            return x,y,z,t,torch.stack(
                [u_func_torch(x,y,z,t),
                    v_func_torch(x,y,z,t),
                    w_func_torch(x,y,z,t),
                    p_func_torch(x,y,z,t)], dim=-1)
        else:
            return x,y,z,t
     
    def _sample_internal(self):
        with torch.no_grad():
            sample=torch.rand(4,self.n_internal,device=self.device)
            x=sample[0]*(self.x_end-self.x_start)+self.x_start
            y=sample[1]*(self.y_end-self.y_start)+self.y_start
            z=sample[2]*(self.z_end-self.z_start)+self.z_start
            t=sample[3]*self.simulation_time
        x.requires_grad=True
        y.requires_grad=True
        z.requires_grad=True
        t.requires_grad=True
        return x,y,z,t

def BeltramiSampler(n_internal:int=N_INTERNAL,
                            n_boundary:int=N_BOUNDARY,
                            n_initial:int=N_INITIAL,
                            device:Union[str,torch.device]="cuda:0",
                            update_data:bool=False,seed:int=21339,
                            x_start:float=X_START,x_end:float=X_END,
                            y_start:float=Y_START,y_end:float=Y_END,
                            z_start:float=Z_START,z_end:float=Z_END,
                            simulation_time:float=SIMULATION_TIME,
                            data_sampler:Literal["latin_hypercube","monte_carlo"]="latin_hypercube"):
     if data_sampler=="latin_hypercube":
          return BeltramiSamplerLH(n_internal, n_boundary, n_initial, device, update_data, seed, x_start, x_end, y_start, y_end, z_start, z_end, simulation_time)
     elif data_sampler=="monte_carlo":
          return BeltramiSamplerMC(n_internal, n_boundary, n_initial, device, update_data, seed, x_start, x_end, y_start, y_end, z_start, z_end, simulation_time)
     else:
          raise ValueError(f"Invalid sampler type:{data_sampler}")

def uniform_mesh(n_x, n_y, n_z,n_t,
                 x_start:float=X_START,x_end:float=X_END,
                 y_start:float=Y_START,y_end:float=Y_END,
                 z_start:float=Z_START,z_end:float=Z_END,
                 simulation_time:float=SIMULATION_TIME) -> None:
        x,y,z=np.meshgrid(
            np.linspace(x_start,x_end,n_x),
            np.linspace(y_start,y_end,n_y),
            np.linspace(z_start,z_end,n_z)
            )
        t_values=np.linspace(0,simulation_time,n_t)
        t=np.stack([np.ones_like(x)*t_i for t_i in t_values],axis=0)
        x=np.repeat(np.expand_dims(x,axis=0),n_t,axis=0)
        y=np.repeat(np.expand_dims(y,axis=0),n_t,axis=0)
        z=np.repeat(np.expand_dims(z,axis=0),n_t,axis=0)
        return x,y,z,t
 
class BeltramiValidationDataSet():
    
    def __init__(self,
                 x_start:float=X_START,x_end:float=X_END,
                 y_start:float=Y_START,y_end:float=Y_END,
                 z_start:float=Z_START,z_end:float=Z_END,
                 simulation_time:float=SIMULATION_TIME,
                 n_t=5,
                 n_point=2001) -> None:
        n_point=n_point//n_t
        x_length=x_end-x_start
        y_length=y_end-y_start
        z_length=z_end-z_start
        n_point_x=int(np.power(n_point/(y_length/x_length)/(z_length/x_length),1/3))
        n_point_y=int(n_point_x*y_length/x_length)
        n_point_z=int(n_point/n_point_x/n_point_y)
        self.x,self.y,self.z,self.t=uniform_mesh(
            n_point_x,n_point_y,n_point_z,n_t,x_start,x_end,y_start,y_end,z_start,z_end,simulation_time)
        self.mesh_shape=self.x.shape
        self.x=self.x.reshape(-1)
        self.y=self.y.reshape(-1)
        self.z=self.z.reshape(-1)
        self.t=self.t.reshape(-1)
        self.uvwp=np.stack(
            [u_func_np(self.x,self.y,self.z,self.t),
             v_func_np(self.x,self.y,self.z,self.t),
             w_func_np(self.x,self.y,self.z,self.t),
             p_func_np(self.x,self.y,self.z,self.t)],axis=-1)
        
        
class BeltramiValidationDataLoader():
    
    def __init__(self,validation_dataset:BeltramiValidationDataSet,device:Union[str,torch.device]="cuda:0") -> None:
        self.fake_data=[0]
        self.xs=torch.from_numpy(validation_dataset.x).float().to(device)
        self.ys=torch.from_numpy(validation_dataset.y).float().to(device)
        self.zs=torch.from_numpy(validation_dataset.z).float().to(device)
        self.ts=torch.from_numpy(validation_dataset.t).float().to(device)
        self.uvwps=torch.from_numpy(validation_dataset.uvwp).float().to(device)

    def __len__(self):
        return 1
    
    def __getitem__(self,idx):
        return self.fake_data[idx]
