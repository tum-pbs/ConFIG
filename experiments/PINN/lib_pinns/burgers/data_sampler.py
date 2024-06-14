from .simulation_paras import *
import torch
from scipy.stats.qmc import LatinHypercube
from typing import *

class BurgersSamplerBase():
    
    def __init__(self,n_internal:int,n_initial:int,n_boundary:int,device:Union[str,torch.device],
                 update_data:bool=False,seed:int=21339,
                 x_start:float=X_START,x_end:float=X_END,ini_boundary:Callable=INITIAL_BOUNDAR,simulation_time:float=SIMULATION_TIME) -> None:
        self.fake_data=[0]
        self.ini_boundary=ini_boundary
        self.x_start=x_start
        self.x_end=x_end
        self.n_internal=n_internal
        self.n_boundary=n_boundary
        self.n_initial=n_initial
        self.simulation_time=simulation_time
        self.device=device
        self.update_data=update_data
        if self.update_data:
            self.sample_initial_boundary=self._sample_initial_boundary
            self.sample_initial=self._sample_initial
            self.sample_boundary=self._sample_boundary
            self.sample_internal=self._sample_internal
        else:
            x_b,t_b,v_b=self._sample_boundary()
            self.sample_boundary=lambda: (x_b,t_b,v_b)
            x_ini,t_ini,v_ini=self._sample_initial()
            self.sample_initial=lambda: (x_ini,t_ini,v_ini)
            x_bi=torch.cat([x_b,x_ini])
            t_bi=torch.cat([t_b,t_ini])
            v_bi=torch.cat([v_b,v_ini])
            self.sample_initial_boundary=lambda: (x_bi,t_bi,v_bi)
            x_i,t_i=self._sample_internal()
            self.sample_internal=lambda: (x_i,t_i)
        
    def _sample_initial_boundary(self):
        with torch.no_grad():
            x_boundary,t_boundary,value_boundary=self._sample_boundary()
            x_initial,t_initial,value_initial=self._sample_initial()
            return torch.cat([x_initial,x_boundary]),torch.cat([t_initial,t_boundary]),torch.cat([value_initial,value_boundary])  
    
    def _sample_initial(self):
        raise NotImplementedError("This method should be implemented by the derived class")
    
    def _sample_boundary(self):
        raise NotImplementedError("This method should be implemented by the derived class")
    
    def _sample_internal(self):
        raise NotImplementedError("This method should be implemented by the derived class")
    
    def __len__(self):
        return 1
    
    def __getitem__(self,idx):
        return self.fake_data[idx]


class BurgersSamplerLH(BurgersSamplerBase):
    
    
    def __init__(self, n_internal: int, n_initial: int, n_boundary: int, device:Union[str,torch.device], update_data: bool = False, seed: int = 21339, x_start: float = X_START, x_end: float = X_END, ini_boundary: Callable[..., Any] = INITIAL_BOUNDAR, simulation_time: float = SIMULATION_TIME) -> None:
        self.random_engine_initial=LatinHypercube(d=1,seed=seed)
        self.random_engine_internal=LatinHypercube(d=2,seed=seed)
        super().__init__(n_internal, n_initial, n_boundary, device, update_data, seed, x_start, x_end, ini_boundary, simulation_time)
        
    
    def _sample_initial(self):
        with torch.no_grad():
            x_initial = torch.tensor(self.random_engine_initial.random(n=self.n_initial)[:,0],device=self.device,dtype=torch.float32)*(self.x_end-self.x_start)+self.x_start
            t_initial = torch.zeros_like(x_initial)
            value_initial = self.ini_boundary(x_initial)
            return x_initial,t_initial,value_initial
    
    def _sample_boundary(self):
        with torch.no_grad():
            x_boundary = torch.cat([self.x_start*torch.ones(self.n_boundary//2,device=self.device),self.x_end*torch.ones(self.n_boundary//2,device=self.device)])
            t_boundary = torch.tensor(self.random_engine_initial.random(n=self.n_boundary)[:,0],device=self.device,dtype=torch.float32)
            value_boundary = torch.zeros_like(t_boundary,device=self.device)
            return x_boundary,t_boundary,value_boundary
    
    def _sample_internal(self):
        sample = self.random_engine_internal.random(n=self.n_internal)
        x=torch.tensor(sample[:,0],device=self.device,dtype=torch.float32)*(self.x_end-self.x_start)+self.x_start
        x.requires_grad=True
        t=torch.tensor(sample[:,1],device=self.device,dtype=torch.float32)*self.simulation_time
        t.requires_grad=True
        return x,t
    

class BurgersSamplerMC(BurgersSamplerBase):
    
    def __init__(self, n_internal: int, n_initial: int, n_boundary: int, device: Union[str,torch.device], update_data: bool = False, seed: int = 21339, x_start: float = X_START, x_end: float = X_END, ini_boundary: Callable[..., Any] = INITIAL_BOUNDAR, simulation_time: float = SIMULATION_TIME) -> None:
        super().__init__(n_internal, n_initial, n_boundary, device, update_data, seed, x_start, x_end, ini_boundary, simulation_time)
    
    def _sample_initial(self):
        with torch.no_grad():
            x_initial = (torch.rand(self.n_initial,device=self.device))*(self.x_end-self.x_start)+self.x_start
            t_initial = torch.zeros_like(x_initial)
            value_initial = self.ini_boundary(x_initial)
            return x_initial,t_initial,value_initial
        
    def _sample_boundary(self):
        with torch.no_grad():
            x_boundary = torch.cat([self.x_start*torch.ones(self.n_boundary//2,device=self.device),self.x_end*torch.ones(self.n_boundary//2,device=self.device)])
            t_boundary = torch.rand(self.n_boundary,device=self.device)
            value_boundary = torch.zeros_like(t_boundary,device=self.device)
            return x_boundary,t_boundary,value_boundary
    
    def _sample_internal(self):
        x=torch.rand(self.n_internal,device=self.device)*(self.x_end-self.x_start)+self.x_start
        x.requires_grad=True
        t=torch.rand(self.n_internal,device=self.device,requires_grad=True)   
        return x,t
    

def BurgersSampler(n_internal:int,n_initial:int,n_boundary:int,device:Union[str,torch.device],
                 update_data:bool=False,seed:int=21339,
                 data_sampler:Literal["latin_hypercube","monte_carlo"]="latin_hypercube",
                 x_start:float=X_START,x_end:float=X_END,ini_boundary:Callable=INITIAL_BOUNDAR,simulation_time:float=SIMULATION_TIME):
    """
    A function that samples data for the Burgers equation.

    Parameters:
    - n_internal (int): Number of internal points.
    - n_initial (int): Number of initial points.
    - n_boundary (int): Number of boundary points.
    - device: The device to be used for computation.
    - use_LHS (bool): Flag indicating whether to use Latin Hypercube Sampling (LHS) or Monte Carlo Sampling (MC).
    - update_data (bool): Flag indicating whether to update existing data.
    - seed (int): Seed for random number generation.
    - x_start: Start value for x.
    - x_end: End value for x.
    - ini_boundary: Initial boundary value.
    - simulation_time: Simulation time.

    Returns:
    - The data sampler for Burgers equation. the sampler has two key methods:
        - sample_initial_boundary: A method that samples initial and boundary points.
        - sample_internal: A method that samples internal points.
    """
    if data_sampler=="latin_hypercube":
        return BurgersSamplerLH(n_internal,n_initial,n_boundary,device,update_data,seed,x_start,x_end,ini_boundary,simulation_time)
    else:
        return BurgersSamplerMC(n_internal,n_initial,n_boundary,device,update_data,seed,x_start,x_end,ini_boundary,simulation_time)

class BurgersValidationDataSet():
    
    def __init__(self,dataset_path:str,xs:np.ndarray=X_TEST,ts:np.ndarray=T_TEST,) -> None:
        self.simulation_data=np.load(dataset_path)
        self.xs=xs
        self.ts=ts
        
class BurgersValidationDataLoader():
    
    def __init__(self,validation_dataset:BurgersValidationDataSet,device:Union[str,torch.device]="cuda:0") -> None:
        self.fake_data=[0]
        self.simulation_data=torch.from_numpy(validation_dataset.simulation_data).float().to(device)
        self.xs=torch.from_numpy(validation_dataset.xs).float().to(device).unsqueeze(0).repeat(self.simulation_data.shape[0],1)
        self.ts=torch.from_numpy(validation_dataset.ts).float().to(device).unsqueeze(1).repeat(1,self.simulation_data.shape[1])

    def __len__(self):
        return 1
    
    def __getitem__(self,idx):
        return self.fake_data[idx]
