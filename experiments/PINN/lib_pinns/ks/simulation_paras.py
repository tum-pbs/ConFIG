import numpy as np
import torch

X_START=0.0
X_END=2*np.pi
SIMULATION_TIME=1.0
N_X=256
N_T=200
X_TEST=np.arange(0,N_X)/N_X*(X_END-X_START)
T_TEST=np.arange(0,N_T)/N_T
T_TEST=T_TEST+T_TEST[1]
INITIAL_CONDITION=lambda x: torch.cos(x)*(1+torch.sin(x))