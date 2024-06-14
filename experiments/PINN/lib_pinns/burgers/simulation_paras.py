import numpy as np
import torch

X_START=-1.0
X_END=1.0
SIMULATION_TIME=1.0
NU=0.01/np.pi
N_T=100
N_X=256

DOMAIN_LENGTH=X_END-X_START
DX=DOMAIN_LENGTH/N_X
X_TEST=np.linspace(X_START+DX/2,X_END-DX/2,N_X)
T_TEST =np.linspace(0,SIMULATION_TIME,N_T)

INITIAL_BOUNDAR= lambda x: -1*torch.sin(np.pi*x)