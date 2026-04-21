import torch
import numpy as np  
from .simulation_paras import N_X,N_T,X_TEST,T_TEST

def run_test(network,
             simulation_data,device="cuda"):
    network.to(device)
    network.eval()
    with torch.no_grad():
        xs_test_torch=torch.from_numpy(X_TEST).float().to(device).unsqueeze(0).repeat(N_T,1)
        ts_test_torch=torch.from_numpy(T_TEST).float().to(device).unsqueeze(1).repeat(1,N_X)
        prediction=network(xs_test_torch,ts_test_torch)
        prediction=prediction.detach().cpu().numpy()
    mse=(simulation_data-prediction)**2
    mse_value=np.mean(mse)
    return mse_value,mse,prediction


def validation(network,
               simulation_data,
               device="cuda"):
    network.to(device)
    network.eval()
    with torch.no_grad():
        xs_test_torch=torch.from_numpy(X_TEST).float().to(device).unsqueeze(0).repeat(N_T,1)
        ts_test_torch=torch.from_numpy(T_TEST).float().to(device).unsqueeze(1).repeat(1,N_X)
        prediction=network(xs_test_torch,ts_test_torch)
        prediction=prediction
        mse=((torch.tensor(simulation_data,device=device)-prediction)**2).mean()
    return mse
