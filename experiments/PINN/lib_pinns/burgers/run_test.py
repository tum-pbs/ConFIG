import torch
from .simulation_paras import *

def run_test(network,simulation_data,n_t=N_T,n_x=N_X,x_test=X_TEST,t_test=T_TEST,device="cuda"):
    network.to(device)
    network.eval()
    with torch.no_grad():
        xs_test_torch=torch.from_numpy(x_test).float().to(device).unsqueeze(0).repeat(n_t,1)
        ts_test_torch=torch.from_numpy(t_test).float().to(device).unsqueeze(1).repeat(1,n_x)
        prediction=network(xs_test_torch,ts_test_torch)
        prediction=prediction.detach().cpu().numpy()
    mse=(simulation_data-prediction)**2
    mse_value=np.mean(mse)
    return mse_value,mse,prediction


def validation(network,simulation_data,n_t=N_T,n_x=N_X,x_test=X_TEST,t_test=T_TEST,device="cuda"):
    network.eval()
    with torch.no_grad():
        xs_test_torch=torch.from_numpy(x_test).float().to(device).unsqueeze(0).repeat(n_t,1)
        ts_test_torch=torch.from_numpy(t_test).float().to(device).unsqueeze(1).repeat(1,n_x)
        prediction=network(xs_test_torch,ts_test_torch)
        prediction=prediction
        mse=((torch.tensor(simulation_data,device=device)-prediction)**2).mean()
    return mse
