import torch
from .simulation_paras import *
from .data_sampler import PNdValidationDataSet

def run_test(network,
             x_start:float=X_START,
             x_end=X_END,
             n_dim=N_DIM,
             n_point=1001,
             device="cuda:0",
             return_ground_truth=False):
    network.eval()
    network.to(device)
    dataset=PNdValidationDataSet(x_start=x_start,x_end=x_end,
                                 n_dim=n_dim,n_point=n_point)
    with torch.no_grad():
        prediction=network(dataset.x.to(device))
    mse=(dataset.u.to(device)-prediction)**2
    mse_value=torch.mean(mse).item()
    if return_ground_truth:
        return mse_value,mse,prediction,dataset.u
    else:
        return mse_value,mse,prediction
