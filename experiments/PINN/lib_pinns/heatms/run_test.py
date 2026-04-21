import torch
from .simulation_paras import *
from .data_sampler import HeatMSValidationDataSet

def run_test(network,
             dataset:HeatMSValidationDataSet,
             device="cuda:0",
             return_ground_truth=False):
    with torch.no_grad():
        prediction=network(dataset.x.to(device),dataset.t.to(device))
    mse=(dataset.u.to(device)-prediction)**2
    mse_value=torch.mean(mse).item()
    if return_ground_truth:
        return mse_value,mse,prediction,dataset.u
    else:
        return mse_value,mse,prediction
