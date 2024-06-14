import torch
from .simulation_paras import *
from .data_sampler import KovasznayValidationDataSet

def run_test(network,x_start:float=X_START,
                 x_end=X_END,y_start=Y_START,y_end=Y_END,n_point=1001,device="cuda:0",return_ground_truth=False):
    network.eval()
    network.to(device)
    dataset=KovasznayValidationDataSet(x_start=x_start,x_end=x_end,y_start=y_start,y_end=y_end,n_point=n_point)
    with torch.no_grad():
        prediction=network(torch.from_numpy(dataset.x).float().to(device),torch.from_numpy(dataset.y).float().to(device))
        prediction=prediction.detach().cpu().numpy()
    mse=(dataset.uvp-prediction)**2
    mse_value=np.mean(mse)
    if return_ground_truth:
        return mse_value,mse,prediction,dataset.uvp
    else:
        return mse_value,mse,prediction
