import torch
from .simulation_paras import *
from .data_sampler import BeltramiValidationDataSet

def run_test(network,
             x_start:float=X_START,x_end=X_END,
                 y_start=Y_START,y_end=Y_END,
                 z_start=Z_START,z_end=Z_END,
                 simulation_time=SIMULATION_TIME,
                 n_t=5,
                 n_point=2001,device="cuda:0",return_ground_truth=False):
    network.eval()
    network.to(device)
    dataset=BeltramiValidationDataSet(x_start=x_start,
                                        x_end=x_end,
                                        y_start=y_start,
                                        y_end=y_end,
                                        z_start=z_start,
                                        z_end=z_end,
                                        simulation_time=simulation_time,
                                        n_t=n_t,
                                        n_point=n_point,
                                      )
    with torch.no_grad():
        prediction=network(torch.from_numpy(dataset.x).float().to(device),
                           torch.from_numpy(dataset.y).float().to(device),
                           torch.from_numpy(dataset.z).float().to(device),
                           torch.from_numpy(dataset.t).float().to(device)
                            )
        prediction=prediction.detach().cpu().numpy()
    mse=(dataset.uvwp-prediction)**2
    mse_value=np.mean(mse)
    shape=dataset.mesh_shape+(4,)
    if return_ground_truth:
        return mse_value,mse.reshape(shape),prediction.reshape(shape),dataset.uvwp.reshape(shape)
    else:
        return mse_value,mse.reshape(shape),prediction.reshape(shape)
