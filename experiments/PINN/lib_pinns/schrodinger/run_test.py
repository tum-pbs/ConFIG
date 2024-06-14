import torch
import numpy as np
import scipy

def run_test(network,dataset_path,device=None):
    if device is None:
        device=network.parameters().__next__().device
    simulation_data=scipy.io.loadmat(dataset_path)
    uu_gd = simulation_data['uu']
    u_gd = np.real(uu_gd)
    v_gd = np.imag(uu_gd)
    h_gd= np.sqrt(u_gd**2+v_gd**2)
    ground_truth_data=np.stack([u_gd,v_gd,h_gd],axis=0)
    x=simulation_data['x'].flatten()[:,None]
    t=simulation_data['tt'].flatten()[:,None]
    x,t=np.meshgrid(x.flatten(),t.flatten())    
    x=torch.from_numpy(x).float().to(device).T.reshape(-1)
    t=torch.from_numpy(t).float().to(device).T.reshape(-1)
    network.eval()
    with torch.no_grad():
        prediction=network(x,t)
        prediction=prediction.detach().cpu().numpy()
    prediction=prediction.reshape((ground_truth_data.shape[1],ground_truth_data.shape[2],2))
    u_prediction=prediction[:,:,0]
    v_prediction=prediction[:,:,1]
    h_prediction= np.sqrt(u_prediction**2+v_prediction**2)
    prediction=np.stack([u_prediction,v_prediction,h_prediction],axis=0)
    mse=(ground_truth_data-prediction)**2
    mse_loss=np.mean(mse,axis=(-1,-2))
    return mse_loss,mse,prediction,ground_truth_data

'''
def validation(network,ground_truth_data,x,t,device=None):
    if device is None:
        device=network.parameters().__next__().device
    x,t=np.meshgrid(x.flatten(),t.flatten())
    x=torch.from_numpy(x).float().to(device).T.reshape(-1)
    t=torch.from_numpy(t).float().to(device).T.reshape(-1)
    network.eval()
    with torch.no_grad():
        prediction=network(x,t)
        prediction=prediction
        prediction=prediction.reshape((ground_truth_data.shape[1],ground_truth_data.shape[2],2))
        u_prediction=prediction[:,:,0]
        v_prediction=prediction[:,:,1]
        h_prediction= torch.sqrt(u_prediction**2+v_prediction**2)
        ground_truth_data=torch.tensor(ground_truth_data,device=device)
        u_gd=ground_truth_data[0,:,:]
        v_gd=ground_truth_data[1,:,:]
        h_gd= torch.sqrt(u_gd**2+v_gd**2)
        mse=((h_gd-h_prediction)**2).mean()
    return mse
'''
