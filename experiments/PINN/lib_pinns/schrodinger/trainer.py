import torch

from .data_sampler import SchrodingerSampler,SchrodingerValidationDataSet,SchrodingerValidationDataLoader
from .simulation_paras import *
from .physical_residual import *
from .run_test import *
from .networks import *

from ..trainer_basis import *

class SchrodingerTrainerBasis():
    
    def set_configs_type_dataloader(self):
        self.configs_handler.add_config_item("n_internal",mandatory=True,value_type=int,description="Num of samples in internal domain")
        self.configs_handler.add_config_item("n_boundary",mandatory=True,value_type=int,description="Num of samples for boundary conditions")
        self.configs_handler.add_config_item("n_initial",mandatory=True,value_type=int,description="Num of samples for initial conditions")
        self.configs_handler.add_config_item("x_start",mandatory=True,value_type=float,description="Start of x axis")
        self.configs_handler.add_config_item("x_end",mandatory=True,value_type=float,description="End of x axis")
        self.configs_handler.add_config_item("simulation_time",mandatory=True,value_type=float,description="simulation time of burgers equation")
        
    def generate_dataloader(self, train_dataset, validation_dataset):
        train_dataloader= SchrodingerSampler(n_internal=self.configs.n_internal,n_initial=self.configs.n_initial,
                n_boundary_left=self.configs.n_boundary//2,n_boundary_right=self.configs.n_boundary//2,
                device=self.configs.device,
                data_sampler=self.configs.data_sampler,
                x_start=self.configs.x_start,x_end=self.configs.x_end,simulation_time=self.configs.simulation_time,
                seed=self.configs.random_seed,
                update_data=self.configs.update_training_data,
        )
        if validation_dataset is not None:
            vali_dataloader=SchrodingerValidationDataLoader(validation_dataset,device=self.configs.device)
        else:
            vali_dataloader=None
        return train_dataloader,vali_dataloader
    
    def validation_step(self, network, batched_data, idx_batch: int, num_batches: int, idx_epoch: int, num_epoch: int):
        prediction=network(self.validate_dataloader.xs,self.validate_dataloader.ts)
        prediction=prediction.reshape((self.validate_dataloader.hs.shape[0],self.validate_dataloader.hs.shape[1],2))
        u_prediction=prediction[:,:,0]
        v_prediction=prediction[:,:,1]
        h_mag= torch.sqrt(u_prediction**2+v_prediction**2)
        self.recorder.add_scalar("separated_vloss/mag",torch.nn.functional.mse_loss(h_mag,self.validate_dataloader.hsm),idx_epoch)
        self.recorder.add_scalar("separated_vloss/u",torch.nn.functional.mse_loss(u_prediction,self.validate_dataloader.us),idx_epoch)
        self.recorder.add_scalar("separated_vloss/v",torch.nn.functional.mse_loss(v_prediction,self.validate_dataloader.vs),idx_epoch)
        return torch.nn.functional.mse_loss(prediction,self.validate_dataloader.hs)
    
    def get_internal_loss(self,network,idx_epoch:int):
        x_int,t_int=self.train_dataloader.sample_internal()
        internal_loss= squared_physical_residual_internal(network(x_int,t_int),x_int,t_int).mean()
        self.recorder.add_scalar("separated_loss/internal_loss",internal_loss.item(),idx_epoch)
        return internal_loss
    
    def get_bound_ini_loss(self,network,idx_epoch:int):
        x_left,t_left,x_right,t_right=self.train_dataloader.sample_boundary()
        boundary_loss_v,boundary_loss_f=squared_residual_boundary(network(x_left,t_left),network(x_right,t_right),x_left,x_right)
        boundary_loss_f=boundary_loss_f.mean()
        boundary_loss_v=boundary_loss_v.mean()
        self.recorder.add_scalar("separated_loss/boundary_loss_derivate",boundary_loss_f.item(),idx_epoch)
        self.recorder.add_scalar("separated_loss/boundary_loss_data",boundary_loss_v.item(),idx_epoch)
        x_ini,t_ini,h_ini=self.train_dataloader.sample_initial()
        initial_loss = squared_residual_initial(network(x_ini,t_ini),h_ini).mean()
        self.recorder.add_scalar("separated_loss/initial_loss",initial_loss.item(),idx_epoch) 
        #bound_ini_loss=boundary_loss_v+initial_loss+boundary_loss_f
        bound_ini_loss=boundary_loss_v+initial_loss
        self.recorder.add_scalar("separated_loss/bound_ini_loss",bound_ini_loss.item(),idx_epoch)
        return bound_ini_loss
    
    def get_boundary_loss(self,network,idx_epoch:int):
        x_left,t_left,x_right,t_right=self.train_dataloader.sample_boundary()
        boundary_loss_v,boundary_loss_f=squared_residual_boundary(network(x_left,t_left),network(x_right,t_right),x_left,x_right)
        boundary_loss_f=boundary_loss_f.mean()
        boundary_loss_v=boundary_loss_v.mean()
        self.recorder.add_scalar("separated_loss/boundary_loss_derivate",boundary_loss_f.item(),idx_epoch)
        self.recorder.add_scalar("separated_loss/boundary_loss_data",boundary_loss_v.item(),idx_epoch)
        #return boundary_loss_v+boundary_loss_f
        return boundary_loss_v
    
    def get_initial_loss(self,network,idx_epoch:int):
        x_ini,t_ini,h_ini=self.train_dataloader.sample_initial()
        initial_loss = squared_residual_initial(network(x_ini,t_ini),h_ini).mean()
        self.recorder.add_scalar("separated_loss/initial_loss",initial_loss.item(),idx_epoch) 
        return initial_loss    
    
    def _loss_funcs(self):
        if self.configs.n_losses==2:
            return [self.get_internal_loss,self.get_bound_ini_loss]
        else:
            return [self.get_internal_loss,self.get_boundary_loss,self.get_initial_loss]

class BaseLineTrainer(SchrodingerTrainerBasis,BaseLineTrainerBasis):
    pass

def training(epochs,trainer:SchrodingerTrainerBasis,
             random_seeds:Sequence,
             path_config_file,name,num_run=3,device="cuda:0",**kwargs):    
    simulation_path=package_path()+"data/schrodinger/NLS.mat"
    if not os.path.exists(simulation_path):
        raise FileNotFoundError("Simulation data not found.")
    vali_dataset=SchrodingerValidationDataSet(dataset_path=simulation_path)
    errors=[]
    for i in range(num_run):
        set_random_seed(random_seeds[i])
        network=SchrodingerNet()
        trainer.train_from_scratch(
            network=network,
            train_dataset=None,
            validation_dataset=vali_dataset,
            run_in_silence=True,
            path_config_file=path_config_file,
            name=name,
            epochs=epochs,
            warmup_epoch=min(100,int(epochs*0.01)),
            random_seed=random_seeds[i],
            device=device,
            **kwargs
            )
        network.eval()
        network.to(device)
        mse_loss,_,_,_=run_test(network,simulation_path,device=device)
        errors.append(mse_loss)
        print("mse_loss real:%.5e, mse_loss image:%.5e, mse_loss all:%.5e" % (mse_loss[0],mse_loss[1],mse_loss[2]))
    mean_error_real=np.mean([error[0] for error in errors])
    std_error_real=np.std([error[0] for error in errors])
    mean_error_image=np.mean([error[1] for error in errors])
    std_error_image=np.std([error[1] for error in errors])
    mean_error_all=np.mean([error[2] for error in errors])
    std_error_all=np.std([error[2] for error in errors])
    errors_real_image=[error[0]+error[1] for error in errors]
    mean_error_real_image=np.mean(errors_real_image)
    std_error_real_image=np.std(errors_real_image)
    print("mse_loss real:%.5e±%.5e" % (mean_error_real,std_error_real))
    print("mse_loss image:%.5e±%.5e" % (mean_error_image,std_error_image))
    print("mse_loss all:%.5e±%.5e" % (mean_error_all,std_error_all))
    print("mse_loss real+image:%.5e±%.5e" % (mean_error_real_image,std_error_real_image))
    return errors,mean_error_real_image,std_error_real_image

def run_schrodinger(
    name,
    trainer,
    epochs=100000,
    num_run=3,
    save_path=None,
    config_file=None,
    update_training_data=True,
    **kwargs):
    run_training(
        run_name=name,
        equation_name="schrodinger",
        training_func=training,
        trainer=trainer,
        epochs=epochs,
        num_run=num_run,
        save_path=save_path,
        config_file=config_file,
        update_training_data=update_training_data,
        **kwargs
    )