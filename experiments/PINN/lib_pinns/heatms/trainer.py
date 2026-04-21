import torch
from .data_sampler import HeatMSSampler,HeatMSValidationDataLoader,HeatMSValidationDataSet
from .networks import *
from .simulation_paras import *
from .physical_residual import *
from .run_test import *

from ..trainer_basis import *

class HeatMSTrainerBasis():
        
    def set_configs_type_dataloader(self):

        self.configs_handler.add_config_item("n_internal",mandatory=True,value_type=int,
                                             description="Num of samples in internal domain")
        self.configs_handler.add_config_item("n_boundary",mandatory=True,value_type=int,
                                             description="Num of samples for boundary conditions")
        self.configs_handler.add_config_item("n_initial",mandatory=True,value_type=int,
                                             description="Num of samples for boundary conditions")
        self.configs_handler.add_config_item("x_start",mandatory=True,value_type=float,
                                             description="Start of x axis")
        self.configs_handler.add_config_item("x_end",mandatory=True,value_type=float,
                                             description="End of x axis")
        self.configs_handler.add_config_item("simulation_time",mandatory=True,value_type=float,
                                             description="Simulation time")
    
    def generate_dataloader(self, train_dataset, validation_dataset):
        train_dataloader=HeatMSSampler(n_internal=self.configs.n_internal,
                                          n_boundary=self.configs.n_boundary,
                                          n_initial=self.configs.n_initial,
                                          device=self.configs.device,
                                          update_data=self.configs.update_training_data,
                                          seed=self.configs.random_seed,
                                          data_sampler=self.configs.data_sampler,
                                          x_start=self.configs.x_start,
                                          x_end=self.configs.x_end,
                                          simulation_time=self.configs.simulation_time)
        if validation_dataset is not None:
            vali_dataloader=HeatMSValidationDataLoader(validation_dataset,device=self.configs.device)
        else:
            vali_dataloader=None
        return train_dataloader,vali_dataloader
        
    def validation_step(self, network, batched_data, idx_batch: int, num_batches: int, idx_epoch: int, num_epoch: int):  
        return ((self.validate_dataloader.u-network(self.validate_dataloader.x,self.validate_dataloader.t))**2).mean()
    
    def get_internal_loss(self,network,idx_epoch:int):
        x_internal,t_internal = self.train_dataloader.sample_internal()
        internal_loss=physical_residual(network(x_internal,t_internal),x_internal,t_internal).pow(2).mean()
        self.recorder.add_scalar("separated_loss/internal",internal_loss.item(),idx_epoch)
        return internal_loss

    def get_bound_loss(self,network,idx_epoch:int): 
        x_b,t_b,u_b=self.train_dataloader.sample_boundary()
        bound_loss=torch.mean((network(x_b,t_b)-u_b)**2)
        self.recorder.add_scalar("separated_loss/bound_loss",bound_loss.item(),idx_epoch)
        return bound_loss

    def get_initial_loss(self,network,idx_epoch:int):      
        x_i,t_i,value_i= self.train_dataloader.sample_initial()
        initial_loss=torch.mean((network(x_i,t_i)-value_i)**2)
        self.recorder.add_scalar("separated_loss/initial_loss",initial_loss.item(),idx_epoch)
        return initial_loss
    
    def get_bound_ini_loss(self,network,idx_epoch:int):
        x_b,t_b,u_b=self.train_dataloader.sample_boundary()
        bound_loss=torch.mean((network(x_b,t_b)-u_b)**2)
        self.recorder.add_scalar("separated_loss/bound_loss",bound_loss.item(),idx_epoch)
        x_i,t_i,value_i= self.train_dataloader.sample_initial()
        initial_loss=torch.mean((network(x_i,t_i)-value_i)**2)
        self.recorder.add_scalar("separated_loss/initial_loss",initial_loss.item(),idx_epoch)
        return bound_loss+initial_loss
      
    def _loss_funcs(self):
        if self.configs.n_losses==2:
            return [self.get_internal_loss,self.get_bound_ini_loss]
        else:
            return [self.get_internal_loss,self.get_bound_loss,self.get_initial_loss] 
         
# Parent classes
class StandardTrainer(HeatMSTrainerBasis,StandardTrainerBasis):
    pass

def training(epochs,trainer:HeatMSTrainerBasis,
             random_seeds:Sequence,
             path_config_file,name,num_run=3,
             device="cuda:0",**kwargs):
    simulation_path=package_path()+"data/heatms/simulation_data.npz"
    validataset=HeatMSValidationDataSet(simulation_path)
    errors=[]
    for i in range(num_run):
        set_random_seed(random_seeds[i])
        network=HeatMSNet()
        trainer.train_from_scratch(
            network=network,
            train_dataset=None,
            validation_dataset=validataset,
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
        mse_loss,_,_=run_test(network,validataset)
        errors.append(mse_loss)
        print("mse_loss:%.5e" % mse_loss)
    mean_error=np.mean(errors)
    std_error=np.std(errors)
    print("mse_loss:%.5e±%.5e" % (mean_error,std_error))
    return errors,mean_error,std_error


def run_heatms(
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
        equation_name="heatms",
        training_func=training,
        trainer=trainer,
        epochs=epochs,
        num_run=num_run,
        save_path=save_path,
        config_file=config_file,
        update_training_data=update_training_data,
        **kwargs
    )