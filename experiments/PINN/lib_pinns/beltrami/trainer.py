
import torch

from .data_sampler import BeltramiSampler,BeltramiValidationDataLoader,BeltramiValidationDataSet
from .networks import *
from .simulation_paras import *
from .physical_residual import *
from .run_test import *

from ..trainer_basis import *



class BeltramiTrainerBasis():
        
    def set_configs_type_dataloader(self):

        self.configs_handler.add_config_item("n_internal",mandatory=True,value_type=int,
                                             description="Num of samples in internal domain")
        self.configs_handler.add_config_item("n_boundary",mandatory=True,value_type=int,
                                             description="Num of samples for boundary conditions")
        self.configs_handler.add_config_item("n_initial",mandatory=True,value_type=int,
                                             description="Num of samples for initial conditions")
        self.configs_handler.add_config_item("x_start",mandatory=True,value_type=float,
                                             description="Start of x axis")
        self.configs_handler.add_config_item("x_end",mandatory=True,value_type=float,
                                             description="End of x axis")
        self.configs_handler.add_config_item("y_start",mandatory=True,value_type=float,
                                             description="Start of y axis")
        self.configs_handler.add_config_item("y_end",mandatory=True,value_type=float,
                                             description="End of y axis")
        self.configs_handler.add_config_item("z_start",mandatory=True,value_type=float,
                                             description="Start of z axis")
        self.configs_handler.add_config_item("z_end",mandatory=True,value_type=float,
                                             description="End of z axis")
        self.configs_handler.add_config_item("simulation_time",mandatory=True,value_type=float,
                                             description="Simulation time")
    
    def generate_dataloader(self, train_dataset, validation_dataset):
        train_dataloader=BeltramiSampler(n_internal=self.configs.n_internal,
                                          n_boundary=self.configs.n_boundary,
                                          n_initial=self.configs.n_initial,
                                          device=self.configs.device,
                                          update_data=self.configs.update_training_data,
                                          seed=self.configs.random_seed,
                                          data_sampler=self.configs.data_sampler,
                                          x_start=self.configs.x_start,
                                          x_end=self.configs.x_end,
                                          y_start=self.configs.y_start,
                                          y_end=self.configs.y_end,
                                          z_start=self.configs.z_start,
                                          z_end=self.configs.z_end,
                                          simulation_time=self.configs.simulation_time)
        if validation_dataset is not None:
            vali_dataloader=BeltramiValidationDataLoader(validation_dataset,
                                                          device=self.configs.device)
        else:
            vali_dataloader=None
        return train_dataloader,vali_dataloader
        
    def validation_step(self, network, batched_data, idx_batch: int, num_batches: int, idx_epoch: int, num_epoch: int):  
        prediction=network(self.validate_dataloader.xs,
                           self.validate_dataloader.ys,
                           self.validate_dataloader.zs,
                           self.validate_dataloader.ts)
        return ((self.validate_dataloader.uvwps-prediction)**2).mean()
    
    def get_internal_loss(self,network,idx_epoch:int):
        x_internal,y_internal,z_internal,t_internal = self.train_dataloader.sample_internal()
        loss_mx,loss_my,loss_mz,loss_c=physical_residual(network(x_internal,y_internal,z_internal,t_internal),
                                                         x_internal,y_internal,z_internal,t_internal)
        loss_mx=torch.mean(loss_mx**2)
        loss_my=torch.mean(loss_my**2)
        loss_mz=torch.mean(loss_mz**2)
        loss_c=torch.mean(loss_c**2)
        internal_loss=loss_mx+loss_my+loss_mz+loss_c
        self.recorder.add_scalar("separated_loss/momentum_x",loss_mx.item(),idx_epoch)
        self.recorder.add_scalar("separated_loss/momentum_y",loss_my.item(),idx_epoch)
        self.recorder.add_scalar("separated_loss/momentum_z",loss_mz.item(),idx_epoch)
        self.recorder.add_scalar("separated_loss/continuity",loss_c.item(),idx_epoch)
        self.recorder.add_scalar("separated_loss/internal_loss",internal_loss.item(),idx_epoch)
        return internal_loss

    def get_bound_ini_loss(self,network,idx_epoch:int): 
        x_bi,y_bi,z_bi,t_bi,uvwp_bi=self.train_dataloader.sample_initial_boundary()
        bound_ini_loss=torch.mean((network(x_bi,y_bi,z_bi,t_bi)-uvwp_bi)**2)
        self.recorder.add_scalar("separated_loss/bound_ini_loss",bound_ini_loss.item(),idx_epoch)
        return bound_ini_loss

    def get_boundary_loss(self,network,idx_epoch:int): 
        x_b,y_b,z_b,t_b,uvwp_b=self.train_dataloader.sample_boundary()
        boundary_loss=torch.mean((network(x_b,y_b,z_b,t_b)-uvwp_b)**2)
        self.recorder.add_scalar("separated_loss/boundary_loss",boundary_loss.item(),idx_epoch)   
        return boundary_loss

    def get_initial_loss(self,network,idx_epoch:int):
        x_i,y_i,z_i,t_i,uvwp_i= self.train_dataloader.sample_initial()
        initial_loss=torch.mean((network(x_i,y_i,z_i,t_i)-uvwp_i)**2)
        self.recorder.add_scalar("separated_loss/initial_loss",initial_loss.item(),idx_epoch)
        return initial_loss
    
    def _loss_funcs(self):
        if self.configs.n_losses==2:
            return [self.get_internal_loss,self.get_bound_ini_loss]
        else:
            return [self.get_internal_loss,self.get_boundary_loss,self.get_initial_loss]
# Parent classes

class StandardTrainer(BeltramiTrainerBasis,StandardTrainerBasis):
    pass

def training(epochs,trainer:BeltramiTrainerBasis,random_seeds:Sequence,
                 path_config_file,name,num_run=3,device="cuda:0",n_validation_point=2001,**kwargs):
    errors=[]
    for i in range(num_run):
        set_random_seed(random_seeds[i])
        network=BeltramiNet()
        trainer.train_from_scratch(
            network=network,
            train_dataset=None,
            validation_dataset=BeltramiValidationDataSet(n_point=n_validation_point),
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
        mse_loss,_,_=run_test(network,n_point=n_validation_point)
        errors.append(mse_loss)
        print("mse_loss:%.5e" % mse_loss)
    mean_error=np.mean(errors)
    std_error=np.std(errors)
    print("mse_loss:%.5e±%.5e" % (mean_error,std_error))
    return errors,mean_error,std_error

def run_beltrami(
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
        equation_name="beltrami",
        training_func=training,
        trainer=trainer,
        epochs=epochs,
        num_run=num_run,
        save_path=save_path,
        config_file=config_file,
        update_training_data=update_training_data,
        **kwargs
    )