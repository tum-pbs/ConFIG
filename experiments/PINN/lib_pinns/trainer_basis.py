import torch
from .foxutils.trainerX import *
from torch.optim.optimizer import Optimizer as Optimizer
from .network_initialization import *
from .helpers import *
from .loss_weighter import *

from conflictfree.grad_operator import *
from conflictfree.momentum_operator import *

def get_cosine_constant_lambda(initial_lr,final_lr,epochs,warmup_epoch,constant_start_epoch):
    """
    Returns a lambda function that calculates the learning rate based on the cosine schedule.

    Args:
        initial_lr (float): The initial learning rate.
        final_lr (float): The final learning rate.
        epochs (int): The total number of epochs.
        warmup_epoch (int): The number of warm-up epochs.

    Returns:
        function: The lambda function that calculates the learning rate.
    """
    cos_f=get_cosine_lambda(initial_lr,final_lr,constant_start_epoch,warmup_epoch)
    def cosine_constant_lambda(idx_epoch):
        if idx_epoch<constant_start_epoch:
            return cos_f(idx_epoch)
        else:
            return final_lr/initial_lr
    return cosine_constant_lambda

class TrainerBasis(Trainer):
    def __init__(self) -> None:
        super().__init__()
        
    def set_configs_type(self):
        super().set_configs_type()
        self.configs_handler.add_config_item("bound_ini_normalize_base",mandatory=False,
                                             default_value="none",value_type=str,option=["none","minimum","maximum"],
                                             description="Option for normalizing gradient vector.")
        self.configs_handler.add_config_item("normalize_base",mandatory=False,
                                             default_value="none",value_type=str,option=["none","internal","bound_ini","minimum","scale1"],
                                             description="Option for normalizing gradient vector.")
        self.configs_handler.add_config_item("update_training_data",mandatory=False,default_value=True,
                                             value_type=bool,description="Weather to update bound_ini samples during training.")
        self.configs_handler.add_config_item("data_sampler",default_value="latin_hypercube",value_type=str,
                                             description="Sampler for training bound_ini.",option=["latin_hypercube","monte_carlo"])
        self.configs_handler.add_config_item("network_initialization",default_value="xavier",value_type=str,
                                             description="Initialization method for the network.",option=["xavier","kaiming"])
        self.configs_handler.add_config_item("n_losses",default_value=2,value_type=int,
                                             description="Number of loss terms.")
        self.configs_handler.add_config_item("constant_start_epoch",default_value=100000,value_type=int,
                                             description="Epoch to start constant learning rate.")
        self.configs_handler.add_config_item("lr_scheduler",default_value="cosine",value_type=str,description="Learning rate scheduler for training",option=["cosine","linear","constant","cosine_constant"])
    
    def get_lr_scheduler(self,optimizer):
        """
        Get the learning rate scheduler based on the configuration.

        Args:
            optimizer (torch.optim.Optimizer): The optimizer.

        Returns:
            torch.optim.lr_scheduler._LRScheduler: The learning rate scheduler.
        
        Raises:
            ValueError: If the learning rate scheduler is not supported.
        """
        if self.configs.lr_scheduler=="cosine":
            return torch.optim.lr_scheduler.LambdaLR(optimizer,get_cosine_lambda(initial_lr=self.configs.lr,final_lr=self.configs.final_lr,epochs=self.configs.epochs,warmup_epoch=self.configs.warmup_epoch))
        elif self.configs.lr_scheduler=="linear":
            return torch.optim.lr_scheduler.LambdaLR(optimizer,get_linear_lambda(initial_lr=self.configs.lr,final_lr=self.configs.final_lr,epochs=self.configs.epochs,warmup_epoch=self.configs.warmup_epoch))
        elif self.configs.lr_scheduler=="constant":
            return torch.optim.lr_scheduler.LambdaLR(optimizer,get_constant_lambda(initial_lr=self.configs.lr,final_lr=self.configs.final_lr,epochs=self.configs.epochs,warmup_epoch=self.configs.warmup_epoch))
        elif self.configs.lr_scheduler=="cosine_constant":
            return torch.optim.lr_scheduler.LambdaLR(optimizer,get_cosine_constant_lambda(initial_lr=self.configs.lr,final_lr=self.configs.final_lr,epochs=self.configs.epochs,warmup_epoch=self.configs.warmup_epoch,constant_start_epoch=self.configs.constant_start_epoch))
        else:
            raise ValueError("Learning rate scheduler '{}' not supported".format(self.configs.lr_scheduler))
                
    def event_before_training(self, network):
        if self.configs.network_initialization=="xavier":
            network.apply(xavier_init_weights)
    
    def _loss_funcs(self):
        raise NotImplementedError("The method _loss_funcs should be implemented in the bound_iniclass.")          

    def get_losses(self,network,idx_epoch:int):
        return torch.stack([loss_f(network,idx_epoch) for loss_f in self._loss_funcs()])
        
    def get_gradients(self,network,idx_epoch:int,return_loss=False):
        loss_func=self._loss_funcs()
        grads=[]
        if return_loss:
            losses=[]
        for loss_f in loss_func:
            loss_i=loss_f(network,idx_epoch)
            self.optimizer.zero_grad()
            loss_i.backward()
            grads.append(get_gradient_vector(network))
            if return_loss:
                losses.append(loss_i)
        if return_loss:
            return torch.stack(grads,dim=0),torch.stack(losses)
        else:
            return torch.stack(grads,dim=0)
    
    def get_loss_i(self,idx_loss,network,idx_epoch:int):
        return self._loss_funcs()[idx_loss](network,idx_epoch)
    
    def get_gradient_i(self,idx_grad,network,idx_epoch:int,return_loss=False):
        loss_i=self._loss_funcs()[idx_grad](network,idx_epoch)
        self.optimizer.zero_grad()
        loss_i.backward()
        grad_i=get_gradient_vector(network)
        if return_loss:
            return grad_i,loss_i
        else:
            return grad_i

    def train_step(self, network, batched_data, idx_batch: int, num_batches: int, idx_epoch: int, num_epoch: int):
        raise NotImplementedError("The method train_step should be implemented in the bound_iniclass.")

class StandardTrainerBasis(TrainerBasis):
    
    def set_configs_type(self):
        super().set_configs_type()
        self.configs_handler.add_config_item("optimizer",default_value="Adam",value_type=str,description="Optimizer for training.",option=["Adam","AdaMod"])
    
    def train_step(self, network, batched_data, idx_batch: int, num_batches: int, idx_epoch: int, num_epoch: int):
        total_loss=torch.sum(self.get_losses(network,idx_epoch))
        return total_loss

class GradVecTrainerBasis(TrainerBasis):
    
    def set_configs_type(self):
        super().set_configs_type()
        self.configs_handler.add_config_item("optimizer",default_value="Adam",value_type=str,description="Optimizer for training.",option=["Adam","AdaMod"])
    
    def initialize_gradient_operator(self,operator:GradientOperator):
        self.operator=None
    
    def gradient_operation(self,grads,idx_epoch:int):
        raise NotImplementedError("The method train_step should be implemented in the bound_iniclass.")
    
    def event_before_training(self, network):
        super().event_before_training(network)
        self.initialize_gradient_operator()
    
    def train_step(self, network, batched_data, idx_batch: int, num_batches: int, idx_epoch: int, num_epoch: int):
        grads,losses=self.get_gradients(network,idx_epoch,return_loss=True)
        self.operator.update_gradient(network,grads,losses)
        self.optimizer.step()
        return torch.sum(losses)
  
    def back_propagate(self, loss: torch.Tensor, optimizer: torch.optim.Optimizer):
        pass


class GradPlusTrainerBasis(TrainerBasis):
    
    def set_configs_type(self):
        super().set_configs_type()
        self.configs_handler.add_config_item("optimizer",default_value="Adam",value_type=str,description="Optimizer for training.",option=["Adam","AdaMod"])
    
    def event_before_training(self, network):
        super().event_before_training(network)
    
    def train_step(self, network, batched_data, idx_batch: int, num_batches: int, idx_epoch: int, num_epoch: int):
        grads,losses=self.get_gradients(network,idx_epoch,return_loss=True)
        apply_gradient_vector(network,torch.sum(grads,dim=0))
        self.optimizer.step()
        return torch.sum(losses)
  
    def back_propagate(self, loss: torch.Tensor, optimizer: torch.optim.Optimizer):
        pass

class MomentumGradVecTrainerBasis(TrainerBasis):

    def set_configs_type(self):
        super().set_configs_type()
        self.configs_handler.add_config_item("optimizer",default_value="SGD",value_type=str,description="Optimizer for training.",option=["SGD"])

    def initialize_momentum_handler(self,network):
        self.momentum_handler=None
        raise NotImplementedError("The method train_step should be implemented in the bound_iniclass.")
    
    def event_before_training(self, network):
        super().event_before_training(network)
        self.pre_losses=torch.zeros(self.configs.n_losses)
        self.initialize_momentum_handler(network)
    
    def train_step(self, network, batched_data, idx_batch: int, num_batches: int, idx_epoch: int, num_epoch: int):
        idx_grad=idx_epoch%self.configs.n_losses
        grad_i,loss_i=self.get_gradient_i(idx_grad,network,idx_epoch,return_loss=True)
        self.pre_losses[idx_grad]=loss_i.detach()
        self.momentum_handler.update_gradient(network,idx_grad,grad_i)
        self.optimizer.step()
        return torch.sum(self.pre_losses)
    
    def back_propagate(self, loss: torch.Tensor, optimizer: torch.optim.Optimizer):
        pass   

class PerfectMomentumGradVecTrainerBasis(TrainerBasis):

    def set_configs_type(self):
        super().set_configs_type()
        self.configs_handler.add_config_item("optimizer",default_value="SGD",value_type=str,description="Optimizer for training.",option=["SGD"])

    def initialize_momentum_handler(self,network):
        self.momentum_handler=None
        raise NotImplementedError("The method train_step should be implemented in the bound_iniclass.")
    
    def event_before_training(self, network):
        super().event_before_training(network)
        self.initialize_momentum_handler(network)
    
    def train_step(self, network, batched_data, idx_batch: int, num_batches: int, idx_epoch: int, num_epoch: int):
        grads,losses=self.get_gradients(network,idx_epoch,return_loss=True)
        self.momentum_handler.update_gradient(grads,[i for i in range(self.configs.n_losses)])
        self.optimizer.step()
        return torch.sum(self.pre_losses)
    
    def back_propagate(self, loss: torch.Tensor, optimizer: torch.optim.Optimizer):
        pass   

class LRAWeightTrainerBasis(TrainerBasis):

    def set_configs_type(self):
        super().set_configs_type()
        self.configs_handler.add_config_item("optimizer",default_value="Adam",value_type=str,description="Optimizer for training.",option=["Adam","AdaMod"])
        self.configs_handler.add_config_item("separate_boundary_initial",default_value=False,value_type=bool,description="Whether to separate boundary and initial losses.")
        self.configs_handler.add_config_item("beta_LRA",default_value=0.999,value_type=float,description="Beta for exponential moving average in LRA weighter.")
    
    def event_before_training(self, network):
        super().event_before_training(network)
        self.weighter=LRAWeighter(self.configs.beta_LRA)
        
    def train_step(self, network, batched_data, idx_batch: int, num_batches: int, idx_epoch: int, num_epoch: int):
        grads,losses=self.get_gradients(network,idx_epoch,return_loss=True)
        weight=self.weighter.get_weights(grads=grads)
        self.recorder.add_scalar("losses/loss_before_weight",torch.sum(losses).item(),idx_epoch)
        apply_gradient_vector(network,torch.sum(weight.unsqueeze(1)*grads,dim=0))
        self.optimizer.step()
        return torch.sum(losses*weight)

    def back_propagate(self, loss: torch.Tensor, optimizer: torch.optim.Optimizer):
        pass

class ReLoWeightTrainerBasis(TrainerBasis):

    def set_configs_type(self):
        super().set_configs_type()
        self.configs_handler.add_config_item("optimizer",default_value="Adam",value_type=str,description="Optimizer for training.",option=["Adam","AdaMod"])
        self.configs_handler.add_config_item("separate_boundary_initial",default_value=False,value_type=bool,description="Whether to separate boundary and initial losses.")
        self.configs_handler.add_config_item("beta_ReLo",default_value=0.999,value_type=float,description="Beta for exponential moving average in ReLo weighter.")
        self.configs_handler.add_config_item("tau_ReLo",default_value=0.1,value_type=float,description="Tau for softmax in ReLo weighter.")
        self.configs_handler.add_config_item("rou_ReLo",default_value=0.999,value_type=float,description="Rou for binomial in ReLo weighter.")
    
    def event_before_training(self, network):
        super().event_before_training(network)
        self.weighter=ReLoWeighter(beta=self.configs.beta_ReLo,tau=self.configs.tau_ReLo,rou=self.configs.rou_ReLo)
        
    def train_step(self, network, batched_data, idx_batch: int, num_batches: int, idx_epoch: int, num_epoch: int):
        losses=self.get_losses(network,idx_epoch)
        self.recorder.add_scalar("losses/loss_before_weight",torch.sum(losses).item(),idx_epoch)
        weight=self.weighter.get_weights(losses=losses)
        return torch.sum(weight*losses)

class MinMaxWeightTrainerBasis(TrainerBasis):
    def set_configs_type(self):
        super().set_configs_type()
        self.configs_handler.add_config_item("optimizer",default_value="Adam",value_type=str,description="Optimizer for training.",option=["Adam","AdaMod"])
        self.configs_handler.add_config_item("separate_boundary_initial",default_value=False,value_type=bool,description="Whether to separate boundary and initial losses.")   
        self.configs_handler.add_config_item("lr_MinMax",default_value=0.001,value_type=float,description="Learning rate for MinMaxWeighter.")
        self.configs_handler.add_config_item("use_adam_MinMax",default_value=True,value_type=bool,description="Whether to use Adam for MinMaxWeighter.")

    def event_before_training(self, network):
        super().event_before_training(network)
        self.weighter=MinMaxWeighter(lr=self.configs.lr_MinMax,use_adam=self.configs.use_adam_MinMax)
        
    def train_step(self, network, batched_data, idx_batch: int, num_batches: int, idx_epoch: int, num_epoch: int):
        losses=self.get_losses(network,idx_epoch)
        self.recorder.add_scalar("losses/loss_before_weight",torch.sum(losses).item(),idx_epoch)
        weight=self.weighter.get_weights(losses=losses)
        return torch.sum(weight*losses)      


def get_gradvec_trainer(sub_trainer,operator):
    class GradVecTrainer(GradVecTrainerBasis):
        def initialize_gradient_operator(self):
            self.operator=operator
    class Trainer(sub_trainer,GradVecTrainer):
        pass
    return Trainer()

def get_momentum_trainer(sub_trainer,operator):
    class MomentumTrainer(MomentumGradVecTrainerBasis):
        def initialize_momentum_handler(self,network):
            self.momentum_handler=PseudoMomentumOperator(num_vectors=self.configs.n_losses,
                                                         network=network,
                                                         gradient_operator=operator,
                                                         loss_recorder=LatestLossRecorder(self.configs.n_losses))
    class Trainer(sub_trainer,MomentumTrainer):
        pass
    return Trainer()

def get_separate_momentum_trainer(sub_trainer,operator):
    class MomentumTrainer(MomentumGradVecTrainerBasis):
        def initialize_momentum_handler(self,network):
            self.momentum_handler=SeparateMomentumOperator(num_vectors=self.configs.n_losses,
                                                network=network,
                                                gradient_operator=operator,
                                                loss_recorder=LatestLossRecorder(self.configs.n_losses))
    class Trainer(sub_trainer,MomentumTrainer):
        pass
    return Trainer()

def get_perfect_momentum_trainer(sub_trainer,operator):
    class PerfectMomentumTrainer(PerfectMomentumGradVecTrainerBasis):
        def initialize_momentum_handler(self,network):
            self.momentum_handler=PseudoMomentumOperator(num_vectors=self.configs.n_losses,
                                                         network=network,
                                                         gradient_operator=operator,
                                                         loss_recorder=LatestLossRecorder(self.configs.n_losses))
    class Trainer(sub_trainer,PerfectMomentumTrainer):
        pass
    return Trainer()

def get_LRAWeight_trainer(sub_trainer):
    class Trainer(sub_trainer,LRAWeightTrainerBasis):
        pass
    return Trainer()

def get_ReLoWeight_trainer(sub_trainer):
    class Trainer(sub_trainer,ReLoWeightTrainerBasis):
        pass
    return Trainer()

def get_MinMaxWeight_trainer(sub_trainer):
    class Trainer(sub_trainer,MinMaxWeightTrainerBasis):
        pass
    return Trainer()


def get_plus_trainer(sub_trainer):
    class Trainer(sub_trainer,GradPlusTrainerBasis):
        pass
    return Trainer()

def run_training(
    run_name,
    equation_name,
    training_func,
    trainer,
    epochs=100000,
    num_run=3,
    save_path=None,
    config_file=None,
    update_training_data=True,
    **kwargs
):

    set_random_seed(21339)
    random_seeds=[]
    seed_path=package_path()+"random_seed.txt"
    if not os.path.exists(seed_path):    
        print("Warning: random_seed.txt not found, generating random seeds.")
        np.savetxt(seed_path,np.random.randint(9999, 99999, (100,)), fmt='%d')
    with open(seed_path,"r") as f:
        for line in f.readlines():
            random_seeds.append(int(line.strip()))
    f.close()
    
    print(f"Running {equation_name}:{run_name}")
    if save_path is None:
        save_path="./PINN_trained/"+equation_name+"/"
    if config_file is None:
        config_file=package_path()+f"training_configs/{equation_name}.yaml"
        
    training_func(
        epochs=epochs,
        trainer=trainer,
        path_config_file=config_file,
        save_path=save_path,
        update_training_data=update_training_data,
        name=run_name,
        num_run=num_run,
        random_seeds=random_seeds,
        **kwargs
    )
