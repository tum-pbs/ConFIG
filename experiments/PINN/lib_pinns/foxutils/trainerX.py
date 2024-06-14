# usr/bin/python3

#version:0.0.18
#last modified:20240424
#TODO: loss prediction

import os,torch,time,math,logging,yaml
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
from tensorboard.backend.event_processing import event_accumulator
from torch.utils.data import DataLoader 
from .helper.coding import *
from .helper.network import *
import copy,numpy,random
from tqdm import tqdm


#NOTE: the range of the lambda output should be [0,1]                   
def get_cosine_lambda(initial_lr,final_lr,epochs,warmup_epoch):
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
    def cosine_lambda(idx_epoch):
        if idx_epoch < warmup_epoch:
            return idx_epoch / warmup_epoch
        else:
            return 1-(1-(math.cos((idx_epoch-warmup_epoch)/(epochs-warmup_epoch)*math.pi)+1)/2)*(1-final_lr/initial_lr)
    return cosine_lambda
    
def get_linear_lambda(initial_lr,final_lr,epochs,warmup_epoch):
    """
    Returns a lambda function that calculates the learning rate based on the linear schedule.

    Args:
        initial_lr (float): The initial learning rate.
        final_lr (float): The final learning rate.
        epochs (int): The total number of epochs.
        warmup_epoch (int): The number of warm-up epochs.

    Returns:
        function: The lambda function that calculates the learning rate.
    """
    def linear_lambda(idx_epoch):
        if idx_epoch < warmup_epoch:
            return idx_epoch / warmup_epoch
        else:
            return 1-((idx_epoch-warmup_epoch)/(epochs-warmup_epoch))*(1-final_lr/initial_lr)
    return linear_lambda

def get_constant_lambda(initial_lr,final_lr,epochs,warmup_epoch):
    """
    Returns a lambda function that calculates the learning rate based on the constant schedule.

    Args:
        initial_lr (float): Just a placeholder, no actual use.
        final_lr (float): Just a placeholder, no actual use.
        epochs (int): Just a placeholder, no actual use.
        warmup_epoch (int): The number of warm-up epochs.

    Returns:
        function: The lambda function that calculates the learning rate.
    """
    def constant_lambda(idx_epoch):
        if idx_epoch < warmup_epoch:
            return idx_epoch / warmup_epoch
        else:
            return 1
    return constant_lambda

class Trainer():
    """
    A class that handles the training process.

    Attributes:
        set_configs_type: Set the type of configurations.
        configs: The configurations for training.
        project_path: The path to the project folder.
        logger: The logger for logging training events.
        recorder: The recorder for recording training progress.
        train_dataloader: The dataloader for training data.
        validate_dataloader: The dataloader for validation data.
        start_epoch: The starting epoch for training.
        optimizer: The optimizer for updating model parameters.
        lr_scheduler: The learning rate scheduler.

    Methods:
        train_from_scratch: Train the network from scratch.
        train_from_checkpoint: Train the network from a checkpoint.
        show_config_options: Show the options of training configurations.
        show_current_configs: Show the current training configurations.
        get_optimizer: Get the optimizer based on the configuration. Recommend to override this method.
        get_lr_scheduler: Get the learning rate scheduler based on the configuration. Recommend to override this method.
        set_configs_type: Set the type of configurations. Recommend to override this method. Recommend to override this method.
        set configs_type_dataloader: Sets the configurations for the dataloader. Recommend to override this method if you want to use custom dataloader.
        train_step: The training step. Highly recommend to override this method.
        validation_step: The validation step. Highly recommend to override this method.
        back_propagate: Back propagate the loss to update the model parameters. Recommend to override this method.
        event_before_training: Event before training. Recommend to override this method.
        event_after_training: Event after training. Recommend to override this method.
        event_after_training_epoch: Event after training an epoch. Recommend to override this method.
        event_after_training_iteration: Event after training an iteration. Recommend to override this method.
        event_after_validation_epoch: Event after validating an epoch. Recommend to override this method.
        event_after_validation_iteration: Event after validating an iteration. Recommend to override this method.
        generate_dataloader: Generates dataloaders for training and validation datasets. Recommend to override this method if you want to use custom dataloader.
    """
    
    def __init__(self) -> None:
        self.set_configs_type()
        self._train_from_checkpoint=False
        self.configs=None
        self.project_path=""
        self.logger=None
        self.recorder=None
        self.train_dataloader=None
        self.validate_dataloader=None
        self.start_epoch=1
        self.optimizer=None
        self.lr_scheduler=None
        self.run_in_silence=False

    def __get_logger_recorder(self):
        """
        Get the logger and recorder for logging and recording training progress.

        Returns:
            logger: The logger for logging training events.
            recorder: The recorder for recording training progress.
        """
        logger=logging.getLogger("logger_{}".format(self.configs.name))
        logger.setLevel(logging.INFO)
        logger.handlers = []
        disk_handler = logging.FileHandler(filename=self.project_path+"training_event.log", mode='a')
        disk_handler.setFormatter(logging.Formatter(fmt="%(asctime)s : %(message)s", datefmt="%Y-%m-%d %H:%M:%S"))
        logger.addHandler(disk_handler)
        screen_handler = logging.StreamHandler()
        screen_handler.setFormatter(logging.Formatter(fmt="%(message)s"))
        if not self.run_in_silence:
            logger.addHandler(screen_handler)

        os.makedirs(self.records_path,exist_ok=True)
        recorder=SummaryWriter(log_dir=self.records_path)
        
        return logger,recorder

    def get_optimizer(self,network):
        """
        Get the optimizer based on the configuration. 
        Recommend to override this method. 
        When overriding this method, you do not need to call the parent method.

        Args:
            network (torch.nn.Module): The network to be optimized.

        Returns:
            torch.optim.Optimizer: The optimizer.
        
        Raises:
            ValueError: If the optimizer is not supported.
        """
        if self.configs.optimizer=="AdamW":
            return torch.optim.AdamW(network.parameters(),lr=self.configs.lr)
        elif self.configs.optimizer=="Adam":
            return torch.optim.Adam(network.parameters(),lr=self.configs.lr)
        elif self.configs.optimizer=="SGD":
            return torch.optim.SGD(network.parameters(),lr=self.configs.lr)
        else:
            raise ValueError("Optimizer '{}' not supported".format(self.configs.optimizer))

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
        else:
            raise ValueError("Learning rate scheduler '{}' not supported".format(self.configs.lr_scheduler))

    def __train(self,network:torch.nn.Module,train_dataset,validation_dataset=None):
        """
        The main training loop.

        Args:
            network (torch.nn.Module): The network to be trained.
            train_dataset: The dataset for training.
            validation_dataset: The dataset for validation (optional).
        """
        #set random seed  
        set_random_seed(self.configs.random_seed)
        training_time=0.0;start_time=time.time()
        # create project folder
        time_label = time.strftime("%Y-%m-%d-%H_%M_%S", time.localtime(time.time()))
        if not self._train_from_checkpoint:
            self.project_path=self.configs.save_path+self.configs.name+os.sep+time_label+os.sep
            os.makedirs(self.project_path, exist_ok=True)
        self.records_path=self.project_path+"records"+os.sep
        self.checkpoints_path=self.project_path+"checkpoints"+os.sep
        os.makedirs(self.checkpoints_path, exist_ok=True)
        #get logger and recorder
        self.logger,self.recorder=self.__get_logger_recorder()
        self.logger.info("Trainer created at {}".format(time_label))
        if self._train_from_checkpoint:
            self.logger.info("Training from checkpoint, checkpoint epoch:{}".format(self.start_epoch))
        self.logger.info("Working path:{}".format(self.project_path))
        self.logger.info("Random seed: {}".format(self.configs.random_seed))      
        # save configs if not train from checkpoint
        if not self._train_from_checkpoint:
            self.configs_handler.save_config_items_to_yaml(self.project_path+"configs.yaml")
        self.logger.info("Training configurations saved to {}".format(self.project_path+"configs.yaml"))
        # show model paras and save model structure
        self.logger.info("Network has {} trainable parameters".format(show_paras(network,print_result=False)))
        torch.save(network, self.project_path + "network_structure.pt")
        self.logger.info("Network structure saved to {}".format(self.project_path + "network_structure.pt"))
        # generate dataloader
        self.train_dataloader,self.validate_dataloader=self.generate_dataloader(train_dataset,validation_dataset)
        if self.validate_dataloader is not None:
            num_batches_validation=len(self.validate_dataloader)
        num_batches_train=len(self.train_dataloader)
        self.logger.info("There are {} training batches in each epoch".format(num_batches_train))
        if hasattr(self.configs,"batch_size_train"):
            self.logger.info("Batch size for training:{}".format(self.configs.batch_size_train))
        self.logger.info("Training epochs:{}".format(self.configs.epochs-self.start_epoch+1))
        self.logger.info("Total training iterations:{}".format(len(self.train_dataloader)*(self.configs.epochs-self.start_epoch+1)))
        # set optimizer and lr scheduler
        self.optimizer = self.get_optimizer(network)
        self.lr_scheduler = self.get_lr_scheduler(self.optimizer)
        self.logger.info("learning rate:{}".format(self.configs.lr))
        self.logger.info("Optimizer:{}".format(self.configs.optimizer))
        self.logger.info("Learning rate scheduler:{}".format(self.configs.lr_scheduler))
        if self.configs.warmup_epoch!=0:
            self.logger.info("Use learning rate warm up, warmup epoch:{}".format(self.configs.warmup_epoch))
        if self._train_from_checkpoint:
            checkpoint_file_path=self.checkpoints_path+"checkpoint_{}.pt".format(self.start_epoch-1)
            self.logger.info("Loading checkpoint from {}".format(checkpoint_file_path))
            checkpoint=torch.load(checkpoint_file_path)
            set_random_state(checkpoint["random_state"])
            network.load_state_dict(checkpoint["network"])
            self.optimizer.load_state_dict(checkpoint["optimizer"])
            self.lr_scheduler.load_state_dict(checkpoint["lr_scheduler"])
        # main train loop
        if self.configs.record_iteration_loss:
            loss_tag="Loss_epoch"
        else:
            loss_tag="Loss"
        losses_train_final_epochs=[]
        losses_validation_final_epochs=[]
        network.to(self.configs.device)
        self.event_before_training(network)
        self.logger.info("Training start!")
        p_bar=tqdm(range(self.start_epoch,self.configs.epochs+1))
        best_validation_loss=None
        best_validation_loss_epoch=None
        for idx_epoch in p_bar:
            train_losses_epoch=[]
            lr_now=self.optimizer.param_groups[0]["lr"]
            info_epoch="lr:{:.3e}".format(lr_now)
            self.recorder.add_scalar("learning_rate",lr_now,idx_epoch)
            network.train()
            for idx_batch,batched_data in enumerate(self.train_dataloader):
                time_now=time.time()
                loss = self.train_step(network=network, batched_data=batched_data,idx_batch=idx_batch,num_batches=num_batches_train,idx_epoch=idx_epoch,num_epoch=self.configs.epochs)
                self.back_propagate(loss,self.optimizer)
                self.event_after_training_iteration(network,idx_epoch,idx_batch)
                training_time+=time.time()-time_now
                if self.configs.record_epoch_loss or self.configs.record_iteration_loss:
                    train_losses_epoch.append(loss.item())
                if self.configs.record_iteration_loss:
                    self.recorder.add_scalar("Loss_iteration/train",train_losses_epoch[-1],(idx_epoch-1)*num_batches_train+idx_batch)
            if self.configs.record_epoch_loss:
                train_losses_epoch_average=sum(train_losses_epoch)/len(train_losses_epoch)
                if train_losses_epoch_average>1e-5:
                    info_epoch+=" train loss:{:.5f}".format(train_losses_epoch_average)
                else: 
                    info_epoch+=" train loss:{:.3e}".format(train_losses_epoch_average)
                self.recorder.add_scalar("{}/train".format(loss_tag),train_losses_epoch_average,idx_epoch)
            self.event_after_training_epoch(network,idx_epoch)
            if self.validate_dataloader is not None:
                if idx_epoch%self.configs.validation_epoch_frequency==0 or idx_epoch==self.configs.epochs:
                    validation_losses_epoch=[]
                    network.eval()
                    with torch.no_grad():
                        for idx_batch,batched_data in enumerate(self.validate_dataloader):
                            loss_validation=self.validation_step(network=network,batched_data=batched_data,idx_batch=idx_batch,num_batches=num_batches_validation,idx_epoch=idx_epoch,num_epoch=self.configs.epochs)
                            if self.configs.record_epoch_loss or self.configs.record_iteration_loss:
                                validation_losses_epoch.append(loss_validation.item())
                            if self.configs.record_iteration_loss:
                                self.recorder.add_scalar("Loss_iteration/validation",validation_losses_epoch[-1],(idx_epoch-1)*num_batches_validation+idx_batch)
                            self.event_after_validation_iteration(network,idx_epoch,idx_batch)
                        validation_losses_epoch_average=sum(validation_losses_epoch)/len(validation_losses_epoch)
                        if best_validation_loss is None:
                            best_validation_loss=sum(validation_losses_epoch)/len(validation_losses_epoch)
                            best_validation_loss_epoch=idx_epoch
                        elif validation_losses_epoch_average<best_validation_loss:
                            best_validation_loss=validation_losses_epoch_average
                            best_validation_loss_epoch=idx_epoch
                        if self.configs.record_epoch_loss:
                            if validation_losses_epoch_average>1e-5:
                                info_epoch+=" validation loss:{:.5f}".format(validation_losses_epoch_average)
                            else:
                                info_epoch+=" validation loss:{:.3e}".format(validation_losses_epoch_average)
                            self.recorder.add_scalar("{}/validation".format(loss_tag),validation_losses_epoch_average,idx_epoch)
                        self.event_after_validation_epoch(network,idx_epoch)
            p_bar.set_description(info_epoch)
            self.lr_scheduler.step()
            if idx_epoch%self.configs.save_epoch==0:
                checkpoint_now={
                    "epoch":idx_epoch,
                    "network":network.state_dict(),
                    "optimizer":self.optimizer.state_dict(),
                    "lr_scheduler":self.lr_scheduler.state_dict(),
                    "random_state":get_random_state()
                }
                torch.save(checkpoint_now,self.checkpoints_path+"checkpoint_{}.pt".format(idx_epoch))
            if self.configs.record_final_losses and idx_epoch>self.configs.epochs-self.configs.final_record_epoch:
                losses_train_final_epochs.append([idx_epoch,train_losses_epoch_average])
                if self.validate_dataloader is not None:
                    losses_validation_final_epochs.append([idx_epoch,validation_losses_epoch_average])
        self.event_after_training(network)
        network.to("cpu")
        torch.save(network.state_dict(),self.project_path+"trained_network_weights.pt")
        self.logger.info("Training finished!")
        total_time=time.time()-start_time
        self.logger.info("Final training loss: {}".format(train_losses_epoch_average))
        if self.validate_dataloader is not None:
            self.logger.info("Final validation loss: {}".format(validation_losses_epoch_average))
            self.logger.info("Best validation loss: {} at epoch {}".format(best_validation_loss,best_validation_loss_epoch))
        # record_loss 
        if self.configs.record_final_losses:
            losses=[t_loss[1] for t_loss in losses_train_final_epochs]
            self.logger.info(
                "Training losses of the final {} epochs: min:{:.5e}, max:{:.5e}, average:{:.5e}, median:{:.5e}".format(
                    len(losses),
                    min(losses),max(losses),
                    sum(losses)/len(losses),numpy.median(losses))
                )
            with open(self.project_path+"final_losses.txt","w") as f:
                if self.validate_dataloader is not None:
                    f.write("Epoch\tTrain loss\tValidation loss\n")
                    for i in range(len(losses_train_final_epochs)):
                        f.write("{}\t{}\t{}\n".format(losses_train_final_epochs[i][0],losses_train_final_epochs[i][1],losses_validation_final_epochs[i][1]))
                    v_losses=[v_loss[1] for v_loss in losses_validation_final_epochs]
                    self.logger.info(
                        "Validation losses of the final {} epochs: min:{:.5e}, max:{:.5e}, average:{:.5e}, median:{:.5e}".format(
                            len(v_losses),
                            min(v_losses),max(v_losses),
                            sum(v_losses)/len(v_losses),numpy.median(v_losses))
                        )
                else:
                    f.write("Epoch\tTrain loss\n")
                    for i in range(len(losses_train_final_epochs)):
                        f.write("{}\t{}\n".format(losses_train_final_epochs[i][0],losses_train_final_epochs[i][1]))
        self.logger.info("Total running time: {}".format(seconds_to_hms(total_time)))
        self.logger.info("Total training time: {}".format(seconds_to_hms(training_time)))
        self.logger.info("Training speed: {:.5f} s/iteration".format(training_time/(self.configs.epochs-self.start_epoch+1)/num_batches_train))

    def set_configs_type(self):
        '''
        Set the type of configurations. Supported training configurations can be shown through show_configs() function,
        Recommend to override this method.
        When overriding this method, you must to call the parent method.

        e.g.
        ```
        super().set_configs_type()
        self.configs_handler.add_config_item("training os",value_type=str,mandatory=True,description="Name of the training os.")
        ```

        '''
        self.configs_handler=ConfigurationsHandler()
        self.configs_handler.add_config_item("name",value_type=str,mandatory=True,description="Name of the training.")
        self.configs_handler.add_config_item("save_path",value_type=str,mandatory=True,description="Path to save the training results.")
        self.configs_handler.add_config_item("epochs",mandatory=True,value_type=int,description="Number of epochs for training.")
        self.configs_handler.add_config_item("lr",mandatory=True,value_type=float,description="Initial learning rate.")
        self.configs_handler.add_config_item("device",default_value="cpu",value_type=str,description="Device for training.",in_func=lambda x,other_config:torch.device(x),out_func=lambda x,other_config:str(x))
        self.configs_handler.add_config_item("random_seed",default_value_func=lambda x:int(time.time()),value_type=int,description="Random seed for training. Default is the same as batch_size_train.")# need func
        self.configs_handler.add_config_item("validation_epoch_frequency",default_value=1,value_type=int,description="Frequency of validation.")
        self.configs_handler.add_config_item("optimizer",default_value="AdamW",value_type=str,description="Optimizer for training.",option=["AdamW","Adam","SGD"])
        self.configs_handler.add_config_item("lr_scheduler",default_value="cosine",value_type=str,description="Learning rate scheduler for training",option=["cosine","linear","constant"])
        self.configs_handler.add_config_item("final_lr",default_value_func=lambda configs:configs["lr"],value_type=float,description="Final learning rate for lr_scheduler.")
        self.configs_handler.add_config_item("warmup_epoch",default_value=0,value_type=int,description="Number of epochs for learning rate warm up.")
        self.configs_handler.add_config_item("record_iteration_loss",default_value=False,value_type=bool,description="Whether to record iteration loss.")
        self.configs_handler.add_config_item("record_epoch_loss",default_value=True,value_type=bool,description="Whether to record epoch loss.")
        self.configs_handler.add_config_item("record_learning_rate",default_value=True,value_type=bool,description="Whether to record learning rate.")
        self.configs_handler.add_config_item("save_epoch",default_value_func=lambda configs:configs["epochs"]//10,value_type=int,description="Frequency of saving checkpoints.")
        self.configs_handler.add_config_item("record_final_losses",default_value=True,value_type=bool,description="Whether to record losses at the final period of training. The number of epochs can be set through `final_record_epoch`")
        self.configs_handler.add_config_item("final_record_epoch",default_value=100,value_type=int,description="Number of epochs to record at the end of training")
        self.set_configs_type_dataloader()
        
    def set_configs_type_dataloader(self):
        """
        Sets the configurations for the dataloader. Recommend to override this method if you want to use custom dataloader.

        Configurations:
        - batch_size_train: Batch size for training.
        - batch_size_validation: Batch size for validation. Default is the same as batch_size_train.
        - shuffle_train: Whether to shuffle the training dataset. Default is True.
        - shuffle_validation: Whether to shuffle the validation dataset. Default is the same as shuffle_train.
        - num_workers_train: Number of workers for training. Default is 0.
        - num_workers_validation: Number of workers for validation. Default is the same as num_workers_train.
        """
        
        self.configs_handler.add_config_item("batch_size_train",mandatory=True,value_type=int,description="Batch size for training.")
        self.configs_handler.add_config_item("batch_size_validation",default_value_func=lambda configs:configs["batch_size_train"],value_type=int,description="Batch size for validation. Default is the same as batch_size_train.")
        self.configs_handler.add_config_item("shuffle_train",default_value=True,value_type=bool,description="Whether to shuffle the training dataset.")
        self.configs_handler.add_config_item("shuffle_validation",default_value_func=lambda configs:configs["shuffle_train"],value_type=bool,description="Whether to shuffle the validation dataset. Default is the same as shuffle_train.")
        self.configs_handler.add_config_item("num_workers_train",default_value=0,value_type=int,description="Number of workers for training.")
        self.configs_handler.add_config_item("num_workers_validation",default_value_func=lambda configs:configs["num_workers_train"],value_type=int,description="Number of workers for validation. Default is the same as num_workers_train.")
        
    def generate_dataloader(self, train_dataset, validation_dataset):
        """
        Generates dataloaders for training and validation datasets. Recommend to override this method if you want to use custom dataloader.

        Args:
            train_dataset (torch.utils.data.Dataset): The training dataset.
            validation_dataset (torch.utils.data.Dataset): The validation dataset.

        Returns:
            train_dataloader (torch.utils.data.DataLoader): The dataloader for the training dataset.
            validate_dataloader (torch.utils.data.DataLoader): The dataloader for the validation dataset, or None if no validation dataset is provided.
        """
        def seed_worker(worker_id):
            worker_seed = torch.initial_seed() % 2**32
            numpy.random.seed(worker_seed)
            random.seed(worker_seed)

        dataloader_genrator = torch.Generator()
        dataloader_genrator.manual_seed(self.configs.random_seed)

        train_dataloader = DataLoader(train_dataset, 
                                      batch_size=self.configs.batch_size_train, 
                                      shuffle=self.configs.shuffle_train,
                                      num_workers=self.configs.num_workers_train,
                                      worker_init_fn=seed_worker,
                                      generator=dataloader_genrator
                                      )

        if validation_dataset is not None:
            validate_dataloader = DataLoader(validation_dataset, 
                                             batch_size=self.configs.batch_size_validation, 
                                             shuffle=self.configs.shuffle_validation,
                                             num_workers=self.configs.num_workers_validation,
                                             worker_init_fn=seed_worker,
                                             generator=dataloader_genrator
                                             )
            self.logger.info("Validation will be done every {} epochs".format(self.configs.validation_epoch_frequency))
            self.logger.info("Batch size for validation:{}".format(self.configs.batch_size_validation))
        else:
            validate_dataloader = None
            self.logger.info("No validation will be done")

        return train_dataloader, validate_dataloader
        
    def train_step(self,network:torch.nn.Module,batched_data,idx_batch:int,num_batches:int,idx_epoch:int,num_epoch:int):
        '''
        Train the network for one step.
        It is highly to recommend to override this method when developing a new trainer.
        When overriding this method, you don't need to call the parent method.
        Gradient will be automatically calculated in this operation.
        Note that you need to move the 'batched_data' to the device first when overriding this method.

        Args:
            network: torch.nn.Module, the network to be trained
            batched_data: torch.Tensor or tuple of torch.Tensor, the data for training.
            idx_batch: int, index of the current batch
            num_batches: int, total number of batches
            idx_epoch: int, index of the current epoch
            num_epoch: int, total number of epochs

        Returns:
            torch.Tensor: The loss of the current step.
        '''
        inputs=batched_data[0].to(self.configs.device)
        targets=batched_data[1].to(self.configs.device)
        predictions=network(inputs)
        loss=torch.nn.functional.mse_loss(predictions,targets)
        return loss
    
    def validation_step(self,network:torch.nn.Module,batched_data,idx_batch:int,num_batches:int,idx_epoch:int,num_epoch:int):
        '''
        Validate the network for one step.
        It is highly to recommend to override this method when developing a new trainer.
        When overriding this method, you don't need to call the parent method.
        All the operation here is gradient free.
        Note that you need to move the 'batched_data' to the device first when overriding this method.

        Args:
            network: torch.nn.Module, the network to be trained
            batched_data: torch.Tensor or tuple of torch.Tensor, the data for validation,need to be moved to the device first!
            idx_batch: int, index of the current batch
            num_batches: int, total number of batches
            idx_epoch: int, index of the current epoch
            num_epoch: int, total number of epochs
        
        Returns:
            torch.Tensor: The validation loss of the current step.
        '''
        inputs=batched_data[0].to(self.configs.device)
        targets=batched_data[1].to(self.configs.device)
        predictions=network(inputs)
        loss=torch.nn.functional.mse_loss(predictions,targets)
        return loss

    def train_from_scratch(self,network,train_dataset,validation_dataset=None,path_config_file:str="",run_in_silence=False,**kwargs):
        '''
        Train the network from scratch. Supported training configurations can be shown through show_configs() function.

        Args:
            network: torch.nn.Module, the network to be trained, mandatory
            train_dataset: torch.utils.data.Dataset, the training dataset, mandatory
            validation_dataset: torch.utils.data.Dataset, the validation dataset, default is None. If None, no validation will be done.
            path_config_file: str, path to the yaml file of the training configurations, default is ""
            kwargs: dict, the training configurations, default is {}, will overwrite the configurations in the yaml file.
        '''
        self.run_in_silence=run_in_silence
        self._train_from_checkpoint=False
        if path_config_file != "":
            self.configs_handler.set_config_items_from_yaml(path_config_file)
        self.configs_handler.set_config_items(**kwargs)
        self.configs=self.configs_handler.configs()
        self.__train(network,train_dataset,validation_dataset)

    def train_from_checkpoint(self,project_path,train_dataset,validation_dataset=None,restart_epoch=None,run_in_silence=False):
        '''
        Train the network from a checkpoint. The checkpoint should be in the project folder.
        Supported training configurations can be shown through show_configs() function.

        Args:
            project_path: str, path to the project folder, mandatory
            train_dataset: torch.utils.data.Dataset, the training dataset, mandatory
            validation_dataset: torch.utils.data.Dataset, the validation dataset, default is None. If None, no validation will be done.
            restart_epoch: int, the epoch to restart training, default is None, which means the latest checkpoint will be used.
        '''
        self.run_in_silence=run_in_silence
        self._train_from_checkpoint=True
        self.project_path=project_path
        # get checkpoint epoch
        if self.project_path[-1]!=os.sep:
            self.project_path+=os.sep
        if not os.path.exists(self.project_path+"configs.yaml"):
            print("No configs.yaml found in {}".format(self.project_path))
            print("Trying to use the latest subfolder as project path")
            dir_list = [folder_name for folder_name in os.listdir(self.project_path) if os.path.isdir(self.project_path+folder_name)]
            folder_name = sorted(dir_list,  key=lambda x: os.path.getmtime(os.path.join(self.project_path, x)))[-1]
            if not os.path.exists(self.project_path+folder_name+os.sep+"configs.yaml"):
                raise FileNotFoundError("No configs.yaml found in {}".format(self.project_path+folder_name+os.sep))
            self.project_path+=folder_name+os.sep
            print("Project path set to {}".format(self.project_path))
        if restart_epoch is None:
            check_points_names=os.listdir(self.project_path+"checkpoints")
            latest_check_point_name = sorted(check_points_names,  key=lambda x: os.path.getmtime(os.path.join(self.project_path+"checkpoints"+os.sep, x)))[-1]
            restart_epoch=int(latest_check_point_name.split("_")[-1].split(".")[0])
        # check files
        if not os.path.exists(self.project_path+"checkpoints"+os.sep+"checkpoint_{}.pt".format(restart_epoch)):
            raise FileNotFoundError("No 'checkpoint_{}.pt' found in {}".format(restart_epoch,self.project_path+"checkpoints"+os.sep))
        if not os.path.exists(self.project_path+"network_structure.pt"):
            raise FileNotFoundError("No network_structure.pt found in {}".format(self.project_path))
        # read configs and network
        self.configs_handler.set_config_items_from_yaml(self.project_path+"configs.yaml")
        self.configs=self.configs_handler.configs()
        network=torch.load(self.project_path+"network_structure.pt")
        self.start_epoch=restart_epoch+1
        self.__train(network,train_dataset,validation_dataset)

    def back_propagate(self,loss:torch.Tensor,optimizer:torch.optim.Optimizer):
        '''
        Back propagate the loss to update the model parameters.

        Args:
            loss: torch.Tensor, the loss to be back propagated.
            optimizer: torch.optim.Optimizer, the optimizer for updating model parameters.
        '''

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    def event_before_training(self,network):
        '''
        Event before training. Will do nothing by default.
        Recommend to override this method.
        When overriding this method, you don't need to call the parent method.

        Args:
            network: torch.nn.Module, the network to be trained.
        
        '''
        pass
    
    def event_after_training(self,network):
        '''
        Event after training. Will do nothing by default.
        Recommend to override this method.
        When overriding this method, you don't need to call the parent method.
        
        Args:
            network: torch.nn.Module, the network to be trained.
        
        '''
        pass
    
    def event_after_training_epoch(self,network,idx_epoch):
        '''
        Event after training an epoch. Will do nothing by default.
        Recommend to override this method.
        When overriding this method, you don't need to call the parent method.

        Args:
            network: torch.nn.Module, the network to be trained.
            idx_epoch: int, index of the current epoch.
        
        '''
        pass
    
    def event_after_training_iteration(self,network,idx_epoch,idx_batch):
        '''
        Event after training an iteration. Will do nothing by default.
        Recommend to override this method.
        When overriding this method, you don't need to call the parent method.

        Args:
            network: torch.nn.Module, the network to be trained.
            idx_epoch: int, index of the current epoch.
            idx_batch: int, index of the current batch.
        
        '''
        pass
    
    def event_after_validation_epoch(self,network,idx_epoch):
        '''
        Event after validating an epoch. Will do nothing by default.
        Recommend to override this method.
        When overriding this method, you don't need to call the parent method.

        Args:
            network: torch.nn.Module, the network to be trained.
            idx_epoch: int, index of the current epoch.
        
        '''
        pass
    
    def event_after_validation_iteration(self,network,idx_epoch,idx_batch):
        '''
        Event after validating an iteration. Will do nothing by default.
        Recommend to override this method.
        When overriding this method, you don't need to call the parent method.

        Args:
            network: torch.nn.Module, the network to be trained.
            idx_epoch: int, index of the current epoch.
            idx_batch: int, index of the current batch.

        '''
        pass

    def show_config_options(self):
        '''
        Show the options of training configurations.
        '''
        self.configs_handler.show_config_features()
    
    def show_current_configs(self):
        '''
        Show the current training configurations.
        '''
        self.configs_handler.show_config_items()



class TrainedProject():
    '''
    A class that handles the project trained by a 'Trainner'.

    Attributes:
        project_path: The path to the project folder.
    
    Methods:
        get_configs: Get the training configurations.
        get_network_structure: Get the network structure.
        get_trained_network_weights: Get the trained network weights.
        get_checkpoints: Get the checkpoints.
        get_records: Get the records.
        get_saved_network: Get the saved network.
    '''
    
    def __init__(self,project_path) -> None:
        if project_path[-1]!=os.sep:
            project_path+=os.sep
        if not os.path.exists(project_path+"network_structure.pt"):
            print("Warning: No network structure found in {}".format(project_path),flush=True)
            print("Trying to use the latest subfolder as project path",flush=True)
            dir_list = [folder_name for folder_name in os.listdir(project_path) if os.path.isdir(project_path+folder_name)]
            folder_name = sorted(dir_list,  key=lambda x: os.path.getmtime(os.path.join(project_path, x)))[-1]
            if not os.path.exists(project_path+folder_name+os.sep+"configs.yaml"):
                raise FileNotFoundError("No configs.yaml found in {}".format(project_path+folder_name+os.sep))
            project_path+=folder_name+os.sep
        self.project_path=project_path
    
    def get_configs(self,only_path=False):
        '''
        Get the training configurations.

        Args:
            only_path: bool, whether only to return the path of the configs.yaml file, default is False.
        
        Returns:
            dict or str: The configurations or the path of the configs.yaml file.
        '''
        if not os.path.exists(self.project_path+"configs.yaml"):
            raise FileNotFoundError("No configs.yaml found in {}".format(self.project_path))
        if only_path:
            return self.project_path+"configs.yaml"
        else:
            with open(self.project_path+"configs.yaml","r") as f:
                configs_dict=yaml.safe_load(f)
            return configs_dict
    
    def get_network_structure(self,only_path=False):
        '''
        Get the network structure.

        Args:
            only_path: bool, whether only to return the path of the network_structure.pt file, default is False.

        Returns:
            torch.nn.Module or str: The network structure or the path of the network_structure.pt file.
        '''
        if not os.path.exists(self.project_path+"network_structure.pt"):
            raise FileNotFoundError("No network_structure.pt found in {}".format(self.project_path))
        if only_path:
            return self.project_path+"network_structure.pt"
        else:
            return torch.load(self.project_path+"network_structure.pt")
    
    def get_trained_network_weights(self,only_path=False):
        '''
        Get the trained network weights.

        Args:
            only_path: bool, whether only to return the path of the trained_network_weights.pt file, default is False.
        
        Returns:
            torch.nn.Module or str: The trained network weights or the path of the trained_network_weights.pt file.
        '''
        if not os.path.exists(self.project_path+"trained_network_weights.pt"):
            raise FileNotFoundError("No trained_network_weights.pt found in {}".format(self.project_path))
        if only_path:
            return self.project_path+"trained_network_weights.pt"
        else:
            return torch.load(self.project_path+"trained_network_weights.pt")
    
    def get_checkpoints(self,check_point=None,only_path=False):
        '''
        Get the checkpoints.

        Args:
            check_point: int, the checkpoint epoch, default is None, which means the latest checkpoint will be used.
            only_path: bool, whether only to return the path of the checkpoints, default is False.

        Returns:
            dict or str: The checkpoints or the path of the checkpoints.
        '''
        if check_point is not None:
            if not os.path.exists(self.project_path+"checkpoints"+os.sep+"checkpoint_{}.pt".format(check_point)):
                check_points_names=[case.split("_")[1].split(".")[0] for case in os.listdir(self.project_path+"checkpoints")]
                raise FileNotFoundError("No 'checkpoint_{}.pt' found in {}".format(check_point,self.project_path+"checkpoints"+os.sep)
                                        +os.linesep+"Possible checkpoints:"+os.linesep+str(check_points_names))
            check_point_path=self.project_path+"checkpoints"+os.sep+"checkpoint_{}.pt".format(check_point)
        else:
            print("No checkpoint specified, using the latest checkpoint",flush=True)
            print("Trying to load checkpoints from the latest checkpoint",flush=True)
            check_points_names=os.listdir(self.project_path+"checkpoints")
            latest_check_point_name = sorted(check_points_names,  key=lambda x: os.path.getmtime(os.path.join(self.project_path+"checkpoints"+os.sep, x)))[-1]
            check_point_path=self.project_path+"checkpoints"+os.sep+latest_check_point_name
        if only_path:
            return check_point_path
        else:
            return torch.load(check_point_path)
    
    def get_records(self,only_path=False):
        '''
        Get the records.
        This will return an Tensorboard EventAccumulator or the path of the tensorboard records.
        Note that Tensorboard will only read part of the records if the records are too large.
        If you want to read the full records, please use read_full_records() function.

        Args:
            only_path: bool, whether only to return the path of the records, default is False.
        
        Returns:
            str: The records or the path of the records(Tensorboard EventAccumulator).
        '''
        records=os.listdir(self.project_path+"records"+os.sep)
        if len(records)==0:
            raise FileNotFoundError("No records found in {}".format(self.project_path+"records"+os.sep))
        records_path=self.project_path+"records"+os.sep+records[0]
        if only_path:
            return records_path
        else:
            ea= event_accumulator.EventAccumulator(records_path)
            ea.Reload()
            return ea
    
    def get_full_records(self, key, return_wall_time=False):
        """
        Reads the full records from the specified key. Note that this function requires TensorFlow to be installed.

        Args:
            key (str): The tag key to filter the records.
            return_wall_time (bool, optional): Whether to return the wall times of each record. 
                Defaults to False.

        Returns:
            tuple: A tuple containing the steps and losses from the records. If `return_wall_time` 
                is True, it also includes the wall times.

        Raises:
            ImportError: If TensorFlow is not installed.
            FileNotFoundError: If no records are found in the specified record path.

        """
        try:
            import tensorflow as tf
        except ImportError:
            raise ImportError("Please install tensorflow to use this function.")
        record_path=self.project_path+"records"+os.sep
        logs=os.listdir(record_path)
        if len(logs)==0:
            raise FileNotFoundError("No records found in {}".format(record_path))
        if len(logs)>1:
            print("Warning: More than one records found in {}, using the first one".format(record_path),flush=True)
        record_path=os.path.join(record_path,logs[0])
        v_losses=[]
        steps=[]
        if return_wall_time:
            wall_times=[]
        for e in tf.compat.v1.train.summary_iterator(record_path):
            for v in e.summary.value:
                if v.tag == key:
                    v_losses.append(v.simple_value)
                    steps.append(e.step)
                    if return_wall_time:
                        wall_times.append(e.wall_time)
        if return_wall_time:
            return steps,v_losses,wall_times
        else:
            return steps,v_losses
    
    def get_saved_network(self,check_point=None):
        '''
        Get the saved network with trained weights.

        Args:
            check_point: int, the checkpoint epoch, default is None, which means the latest checkpoint will be used.
        
        Returns:
            torch.nn.Module: The saved network.
        '''
        network=self.get_network_structure(only_path=False)
        if check_point is not None:
            weights=self.get_checkpoints(check_point=check_point)["network"]
        else:
            if os.path.exists(self.project_path+"trained_network_weights.pt"):
                weights=torch.load(self.project_path+"trained_network_weights.pt")        
            else:
                print("Warning: No trained_network_weights.pt found in {}".format(self.project_path),flush=True)
                weights=self.get_checkpoints(check_point=None)["network"]
        network.load_state_dict(weights)
        return network

class TrainedProjects():
    """
    A class representing a collection of trained projects.

    Attributes:
    - projects_path: The path to the directory containing the trained projects.

    Methods:
    - __init__(self, projects_path): Initializes the TrainedProjects object.
    - __len__(self): Returns the number of trained projects.
    - __getitem__(self, index): Returns the trained project at the specified index.
    - get_projects_path(self): Returns the path to the directory containing the trained projects.
    """

    def __init__(self, projects_path) -> None:
        """
        Initializes the TrainedProjects object.

        Parameters:
        - projects_path: The path to the directory containing the trained projects.
        """
        folder_names = [folder_name for folder_name in os.listdir(projects_path) if os.path.isdir(os.path.join(projects_path, folder_name))]
        self.folders_name = sorted(folder_names,  key=lambda x: os.path.getmtime(os.path.join(projects_path, x)))
        for folder in self.folders_name:
            path=os.path.join(projects_path,folder)
            if not os.path.exists(os.path.join(path,"configs.yaml")):
                raise FileNotFoundError("No configs.yaml found in {}".format(path))
        self.folders_name = [os.path.join(projects_path,folder) for folder in self.folders_name]
            
    def __len__(self):
        """
        Returns the number of trained projects.

        Returns:
        - The number of trained projects.
        """
        return len(self.folders_name)
    
    def __getitem__(self, index):
        """
        Returns the trained project at the specified index.

        Parameters:
        - index: The index of the trained project to retrieve.

        Returns:
        - If index is an integer, returns the TrainedProject object at the specified index.
        - If index is a slice, returns a list of TrainedProject objects corresponding to the slice.
        """
        if type(index) is int:
            return TrainedProject(self.folders_name[index])
        else:
            return [TrainedProject(folder_name) for folder_name in self.folders_name[index]]
        
    def get_projects_path(self):
        """
        Returns the path to the directory containing the trained projects.

        Returns:
        - The path to the directory containing the trained projects.
        """
        return self.folders_name

def read_configs(path_config_file):
    '''
    Read the training configurations from a yaml file.
    
    Args:
        path_config_file: str, path to the yaml file of the training configurations.
    
    Returns:
        dict: The training configurations.
    '''
    with open(path_config_file,"r") as f:
        yaml_configs=yaml.safe_load(f)
    return yaml_configs

def get_saved_network(project_path,check_point=None):
    '''
    Get the saved network with trained weights.

    Args:
        project_path: str, path to the project folder, mandatory
        check_point: int, the checkpoint epoch, default is None, which means the latest checkpoint will be used.
    
    Returns:
        torch.nn.Module: The saved network.
    '''
    return TrainedProject(project_path).get_saved_network(check_point=check_point)
    '''
    if project_path[-1]!=os.sep:
        project_path+=os.sep
    if not os.path.exists(project_path+"network_structure.pt"):
        print("Warning: No network structure found in {}".format(project_path),flush=True)
        print("Trying to use the latest subfolder as project path",flush=True)
        dir_list = [folder_name for folder_name in os.listdir(project_path) if os.path.isdir(project_path+folder_name)]
        folder_name = sorted(dir_list,  key=lambda x: os.path.getmtime(os.path.join(project_path, x)))[-1]
        if not os.path.exists(project_path+folder_name+os.sep+"network_structure.pt"):
            raise FileNotFoundError("No configs.yaml found in {}".format(project_path+folder_name+os.sep))
        project_path+=folder_name+os.sep
    if check_point is not None:
        network_weights=torch.load(project_path+"checkpoints"+os.sep+"checkpoint_{}.pt".format(check_point))["network"]
    else:
        if os.path.exists(project_path+"trained_network_weights.pt"):
            network_weights=torch.load(project_path+"trained_network_weights.pt")
        else:
            print("Warning: No trained_network_weights.pt found in {}".format(project_path),flush=True)
            print("Trying to load weights from the latest checkpoint",flush=True)
            check_points_names=os.listdir(project_path+"checkpoints")
            latest_check_point_name = sorted(check_points_names,  key=lambda x: os.path.getmtime(os.path.join(project_path+"checkpoints"+os.sep, x)))[-1]
            network_weights=torch.load(project_path+"checkpoints"+os.sep+latest_check_point_name)["network"]        
    network=torch.load(project_path+"network_structure.pt")
    network.load_state_dict(network_weights)
    return network
    '''
