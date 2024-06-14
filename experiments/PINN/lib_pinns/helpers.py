from inspect import isfunction
import torch,random,os
import numpy as np
from .foxutils.trainerX import TrainedProjects,TrainedProject

def exists(val):
    return val is not None

def default(val, d):
    if exists(val):
        return val
    return d() if isfunction(d) else d

def set_random_seed(random_seed):
    torch.manual_seed(random_seed)
    torch.cuda.manual_seed(random_seed)
    torch.cuda.manual_seed_all(random_seed)
    np.random.seed(random_seed)
    random.seed(random_seed)

# Adapted from:https://github.com/erfanhamdi/pinn-torch/blob/main/Schrodingers_Equation/main.py
def derivative(y: torch.Tensor, x: torch.Tensor, order: int = 1) -> torch.Tensor:
    for i in range(order):
        y = torch.autograd.grad(
            y, x, grad_outputs = torch.ones_like(y), create_graph=True, retain_graph=True
        )[0]
    return y
    
def package_path():
    #return os.path.dirname(os.path.abspath(__file__))+os.sep
    return os.path.dirname(os.path.dirname(os.path.abspath(__file__)))+os.sep

def read_project_records(project_path,project_item):
    values=[]
    for project in TrainedProjects(project_path):
        recorder=project.get_records()
        losses=recorder.scalars.Items(project_item)
        wall_time=[loss.wall_time for loss in losses]
        step=[loss.step for loss in losses]
        values.append([loss.value for loss in losses])
    values=np.asarray(values)
    return wall_time,step,values,np.mean(values,axis=0),np.std(values,axis=0)

def read_project_record(project_path,project_item):
    project=TrainedProject(project_path)
    recorder=project.get_records()
    losses=recorder.scalars.Items(project_item)
    return [loss.wall_time for loss in losses],[loss.step for loss in losses],[loss.value for loss in losses]
    