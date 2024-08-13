import torch
import numpy as np
import random
from typing import Sequence

def get_cos_similarity(vector1,vector2):
    with torch.no_grad():
        return torch.dot(vector1,vector2)/vector1.norm()/vector2.norm()

def unit_vector(vector):
    with torch.no_grad():
        return vector/vector.norm()

def get_gradient_vector(network):
    with torch.no_grad():
        grad_vec = None
        for par in network.parameters():
            viewed=par.grad.data.view(-1)
            if grad_vec is None:
                grad_vec = viewed
            else:
                grad_vec = torch.cat((grad_vec, viewed))
        return grad_vec
    
def get_para_vector(network):
    with torch.no_grad():
        para_vec = None
        for par in network.parameters():
            viewed=par.data.view(-1)
            if para_vec is None:
                para_vec = viewed
            else:
                para_vec = torch.cat((para_vec, viewed))
        return para_vec

def apply_gradient_vector(network,grad_vec):
    with torch.no_grad():
        start=0
        for par in network.parameters():
            end=start+par.grad.data.view(-1).shape[0]
            par.grad.data=grad_vec[start:end].view(par.grad.data.shape)
            start=end

def update_weights(network,lr):
    with torch.no_grad():
        for par in network.parameters():
            par.data-=lr*par.grad.data

        
def config_update_double(grads,return_cos=True):
    with torch.no_grad():
        norm_1=grads[0].norm();norm_2=grads[1].norm()
        if norm_1==0 or norm_2==0:
            if return_cos:
                return torch.zeros_like(grads[0]),torch.zeros(1,device=grads.device)
            else:
                return torch.zeros_like(grads[0])
        else:
            cos_angle=torch.dot(grads[0],grads[1])/norm_1/norm_2
            if cos_angle==1:
                if return_cos:
                    return grads[0]+grads[1],cos_angle
                else:
                    return grads[0]+grads[1]
            or_2=grads[0]-norm_1*cos_angle*(grads[1]/norm_2)
            or_1=grads[1]-norm_2*cos_angle*(grads[0]/norm_1)
            best_direction=unit_vector(or_1/or_1.norm()+or_2/or_2.norm())
            cos_1=torch.dot(grads[0],best_direction)
            cos_2=torch.dot(grads[1],best_direction)
            if return_cos:
                return (cos_1+cos_2)*best_direction,get_cos_similarity(grads[0],best_direction)
            else:
                return (cos_1+cos_2)*best_direction

def config_update_double_coefs(vector1,vector2,coef1,coef2,length_coef=None,return_cos=True):
    with torch.no_grad():
        norm_1=vector1.norm();norm_2=vector2.norm()
        cos_angle=get_cos_similarity(vector1,vector2)
        or_2=vector1-norm_1*cos_angle*(vector2/norm_2)
        or_1=vector2-norm_2*cos_angle*(vector1/norm_1)
        best_direction=unit_vector(coef2*or_1/or_1.norm()+coef1*or_2/or_2.norm())
        if length_coef is None:
            cos_1=torch.dot(vector1,best_direction)
            cos_2=torch.dot(vector2,best_direction)
            best_direction*=cos_1+cos_2
        else:
            best_direction*=length_coef
        if return_cos:
            return best_direction,get_cos_similarity(vector1,best_direction)
        else:
            return best_direction
        
def config_update_multi(grads,return_cos=True):
    with torch.no_grad():
        best_direction=torch.ones(grads.shape[0],device=grads.device)@torch.linalg.pinv(torch.nan_to_num((grads/(grads.norm(dim=1)).unsqueeze(1)).T,0))# optimal point, grad=0, /grad.norm>nan
        best_direction=torch.nan_to_num(best_direction/(best_direction.norm()),0)
        compoents=[torch.dot(grad_i,best_direction) for grad_i in grads]
        if return_cos:
            return torch.sum(torch.stack(compoents))*best_direction,compoents[0]/grads[0].norm()
        else:
            return torch.sum(torch.stack(compoents))*best_direction
        
def config_update_multi_coefs(grads,coefs,coef_length=None,return_cos=True):
    with torch.no_grad():
        best_direction=coefs@torch.linalg.pinv(
            (grads/grads.norm(dim=1).unsqueeze(1)).T)
        best_direction=best_direction/best_direction.norm()
        compoents=[torch.dot(grad_i,best_direction) for grad_i in grads]
        if coef_length is None:
            coef_length=torch.sum(torch.stack(compoents))
        if return_cos:
            return coef_length*best_direction,compoents[0]/grads[0].norm()
        else:
            return coef_length*torch.sum(torch.stack(compoents))*best_direction
        
def config_update(grads,return_cos=True):
    if grads.shape[0]==2:
        return config_update_double(grads,return_cos)
    else:
        return config_update_multi(grads,return_cos)

def config_scale1_double(grads,return_cos=True):
    with torch.no_grad():
        norm_1=grads[0].norm();norm_2=grads[1].norm()
        if norm_1==0 or norm_2==0:
            print('zero grad')
            if return_cos:
                return torch.zeros_like(grads[0]),torch.zero(0,device=grads.device)
            else:
                return torch.zeros_like(grads[0])
        else:
            cos_angle=torch.dot(grads[0],grads[1])/norm_1/norm_2
            if cos_angle==1:
                if return_cos:
                    return grads[0]+grads[1],cos_angle
                else:
                    return grads[0]+grads[1]
            or_2=grads[0]-norm_1*cos_angle*(grads[1]/norm_2)
            or_1=grads[1]-norm_2*cos_angle*(grads[0]/norm_1)
            best_direction=unit_vector(or_1/or_1.norm()+or_2/or_2.norm())
            cos=get_cos_similarity(grads[0],best_direction)
            length_scale=(norm_1*norm_2)**0.5
            if return_cos:
                return 2*length_scale*cos*best_direction,cos
            else:
                return 2*length_scale*cos*best_direction
        
def config_scale1_multi(grads,return_cos=True):
    with torch.no_grad():
        best_direction=torch.ones(grads.shape[0],device=grads.device)@torch.linalg.pinv(torch.nan_to_num((grads/(grads.norm(dim=1)).unsqueeze(1)).T,0))# optimal point, grad=0, /grad.norm>nan
        best_direction=torch.nan_to_num(best_direction/(best_direction.norm()),0)
        cos=torch.dot(grads[0],best_direction)/grads[0].norm()
        length_scale=torch.prod(torch.stack([grad_i.norm() for grad_i in grads],0),0)**(1/grads.shape[0])
        if return_cos:
            return grads.shape[0]*length_scale*cos*best_direction,cos
        else:
            return grads.shape[0]*length_scale*cos*best_direction
        
def config_scale1_update(grads,return_cos=True):
    if grads.shape[0]==2:
        return config_scale1_double(grads,return_cos)
    else:
        return config_scale1_multi(grads,return_cos)
    
def config_scale2_double(grads,return_cos=True):
    with torch.no_grad():
        norm_1=grads[0].norm();norm_2=grads[1].norm()
        if norm_1==0 or norm_2==0:
            print('zero grad')
            if return_cos:
                return torch.zeros_like(grads[0]),torch.zero(0,device=grads.device)
            else:
                return torch.zeros_like(grads[0])
        else:
            cos_angle=torch.dot(grads[0],grads[1])/norm_1/norm_2
            if cos_angle==1:
                if return_cos:
                    return grads[0]+grads[1],cos_angle
                else:
                    return grads[0]+grads[1]
            or_2=grads[0]-norm_1*cos_angle*(grads[1]/norm_2)
            or_1=grads[1]-norm_2*cos_angle*(grads[0]/norm_1)
            best_direction=unit_vector(or_1/or_1.norm()+or_2/or_2.norm())
            cos=get_cos_similarity(grads[0],best_direction)
            length_scale=2*norm_1
            if return_cos:
                return length_scale*cos*best_direction,cos
            else:
                return length_scale*cos*best_direction
        
def config_scale2_multi(grads,return_cos=True):
    with torch.no_grad():
        best_direction=torch.ones(grads.shape[0],device=grads.device)@torch.linalg.pinv(torch.nan_to_num((grads/(grads.norm(dim=1)).unsqueeze(1)).T,0))# optimal point, grad=0, /grad.norm>nan
        best_direction=torch.nan_to_num(best_direction/(best_direction.norm()),0)
        cos=torch.dot(grads[0],best_direction)/grads[0].norm()
        length_scale=grads.shape[0]*grads[0].norm()
        if return_cos:
            return length_scale*cos*best_direction,cos
        else:
            return length_scale*cos*best_direction
        
def config_scale2_update(grads,return_cos=True):
    if grads.shape[0]==2:
        return config_scale2_double(grads,return_cos)
    else:
        return config_scale2_multi(grads,return_cos)

def config_scale3_double(grads,return_cos=True):
    with torch.no_grad():
        norm_1=grads[0].norm();norm_2=grads[1].norm()
        if norm_1==0 or norm_2==0:
            print('zero grad')
            if return_cos:
                return torch.zeros_like(grads[0]),torch.zero(0,device=grads.device)
            else:
                return torch.zeros_like(grads[0])
        else:
            cos_angle=torch.dot(grads[0],grads[1])/norm_1/norm_2
            if cos_angle==1:
                if return_cos:
                    return grads[0]+grads[1],cos_angle
                else:
                    return grads[0]+grads[1]
            or_2=grads[0]-norm_1*cos_angle*(grads[1]/norm_2)
            or_1=grads[1]-norm_2*cos_angle*(grads[0]/norm_1)
            best_direction=unit_vector(or_1/or_1.norm()+or_2/or_2.norm())
            cos=get_cos_similarity(grads[0],best_direction)
            length_scale=torch.max(norm_1,norm_2)*2
            if return_cos:
                return length_scale*cos*best_direction,cos
            else:
                return length_scale*cos*best_direction
        
def config_scale3_multi(grads,return_cos=True):
    with torch.no_grad():
        best_direction=torch.ones(grads.shape[0],device=grads.device)@torch.linalg.pinv(torch.nan_to_num((grads/(grads.norm(dim=1)).unsqueeze(1)).T,0))# optimal point, grad=0, /grad.norm>nan
        best_direction=torch.nan_to_num(best_direction/(best_direction.norm()),0)
        cos=torch.dot(grads[0],best_direction)/grads[0].norm()
        length_scale=torch.max(torch.stack([grad_i.norm() for grad_i in grads],0),0)*grads.shape[0]
        if return_cos:
            return length_scale*cos*best_direction,cos
        else:
            return length_scale*cos*best_direction
        
def config_scale3_update(grads,return_cos=True):
    if grads.shape[0]==2:
        return config_scale3_double(grads,return_cos)
    else:
        return config_scale3_multi(grads,return_cos)

def config_scale4_double(grads,return_cos=True):
    with torch.no_grad():
        norm_1=grads[0].norm();norm_2=grads[1].norm()
        if norm_1==0 or norm_2==0:
            print('zero grad')
            if return_cos:
                return torch.zeros_like(grads[0]),torch.zero(0,device=grads.device)
            else:
                return torch.zeros_like(grads[0])
        else:
            cos_angle=torch.dot(grads[0],grads[1])/norm_1/norm_2
            if cos_angle==1:
                if return_cos:
                    return grads[0]+grads[1],cos_angle
                else:
                    return grads[0]+grads[1]
            or_2=grads[0]-norm_1*cos_angle*(grads[1]/norm_2)
            or_1=grads[1]-norm_2*cos_angle*(grads[0]/norm_1)
            best_direction=unit_vector(or_1/or_1.norm()+or_2/or_2.norm())
            cos=get_cos_similarity(grads[0],best_direction)
            length_scale=2*(norm_1*norm_2)/(norm_1+norm_2)
            if return_cos:
                return length_scale*cos*best_direction,cos
            else:
                return length_scale*cos*best_direction
        
def config_scale4_multi(grads,return_cos=True):
    with torch.no_grad():
        best_direction=torch.ones(grads.shape[0],device=grads.device)@torch.linalg.pinv(torch.nan_to_num((grads/(grads.norm(dim=1)).unsqueeze(1)).T,0))# optimal point, grad=0, /grad.norm>nan
        best_direction=torch.nan_to_num(best_direction/(best_direction.norm()),0)
        cos=torch.dot(grads[0],best_direction)/grads[0].norm()
        norms=torch.stack([grad_i.norm() for grad_i in grads],0)
        length_scale=torch.prod(norms,0)*grads.shape[0]/torch.sum(norms,0)
        if return_cos:
            return length_scale*cos*best_direction,cos
        else:
            return length_scale*cos*best_direction
        
def config_scale4_update(grads,return_cos=True):
    if grads.shape[0]==2:
        return config_scale4_double(grads,return_cos)
    else:
        return config_scale4_multi(grads,return_cos)

def config_scale5_double(grads,return_cos=True):
    with torch.no_grad():
        norm_1=grads[0].norm();norm_2=grads[1].norm()
        if norm_1==0 or norm_2==0:
            print('zero grad')
            if return_cos:
                return torch.zeros_like(grads[0]),torch.zero(0,device=grads.device)
            else:
                return torch.zeros_like(grads[0])
        else:
            cos_angle=torch.dot(grads[0],grads[1])/norm_1/norm_2
            if cos_angle==1:
                if return_cos:
                    return grads[0]+grads[1],cos_angle
                else:
                    return grads[0]+grads[1]
            or_2=grads[0]-norm_1*cos_angle*(grads[1]/norm_2)
            or_1=grads[1]-norm_2*cos_angle*(grads[0]/norm_1)
            best_direction=unit_vector(or_1/or_1.norm()+or_2/or_2.norm())
            cos=get_cos_similarity(grads[0],best_direction)
            length_scale=torch.min(norm_1,norm_2)*2
            if return_cos:
                return length_scale*cos*best_direction,cos
            else:
                return length_scale*cos*best_direction
        
def config_scale5_multi(grads,return_cos=True):
    with torch.no_grad():
        best_direction=torch.ones(grads.shape[0],device=grads.device)@torch.linalg.pinv(torch.nan_to_num((grads/(grads.norm(dim=1)).unsqueeze(1)).T,0))# optimal point, grad=0, /grad.norm>nan
        best_direction=torch.nan_to_num(best_direction/(best_direction.norm()),0)
        cos=torch.dot(grads[0],best_direction)/grads[0].norm()
        length_scale=torch.min(torch.stack([grad_i.norm() for grad_i in grads],0),0)*grads.shape[0]
        if return_cos:
            return length_scale*cos*best_direction,cos
        else:
            return length_scale*cos*best_direction
        
def config_scale5_update(grads,return_cos=True):
    if grads.shape[0]==2:
        return config_scale5_double(grads,return_cos)
    else:
        return config_scale5_multi(grads,return_cos)
    
def config_scale6_double(grads,return_cos=True):
    with torch.no_grad():
        norm_1=grads[0].norm();norm_2=grads[1].norm()
        if norm_1==0 or norm_2==0:
            if return_cos:
                return torch.zeros_like(grads[0]),torch.zero(0,device=grads.device)
            else:
                return torch.zeros_like(grads[0])
        else:
            cos_angle=torch.dot(grads[0],grads[1])/norm_1/norm_2
            if cos_angle==1:
                if return_cos:
                    return grads[0]+grads[1],cos_angle
                else:
                    return grads[0]+grads[1]
            or_2=grads[0]-norm_1*cos_angle*(grads[1]/norm_2)
            or_1=grads[1]-norm_2*cos_angle*(grads[0]/norm_1)
            best_direction=unit_vector(or_1/or_1.norm()+or_2/or_2.norm())
            cos=get_cos_similarity(grads[0],best_direction)
            if return_cos:
                return (norm_1+norm_2)/2/cos*best_direction,get_cos_similarity(grads[0],best_direction)
            else:
                return (norm_1+norm_2)/2/cos*best_direction
        
def config_scale6_multi(grads,return_cos=True):
    with torch.no_grad():
        best_direction=torch.ones(grads.shape[0],device=grads.device)@torch.linalg.pinv(torch.nan_to_num((grads/(grads.norm(dim=1)).unsqueeze(1)).T,0))# optimal point, grad=0, /grad.norm>nan
        best_direction=torch.nan_to_num(best_direction/(best_direction.norm()),0)
        length=torch.sum(torch.stack([grad_i.norm() for grad_i in grads],0),0)/grads.shape[0]
        cos=get_cos_similarity(grads[0],best_direction)
        if return_cos:
            return length/cos*best_direction,cos
        else:
            return length/cos*best_direction
        
def config_scale6_update(grads,return_cos=True):
    if grads.shape[0]==2:
        return config_scale6_double(grads,return_cos)
    else:
        return config_scale6_multi(grads,return_cos)


def config_scale7_double(grads,return_cos=True):
    with torch.no_grad():
        norm_1=grads[0].norm();norm_2=grads[1].norm()
        if norm_1==0 or norm_2==0:
            print('zero grad')
            if return_cos:
                return torch.zeros_like(grads[0]),torch.zero(0,device=grads.device)
            else:
                return torch.zeros_like(grads[0])
        else:
            cos_angle=torch.dot(grads[0],grads[1])/norm_1/norm_2
            if cos_angle==1:
                if return_cos:
                    return grads[0]+grads[1],cos_angle
                else:
                    return grads[0]+grads[1]
            or_2=grads[0]-norm_1*cos_angle*(grads[1]/norm_2)
            or_1=grads[1]-norm_2*cos_angle*(grads[0]/norm_1)
            best_direction=unit_vector(or_1/or_1.norm()+or_2/or_2.norm())
            cos=get_cos_similarity(grads[0],best_direction)
            length_scale=(norm_1*norm_2)**0.5
            if return_cos:
                return length_scale/cos*best_direction,cos
            else:
                return length_scale/cos*best_direction
        
def config_scale7_multi(grads,return_cos=True):
    with torch.no_grad():
        best_direction=torch.ones(grads.shape[0],device=grads.device)@torch.linalg.pinv(torch.nan_to_num((grads/(grads.norm(dim=1)).unsqueeze(1)).T,0))# optimal point, grad=0, /grad.norm>nan
        best_direction=torch.nan_to_num(best_direction/(best_direction.norm()),0)
        cos=torch.dot(grads[0],best_direction)/grads[0].norm()
        length_scale=torch.prod(torch.stack([grad_i.norm() for grad_i in grads],0),0)**(1/grads.shape[0])
        if return_cos:
            return length_scale/cos*best_direction,cos
        else:
            return length_scale/cos*best_direction
        
def config_scale7_update(grads,return_cos=True):
    if grads.shape[0]==2:
        return config_scale7_double(grads,return_cos)
    else:
        return config_scale7_multi(grads,return_cos)
    
def config_scale8_double(grads,return_cos=True):
    with torch.no_grad():
        norm_1=grads[0].norm();norm_2=grads[1].norm()
        if norm_1==0 or norm_2==0:
            print('zero grad')
            if return_cos:
                return torch.zeros_like(grads[0]),torch.zero(0,device=grads.device)
            else:
                return torch.zeros_like(grads[0])
        else:
            cos_angle=torch.dot(grads[0],grads[1])/norm_1/norm_2
            if cos_angle==1:
                if return_cos:
                    return grads[0]+grads[1],cos_angle
                else:
                    return grads[0]+grads[1]
            or_2=grads[0]-norm_1*cos_angle*(grads[1]/norm_2)
            or_1=grads[1]-norm_2*cos_angle*(grads[0]/norm_1)
            best_direction=unit_vector(or_1/or_1.norm()+or_2/or_2.norm())
            cos=get_cos_similarity(grads[0],best_direction)
            length_scale=norm_1
            if return_cos:
                return length_scale/cos*best_direction,cos
            else:
                return length_scale/cos*best_direction
        
def config_scale8_multi(grads,return_cos=True):
    with torch.no_grad():
        best_direction=torch.ones(grads.shape[0],device=grads.device)@torch.linalg.pinv(torch.nan_to_num((grads/(grads.norm(dim=1)).unsqueeze(1)).T,0))# optimal point, grad=0, /grad.norm>nan
        best_direction=torch.nan_to_num(best_direction/(best_direction.norm()),0)
        cos=torch.dot(grads[0],best_direction)/grads[0].norm()
        length_scale=grads[0].norm()
        if return_cos:
            return length_scale/cos*best_direction,cos
        else:
            return length_scale/cos*best_direction
        
def config_scale8_update(grads,return_cos=True):
    if grads.shape[0]==2:
        return config_scale8_double(grads,return_cos)
    else:
        return config_scale8_multi(grads,return_cos)

def config_scale9_double(grads,return_cos=True):
    with torch.no_grad():
        norm_1=grads[0].norm();norm_2=grads[1].norm()
        if norm_1==0 or norm_2==0:
            print('zero grad')
            if return_cos:
                return torch.zeros_like(grads[0]),torch.zero(0,device=grads.device)
            else:
                return torch.zeros_like(grads[0])
        else:
            cos_angle=torch.dot(grads[0],grads[1])/norm_1/norm_2
            if cos_angle==1:
                if return_cos:
                    return grads[0]+grads[1],cos_angle
                else:
                    return grads[0]+grads[1]
            or_2=grads[0]-norm_1*cos_angle*(grads[1]/norm_2)
            or_1=grads[1]-norm_2*cos_angle*(grads[0]/norm_1)
            best_direction=unit_vector(or_1/or_1.norm()+or_2/or_2.norm())
            cos=get_cos_similarity(grads[0],best_direction)
            length_scale=torch.max(norm_1,norm_2)
            if return_cos:
                return length_scale/cos*best_direction,cos
            else:
                return length_scale/cos*best_direction
        
def config_scale9_multi(grads,return_cos=True):
    with torch.no_grad():
        best_direction=torch.ones(grads.shape[0],device=grads.device)@torch.linalg.pinv(torch.nan_to_num((grads/(grads.norm(dim=1)).unsqueeze(1)).T,0))# optimal point, grad=0, /grad.norm>nan
        best_direction=torch.nan_to_num(best_direction/(best_direction.norm()),0)
        cos=torch.dot(grads[0],best_direction)/grads[0].norm()
        length_scale=torch.max(torch.stack([grad_i.norm() for grad_i in grads],0),0).values
        if return_cos:
            return length_scale/cos*best_direction,cos
        else:
            return length_scale/cos*best_direction
        
def config_scale9_update(grads,return_cos=True):
    if grads.shape[0]==2:
        return config_scale9_double(grads,return_cos)
    else:
        return config_scale9_multi(grads,return_cos)

def config_scale10_double(grads,return_cos=True):
    with torch.no_grad():
        norm_1=grads[0].norm();norm_2=grads[1].norm()
        if norm_1==0 or norm_2==0:
            print('zero grad')
            if return_cos:
                return torch.zeros_like(grads[0]),torch.zero(0,device=grads.device)
            else:
                return torch.zeros_like(grads[0])
        else:
            cos_angle=torch.dot(grads[0],grads[1])/norm_1/norm_2
            if cos_angle==1:
                if return_cos:
                    return grads[0]+grads[1],cos_angle
                else:
                    return grads[0]+grads[1]
            or_2=grads[0]-norm_1*cos_angle*(grads[1]/norm_2)
            or_1=grads[1]-norm_2*cos_angle*(grads[0]/norm_1)
            best_direction=unit_vector(or_1/or_1.norm()+or_2/or_2.norm())
            cos=get_cos_similarity(grads[0],best_direction)
            length_scale=(norm_1*norm_2)/(norm_1+norm_2)
            if return_cos:
                return length_scale/cos*best_direction,cos
            else:
                return length_scale/cos*best_direction
        
def config_scale10_multi(grads,return_cos=True):
    with torch.no_grad():
        best_direction=torch.ones(grads.shape[0],device=grads.device)@torch.linalg.pinv(torch.nan_to_num((grads/(grads.norm(dim=1)).unsqueeze(1)).T,0))# optimal point, grad=0, /grad.norm>nan
        best_direction=torch.nan_to_num(best_direction/(best_direction.norm()),0)
        cos=torch.dot(grads[0],best_direction)/grads[0].norm()
        norms=torch.stack([grad_i.norm() for grad_i in grads],0)
        length_scale=torch.prod(norms,0)/torch.sum(norms,0)
        if return_cos:
            return length_scale/cos*best_direction,cos
        else:
            return length_scale/cos*best_direction
        
def config_scale10_update(grads,return_cos=True):
    if grads.shape[0]==2:
        return config_scale10_double(grads,return_cos)
    else:
        return config_scale10_multi(grads,return_cos)

def config_scale11_double(grads,return_cos=True):
    with torch.no_grad():
        norm_1=grads[0].norm();norm_2=grads[1].norm()
        if norm_1==0 or norm_2==0:
            print('zero grad')
            if return_cos:
                return torch.zeros_like(grads[0]),torch.zero(0,device=grads.device)
            else:
                return torch.zeros_like(grads[0])
        else:
            cos_angle=torch.dot(grads[0],grads[1])/norm_1/norm_2
            if cos_angle==1:
                if return_cos:
                    return grads[0]+grads[1],cos_angle
                else:
                    return grads[0]+grads[1]
            or_2=grads[0]-norm_1*cos_angle*(grads[1]/norm_2)
            or_1=grads[1]-norm_2*cos_angle*(grads[0]/norm_1)
            best_direction=unit_vector(or_1/or_1.norm()+or_2/or_2.norm())
            cos=get_cos_similarity(grads[0],best_direction)
            length_scale=torch.min(norm_1,norm_2)
            if return_cos:
                return length_scale/cos*best_direction,cos
            else:
                return length_scale/cos*best_direction
        
def config_scale11_multi(grads,return_cos=True):
    with torch.no_grad():
        best_direction=torch.ones(grads.shape[0],device=grads.device)@torch.linalg.pinv(torch.nan_to_num((grads/(grads.norm(dim=1)).unsqueeze(1)).T,0))# optimal point, grad=0, /grad.norm>nan
        best_direction=torch.nan_to_num(best_direction/(best_direction.norm()),0)
        cos=torch.dot(grads[0],best_direction)/grads[0].norm()
        length_scale=torch.min(torch.stack([grad_i.norm() for grad_i in grads],0),0).values
        if return_cos:
            return length_scale/cos*best_direction,cos
        else:
            return length_scale/cos*best_direction
        
def config_scale11_update(grads,return_cos=True):
    if grads.shape[0]==2:
        return config_scale11_double(grads,return_cos)
    else:
        return config_scale11_multi(grads,return_cos)

def coef_config_update(grads,coefs,coef_length=None,return_cos=True):
    if grads.shape[0]==2:
        return config_update_double_coefs(grads[0],grads[1],coefs[0],coefs[1],coef_length,return_cos)
    else:
        return config_update_multi_coefs(grads,coefs,coef_length,return_cos)

def pcgrad_update(grads,return_cos=True):
    with torch.no_grad():
        grads_pc=torch.clone(grads)
        length=grads.shape[0]
        for i in range(length):
            for j in range(length):
                if j !=i:
                    dot=grads_pc[i].dot(grads[j])
                    if dot<0:
                        grads_pc[i]-=dot*grads[j]/((grads[j].norm())**2)
        update_vector=torch.sum(grads_pc,dim=0)
        if return_cos:
            cos=update_vector.dot(grads[0])/grads[0].norm()/update_vector.norm()
            return update_vector,cos
        else:
            return update_vector

def imtlg_update(grads,return_cos=True):
    with torch.no_grad():
        ut_norm=grads/grads.norm(dim=1).unsqueeze(1)
        ut_norm=torch.nan_to_num(ut_norm,0)
        ut=torch.stack([ut_norm[0]-ut_norm[i+1] for i in range(grads.shape[0]-1)],dim=0).T
        d=torch.stack([grads[0]-grads[i+1] for i in range(grads.shape[0]-1)],dim=0)
        at=grads[0]@ut@torch.linalg.pinv(d@ut)
        final_grad=(1-torch.sum(at))*grads[0]+torch.sum(at.unsqueeze(1)*grads[1:],dim=0)
        if return_cos:
            cos=final_grad.dot(grads[0])/grads[0].norm()/final_grad.norm()
            return final_grad,cos
        else:
            return final_grad

    
class MomentumManipulation():
    
    def __init__(self,network,num_grads=2,betas_1=0.9, betas_2=0.999,device=None):#, normalized_basis_vector=None) -> None:
        if device is None:
            device=network.parameters().__next__().device
        self.network=network    
        with torch.no_grad():
            vec = None
            for par in network.parameters():
                viewed=par.data.view(-1)
                if vec is None:
                    vec = viewed
                else:
                    vec = torch.cat((vec, viewed))
            shape=vec.shape
        self.num_grads=num_grads
        self.m=[torch.zeros(shape,device=device) for i in range(num_grads)]
        self.s=torch.zeros(shape,device=device) #second moment
        self.fake_m=torch.zeros(shape,device=device)
        self.betas_1=betas_1
        self.betas_2=betas_2
        self.t=0
        self.t_grads=[0 for i in range(num_grads)]
        self.index=0
    
    def update_gradient(self, grad_i=None, index_i=None):
        with torch.no_grad():
            self.index+=1
            if not isinstance(index_i,Sequence):
                index_i=[index_i]
                grad_i=[grad_i]
            for i in range(len(index_i)):
                self.t_grads[index_i[i]]+=1
                self.m[index_i[i]]=self.betas_1*self.m[index_i[i]]+(1-self.betas_1)*grad_i[i] 
            if self.index<self.num_grads: 
                return torch.zeros_like(self.s),0
            else:  
                self.t+=1
                m_hats=torch.stack([self.m[i]/(1-self.betas_1**self.t_grads[i]) for i in range(self.num_grads)],dim=0)
                final_grad,cos_angle=self.gradient_operation(m_hats)
                fake_m=final_grad*(1-self.betas_1**self.t)
                fake_grad=(fake_m-self.betas_1*self.fake_m)/(1-self.betas_1)
                self.fake_m=fake_m
                self.s=self.betas_2*self.s+(1-self.betas_2)*fake_grad**2
                s_hat=self.s/(1-self.betas_2**self.t) 
                final_grad=final_grad/(torch.sqrt(s_hat)+1e-8)                   
            return final_grad,cos_angle 

    def gradient_operation(self,grads):
        raise NotImplementedError("This method should be implemented in the bound_iniclass")

class SeparateMomentumManipulation():
    
    def __init__(self,network,num_grads=2,betas_1=0.9, betas_2=0.999,device=None):#, normalized_basis_vector=None) -> None:
        if device is None:
            device=network.parameters().__next__().device
        self.network=network    
        with torch.no_grad():
            vec = None
            for par in network.parameters():
                viewed=par.data.view(-1)
                if vec is None:
                    vec = viewed
                else:
                    vec = torch.cat((vec, viewed))
            shape=vec.shape
        self.num_grads=num_grads
        self.m=[torch.zeros(shape,device=device) for i in range(num_grads)]
        self.s=[torch.zeros(shape,device=device) for i in range(num_grads)] #second moment
        self.fake_m=torch.zeros(shape,device=device)
        self.betas_1=betas_1
        self.betas_2=betas_2
        self.t=0
        self.t_grads=[0 for i in range(num_grads)]
        self.index=0
    
    def update_gradient(self, grad_i=None, index_i=None):
        with torch.no_grad():
            self.index+=1
            if not isinstance(index_i,Sequence):
                index_i=[index_i]
                grad_i=[grad_i]
            for i in range(len(index_i)):
                self.t_grads[index_i[i]]+=1
                self.m[index_i[i]]=self.betas_1*self.m[index_i[i]]+(1-self.betas_1)*grad_i[i] 
                self.s[index_i[i]]=self.betas_2*self.s[index_i[i]]+(1-self.betas_2)*grad_i[i]**2
            if self.index<self.num_grads: 
                return torch.zeros_like(self.s[0]),0
            else:  
                self.t+=1
                m_hats=torch.stack([self.m[i]/(1-self.betas_1**self.t_grads[i]) for i in range(self.num_grads)],dim=0)
                s_hats=torch.stack([self.s[i]/(1-self.betas_2**self.t_grads[i]) for i in range(self.num_grads)],dim=0)
                final_grad,cos_angle=self.gradient_operation(m_hats/(torch.sqrt(s_hats)+1e-8))             
            return final_grad,cos_angle 

    def gradient_operation(self,grads):
        raise NotImplementedError("This method should be implemented in the bound_iniclass")

def get_momentum_manipulation(network,operation,num_grads=2,betas_1=0.9, betas_2=0.999,device=None):
    class certain_momentum_manipulation(MomentumManipulation):
        def gradient_operation(self,grads):
            return operation(grads)
    return certain_momentum_manipulation(network,num_grads,betas_1,betas_2,device)

def get_separate_momentum_manipulation(network,operation,num_grads=2,betas_1=0.9, betas_2=0.999,device=None):
    class certain_momentum_manipulation(SeparateMomentumManipulation):
        def gradient_operation(self,grads):
            return operation(grads)
    return certain_momentum_manipulation(network,num_grads,betas_1,betas_2,device)
