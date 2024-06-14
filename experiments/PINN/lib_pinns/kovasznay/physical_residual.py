from .simulation_paras import *
from ..helpers import derivative

def physical_residual(uvp, x, y,nu=NU):
    x.grad = None
    y.grad = None
    u=uvp[:,0]
    v=uvp[:,1]
    p=uvp[:,2]
    u_x = derivative(u, x, order=1)
    u_y = derivative(u, y, order=1)
    u_xx = derivative(u_x, x, order=1)
    u_yy = derivative(u_y, y, order=1)
    v_x = derivative(v, x, order=1)
    v_y = derivative(v, y, order=1)
    v_xx = derivative(v_x, x, order=1)
    v_yy = derivative(v_y, y, order=1)
    p_x = derivative(p, x, order=1)
    p_y = derivative(p, y, order=1)
    
    loss_mx= u*u_x + v*u_y + p_x - nu*(u_xx + u_yy)
    loss_my= u*v_x + v*v_y + p_y - nu*(v_xx + v_yy)
    loss_c= u_x + v_y
    
    return loss_mx,loss_my,loss_c
'''
def mx_loss(uvp, x, y,nu=NU):
    x.grad = None
    y.grad = None
    u=uvp[:,0]
    v=uvp[:,1]
    p=uvp[:,2]
    u_x = derivative(u, x, order=1)
    u_y = derivative(u, y, order=1)
    u_xx = derivative(u_x, x, order=1)
    u_yy = derivative(u_y, y, order=1)
    p_x = derivative(p, x, order=1)
    loss_mx= u*u_x + v*u_y + p_x - nu*(u_xx + u_yy)
    return loss_mx 

def my_loss(uvp, x, y,nu=NU):
    x.grad = None
    y.grad = None
    u=uvp[:,0]
    v=uvp[:,1]
    p=uvp[:,2]
    v_x = derivative(v, x, order=1)
    v_y = derivative(v, y, order=1)
    v_xx = derivative(v_x, x, order=1)
    v_yy = derivative(v_y, y, order=1)
    p_y = derivative(p, y, order=1)
    loss_my= u*v_x + v*v_y + p_y - nu*(v_xx + v_yy)   
    return loss_my

def c_loss(uvp, x, y):
    x.grad = None
    y.grad = None
    u=uvp[:,0]
    v=uvp[:,1]
    u_x = derivative(u, x, order=1)
    v_y = derivative(v, y, order=1)
    loss_c= u_x + v_y 
    return loss_c
'''