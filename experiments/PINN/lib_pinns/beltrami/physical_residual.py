from .simulation_paras import *
from ..helpers import derivative

def physical_residual(uvwp, x, y,z,t,nu=NU):
    x.grad = None
    y.grad = None
    z.grad = None
    t.grad = None
    u=uvwp[:,0]
    v=uvwp[:,1]
    w=uvwp[:,2]
    p=uvwp[:,3]
    
    u_x = derivative(u, x, order=1)
    u_y = derivative(u, y, order=1)
    u_z = derivative(u, z, order=1)
    
    v_x = derivative(v, x, order=1)
    v_y = derivative(v, y, order=1)
    v_z = derivative(v, z, order=1)
    
    w_x = derivative(w, x, order=1)
    w_y = derivative(w, y, order=1)
    w_z = derivative(w, z, order=1)
    
    u_xx = derivative(u_x, x, order=1)
    u_yy = derivative(u_y, y, order=1)
    u_zz = derivative(u_z, z, order=1)
    
    v_xx = derivative(v_x, x, order=1)
    v_yy = derivative(v_y, y, order=1)
    v_zz = derivative(v_z, z, order=1)
    
    w_xx = derivative(w_x, x, order=1)
    w_yy = derivative(w_y, y, order=1)
    w_zz = derivative(w_z, z, order=1)

    p_x = derivative(p, x, order=1)
    p_y = derivative(p, y, order=1)
    p_z = derivative(p, z, order=1)
    
    u_t = derivative(u, t, order=1)
    v_t = derivative(v, t, order=1)
    w_t = derivative(w, t, order=1)
    
    loss_mx= u_t + u*u_x + v*u_y + w*u_z + p_x - nu*(u_xx + u_yy + u_zz)
    loss_my= v_t + u*v_x + v*v_y + w*v_z + p_y - nu*(v_xx + v_yy + v_zz)
    loss_mz= w_t + u*w_x + v*w_y + w*w_z + p_z - nu*(w_xx + w_yy + w_zz)
    loss_c= u_x + v_y + w_z
    
    return loss_mx,loss_my,loss_mz,loss_c