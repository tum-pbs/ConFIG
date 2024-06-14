from .simulation_paras import *
from ..helpers import derivative

def physical_residual(u, x, t,nu=NU):
    x.grad = None
    t.grad = None
    """ Physics-based loss function with Burgers equation """
    u_t = derivative(u, t, order=1)
    u_x = derivative(u, x, order=1)
    u_xx = derivative(u_x, x, order=1)
    return u_t + u*u_x - nu * u_xx