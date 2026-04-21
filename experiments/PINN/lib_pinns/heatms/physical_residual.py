from .simulation_paras import *
from ..helpers import derivative
coef1=1/(500*np.pi)**2
coef2=1/(np.pi)**2

def physical_residual(u, x, t):
    x.grad = None
    t.grad = None
    u_t=derivative(u, t, order=1)
    u_xx=derivative(u, x, order=2)
    return u_t-(coef1*u_xx[:,0:1]+coef2*u_xx[:,1:2])   