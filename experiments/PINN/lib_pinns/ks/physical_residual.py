from ..helpers import derivative

def physical_residual(u, x, t):
    x.grad = None
    t.grad = None
    """ Physics-based loss function with Burgers equation """
    u_t = derivative(u, t, order=1)
    u_x = derivative(u, x, order=1)
    u_xx = derivative(u, x, order=2)
    u_xxxx = derivative(u_xx, x, order=2)
    return u_t + 100/16*u_x*u + 100/(16**2)*u_xx + 100/(16**4)*u_xxxx

def squared_residual_boundary(u_left,u_right,x_left,x_right):
    x_left.grad = None
    x_right.grad = None
    u_x_left = derivative(u_left, x_left, order=1)
    u_x_right = derivative(u_right, x_right, order=1)
    return ((u_left-u_right)**2).mean() , ((u_x_left-u_x_right)**2 ).mean()