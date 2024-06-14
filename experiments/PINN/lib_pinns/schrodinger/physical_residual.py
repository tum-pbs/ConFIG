from ..helpers import derivative

# main PDE loss
def squared_physical_residual_internal(h, x, t):
    x.grad = None
    t.grad = None
    u = h[:, 0]
    v = h[:, 1]
    u_t = derivative(u, t, order=1)
    v_t = derivative(v, t, order=1)
    u_xx = derivative(u, x, order=2)
    v_xx = derivative(v, x, order=2)
    f_u = u_t + 0.5*v_xx + (u**2 + v**2)*v
    f_v = v_t - 0.5*u_xx - (u**2 + v**2)*u 
    # note: already squared   
    return f_u**2 + f_v**2

# periodic boundary condition value_part
def squared_residual_boundary_v(h_left,h_right):
    u_left = h_left[:, 0]
    v_left = h_left[:, 1]
    u_right = h_right[:, 0]
    v_right = h_right[:, 1]
    return (u_left-u_right)**2 + (v_left-v_right)**2

# periodic boundary condition derivative_part
def squared_residual_boundary_f(h_left,h_right,x_left,x_right):
    x_left.grad = None
    x_right.grad = None
    u_left = h_left[:, 0]
    v_left = h_left[:, 1]
    u_right = h_right[:, 0]
    v_right = h_right[:, 1]
    u_x_left = derivative(u_left, x_left, order=1)
    v_x_left = derivative(v_left, x_left, order=1)
    u_x_right = derivative(u_right, x_right, order=1)
    v_x_right = derivative(v_right, x_right, order=1)
    return (u_x_left-u_x_right)**2 + (v_x_left-v_x_right)**2

# periodic boundary condition
def squared_residual_boundary(h_left,h_right,x_left,x_right):
    x_left.grad = None
    x_right.grad = None
    u_left = h_left[:, 0]
    v_left = h_left[:, 1]
    u_right = h_right[:, 0]
    v_right = h_right[:, 1]
    u_x_left = derivative(u_left, x_left, order=1)
    v_x_left = derivative(v_left, x_left, order=1)
    u_x_right = derivative(u_right, x_right, order=1)
    v_x_right = derivative(v_right, x_right, order=1)
    return (u_left-u_right)**2 + (v_left-v_right)**2, (u_x_left-u_x_right)**2 + (v_x_left-v_x_right)**2

def squared_residual_initial(predicted_h,ground_truth_h):
    return (predicted_h-ground_truth_h)**2