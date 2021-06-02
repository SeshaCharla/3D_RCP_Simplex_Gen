import numpy as np

L = 2.468       # Car wheel base
v_max = -1.39
v_min = -2.78
theta_b = np.deg2rad(33)

def f(z, u):
    """The non-linear system
    z = [x    ]     u = [v  ]
        [y    ]         [phi]
        [theta]              """
    theta = z[2, 0]
    v = u[0, 0]
    phi = u[1, 0]
    x_dot = v*np.cos(phi)*np.sin(theta)
    y_dot = v*np.cos(phi)*np.sin(theta)
    theta_dot = (v/L)*np.sin(phi)
    return np.matrix([[x_dot], [y_dot], [theta_dot]])


class affine_sys():
    """affine system class"""
    def __init__(self, A, B, a):
        self.A = A
        self.B = B
        self.a = a

#Linearising system about (x0, y0, \theta_0), (v0 = (v_max + v_min)/2, \phi_0 = 0)

def get_linear(z0):
    """Returns the linearized system parameters"""
    v0 = (v_max + v_min)/2
    theta0 = z0[2, 0]
    A = np.matrix([[0, 0, -v0*np.sin(theta0)],
                   [0, 0,  v0*np.cos(theta0)],
                   [0, 0, 0                 ]])
    B = np.matrix([
        [np.cos(theta0), 0     ],
        [np.sin(theta0), 0     ],
        [0             , (v0/L)]])
    a = np.matrix([[v0*theta0*np.sin(theta0) ],
                   [-v0*theta0*np.cos(theta0)]])
    asys = affine_sys(A, B, a)
    return asys
