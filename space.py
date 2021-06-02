import numpy as np
import pypoman as pp
from numpy import reshape as rs


class ptope():
    """Convex Polytope class"""
    def __init__(self, vMat):
        n, m = np.shape(vMat)
        self.vMat = vMat
        self.vertices = [self.vMat[i, :].A1 for i in range(n)]
        self.A, self.b = pp.duality.compute_polytope_halfspaces(self.vertices)


"""General way of representing vertex sets using matrices: rows -- vertics"""
theta_max = np.deg2rad(60)
theta_min = np.deg2rad(-10)

CarA_vMat = np.matrix([[2, -1.952, theta_min], [6.198, -1.952, theta_min], [6.198, 0, theta_min], [2, 0, theta_min],
                       [2, -1.952, theta_max], [6.198, -1.952, theta_max], [6.198, 0, theta_max], [2, 0, theta_max]])
CarA = ptope(CarA_vMat)

CarB_vMat = np.matrix([[-10.198, 0, theta_min], [-10.198, -1.952, theta_min], [-6, -1.952, theta_min], [-6, 0, theta_min],
                      [-10.198, 0, theta_max], [-10.198, -1.952, theta_max], [-6, -1.952, theta_max], [-6, 0, theta_max]])
CarB = ptope(CarB_vMat)

curb_vMat = np.matrix([[-10.198, -2.002, theta_min], [-10.198, -3.952, theta_min], [6.198, -3.952, theta_min], [6.198, -2.002,      theta_min],
                       [-10.198, -2.002, theta_max], [-10.198, -3.952, theta_max], [6.198, -3.952, theta_max], [6.198, -2.002, theta_max]])
curb = ptope(curb_vMat)

rgn_vMat = np.matrix([[2.819, 1.976, theta_min], [2.819, 1.726, theta_min], [-1.379, -0.976, theta_min], [-5.181, -0.976, theta_min], [-5.181, 1.976, theta_min],
                      [2.819, 1.976, theta_max], [2.819, 1.726, theta_max], [-1.379, -0.976, theta_max], [-5.181, -0.976, theta_max], [-5.181, 1.976, theta_max]])
rgn = ptope(rgn_vMat)


# Initial box
I = np.matrix([[2.569, 1.976, 0], [2.319, 1.726, 0], [2.819, 1.726,
    0]]);

# Eniding Box
E = np.matrix([[-5.181, -0.676, 0], [-5.181, -0.976, 0], [-1.379,
    -0.976, 0], [-1.379, -0.676, 0]]);

W = np.matrix([[ 2.519 ,  1.876 ,  0],
       [ 2.0994,  1.8386,  0.098 ],
       [ 1.6798,  1.8012,  0.196 ],
       [ 1.2574,  1.6712,  0.2945],
       [ 0.835 ,  1.5412,  0.393 ],
       [ 0.4425,  1.3315,  0.491 ],
       [ 0.05  ,  1.1218,  0.589 ],
       [-0.3035,  0.8388,  0.687 ],
       [-0.657 ,  0.5558,  0.785 ],
       [-1.01  ,  0.2725,  0.687 ],
       [-1.363 , -0.0108,  0.589 ],
       [-1.756 , -0.2203,  0.491 ],
       [-2.149 , -0.4298,  0.393 ],
       [-2.5655, -0.5598,  0.2945],
       [-2.982 , -0.6898,  0.196 ],
       [-3.407 , -0.7274,  0.098 ],
       [-3.832 , -0.765 ,  0.    ]])


ptope_list = [CarA, CarB, curb, rgn]


if __name__=="__main__":
    import matplotlib.pyplot as plt
    from plot3D import *

    ax = plt.figure().add_subplot(projection="3d")
    plot3D_prism(ax, CarA)
    plot3D_prism(ax, CarB)
    plot3D_prism(ax, curb)
    plot3D_prism(ax, rgn)
    plot3D_plane(ax, I)
    plot3D_plane(ax, E)
    plot3D_waypts(ax, W)
    plt.show()
