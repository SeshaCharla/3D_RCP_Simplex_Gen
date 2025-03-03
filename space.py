import numpy as np
from numpy.core.fromnumeric import ptp
import pypoman as pp
from numpy import reshape as rs


class ptope():
    """Convex Polytope class"""
    def __init__(self, vMat):
        n, m = np.shape(vMat)
        self.vMat = vMat
        self.vertices = [self.vMat[i, :].A1 for i in range(n)]
        self.A, self.b = pp.duality.compute_polytope_halfspaces(self.vertices)

class theta_ptope():
    """Polytope for linearization"""
    def __init__(self, th0, increasing=True):
        self.A = np.matrix([[0, 0, -1],
                            [0, 0, 1]])
        if increasing:
            self.b = np.array([-(th0-np.deg2rad(1.25)), th0+np.deg2rad(10)])
        else:
            self.b = np.array([-(th0-np.deg2rad(10)), th0+np.deg2rad(1.25)])


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

# rgn_vMat = np.matrix([[2.819, 1.976, theta_min], [2.819, 1.726, theta_min], [-1.379, -0.976, theta_min], [-5.181, -0.976, theta_min], [-5.181, 1.976, theta_min],
#                       [2.819, 1.976, theta_max], [2.819, 1.726, theta_max], [-1.379, -0.976, theta_max], [-5.181, -0.976, theta_max], [-5.181, 1.976, theta_max]])

rgn_vMat = np.matrix([[3, 2.1, theta_min], [3, 1.5, theta_min], [-1.1, -1.1, theta_min], [-5.4, -1.1, theta_min], [-5.4, 2.1, theta_min],
                      [3, 2.1, theta_max], [3, 1.5, theta_max], [-1.1, -1.1, theta_max], [-5.4, -1.1, theta_max], [-5.4, 2.1, theta_max]])
rgn = ptope(rgn_vMat)


ptope_list = [rgn]


## Polytopic Support
# Widenning coefficients
dx_k = [-0.850, -0.834, -0.785, -0.707, -0.707, -0.785, -0.834, -0.850]
dy_k = [-0.094, -0.286, -0.445, -0.592, -0.592, -0.445, -0.286, -0.094]
w_kx = [0,0,0,0,0,0,0,0]
w_ky = [0.09375, 0.12, 0.12, 0.12, 0.12, 0.12, 0.12, 0.09375]

#Coordinates
x = [2.319, 2.819, 2.819, 2.319, 1.496, 1.969, 1.969, 1.469, 0.635, 1.135,
    1.135, 0.635, -0.150, 0.350, 0.350, -0.150, -0.857, -0.357, -0.357, -0.857,
-0.857, -0.357, -0.357, -0.857, -1.563, -1.063, -1.063, -1.563, -2.349, -1.849,
-1.849, -2.349, -3.182, -2.682, -2.682, -3.182, -4.032, -3.532, -3.532, -4.032,
2.114, 1.302, 0.493, -0.253, -0.960, -1.706, -2.516, -3.357]

y = [1.976, 1.976, 1.726, 1.726, 1.976, 1.976, 1.539, 1.539, 1.820, 1.820,
1.123, 1.123, 1.505, 1.505, 0.547, 0.547, 1.043, 1.043, -0.175, -0.175,
1.043, 1.043, -0.175, -0.175, 0.580, 0.580, -0.897, -0.897, 0.265, 0.265,
-1.472, -1.472, 0.109, 0.109, -1.888, -1.888, 0.109, 0.109, -2.076, -2.076,
1.804, 1.614, 1.249, 0.730, 0.138, -0.381, -0.747, -0.936]

theta = [0,0,0,0, 0.196, 0.196, 0.196, 0.196, 0.393, 0.393,
        0.393, 0.393, 0.589, 0.589, 0.589, 0.589, 0.785, 0.785, 0.785, 0.785,
        0.785, 0.785, 0.785, 0.785, 0.589, 0.589, 0.589, 0.589, 0.393, 0.393,
        0.393, 0.393, 0.196, 0.196, 0.196, 0.196, 0, 0, 0, 0,
        0.098, 0.295, 0.491, 0.687, 0.687, 0.491, 0.295, 0.098]


#Vertices of Simplices indices
j1_m = [41, 41, 41, 41, 4,  4,  4,  4,  41, 41, 4,  4,
      42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42,
      43, 43, 43, 43, 43, 43, 43, 43, 43, 43, 43, 43,
      44, 44, 44, 44, 44, 44, 44, 44, 44, 44,
      44, 44, 45, 45, 45, 45, 45, 45, 45, 45,
      45, 45, 45, 45, 30, 30, 28, 28, 28, 27,
      27, 27, 30, 30, 27, 27, 34, 34, 32, 32,
      32, 31, 31, 31, 34, 34, 31, 31, 38, 38,
      38, 38, 36, 36, 36, 36, 38, 38, 36, 36]
j1 = [x-1 for x in j1_m]

j2_m = [6,  3,  1,  6,  3,  41, 8,  41, 7,  8,  41, 1,
      10, 5,  8,  12, 5,  10, 7,  8,  11, 12, 7,  7,
      14, 9,  14, 9,  11, 12, 12, 16, 11, 11, 15, 16,
      16, 15, 16, 20, 13, 18, 13, 18, 20, 19,
      16, 15, 26, 23, 21, 24, 26, 23, 28, 24,
      27, 26, 24, 21, 46, 25, 32, 46, 25, 28,
      46, 30, 32, 46, 46, 46, 47, 29, 36, 47,
      29, 32, 47, 34, 36, 37, 37, 37, 48, 33,
      48, 35, 48, 33, 40, 48, 40, 48, 48, 35]
j2 = [x-1 for x in j2_m]

j3_m = [1, 6,  6,  3, 41,  8, 41,  1,  6,  7,  3, 41,
      5,   8, 12,  8, 10,  7, 10,  7, 10, 11,  5,  8,
      11, 14,  9, 12, 14, 11, 16, 12,  9, 12, 14, 15,
      15, 18, 20, 16, 18, 15, 16, 13, 19, 18,
      13, 16, 21, 26, 26, 21, 23, 24, 24, 28,
      28, 27, 23, 23, 25, 46, 46, 32, 46, 46,
      30, 46, 46, 32, 28, 25, 29, 47, 47, 36,
      47, 47, 34, 47, 47, 36, 32, 29, 33, 48,
      35, 48, 40, 48, 48, 35, 48, 40, 33, 48]
j3 = [x-1 for x in j3_m]

j4_m = [5,   7,  2,  2,  7,  7,  5,  5,  5,  5,  2,  2,
      9,   9,  9, 11,  6,  6, 11, 11,  9,  9,  6,  5,
      10, 10, 13, 13, 15, 15, 13, 15, 10,  9, 13, 13,
      19, 19, 17, 19, 14, 14, 17, 17, 17, 17,
      14, 14, 22, 22, 25, 25, 27, 27, 25, 27,
      25, 25, 21, 22, 29, 26, 31, 29, 29, 31,
      31, 26, 29, 31, 25, 26, 33, 30, 35, 33,
      33, 35, 35, 30, 33, 35, 29, 30, 37, 34,
      34, 39, 37, 37, 39, 39, 37, 39, 34, 34]
j4 = [x-1 for x in j4_m]


# Initial box
I = np.matrix([[2.319, 1.976, 0], [2.319, 1.726, 0],[2.819, 1.726,
    0], [2.819, 1.976, 0]])

# Eniding Box
E =np.matrix([[-5.181, -0.676, 0], [-5.181, -0.976, 0], [-1.379,
    -0.976, 0],[-1.379, -0.676, 0]])


# Polytopes
def P(k):
    """vertex indices of kth polytope k = 0, 1, 2, 3"""
    if (k < 4):
        return [x for x in range(4*k, 4*(k+2))]
    else:
        return [x for x in range(4*(k+1), 4*(k+3))]

def P_inout(k):
    """ vertex indices of in exit facets """
    v = P(k)
    Fin = v[:4]
    Fout = v[4:]
    return [Fin, Fout]

Ph = []
for i in range(8):
    poltope = ptope(np.matrix([[x[v], y[v], theta[v]] for v in P(i)]))
    Ph.append(poltope)


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
    for i in range(8):
        plot3D_prism(ax, Ph[i])
    ax.set_xlim3d(-10, 10)
    ax.set_ylim3d(-10, 10)
    ax.set_zlim3d(-1, 2)
    plt.show()
