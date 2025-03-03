import space as spc
import system as ss
import chain_funcs as cf
import numpy as np
from numpy import reshape as rs
import support_vecs as svc
import matplotlib.pyplot as plt
import pypoman as pp
import plot3D
import space as spc


chain = []
n = 3
F = spc.I
s_in = spc.W[0,:].T
u_max = ss.u_max
u_min = ss.u_min
Sim = cf.init_chain(3, F, s_in,  u_max, u_min, spc.W, spc.ptope_list)
chain.append(Sim)
j = 0
old_spx = Sim
while (svc.which_seg(n, s_in, spc.W) != (np.shape(spc.W))[0] -2) and j<20:
    Sim = cf.prop_chain(n, old_spx, u_max,  u_min, spc.W, spc.ptope_list)
    s_in = Sim.so
    chain.append(Sim)
    old_spx = Sim
    j = j + 1
# term_sim = cf.term_chain(2, ss.lsys, chain[-1], u_max, u_min, spc.W)
# chain.append(term_sim)

if __name__=="__main__":

    # Plot
    import matplotlib.pyplot as plt
    import plot3D

    ax = plt.figure().add_subplot(projection="3d")
    # plot3D.plot3D_prism(ax, spc.CarA)
    # plot3D.plot3D_prism(ax, spc.CarB)
    # plot3D.plot3D_prism(ax, spc.curb)
    # plot3D.plot3D_prism(ax, spc.rgn)
    # plot3D.plot3D_plane(ax, spc.I)
    # #plot3D_plane(ax, E)
    plot3D.plot3D_waypts(ax, spc.W[0:2+1,:])
    for spxi in chain:
        plot3D.plot3D_rcpSpx(ax, spxi)
    plt.show()
