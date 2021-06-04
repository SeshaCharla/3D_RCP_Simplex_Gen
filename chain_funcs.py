import numpy as np
from numpy import reshape as rs
import cvxpy as cvx
from numpy.lib.shape_base import _apply_along_axis_dispatcher
import normals as nr
import rcpSimgen as simgen
import rcpSimplex as rspx
import space as spc
import support_vecs as svc
import system as syst


def init_chain(n, F, s_in, u_max, u_min, phi, ptope_list):
    """ Create initial simplex"""
    eps = 1e-6

    # s_o and xi
    s_o, xi = svc.chain_flow(n, s_in, phi)
    del_s = s_o-s_in
    del_max = np.linalg.norm(del_s)

    # Getting the linear system
    del_th = del_s[2, 0]
    if del_th >= 0:   # => \theta is increasing
        th0 = np.min(np.matrix(F[2,:]).A1) - 1e-8
        thp = spc.theta_ptope(th0, increasing=True)
    else:             # => \theta is decreasing
        th0 = np.max(np.matrix(F[2,:]).A1) + 1e-8
        thp = spc.theta_ptope(th0, increasing=False)
    asys = syst.get_linear(th0)
    *_, m =  np.shape(asys.B)
    ptope_list.append(thp)

    # Optimization problem
    u = [cvx.Variable((m, 1)) for i in range(n)]
    constraints = []
    obj = 0
    for i in range(n):
        obj += xi.T @ (asys.A @ rs(F[i, :], [n, 1]) + asys.B @ u[i] + asys.a)
        # Flow constraints
        constraints += [xi.T @ (asys.A @ rs(F[i, :], [n, 1]) + asys.B @ u[i] + asys.a) >= eps]
        # input constraints
        constraints += [u[i] <= u_max, u[i]>= u_min]
    prob = cvx.Problem(cvx.Maximize(obj), constraints=constraints)
    if not prob.is_dcp():
        raise(ValueError("The problem doesn't follow DCP rules!!"))
    prob.solve()
    if prob.status in ["infeasible", "unbounded"]:
        raise(ValueError("The optimization problem is {}.\nCheck control input Limits!!".format(prob.status)))
    # Possible facets list
    F_list = []
    F_aug = np.append(F, F, axis=0)
    for i in range(n):
        F_list.append(F_aug[i:i+n, :])
    spx_list= []
    spx_alt_list = []
    ld_list = []
    spx_alt_cerr_list = []
    for i in range(n):
        alpha_r = asys.A @ rs(F_list[i][0, :], [n, 1]) + asys.B @ u[i].value + asys.a
        try:
            spx, ld = simgen.rcp_simgen(n, F_list[i], u[i].value, alpha_r, s_in, u_max, u_min, phi, ptope_list)
        except:
            continue
        if ld > 1e-6 and spx.c_err <= 0.5:
            spx_list.append(spx)
            ld_list.append(ld)
        else:
            spx_alt_list.append(spx)
            spx_alt_cerr_list.append(ld - spx.c_err)
    if ld_list:
        return spx_list[np.argmax(ld_list)]
    elif spx_alt_list:
        return spx_alt_list[np.argmax(spx_alt_cerr_list)]
    else:
        raise(ValueError("No possible simplex!!"))

def prop_chain(n, old_spx, u_max, u_min, phi, ptope_list):
    """ Propagates the simplex chain"""
    spx_list= []
    spx_alt_list = []
    ld_list = []
    spx_alt_cerr_list = []
    for F_next,u0_next,alpha0_next in old_spx.next_list:
        try:
            spx, ld = simgen.rcp_simgen(n, F_next, u0_next, alpha0_next, old_spx.so, u_max, u_min, phi, ptope_list)
        except:
            continue
        if ld > 1e-6 and spx.c_err <= 0.5:
            spx_list.append(spx)
            ld_list.append(ld)
        else:
            spx_alt_list.append(spx)
            spx_alt_cerr_list.append(ld)
    if ld_list:
        return spx_list[np.argmax(ld_list)]
    elif spx_alt_list:
        return spx_alt_list[np.argmax(spx_alt_cerr_list)]
    else:
        try:
            spx, ld = simgen.rcp_simgen(n, old_spx.F_last, old_spx.u0_last, old_spx.alpha_last, old_spx.so, u_max, u_min, phi, ptope_list)
        except:
            raise(ValueError("No possible simplex!!"))

def term_chain(n, asys, old_spx, u_max, u_min, phi):
    """Terminate the chain of simplices by creating simplex with equilibrium inside"""
    F = old_spx.vMat[1:, :]
    return rspx.terminalSimplex(n, asys, F, phi, u_max, u_min)
