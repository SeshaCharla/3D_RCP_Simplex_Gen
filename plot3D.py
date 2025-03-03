import matplotlib.pyplot as plt
import numpy as np
from numpy import reshape as rs


def plot3D_spx(ax, spx):
    """plots only the simplex
       Retruns the axis plot of the simplex"""
    xs = np.matrix(spx.vMat[:, 0]).A1
    ys = np.matrix(spx.vMat[:, 1]).A1
    thetas = np.matrix(spx.vMat[:, 2]).A1
    p = [0, 1, 2, 3, 1, 2, 0, 3]
    xp = [xs[i] for i in p]
    yp = [ys[i] for i in p]
    thetap = [thetas[i] for i in p]
    ax.plot(xp, yp, thetap)


def plot3D_flow(ax, spx):
    """plots the 3D flow vectors of the simplex"""
    l = np.mean([np.linalg.norm(spx.vMat[1,:]-spx.vMat[2,:]),
                 np.linalg.norm(spx.vMat[2,:]-spx.vMat[3,:]),
                 np.linalg.norm(spx.vMat[3,:]-spx.vMat[1,:])])
    cen = np.mean(spx.vMat, axis=0)
    for i in range(4):
        ax.plot([spx.vMat[i, 0], 0.2*l*(spx.alphaMat[i, 0]/np.linalg.norm(spx.alphaMat[i,:]))+spx.vMat[i, 0]],
                [spx.vMat[i, 1], 0.2*l*(spx.alphaMat[i, 1]/np.linalg.norm(spx.alphaMat[i,:]))+spx.vMat[i, 1]],
                [spx.vMat[i, 2], 0.2*l*(spx.alphaMat[i, 2]/np.linalg.norm(spx.alphaMat[i,:]))+spx.vMat[i, 2]],
                 "--r", label="Closed-loop Vector Field")

    ax.plot([cen[0, 0], 0.3*l*(spx.xi[0, 0])+cen[0, 0]],
            [cen[0, 1], 0.3*l*(spx.xi[1, 0])+cen[0,1]],
            [cen[0, 2], 0.3*l*(spx.xi[2, 0])+cen[0,2]],
             "--k", label="Flow Vector")


def plot3D_normals(ax, spx):
    """plots the 3D normals of the simplices"""
    l = np.mean([np.linalg.norm(spx.vMat[1,:]-spx.vMat[2,:]),
                 np.linalg.norm(spx.vMat[2,:]-spx.vMat[3,:]),
                 np.linalg.norm(spx.vMat[3,:]-spx.vMat[1,:])])
    for i in range(4):
        v_list = list(spx.vertices).copy()
        vi = v_list.pop(i)
        c = (np.mean(np.array(v_list), axis=0))
        p = c + 0.05*l*spx.h[i,:]
        ax.plot([c[0], p[0]], [c[1], p[1]], [c[2], p[2]], "--g", label="Normals")


def plot3D_rcpSpx(ax, spx):
    """Plot the entire rcpSimplex"""
    plot3D_spx(ax, spx)
    plot3D_flow(ax, spx)
    plot3D_normals(ax, spx)

def plot3D_terminal_flow(ax, spx):
    """plots the 3D flow vectors of the simplex"""
    l = np.mean([np.linalg.norm(spx.vMat[1,:]-spx.vMat[2,:]),
                 np.linalg.norm(spx.vMat[2,:]-spx.vMat[3,:]),
                 np.linalg.norm(spx.vMat[3,:]-spx.vMat[1,:])])
    for i in range(4):
        ax.plot([spx.vMat[i, 0], 0.2*l*(spx.alphaMat[i, 0]/np.linalg.norm(spx.alphaMat[i,:]))+spx.vMat[i, 0]],
                [spx.vMat[i, 1], 0.2*l*(spx.alphaMat[i, 1]/np.linalg.norm(spx.alphaMat[i,:]))+spx.vMat[i, 1]],
                [spx.vMat[i, 2], 0.2*l*(spx.alphaMat[i, 2]/np.linalg.norm(spx.alphaMat[i,:]))+spx.vMat[i, 2]],
                 "--r", label="Closed-loop Vector Field")

def plot3D_term_spx(ax, spx):
    """plot the terminal simplex"""
    plot3D_spx(ax, spx)
    plot3D_terminal_flow(ax, spx)
    plot3D_normals(ax, spx)


def plot3D_prism(ax, prism, color="-k"):
    """Plots a prism"""
    n, _ = np.shape(prism.vMat)
    F1 = prism.vMat[0:int(n/2), :]
    F2 = prism.vMat[int(n/2):n, :]
    p = list(range(int(n/2)))
    p.append(0)
    x1 = [F1[i, 0] for i in p]
    y1 = [F1[i, 1] for i in p]
    z1 = [F1[i, 2] for i in p]
    x2 = [F2[i, 0] for i in p]
    y2 = [F2[i, 1] for i in p]
    z2 = [F2[i, 2] for i in p]
    ax.plot(x1, y1, z1, color)
    ax.plot(x2, y2, z2, color)
    for i in range(int(n/2)):
        ax.plot([F1[i, 0], F2[i, 0]],[F1[i, 1], F2[i, 1]], [F1[i, 2], F2[i, 2]], color)

def plot3D_plane(ax, F, color="-k"):
    """plotting the faces alone"""
    n, _ = np.shape(F)
    p = list(range(n))
    p.append(0)
    ax.plot([F[i, 0] for i in p], [F[i, 1] for i in p], [F[i, 2] for i in p], color)

def plot3D_waypts(ax, W, color="-k"):
    """plotting waypoint sets"""
    n, _ = np.shape(W)
    p = list(range(n))
    ax.plot([W[i, 0] for i in p], [W[i, 1] for i in p], [W[i, 2] for i in p], color)
