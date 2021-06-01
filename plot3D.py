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

def plot3D_term_spx(spx):
    """plot the terminal simplex"""
    plot3D_spx(ax, spx)
    plot3D_terminal_flow(ax, spx)
    plot3D_normals(ax, spx)
