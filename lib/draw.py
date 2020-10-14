import numpy as np
import matplotlib.pyplot as plt


def draw_points(points,values=None,
                ps=500,pc="#CB333A",
                lc="#222222",lw=3):
    xs, ys = points.T
    if values is not None:
        pc = values
    plt.scatter(xs, ys, s = ps,c=pc, edgecolors=lc,lw=lw)


def draw_edges(from_points,
               to_points,
               lw=0.03,
               lc ="#222222"):
    from_xs, from_ys = from_points.T
    to_xs,   to_ys   = to_points.T 
    for i in range(len(from_xs)):
        x0, y0 = from_xs[i], from_ys[i]
        x1, y1 = to_xs[i],   to_ys[i]
        plt.plot([x0,x1],[y0,y1],color=lc,lw=lw)

def draw_vectors(points,
                 vectors,
                 lw=0.005,
                 lc ="#222222",
                 scale=10.0):
    xs,  ys  = points.T
    vxs, vys = vectors.T
    plt.quiver(xs, ys, vxs, vys,
               color=lc, scale=scale,
               width=lw)

