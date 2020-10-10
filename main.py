import numpy as np
import matplotlib.pyplot as plt
from logging import getLogger, StreamHandler, DEBUG

import lib.curve as curve
import lib.shape as shape

logger = getLogger(__name__)
handler = StreamHandler()
handler.setLevel(DEBUG)
logger.setLevel(DEBUG)
logger.addHandler(handler)
logger.propagate = False

def draw_points(points,values=None,
                ps=500,pc="#CB333A",
                ec="#222222",
                lw=3, lc ="#222222",
                showline=True,showpoint=True):
    xs, ys = points.T
    if(showline):
        lxs = np.concatenate([xs,xs[:1]])
        lys = np.concatenate([ys,ys[:1]])
        plt.plot(lxs,lys,'-',lw=lw,c=lc, zorder=1)
    if(showpoint):
        if values is not None:
            pc = values
        plt.scatter(xs, ys, s = ps,c=pc,
                    zorder=2, edgecolors=ec,lw=lw)


def draw_vectors(points, vectors,
                 lw=0.005, lc ="#222222",
                 scale=10.0):
    xs,  ys  = points.T
    vxs, vys = vectors.T
    plt.quiver(xs, ys, vxs, vys,
               color=lc, scale=scale,
               width=lw)

if __name__ == '__main__':
    BC = "#b8fd3f"
    LC = "#4c0cc0"
    PC = "#4c0cc0"

    curves=[]
    curves.append(curve.Curve(shape.circle_points(x= 0,    y=1.5,r=1.8  ,N=2*12)))
    curves.append(curve.Curve(shape.circle_points(x= 0.5,  y= -1,r=1.8,N=2*12)))
    curves.append(curve.Curve(shape.circle_points(x= -2.5, y= -1,r=1  ,N=12)))

    for _ in range(100):
        for c in curves:
            c.step(n=False)


    for j in range(100):
        print(j)

        # conbine
        if j %20 == 0:
            cross_list, inside_list = curve._cross_check(curves)
            print(cross_list)
            points_list = curve._combine_curves(curves, cross_list, inside_list)
            curves=[]
            for points in points_list:
                if len(points) > 0:
                    curves.append(curve.Curve(points))

        for _ in range(10):
            for c in curves:
                c.step(n=False)
        for _ in range(10):
            for c in curves:
                c.step()

        plt.figure(figsize=(10,10))
        plt.rcParams['axes.facecolor'] = BC
        plt.xlim(-5, 5)
        plt.ylim(-5, 5)
        plt.title('step:%03d'%j)
        plt.rcParams['axes.axisbelow'] = True
        for i,c in enumerate(curves):
            draw_points(c.points, ps=120,pc=PC,ec=LC,lc=LC,lw=1)
        plt.savefig('./img/test_%03d.png'%j, dpi=100)
        plt.clf()
        plt.close()
