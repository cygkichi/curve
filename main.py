import curve
import numpy as np
import matplotlib.pyplot as plt

from logging import getLogger, StreamHandler, DEBUG

logger = getLogger(__name__)
handler = StreamHandler()
handler.setLevel(DEBUG)
logger.setLevel(DEBUG)
logger.addHandler(handler)
logger.propagate = False

logger.debug('hello')

def draw_points(points,
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
        plt.scatter(xs, ys, s = ps,c=pc, zorder=2,
                    edgecolors=ec,lw=lw)

def draw_vectors(points, vectors,
                 lw=0.005, lc ="#222222",
                 scale=10.0):
    xs,  ys  = points.T
    vxs, vys = vectors.T
    plt.quiver(xs, ys, vxs, vys,
               color=lc, scale=scale,
               width=lw)

if __name__ == '__main__':
    BC = "#dadada"
    LC = "#204969"
    PC = "#08ffc8"

    c = curve.Curves()
    c.add_curve(curve.circle_points( 2, 2,1.5,16))
    c.add_curve(curve.circle_points(-2,-2,1.5,16))
    c.add_curve(curve.square_points(-2, 2,1.5,16))
    c.add_curve(curve.square_points( 2,-2,1.5,16))
    print(c.curves)
    plt.figure(figsize=(10,10))
    plt.rcParams['axes.facecolor'] = BC
    plt.xlim(-5, 5)
    plt.ylim(-5, 5)
    plt.rcParams['axes.axisbelow'] = True
    for ps in c.curves.values():
        draw_points(ps,ps=200,pc=PC,ec=LC,lc=LC,lw=1)
    plt.savefig('./img/test_000.png')
    plt.clf()
    plt.close()
