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


def update(i, x, y):
    if i != 0:
        plt.cla() #現在描写されているグラフを消去
    plt.plot(x[0:i], y[0:i], "r")
    plt.xlim(0,10)
    plt.ylim(0,100)

def draw():
    fig = plt.figure()
    x = np.arange(0, 10, 1)
    y = x ** 2
    ani = animation.FuncAnimation(fig, update, fargs = (x, y), interval = 100, frames = 10)
    ani.save("test.gif", writer = 'imagemagick')

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
    c = curve.Curve(32)
    BC = "#dadada"
    LC = "#204969"
    PC = "#08ffc8"

    for i in range(50):
        logger.debug('istep:%3d A:%3.4f P0:(%3.2f,%3.2f)'%(i,c.A,c.points[0][0],c.points[0][1]))
        plt.figure(figsize=(10,10))
        plt.rcParams['axes.facecolor'] = BC
        plt.xlim(-5, 5)
        plt.ylim(-5, 5)
        for j in range(500):
            c.step(t=True,n=False)
            if(j%10==0):
                c.step(t=False,n=True)
        plt.rcParams['axes.axisbelow'] = True
        plt.title("istep:%03d"%i)
        draw_points(c.points,ps=200,pc=PC,ec=LC,lc=LC,lw=1)
        draw_points(c.midpoints, showline=False,ps=50,pc=PC,ec=LC,lw=1)
        # draw_vectors(c.points, c.n_vectors * np.array([c.kais,c.kais]).T)
        plt.savefig('./img/test_%03d.png'%(i))
        plt.clf()
        plt.close()
