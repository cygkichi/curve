import curve
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

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
                lw=3, lc ="#222222",
                showline=True,showpoint=True):
    xs, ys = points.T
    if(showline):
        lxs = np.concatenate([xs,xs[:1]])
        lys = np.concatenate([ys,ys[:1]])
        plt.plot(lxs,lys,'-',lw=lw,c=lc, zorder=1)
    if(showpoint):
        plt.scatter(xs, ys, s = ps,c=pc, zorder=2)

def draw_vectors(points, vectors,
                 lw=0.005, lc ="#222222",
                 scale=10.0):
    xs,  ys  = points.T
    vxs, vys = vectors.T
    plt.quiver(xs, ys, vxs, vys,
               color=lc, scale=scale,
               width=lw)

if __name__ == '__main__':
    c = curve.Curve(10)
    plt.figure(figsize=(10,10))
    plt.xlim(-2,2)
    plt.ylim(-2,2)
    plt.grid()
    plt.rcParams['axes.axisbelow'] = True

    draw_points(c.points)
    draw_points(c.midpoints, showline=False,ps=100)
    draw_vectors(c.midpoints, c.nm_vectors)
    plt.savefig('test.png')
