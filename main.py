import numpy as np
import matplotlib.pyplot as plt
import lib.curve
import lib.shape
import lib.draw

if __name__ == "__main__":
    c = lib.curve.Curve()
    c.add_curve(lib.shape.circle_points(-2,-1,1,10))
    c.add_curve(lib.shape.square_points(1,2,1,16))
    c.add_curve(lib.shape.circle_points(2,-2,2,16))
    c.show_property()

    plt.figure(figsize=(10,10))

    BC = "#b8fd3f"
    LC = "#4c0cc0"
    PC = "#4c0cc0"
    plt.rcParams['axes.facecolor'] = BC
    plt.xlim(-5, 5)
    plt.ylim(-5, 5)
    plt.rcParams['axes.axisbelow'] = True
    lib.draw.draw_points(c.points, ps=120,pc=PC,lc=LC,lw=1)
    lib.draw.draw_edges(c.points, c.prev_points,lc=LC,lw=1)
    plt.savefig('./test.png', dpi=100)
    plt.clf()


