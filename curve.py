import numpy as np
from PIL import Image

class Curves(object):
    def __init__(self):
        self.num_curves = 0
        self.curves     = {}

    def add_curve(self, points):
        self.num_curves += 1
        self.curves[self.num_curves] = points

def circle_points(x,y,r,N):
    ps = 2*np.pi*np.arange(N)/N
    xs = r*np.cos(ps) + x
    ys = r*np.sin(ps) + y
    return np.array([xs,ys]).T

def square_points(x,y,d,N):
    def shift_p(p):
        if p > np.pi/4:
            return shift_p(p - np.pi/2)
        else:
            return p
    ps = 2*np.pi*np.arange(N)/N
    rs = np.array([d/np.cos(shift_p(p)) for p in ps])
    xs = rs*np.cos(ps) + x
    ys = rs*np.sin(ps) + y
    return np.array([xs,ys]).T


def init_shape(us):
    points = []
    for u in us:
        p = 2*np.pi*u
        r = 1 + 3*np.random.rand()
        x = r*np.cos(p)
        y = r*np.sin(p)
        points.append([x,y])
    return np.array(points)

def init_shape3(us):
    def shift_p(p):
        if p > np.pi/4:
            return shift_p(p - np.pi/2)
        else:
            return p
    points = []
    for u in us:
        p = 2*np.pi*u
        r = 4/np.cos(shift_p(p))
        x = r*np.cos(p)
        y = r*np.sin(p)
        points.append([x,y])
    return np.array(points)


def init_shape2(us):
    points = []
    for u in us:
        p = 2*np.pi*u
        c1, c2 = 10, 1
        d = c1*np.cos(2*p)
        r = np.sqrt(d+np.sqrt(d**2+c2))
        x = r*np.cos(p)
        y = r*np.sin(p)
        points.append([x,y])
    return np.array(points)

def _generate_us(N):
    return np.arange(N)/N

def _calc_rs(points):
    diff = np.diff(points, axis=0, prepend=points[-1:])
    rs = np.linalg.norm(diff, ord=2, axis=1)
    return rs

def _calc_midpoints(points):
    diff = np.diff(points, axis=0, prepend=points[-1:])
    midpoints = points - diff * 0.5
    return midpoints

def _calc_tm_vectors(points):
    diff = np.diff(points, axis=0, prepend=points[-1:])
    rs = np.linalg.norm(diff, ord=2, axis=1)
    tm_vectors = (diff.T / rs).T
    return tm_vectors

def _calc_nm_vectors(tm_vectors):
    nm_vectors = tm_vectors[:,[1,0]]*np.array([1,-1])
    return nm_vectors

def _calc_thetas(tm_vectors):
    signs = np.where(tm_vectors.T[1]>=0, 1, -1)
    acoss = np.arccos(tm_vectors.T[0])
    thetas = signs*acoss
    return thetas

def _calc_phis(thetas):
    phis  = np.diff(thetas, append=thetas[:1])
    phis += np.where(phis >  np.pi, -2*np.pi, 0)
    phis += np.where(phis < -np.pi,  2*np.pi, 0)
    return phis

def _calc_A(points):
    N = len(points)
    A = 0.0
    for i in range(N):
        x0,y0 = points[i-1]
        x1,y1 = points[i]
        A += (x0*y1 - x1*y0)*0.5
    return A

def _calc_G(points, A):
    N  = len(points)
    Gx = 0.0
    Gy = 0.0
    for i in range(N):
        x0,y0 = points[i-1]
        x1,y1 = points[i]
        det = x0*y1 - x1*y0
        Gx += det*(x1 + x0)/(6*A)
        Gy += det*(y1 + y0)/(6*A)
    return [Gx,Gy]

def _calc_sins(phis):
    return np.sin(phis * 0.5)

def _calc_coss(phis):
    return np.cos(phis * 0.5)

def _calc_tans(phis):
    return np.tan(phis * 0.5)

def _calc_t_vectors(tm_vectors, coss):
    """
    TODO: Refactoring with numpy
    """
    N = len(tm_vectors)
    t_vectors = []
    for i in range(N):
        if i == N-1:
            tx = (tm_vectors[i][0] + tm_vectors[0][0])/(2*coss[i])
            ty = (tm_vectors[i][1] + tm_vectors[0][1])/(2*coss[i])
        else:
            tx = (tm_vectors[i][0] + tm_vectors[i+1][0])/(2*coss[i])
            ty = (tm_vectors[i][1] + tm_vectors[i+1][1])/(2*coss[i])
        t_vectors.append([tx,ty])
    return np.array(t_vectors)

def _calc_n_vectors(t_vectors):
    return _calc_nm_vectors(t_vectors)

def _calc_psis(sins, rs, omega=10.0):
    """
    TODO: Refactoring with numpy
    """
    N = len(sins)
    L = sum(rs)
    psis = [0]
    for i in range(1,N):
        psi = (L/float(N) - rs[i])*omega
        psis.append(psi)
    return psis

def _calc_ws(psis, coss):
    """
    TODO: Refactoring with numpy
    """
    N = len(psis)
    ws = [0]
    for i in range(1,N):
        w = (ws[-1]*coss[i-1]+psis[i])/coss[i]
        ws.append(w)
    mean_w = sum(ws) / float(N)
    ws = [w - mean_w for w in ws]
    return ws

def _calc_kais(rs, sins):
    """
    TODO: Refactoring with numpy
    3 points
    """
    N = len(sins)
    return np.array([2*sins[i]/(rs[i]+rs[i-1]) for i in range(N)])

def _calc_kais2(rs, coss, tans):
    """
    5 points
    """
    N = len(rs)
    kais = np.array([(tans[i] + tans[i-1])/rs[i] for i in range(N)])
    return np.array([(kais[i]+kais[ 0 if i == N-1 else i+1 ])/(2*coss[i]) for i in range(N)])


def _is_intersected(point_a0, point_a1, point_b0, point_b1):
    """
    Check to see if the two line segments intersect.
    if vec(b1-b0) + S*vec(a0-b0) + T*vec(a1-b0),
    then (S+T>1) and (S>0) and (T>0),
    then line_a and lineb are intersected.
    """
    ax0, ay0 = point_a0 - point_b0
    ax1, ay1 = point_a1 - point_b0
    bx1, by1 = point_b1 - point_b0

    detA = ax0 * ay1 - ax1 * ay0

    s = ( ay1*bx1 - ax1*by1)/detA
    t = (-ay0*bx1 + ax0*by1)/detA

    if (s+t>1) and (s>0) and (t>0):
        return True
    else:
        return False
