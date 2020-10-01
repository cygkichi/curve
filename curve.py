import numpy as np

class Curve(object):
    def __init__(self,N):
        self.N = N
        self.us = _generate_us(N)
        self.points = init_shape(self.us)

def init_shape(us):
    points = []
    for u in us:
        p = 2*np.pi*u
        x = 1*np.cos(p)
        y = 1*np.sin(p)
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

def _calc_t_vectors(tm_vectors, coss):
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
    return t_vectors


def _calc_psis(sins, rs, omega=10.0):
    N = len(sins)
    L = sum(rs)
    psis = [0]
    for i in range(1,N):
        psi = (L/float(N) - rs[i])*omega
        psis.append(psi)
    return psis

def _calc_ws(psis, coss):
    N = len(psis)
    ws = [0]
    for i in range(1,N):
        w = (ws[-1]*coss[i-1]+psis[i])/coss[i]
        ws.append(w)
    mean_w = sum(ws) / float(N)
    ws = [w - mean_w + 0.5 for w in ws]
    return ws


def _calc_kais(rs, tans):
    N = len(tans)
    return [(tans[i]+tans[i-1])/rs[i] for i in range(N)]

