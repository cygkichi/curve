import numpy as np
from PIL import Image

class Curve(object):
    def __init__(self,points):
        self.points = points
        self.N = len(self.points)
        self.setup_externalforce('./sample_02.png')
        self.calc()

    def calc(self):
        self.is_inside  = np.zeros(len(self.points), dtype=np.bool)
        self.rs         = _calc_rs(self.points)
        self.midpoints  = _calc_midpoints(self.points)
        self.tm_vectors = _calc_tm_vectors(self.points)
        self.nm_vectors = _calc_nm_vectors(self.tm_vectors)
        self.thetas     = _calc_thetas(self.tm_vectors)
        self.phis       = _calc_phis(self.thetas)
        self.A          = _calc_A(self.points)
        self.G          = _calc_G(self.points, self.A)
        self.sins       = _calc_sins(self.phis)
        self.coss       = _calc_coss(self.phis)
        self.tans       = _calc_tans(self.phis)
        self.t_vectors  = _calc_t_vectors(self.tm_vectors, self.coss)
        self.n_vectors  = _calc_n_vectors(self.t_vectors)
        self.psis       = _calc_psis(self.sins, self.rs)
        self.ws         = _calc_ws(self.psis, self.coss)
        self.kais       = _calc_kais2(self.rs, self.coss, self.tans)


    def step(self,dt=0.001,t=True,n=True):
        for i in range(self.N):
            dx,dy =0,0
            if(t):
                dx += self.t_vectors[i][0]*self.ws[i]*dt
                dy += self.t_vectors[i][1]*self.ws[i]*dt
            if(n):
                x,y = self.points[i]
                #ef  = self.exforce(x,y)
                #dx += 3*(-self.kais[i] + ef)*self.n_vectors[i][0]*dt
                #dy += 3*(-self.kais[i] + ef)*self.n_vectors[i][1]*dt
                dx += 3*(self.kais.mean()-self.kais[i])*self.n_vectors[i][0]*dt
                dy += 3*(self.kais.mean()-self.kais[i])*self.n_vectors[i][1]*dt
            self.points[i][0] += dx
            self.points[i][1] += dy
        self.calc()

    def setup_externalforce(self, imagefile):
        ims = np.array(Image.open(imagefile)).sum(axis=2)
        ims_max = np.max(ims)
        ims_min = np.min(ims)
        self.ex_ims = 1 - (ims - ims_min)/(ims_max - ims_min)
        self.ex_x0 = -5
        self.ex_y0 = -5
        self.ex_x1 = 5
        self.ex_y1 = 5
        self.ex_Nx = self.ex_ims.shape[0]
        self.ex_Ny = self.ex_ims.shape[1]
        self.ex_dx = (self.ex_x1 - self.ex_x0) / self.ex_Nx
        self.ex_dy = (self.ex_y1 - self.ex_y0) / self.ex_Ny

    def exforce(self, x, y):
        """
        x = ix*dx + x0 wo mitasu ix wo keisan suru.
        """
        ix = int(( x-self.ex_x0)/self.ex_dx)
        iy = int((-y-self.ex_y0)/self.ex_dy)

        # hamidashi shusei
        ix = np.clip(ix,0, self.ex_Nx-1)
        iy = np.clip(iy,0, self.ex_Ny-1)

        return (3+12)*self.ex_ims[iy,ix]-3
        # return (2+48)*self.ex_ims[iy,ix]-2


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

    if detA == 0.00:
        return False

    s = ( ay1*bx1 - ax1*by1)/detA
    t = (-ay0*bx1 + ax0*by1)/detA

    if (s+t>1) and (s>0) and (t>0):
        return True
    else:
        return False

def _is_inside(point, points,
               outer_point =np.array([1000.0, 1000.0])):
    """
    あるpointが、閉曲線pointsの内部に存在するか判定する。
    """
    cross_count = 0
    point_a0 = point
    point_a1 = outer_point
    for i in range(len(points)):
        point_b0 = points[i-1]
        point_b1 = points[i]
        if _is_intersected(point_a0,point_a1,
                           point_b0,point_b1):
            cross_count += 1
    is_inside = cross_count%2 == 1
    return is_inside

def _cross_check(curves):
    cross_list = []
    inside_list = []
    for c0_i, c0 in enumerate(curves):
        last_point = c0.points[c0.N - 1]
        is_inside  = np.array([_is_inside(last_point, c.points)
                               for c_i, c in enumerate(curves)
                               if c0_i != c_i]).any()
        inside_list_0 = []
        for line0_i in range(c0.N):
            #これより line0 が交差しいるか判定する。
            point00 = c0.points[line0_i -1]
            point01 = c0.points[line0_i]
            for c1_i, c1 in enumerate(curves):
                for line1_i in range(c1.N):
                    if (c0_i == c1_i) and (line0_i == line1_i):
                        break
                    point10 = c1.points[line1_i - 1]
                    point11 = c1.points[line1_i]
                    if _is_intersected(point00,point01,point10,point11):
                        is_inside = np.invert(is_inside)
                        if c0_i <= c1_i:
                            cross_list.append([(c0_i,line0_i-1 if line0_i > 0 else c0.N-1),(c0_i,line0_i),
                                               (c1_i,line1_i-1 if line1_i > 0 else c1.N-1),(c1_i,line1_i)])
            inside_list_0.append(is_inside)
        inside_list.append(inside_list_0)
    return cross_list, inside_list


def _combine_curves(curves, cross_list, inside_list):
    trace_list =  [[ int(np.invert(inside)) for inside in  ilist]  for ilist in  inside_list]
    # print(trace_list)
    points_list = []
    points = []
    c0_i = 0
    p0_i = 0
    while True:
        # print()
        # print('start',c0_i,p0_i)

        if trace_list[c0_i][p0_i] == 0:
            points_list.append(np.array(points))
            points = []
            for c1_i, t in enumerate(trace_list):
                for p1_i, v in enumerate(t):
                    if v == 1:
                        c0_i = c1_i
                        p0_i = p1_i

        c0 = curves[c0_i]
        p0 = c0.points[p0_i]


        #check cross_list
        is_cross = False
        cross_point = None
        cross_id    = None
        for i,cross in enumerate(cross_list):
            for c1_i, p1_i in cross:
                if (c0_i == c1_i) and (p0_i == p1_i):
                    is_cross = True
                    cross_point = cross
                    cross_id    = i
        # 交差している場合は、内外確認する
        # もし内の場合は、点の入替を行う
        is_inside = None
        next_point = None
        if is_cross:
            is_inside = inside_list[c0_i][p0_i]
            if not is_inside:
                # cross_pointから曲線Noがことなり、外側の点を選択する。
                for c1_i, p1_i in cross_point:
                    if (c0_i != c1_i):
                        if not inside_list[c1_i][p1_i]:
                            # print(cross_id)
                            cross_list.pop(cross_id)
                            next_point = (c1_i, p1_i)

        points.append(p0)
        trace_list[c0_i][p0_i] = 0

        # print(cross_list)
        # print(c0_i,p0_i,is_cross,is_inside,trace_list, next_point)
        # countup
        if next_point is None:
            p0_i += 1
            if p0_i == c0.N:
                p0_i = 0
        else:
            c0_i, p0_i = next_point

        if sum([ sum(t) for t in trace_list]) == 0:
            points_list.append(np.array(points))
            break

        # import time; time.sleep(1)

    return points_list
