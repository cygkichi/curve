import numpy as np

def circle_points(x,y,r,N):
    ps = 2*np.pi*np.arange(N)/N
    xs = r*np.cos(ps) + x
    ys = r*np.sin(ps) + y
    points = np.array([xs,ys]).T
    return points

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
    points = np.array([xs,ys]).T
    return points

def cassini_points(x,y,N,c1=10,c2=1):
    ps = 2*np.pi*np.arange(N)/N
    ds = c1*np.cos(2*ps)
    rs = np.sqrt(ds+np.sqrt(ds**2+c2))
    xs = rs * np.cos(ps) + x
    ys = rs * np.sin(ps) + y
    points = np.array([xs,ys]).T
    return points
