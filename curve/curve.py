import numpy as np

class Curve(object):
    def __init__(self):
        self.num_points  = 0
        self.num_curves  = 0
        self.points      = np.array([])
        self.curve_ids   = np.array([],dtype=np.int)
        self.next_ids    = np.array([],dtype=np.int)

    def add_curve(self, points):
        num_new_points = len(points)
        next_curve_id  = self.num_curves
        self.num_points  += num_new_points
        self.num_curves  += 1
        if next_curve_id == 0:
            self.points = points
        else:
            self.points = np.vstack([self.points, points])
        self.curve_ids = np.hstack([self.curve_ids, np.repeat(next_curve_id, num_new_points)])
        self.next_ids  = np.hstack([self.next_ids, np.roll(np.arange(len(self.next_ids),len(self.next_ids)+num_new_points),-1)])

    @property
    def next_points(self):
        return self.points[self.next_ids]

    @property
    def prev_points(self):
        return self.points[self.prev_ids]

    @property
    def prev_ids(self):
        prev_ids = np.zeros(self.num_points, dtype=int)
        for i in range(self.num_points):
            prev_ids[self.next_ids[i]] = i
        return np.array(prev_ids)

    def show_property(self):
        print("Num Points : %d"%self.num_points)
        print("Num Curves : %d"%self.num_curves)
        print("Points Loc : "+str(self.points))
        print("Curve ID   : "+str(self.curve_ids))
        print("Next ID    : "+str(self.next_ids))
        print("Prev ID    : "+str(self.prev_ids))

    def calc(self):
        self.is_inside  = None
        self.rs         = None
        self.midpoints  = None
        self.tm_vectors = None
        self.nm_vectors = None
        self.thetas     = None
        self.phis       = None
        self.A          = None
        self.G          = None
        self.sins       = None
        self.coss       = None
        self.tans       = None
        self.t_vectors  = None
        self.n_vectors  = None
        self.psis       = None
        self.ws         = None
        self.kais       = None


if __name__ == "__main__":
    c = Curve()
    c.add_curve(np.array([[0,0],[0,1],[1,1],[1,0]]))
    c.add_curve(np.array([[0,0],[0,1],[1,1],[1,0]]))
    c.add_curve(np.array([[0,0],[0,1],[1,1]]))
    c.show_property()

