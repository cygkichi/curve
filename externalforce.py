import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

class ExternalForce(object):
    def __init__(self, imagefile):
        ims = np.array(Image.open(imagefile)).sum(axis=2)
        ims_max = np.max(ims)
        ims_min = np.min(ims)
        self.ims = (ims - ims_min)/(ims_max - ims_min)
        self.x0 = -1
        self.y0 = -1
        self.x1 = 1
        self.y1 = 1
        self.Nx = self.ims.shape[0]
        self.Ny = self.ims.shape[1]
        self.dx = (self.x1 - self.x0) / self.Nx
        self.dy = (self.y1 - self.y0) / self.Ny

    def value(self,x,y):
        """
        x = ix*dx + x0 wo mitasu ix wo keisan suru.
        """
        ix = int((x-self.x0)/self.dx)
        iy = int((-y-self.y0)/self.dy)

        # hamidashi shusei
        ix = np.clip(ix,0, self.Nx-1)
        iy = np.clip(iy,0, self.Ny-1)

        return self.ims[iy,ix]

if __name__ == '__main__':
    imgfile = './sample_02.png'
    print('input file : %s'%(imgfile))
    ef = ExternalForce(imgfile)
    print('Nx : %d'%(ef.Nx))
    print('Ny : %d'%(ef.Ny))
    print('dx : %5f'%(ef.dx))
    print('dy : %5f'%(ef.dy))

    plt.figure(figsize=(5,5))
    xs = np.linspace(-1,1.5,30)
    ys = np.linspace(-1,1.5,30)
    for x in xs:
        for y in ys:
            plt.scatter(x,y,c = ef.value(x,y),vmin=0,vmax=1)

    plt.savefig('test_externalforce.png')
