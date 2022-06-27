import matplotlib.patches as patches
import matplotlib.pyplot as plt
# matplotlib.use('TkAgg')

from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import numpy as np
from itertools import product, combinations
import mpl_toolkits.mplot3d as plt3d
import itertools
from mayavi import mlab


def cube_faces(xmin, xmax, ymin, ymax, zmin, zmax):
    faces = []

    x, y = np.mgrid[xmin:xmax:3j, ymin:ymax:3j]
    z = np.ones(y.shape) * zmin
    faces.append((x, y, z))

    x, y = np.mgrid[xmin:xmax:3j, ymin:ymax:3j]
    z = np.ones(y.shape) * zmax
    faces.append((x, y, z))

    x, z = np.mgrid[xmin:xmax:3j, zmin:zmax:3j]
    y = np.ones(z.shape) * ymin
    faces.append((x, y, z))

    x, z = np.mgrid[xmin:xmax:3j, zmin:zmax:3j]
    y = np.ones(z.shape) * ymax
    faces.append((x, y, z))

    y, z = np.mgrid[ymin:ymax:3j, zmin:zmax:3j]
    x = np.ones(z.shape) * xmin
    faces.append((x, y, z))

    y, z = np.mgrid[ymin:ymax:3j, zmin:zmax:3j]
    x = np.ones(z.shape) * xmax
    faces.append((x, y, z))

    return faces

def mlab_plt_cube(xmin, xmax, ymin, ymax, zmin, zmax):
    faces = cube_faces(xmin, xmax, ymin, ymax, zmin, zmax)
    for grid in faces:
        x, y, z = grid
        mlab.mesh(x, y, z, opacity=0.4)

@mlab.show
def plot(list_3d_rectangles):
    for rect_ind in range(list_3d_rectangles.shape[0]):
        rect = list_3d_rectangles[rect_ind, :, :];
        print("rect: ", rect)
        mlab_plt_cube(rect[0, 0], rect[1, 0], rect[0, 1], rect[1, 1],
                              rect[0, 2], rect[1, 2]);
    print("hello")
    mlab.xlabel('x')
    mlab.ylabel('y')
    mlab.zlabel('z')
    mlab.show()
