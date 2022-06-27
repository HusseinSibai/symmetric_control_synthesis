import numpy as np
import time
import math
from rtree import index
from typing import List, Dict
import numpy.matlib
from z3 import *
import polytope as pc
from yices import *
import random
import copy

import matplotlib

matplotlib.use("macOSX")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, Polygon

def get_rects_inter(rect1, rect2):
    rect_i = np.maximum(rect1[0, :], rect2[0, :]);
    rect_i = np.array([rect_i, np.minimum(rect1[1, :], rect2[1, :])]);
    rect_c = [];
    area_c = 0;
    area_i = np.prod([rect_i[1, :] - rect_i[0, :]]);
    area1 = np.prod([rect1[1, :] - rect1[0, :]]);
    area_c_max = area1 - area_i;
    for dim in range(rect1.shape[1]):
        rect_temp = copy.deepcopy(rect1);
        if rect1[0, dim] < rect_i[0, dim]:
            rect_temp[1, dim] = rect_i[0, dim];
            if len(rect_c) == 0:
                rect_c = np.array([rect_temp]);
            else:
                rect_c = np.concatenate((rect_c, np.array([rect_temp])), 0);
            area_c = area_c + np.prod(rect_temp[1, :] - rect_temp[0, :]);
        rect_temp = copy.deepcopy(rect1);
        if rect1[1, dim] > rect_i[1, dim]:
            rect_temp[0, dim] = rect_i[1, dim];
            # rect_temp[1, dim] = rect1[1, dim];
            if len(rect_c) == 0:
                rect_c = np.array([rect_temp]);
            else:
                rect_c = np.concatenate((rect_c, np.array([rect_temp])), 0);
            area_c = area_c + np.prod(rect_temp[1, :] - rect_temp[0, :]);
    if area_c < area_c_max:
        for dim1 in range(rect1.shape[1]):
            for dim2 in range(rect1.shape[1]):
                rect_temp = copy.deepcopy(rect1);
                if rect1[0, dim1] < rect_i[0, dim1] and rect1[0, dim2] < rect1[0, dim2]:
                    rect_temp[1, dim1] = rect_i[0, dim1];
                    rect_temp[1, dim2] = rect_i[0, dim2];
                    if len(rect_c) == 0:
                        rect_c = np.array([rect_temp]);
                    else:
                        rect_c = np.concatenate((rect_c, np.array([rect_temp])), 0);
                    area_c = area_c + np.prod(rect_temp[1, :] - rect_temp[0, :]);
                rect_temp = copy.deepcopy(rect1);
                if rect1[1, dim1] > rect_i[1, dim1] and rect1[1, dim2] > rect_i[1, dim2]:
                    rect_temp[0, dim1] = rect_i[1, dim1];
                    rect_temp[0, dim2] = rect_i[1, dim2];
                    if len(rect_c) == 0:
                        rect_c = np.array([rect_temp]);
                    else:
                        rect_c = np.concatenate((rect_c, np.array([rect_temp])), 0);
                    area_c = area_c + np.prod(rect_temp[1, :] - rect_temp[0, :]);
    if area_c < area_c_max:
        for dim1 in range(rect1.shape[1]):
            for dim2 in range(rect1.shape[1]):
                for dim3 in range(rect1.shape[1]):
                    rect_temp = copy.deepcopy(rect1);
                    if rect1[0, dim1] < rect_i[0, dim1] and rect1[0, dim2] < rect_i[0, dim2] and rect1[0, dim3] < \
                            rect_i[0, dim3]:
                        rect_temp[1, dim1] = rect_i[0, dim1];
                        rect_temp[1, dim2] = rect_i[0, dim2];
                        rect_temp[1, dim3] = rect_i[0, dim3];
                        if len(rect_c) == 0:
                            rect_c = np.array([rect_temp]);
                        else:
                            rect_c = np.concatenate((rect_c, np.array([rect_temp])), 0);
                    rect_temp = copy.deepcopy(rect1);
                    if rect1[1, dim1] > rect_i[1, dim1] and rect1[1, dim2] > rect_i[1, dim2] and rect1[1, dim3] > \
                            rect_i[1, dim3]:
                        rect_temp[0, dim1] = rect_i[1, dim1];
                        rect_temp[0, dim2] = rect_i[1, dim2];
                        rect_temp[0, dim3] = rect_i[1, dim3];
                        if len(rect_c) == 0:
                            rect_c = np.array([rect_temp]);
                        else:
                            rect_c = np.concatenate((rect_c, np.array([rect_temp])), 0);
    return rect_c, rect_i;

a = np.array([[-1.0,-1.0,-1.0],[1.0,1.0,1.0]]);
b = np.array([[-0.5,-0.5,-0.5],[0.5,0.5,0.5]])
rect_c, rect_i = get_rects_inter(a,b);

print("rect_c: ", rect_c)
print("rect_i: ", rect_i)