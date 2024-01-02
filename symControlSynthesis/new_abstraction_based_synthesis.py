import pdb
import platform

#ignore conversion warnings (uncomment for genuine errors)
import warnings
warnings.simplefilter("ignore", UserWarning)

import numpy as np
import time
import math
from rtree import index
from typing import List, Dict
import numpy.matlib
from z3 import *
import polytope as pc
# from yices import *
import random
import copy
import os
import itertools
from scipy.spatial import ConvexHull
from scipy.optimize import linprog
from qpsolvers import solve_qp
import matlab.engine

import matplotlib

if platform.system() == 'Darwin':
    matplotlib.use("macOSX")

import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, Polygon

import bisect

import csv

#multiprocessing stuff

#cpu-only
if platform.system() == 'Darwin':

    from multiprocess import Process, Queue, shared_memory, Manager
    import multiprocess
    import concurrent.futures
    import signal

else:
    
    from multiprocessing import Process, Queue, shared_memory, Manager
    import multiprocessing as multiprocess
    import concurrent.futures
    import signal

#lock
from shared_memory_dict import SharedMemoryDict
wait_cond = SharedMemoryDict(name='lock', size=128)

# Use 'multiprocessing.cpu_count()' to determine the number of available CPU cores.
cpu_count = multiprocess.cpu_count()

#make a future pool
future_pool = [None] * (cpu_count * 10)

#time we spend spinning
time_spinning = 0

#function name
create_symmetry_abstract_states = None

class ThreadedAbstractState:

    def __init__(self, state_id, quantized_abstract_target, u_idx,
                 abstract_obstacles, concrete_state_indices_in, obstructed_u_idx_set, manager):
        self.id = state_id # tuple of centers
        self.quantized_abstract_target = quantized_abstract_target
        self.u_idx = u_idx
        self.abstract_obstacles = abstract_obstacles
        self.concrete_state_indices = manager.list(concrete_state_indices_in)
        self.obstructed_u_idx_set = obstructed_u_idx_set

class AbstractState:

    def __init__(self, state_id, quantized_abstract_target, u_idx,
                 abstract_obstacles, concrete_state_indices, obstructed_u_idx_set):
        self.id = state_id # tuple of centers
        self.quantized_abstract_target = quantized_abstract_target
        self.u_idx = u_idx
        self.abstract_obstacles = abstract_obstacles
        self.concrete_state_indices = concrete_state_indices
        self.obstructed_u_idx_set = obstructed_u_idx_set


class RelCoordState:
    def __init__(self, concrete_state_idx, abstract_targets,
         abstract_obstacles):
        self.idx = concrete_state_idx
        self.abstract_targets = abstract_targets
        self.abstract_obstacles = abstract_obstacles


def transform_poly_to_abstract(reg: pc.Region, state: np.array, project_to_pos=False):
    # this function takes a polytope in the state space and transforms it to the abstract coordinates.
    # this should be provided by the user as it depends on the symmetries
    # Hussein: unlike the work in SceneChecker, we here rotate then translate, non-ideal, I prefer translation
    # then rotation but this requires changing find_frame which would take time.
    if project_to_pos:
        translation_vector = np.array([-1 * state[0], -1 * state[1]])
        new_reg = project_region_to_position_coordinates(copy.deepcopy(reg))
    else:
        translation_vector = np.array([-1 * state[0], -1 * state[1], -1 * state[2]])
        new_reg = copy.deepcopy(reg)
    rot_angle = -1 * state[2]
    if type(new_reg) == pc.Polytope:
        poly_out: pc.Polytope = new_reg.translation(translation_vector)
        if not project_to_pos:
            poly_out = fix_angle_interval_in_poly(poly_out)
        result = poly_out.rotation(i=0, j=1, theta=rot_angle)
    else:
        result = pc.Region(list_poly=[])
        for poly in new_reg.list_poly:
            poly_out: pc.Polytope = poly.translation(translation_vector)
            if not project_to_pos:
                poly_out = fix_angle_interval_in_poly(poly_out)
            result.list_poly.append(poly_out.rotation(i=0, j=1, theta=rot_angle))
    return result


def transform_poly_to_frame(reg: pc.Region, state: np.array, project_to_pos=False):
    if project_to_pos:
        translation_vector = np.array([state[0], state[1]])
        new_reg = project_region_to_position_coordinates(copy.deepcopy(reg))
    else:
        translation_vector = np.array([state[0], state[1], state[2]])
        new_reg = copy.deepcopy(reg)
    rot_angle = state[2]
    if type(new_reg) == pc.Polytope:
        poly_out: pc.Polytope = new_reg.rotation(i=0, j=1, theta=rot_angle)
        result = poly_out.translation(translation_vector)
        if not project_to_pos:
            result = fix_angle_interval_in_poly(result)
    else:
        result = pc.Region(list_poly=[])
        for poly in new_reg.list_poly:
            poly_out: pc.Polytope = poly.rotation(i=0, j=1, theta=rot_angle)
            poly_out = poly_out.translation(translation_vector)
            if not project_to_pos:
                poly_out = fix_angle_interval_in_poly(poly_out)
            result.list_poly.append(poly_out)
    return result


def get_bounding_box(poly: pc.Polytope, verbose=False) -> np.array:
    if type(poly) != pc.Polytope:
        # print(type(poly))
        raise TypeError("this function only takes polytopes")
    poly.bbox = None
    if verbose:
        print("working")
    return np.column_stack(poly.bounding_box).T


def get_region_bounding_box(reg: pc.Region) -> np.array:
    if type(reg) == pc.Polytope:
        print("warning, called region bbox function on polytope")
        return get_bounding_box(reg)
    elif len(reg.list_poly) <= 0 or pc.is_empty(reg):
        raise ValueError("Passed an empty region, no valid bbox")
    return np.row_stack((np.min(np.stack(map(get_bounding_box, reg.list_poly))[:, 0, :], axis=0),
                         np.max(np.stack(map(get_bounding_box, reg.list_poly))[:, 1, :], axis=0)))


def fix_angle(angle):
    while angle < -1 * math.pi:
        angle += 2 * math.pi
    while angle > math.pi:
        angle -= 2 * math.pi
    return angle


def fix_angle_interval(left_angle, right_angle):
    if abs(right_angle - left_angle) < 0.001:
        left_angle = right_angle - 0.01
    while right_angle < left_angle:
        right_angle += 2 * math.pi
    if right_angle - left_angle >= 2 * math.pi - 0.01:
        return 0, 2 * math.pi
    while left_angle < 0 or right_angle < 0:
        left_angle += 2 * math.pi
        right_angle += 2 * math.pi
    while left_angle > 4 * math.pi or right_angle > 4 * math.pi:
        left_angle -= 2 * math.pi
        right_angle -= 2 * math.pi
    while left_angle > 2 * math.pi and right_angle > 2 * math.pi:
        left_angle -= 2 * math.pi
        right_angle -= 2 * math.pi
    return left_angle, right_angle


def fix_angle_interval_in_rect(rect: np.array):
    rect = copy.deepcopy(rect)
    # rect[0, 2] = fix_angle_to_positive_value(rect[0, 2])
    # rect[1, 2] = fix_angle_to_positive_value(rect[1, 2])
    rect[0, 2], rect[1, 2] = fix_angle_interval(rect[0, 2], rect[1, 2])
    # while rect[1, 2] < rect[0, 2]:
    #    rect[1, 2] += 2 * math.pi
    # rect[0, 2] = rect[0, 2] % (2 * math.pi)
    # rect[1, 2] = rect[1, 2] % (2 * math.pi)
    return rect


def fix_angle_interval_in_poly(poly: pc.Polytope):
    A = copy.deepcopy(poly.A)
    b = copy.deepcopy(poly.b)
    n = A.shape[0]
    i_low = None
    i_up = None
    for i in range(n):
        if A[i, 2] == 1:
            i_up = i
            # b[i] = fix_angle_to_positive_value(b[i])  # fix_angle(b[i])
        if A[i, 2] == -1:
            i_low = i
            # b[i] = fix_angle_to_positive_value(b[i])  # -1 * fix_angle(-b[i])
    b_i_low_neg, b_i_up = fix_angle_interval(-1 * b[i_low], b[i_up])
    b[i_low] = -1 * b_i_low_neg
    b[i_up] = b_i_up
    # while -1 * b[i_low] > b[i_up]:
    # temp = b[i_low]
    # b[i_low] = -1 * b[i_up]
    # b[i_up] += 2 * math.pi  # -1 * temp
    p = pc.Polytope(A, b)
    return p


def get_poly_list_with_decomposed_angle_intervals(poly: pc.Polytope):
    A = copy.deepcopy(poly.A)
    b = copy.deepcopy(poly.b)
    n = A.shape[0]
    i_low = None
    i_up = None
    for i in range(n):
        if A[i, 2] == 1:
            i_up = i
        if A[i, 2] == -1:
            i_low = i
    b_i_low_neg, b_i_up = fix_angle_interval(-1 * b[i_low], b[i_up])
    interval_list = get_decomposed_angle_intervals([-b_i_low_neg, b_i_up])
    result = pc.Region(list_poly=[])
    for interval in interval_list:
        b[i_low] = -1 * interval[0]
        b[i_up] = b_i_up * interval[1]
        result.list_poly.append(pc.Polytope(A, b))
    return result

def project_region_to_position_coordinates(reg: pc.Region):
    if reg.dim <= 2:
        return reg
    if type(reg) == pc.Region:
        poly_list = reg.list_poly
    else:
        poly_list = [reg]
    result = pc.Region(list_poly=[])
    for poly in poly_list:
        A = copy.deepcopy(poly.A)
        b = copy.deepcopy(poly.b)
        n = A.shape[0]
        pos_indices = []
        for i in range(n):
            if not (A[i, 2] == 1 or A[i, 2] == -1):
                pos_indices.append(i)
        A_pos = A[pos_indices, :2]
        b_pos = b[pos_indices]
        p = pc.Polytope(A_pos, b_pos)
        result = pc.union(result, p)
    return result


def add_angle_dimension_to_region(reg: pc.Region, angle_interval: np.array):
    if reg.dim <= 2:
        return reg
    if type(reg) == pc.Region:
        poly_list = reg.list_poly
    else:
        poly_list = [reg]
    result = pc.Region(list_poly=[])
    for poly in poly_list:
        A = copy.deepcopy(poly.A)
        b = copy.deepcopy(poly.b)
        n = A.shape[0]
        pos_indices = []
        for i in range(n):
            if not (A[i, 2] == 1 or A[i, 2] == -1):
                pos_indices.append(i)
        A_pos = A[pos_indices, :2]
        b_pos = b[pos_indices]
        p = pc.Polytope(A_pos, b_pos)
        result = pc.union(result, p)
    return result


def get_poly_intersection(poly_1: pc.Region, poly_2: pc.Region, project_to_pos=False, check_convex=False):
    if project_to_pos:
        return pc.intersect(poly_1, poly_2)
    result = pc.Region(list_poly=[])
    if pc.is_empty(poly_1) or pc.is_empty(poly_2):
        return result
    if type(poly_1) == pc.Region:
        poly_1_list = poly_1.list_poly
    else:
        poly_1_list = [poly_1]
    if type(poly_2) == pc.Region:
        poly_2_list = poly_2.list_poly
    else:
        poly_2_list = [poly_2]
    for poly_1 in poly_1_list:
        for poly_2 in poly_2_list:
            A1 = copy.deepcopy(poly_1.A)
            b1 = copy.deepcopy(poly_1.b)
            A2 = copy.deepcopy(poly_2.A)
            b2 = copy.deepcopy(poly_2.b)
            n = A1.shape[0]
            i_low_1 = None
            i_up_1 = None
            i_low_2 = None
            i_up_2 = None
            pos_indices_1 = []
            pos_indices_2 = []
            for i in range(n):
                if A1[i, 2] == 1:
                    i_up_1 = i
                    # b1[i] = fix_angle_to_positive_value(b[i]) # fix_angle(b[i])
                elif A1[i, 2] == -1:
                    i_low_1 = i
                else:
                    pos_indices_1.append(i)
            A1_pos = A1[pos_indices_1, :2]
            b1_pos = b1[pos_indices_1]
            p1_pos = pc.Polytope(A1_pos, b1_pos)
            for i in range(A2.shape[0]):
                if A2[i, 2] == 1:
                    i_up_2 = i
                elif A2[i, 2] == -1:
                    i_low_2 = i
                else:
                    pos_indices_2.append(i)
            A2_pos = A2[pos_indices_2, :2]
            b2_pos = b2[pos_indices_2]
            p2_pos = pc.Polytope(A2_pos, b2_pos)
            p_inter_pos = pc.intersect(p1_pos, p2_pos)
            # print("p_inter_pos: ", p_inter_pos)
            if not pc.is_empty(p_inter_pos):
                inter_interval = get_intervals_intersection(-1 * b1[i_low_1], b1[i_up_1], -1 * b2[i_low_2], b2[i_up_2])
                if inter_interval is not None:
                    intervals = [inter_interval]  # get_decomposed_angle_intervals(inter_interval)
                    for interval in intervals:
                        b_inter_left = -1 * interval[0]
                        b_inter_right = interval[1]
                        A_new = np.zeros((p_inter_pos.A.shape[0] + 2, A1.shape[1]))
                        b_new = np.zeros((p_inter_pos.A.shape[0] + 2,))
                        for i in range(p_inter_pos.A.shape[0]):
                            for j in range(p_inter_pos.A.shape[1]):
                                A_new[i, j] = p_inter_pos.A[i, j]
                            b_new[i] = p_inter_pos.b[i]
                        A_new[p_inter_pos.A.shape[0], 2] = 1
                        b_new[p_inter_pos.A.shape[0]] = b_inter_right
                        A_new[p_inter_pos.A.shape[0] + 1, 2] = -1
                        b_new[p_inter_pos.A.shape[0] + 1] = b_inter_left
                        result = pc.union(result, pc.Polytope(A_new, b_new), check_convex=check_convex)
    return result


def get_poly_union(poly_1: pc.Region, poly_2: pc.Region, check_convex=False, project_to_pos=False, order_matters=False):
    return pc.union(poly_1, poly_2, check_convex=check_convex)


# https://stackoverflow.com/questions/11406189/determine-if-angle-lies-between-2-other-angles
def is_within_range(angle, a, b):
    a, b = fix_angle_interval(a, b)
    ang = fix_angle(angle)
    while ang < 0:
        ang += 2 * math.pi
    while ang > 2 * math.pi:
        ang -= 2 * math.pi
    if a <= 2 * math.pi <= b:
        if ang >= a or ang <= b - 2 * math.pi:
            return True
    if a <= ang <= b:
        return True
    return False


def does_interval_contain(a_s, b_s, a_l, b_l):
    if is_within_range(a_s, a_l, b_l) and is_within_range(b_s, a_l, b_l) \
            and is_within_range(a_s, a_l, b_s) and is_within_range(b_s, a_s, b_l):
        return True
    return False


def do_intervals_intersect(a_s, b_s, a_l, b_l):
    if does_interval_contain(a_s, b_s, a_l, b_l) or does_interval_contain(a_l, b_l, a_s, b_s) \
            or is_within_range(a_s, a_l, b_l) or is_within_range(b_s, a_l, b_l):
        return True
    return False


def get_intervals_intersection(a_s, b_s, a_l, b_l):
    # TODO: write a function that returns the two intervals
    # resulting from the intersection instead of just one.
    if not do_intervals_intersect(a_s, b_s, a_l, b_l):
        # pdb.set_trace()
        return None
    a_s, b_s = fix_angle_interval(a_s, b_s)
    a_l, b_l = fix_angle_interval(a_l, b_l)
    '''
    a_s = fix_angle_to_positive_value(a_s)
    b_s = fix_angle_to_positive_value(b_s)
    a_l = fix_angle_to_positive_value(a_l)
    b_l = fix_angle_to_positive_value(b_l)
    '''
    if b_s - a_s >= 2 * math.pi - 0.01 and b_l - a_l >= 2 * math.pi - 0.01:
        result = [0, 2 * math.pi]
    elif does_interval_contain(a_s, b_s, a_l, b_l):
        result = [a_s, b_s]
    elif does_interval_contain(a_l, b_l, a_s, b_s):
        result = [a_l, b_l]
    elif is_within_range(a_s, a_l, b_l):
        while a_s <= a_l:
            a_s += 2 * math.pi
        while a_s > b_l:
            b_l += 2 * math.pi
        result = [a_s, b_l]
    elif is_within_range(b_s, a_l, b_l):
        while b_s >= b_l:
            b_s -= 2 * math.pi
        while a_l > b_s:
            b_s += 2 * math.pi
        result = [a_l, b_s]
    result[0], result[1] = fix_angle_interval(result[0], result[1])
    return result


def get_intervals_union(a_s, b_s, a_l, b_l, order_matters=False):
    # TODO: write a function that returns the two intervals
    # resulting from the intersection instead of just one.
    a_s, b_s = fix_angle_interval(a_s, b_s)
    a_l, b_l = fix_angle_interval(a_l, b_l)
    if b_s - a_s >= 2 * math.pi - 0.001 or b_l - a_l >= 2 * math.pi - 0.001:
        # to not lose over-approximation of reachability analysis
        return [[0, 2 * math.pi]]
    if does_interval_contain(a_s, b_s, a_l, b_l):
        result = [[a_l, b_l]]
    elif does_interval_contain(a_l, b_l, a_s, b_s):
        result = [[a_s, b_s]]
    elif is_within_range(a_s, a_l, b_l):
        while a_l > b_s:
            b_s += 2 * math.pi
        result = [[a_l, b_s]]
    elif is_within_range(b_s, a_l, b_l):
        while a_s > b_l:
            b_l += 2 * math.pi
        result = [[a_s, b_l]]
    else:
        if order_matters:
            result = [[a_s, b_s], [a_l, b_l]]
        else:
            while a_s > b_l:
                b_l += 2 * math.pi
            a_s, b_l = fix_angle_interval(a_s, b_l)
            if order_matters or not b_l - a_s > 2 * math.pi:
                result = [[a_s, b_l]]
            else:
                while a_l > b_s:
                    b_s += 2 * math.pi
                result = [[a_l, b_s]]
    # if disjoint, choose a direction to join them
    result_list = []
    if order_matters:
        for interval in result:
            new_interval_l, new_interval_u = fix_angle_interval(interval[0], interval[1])
            new_intervals = get_decomposed_angle_intervals([new_interval_l, new_interval_u])
            for new_interval in new_intervals:
                result_list.append(new_interval)
        # result[0], result[1] = fix_angle_interval(result[0], result[1])
    else:
        result_l, result_u = fix_angle_interval(result[0][0], result[0][1])
        result_list.append([result_l, result_u])
    return result_list


def transform_rect_to_abstract(rect: np.array, state: np.array, overapproximate=False):
    ang = -1 * state[2]  # psi = 0 is North, psi = pi/2 is east

    while ang < 0:
        ang += 2 * math.pi
    while ang > 2 * math.pi:
        ang -= 2 * math.pi

    # ang = fix_angle(ang)
    rect = fix_angle_interval_in_rect(rect)
    # state[2] = fix_angle_to_positive_value(state[2])

    low_red = np.array(
        [(rect[0, 0] - state[0]) * math.cos(ang) -
         (rect[0, 1] - state[1]) * math.sin(ang),
         (rect[0, 0] - state[0]) * math.sin(ang) +
         (rect[0, 1] - state[1]) * math.cos(ang),
         rect[0, 2] - state[2]])
    up_red = np.array(
        [(rect[1, 0] - state[0]) * math.cos(ang) -
         (rect[1, 1] - state[1]) * math.sin(ang),
         (rect[1, 0] - state[0]) * math.sin(ang) +
         (rect[1, 1] - state[1]) * math.cos(ang),
         rect[1, 2] - state[2]])

    if 0 <= ang <= math.pi / 2:
        x_bb_up = up_red[0] + (rect[1, 1] - rect[0, 1]) * math.sin(ang)
        y_bb_up = up_red[1]
        x_bb_low = low_red[0] - (rect[1, 1] - rect[0, 1]) * math.sin(ang)
        y_bb_low = low_red[1]
    elif math.pi / 2 <= ang <= math.pi:
        x_bb_up = low_red[0]
        y_bb_up = low_red[1] + (rect[1, 0] - rect[0, 0]) * math.cos(ang - math.pi / 2)
        x_bb_low = up_red[0]
        y_bb_low = up_red[1] - (rect[1, 0] - rect[0, 0]) * math.cos(ang - math.pi / 2)
    elif math.pi <= ang <= 3 * math.pi / 2.0:
        x_bb_up = low_red[0] + (rect[1, 1] - rect[0, 1]) * math.cos(3 * math.pi / 2 - ang)
        y_bb_up = low_red[1]
        x_bb_low = up_red[0] - (rect[1, 1] - rect[0, 1]) * math.cos(3 * math.pi / 2 - ang)
        y_bb_low = up_red[1]
    else:
        x_bb_up = up_red[0]
        y_bb_up = up_red[1] + (rect[1, 0] - rect[0, 0]) * math.cos(ang - 3 * math.pi / 2)
        x_bb_low = low_red[0]
        y_bb_low = low_red[1] - (rect[1, 0] - rect[0, 0]) * math.cos(ang - 3 * math.pi / 2)

    bb = np.array([[x_bb_low, y_bb_low, low_red[2]], [x_bb_up, y_bb_up, up_red[2]]])
    bb = fix_angle_interval_in_rect(bb)

    if overapproximate:
        return bb
    w = bb[1, 0] - bb[0, 0]
    h = bb[1, 1] - bb[0, 1]

    bb_center = np.average(bb, axis=0)

    new_w, new_h = rotatedRectWithMaxArea(w, h, state[2])

    low_new_rect = np.array([bb_center[0] - new_w / 2.0,
                             bb_center[1] - new_h / 2.0,
                             bb[0, 2]])
    up_new_rect = np.array([bb_center[0] + new_w / 2.0,
                            bb_center[1] + new_h / 2.0,
                            bb[1, 2]])

    # print("low_new_rect: ", low_new_rect)
    # print("up_new_rect: ", up_new_rect)
    result = np.array([low_new_rect, up_new_rect])

    return result


def rectangle_to_vertices(rect: np.array):
    points = []
    for i in range(2):
        for j in range(2):
            for k in range(2):
                points.append([rect[i, 0], rect[j, 1], rect[k, 2]])
    return points


# This is a code from https://stackoverflow.com/a/16778797
def rotatedRectWithMaxArea(w, h, angle):
    """
  Given a rectangle of size wxh that has been rotated by 'angle' (in
  radians), computes the width and height of the largest possible
  axis-aligned rectangle (maximal area) within the rotated rectangle.
  """
    if w <= 0 or h <= 0:
        return 0, 0

    width_is_longer = w >= h
    side_long, side_short = (w, h) if width_is_longer else (h, w)

    # since the solutions for angle, -angle and 180-angle are all the same,
    # if suffices to look at the first quadrant and the absolute values of sin,cos:
    sin_a, cos_a = abs(math.sin(angle)), abs(math.cos(angle))
    if side_short <= 2. * sin_a * cos_a * side_long or abs(sin_a - cos_a) < 1e-10:
        # half constrained case: two crop corners touch the longer side,
        #   the other two corners are on the mid-line parallel to the longer line
        x = 0.5 * side_short
        wr, hr = (x / sin_a, x / cos_a) if width_is_longer else (x / cos_a, x / sin_a)
    else:
        # fully constrained case: crop touches all 4 sides
        cos_2a = cos_a * cos_a - sin_a * sin_a
        wr, hr = (w * cos_a - h * sin_a) / cos_2a, (h * cos_a - w * sin_a) / cos_2a

    return wr, hr


# translated and adapted  from a Javascript code
# from https://codegrepr.com/question/calculate-largest-inscribed-rectangle-in-a-rotated-rectangle/
# A function that rotates a rectangle by angle alpha (the xy-coordinate system by angle -alpha) and translates
# the third coordinate accordingly.
def transform_to_frame(rect: np.array, state: np.array, overapproximate=True):
    # print("rect: ", rect)
    # print("state: ", state)
    ang = state[2]  # psi = 0 is North, psi = pi/2 is east

    while ang < 0:
        ang += 2 * math.pi
    while ang > 2 * math.pi:
        ang -= 2 * math.pi

    low_red = np.array(
        [(rect[0, 0]) * math.cos(ang) -
         (rect[0, 1]) * math.sin(ang) + state[0],
         (rect[0, 0]) * math.sin(ang) +
         (rect[0, 1]) * math.cos(ang) + state[1],
         rect[0, 2] + state[2]])
    up_red = np.array(
        [(rect[1, 0]) * math.cos(ang) -
         (rect[1, 1]) * math.sin(ang) + state[0],
         (rect[1, 0]) * math.sin(ang) +
         (rect[1, 1]) * math.cos(ang) + state[1],
         rect[1, 2] + state[2]])

    if 0 <= ang <= math.pi / 2:
        x_bb_up = up_red[0] + (rect[1, 1] - rect[0, 1]) * math.sin(ang)
        y_bb_up = up_red[1]
        x_bb_low = low_red[0] - (rect[1, 1] - rect[0, 1]) * math.sin(ang)
        y_bb_low = low_red[1]
    elif math.pi / 2 <= ang <= math.pi:
        x_bb_up = low_red[0]
        y_bb_up = low_red[1] + (rect[1, 0] - rect[0, 0]) * math.cos(ang - math.pi / 2)
        x_bb_low = up_red[0]
        y_bb_low = up_red[1] - (rect[1, 0] - rect[0, 0]) * math.cos(ang - math.pi / 2)
    elif math.pi <= ang <= 3 * math.pi / 2.0:
        x_bb_up = low_red[0] + (rect[1, 1] - rect[0, 1]) * math.cos(3 * math.pi / 2 - ang)
        y_bb_up = low_red[1]
        x_bb_low = up_red[0] - (rect[1, 1] - rect[0, 1]) * math.cos(3 * math.pi / 2 - ang)
        y_bb_low = up_red[1]
    else:
        x_bb_up = up_red[0]
        y_bb_up = up_red[1] + (rect[1, 0] - rect[0, 0]) * math.cos(ang - 3 * math.pi / 2)
        x_bb_low = low_red[0]
        y_bb_low = low_red[1] - (rect[1, 0] - rect[0, 0]) * math.cos(ang - 3 * math.pi / 2)

    bb = np.array([[x_bb_low, y_bb_low, low_red[2]], [x_bb_up, y_bb_up, up_red[2]]])

    if overapproximate:
        return bb

    w = bb[1, 0] - bb[0, 0]
    h = bb[1, 1] - bb[0, 1]

    bb_center = np.average(bb, axis=0)

    new_w, new_h = rotatedRectWithMaxArea(w, h, state[2])

    low_new_rect = np.array([bb_center[0] - new_w / 2.0,
                             bb_center[1] - new_h / 2.0,
                             bb[0, 2]])
    up_new_rect = np.array([bb_center[0] + new_w / 2.0,
                            bb_center[1] + new_h / 2.0,
                            bb[1, 2]])

    # print("low_new_rect: ", low_new_rect)
    # print("up_new_rect: ", up_new_rect)
    result = np.array([low_new_rect, up_new_rect])

    return result


def transform_to_frames(low_red, up_red, source_full_low, source_full_up):
    '''
    ang = (source_full_low[2] + source_full_up[2]) / 2
    source_full_low_temp = copy.deepcopy(source_full_low)
    source_full_low_temp[2] = ang
    source_full_up_temp = copy.deepcopy(source_full_up)
    source_full_up_temp[2] = ang
    '''
    box_1 = transform_to_frame(np.array([low_red, up_red]), source_full_low, overapproximate=True)
    box_2 = transform_to_frame(np.array([low_red, up_red]), source_full_up, overapproximate=True)
    box_3 = transform_to_frame(np.array([low_red, up_red]), np.array([source_full_low[0], source_full_low[1],
                                                                      source_full_up[2]]), overapproximate=True)
    box_4 = transform_to_frame(np.array([low_red, up_red]), np.array([source_full_up[0], source_full_up[1],
                                                                      source_full_low[2]]), overapproximate=True)
    result = get_convex_union([box_1, box_2, box_3, box_4])
    # result = get_convex_union([box_1, box_2])
    return result  # np.array([result_low, result_up]);


def transform_poly_to_frames(poly, source_full_low, source_full_up):
    '''
    ang = (source_full_low[2] + source_full_up[2]) / 2
    source_full_low_temp = copy.deepcopy(source_full_low)
    source_full_low_temp[2] = ang
    source_full_up_temp = copy.deepcopy(source_full_up)
    source_full_up_temp[2] = ang
    '''
    poly_1 = transform_poly_to_frame(poly, source_full_low)
    poly_2 = transform_poly_to_frame(poly, source_full_up)
    poly_3 = transform_poly_to_frame(poly, np.array([source_full_low[0], source_full_low[1], source_full_up[2]]))
    poly_4 = transform_poly_to_frame(poly, np.array([source_full_up[0], source_full_up[1], source_full_low[2]]))
    # result = get_poly_list_convex_hull([poly_1, poly_2])
    result = get_poly_list_convex_hull([poly_1, poly_2, poly_3, poly_4])
    return result


def transform_rect_to_abstract_frames(concrete_rect, frames_rect, over_approximate=False, project_to_pos=False):
    '''
    ang = (frames_rect[0, 2] + frames_rect[1, 2]) / 2
    source_full_low_temp = copy.deepcopy(frames_rect[0, :])
    source_full_low_temp[2] = ang
    source_full_up_temp = copy.deepcopy(frames_rect[1, :])
    source_full_up_temp[2] = ang
    '''
    box_1 = transform_rect_to_abstract(concrete_rect, frames_rect[0, :], overapproximate=over_approximate)
    box_2 = transform_rect_to_abstract(concrete_rect, frames_rect[1, :], overapproximate=over_approximate)
    box_3 = transform_rect_to_abstract(concrete_rect, np.array([frames_rect[0, 0], frames_rect[0, 1],
                                                                frames_rect[1, 2]]), overapproximate=over_approximate)
    box_4 = transform_rect_to_abstract(concrete_rect, np.array([frames_rect[1, 0], frames_rect[1, 1],
                                                                frames_rect[0, 2]]), overapproximate=over_approximate)
    if over_approximate:
        result = get_convex_union([box_1, box_4, box_3, box_2],
                                  order_matters=True)  # Hussein: order in the list matters!!!!
        # result = get_convex_union([box_1, box_2], order_matters=True)
        # it determines the direction of the union of the angles
    else:
        result = get_intersection([box_1, box_2, box_3, box_4])
        # result = get_intersection([box_1, box_2])
    if project_to_pos:
        result = result[:, :2]
    return result  # np.array([result_low, result_up]);


def get_poly_list_convex_hull(poly_list: List[pc.Region]):
    result = poly_list[0]
    for ind in range(1, len(poly_list)):
        result = get_poly_union(result, poly_list[ind], project_to_pos=False, order_matters=False)
    if isinstance(result, pc.Region):
        poly_list = result.list_poly
    else:
        poly_list = [result]
    points = None
    interval = None
    for poly in poly_list:
        A = copy.deepcopy(poly.A)
        b = copy.deepcopy(poly.b)
        n = A.shape[0]
        i_low = None
        i_up = None
        pos_indices = []
        for i in range(n):
            if A[i, 2] == 1:
                i_up = i
                # b1[i] = fix_angle_to_positive_value(b[i]) # fix_angle(b[i])
            elif A[i, 2] == -1:
                i_low = i
            else:
                pos_indices.append(i)
        A_pos = A[pos_indices, :2]
        b_pos = b[pos_indices]
        p_pos = pc.Polytope(A_pos, b_pos)
        if points is None:
            points = pc.extreme(p_pos)
        else:
            points = np.concatenate((points, pc.extreme(p_pos)), axis=0)
        original_interval = fix_angle_interval(-1 * b[i_low], b[i_up])
        interval_list = get_decomposed_angle_intervals(list(original_interval))
        # union_interval_list = get_intervals_union(-1 * b[i_low_1], b[i_up_1], -1 * b[i_low_2], b[i_up_2],
        #                                         order_matters=order_matters)
        interval = interval_list[0]
        for interval_idx in range(1, len(interval_list)):
            # b_inter_left = -1 * interval[0]
            # b_inter_right = interval[1]
            # A_new = np.zeros((p_pos.A.shape[0] + 2, A.shape[1]))
            # b_new = np.zeros((p_pos.A.shape[0] + 2,))
            temp_interval_list = get_intervals_union(interval[0], interval[1],
                                                     interval_list[interval_idx][0],
                                                     interval_list[interval_idx][1])
            interval = temp_interval_list[0]
            # A_new[p_pos.A.shape[0], 2] = 1
            # b_new[p_pos.A.shape[0]] = b_inter_right
            # A_new[p_pos.A.shape[0] + 1, 2] = -1
            # b_new[p_pos.A.shape[0] + 1] = b_inter_left
            # result = pc.union(result, pc.Polytope(A_new, b_new), check_convex=check_convex)
    if points is not None and interval is not None:
        hull = pc.qhull(points)
        if points.shape[0] > 20:
            bbox = np.column_stack(pc.bounding_box(hull)).T
            hull = pc.box2poly(bbox.T)
        b_inter_left = -1 * interval[0]
        b_inter_right = interval[1]
        A_new = np.zeros((hull.A.shape[0] + 2, hull.A.shape[1] + 1))
        b_new = np.zeros((hull.A.shape[0] + 2,))
        for i in range(hull.A.shape[0]):
            for j in range(hull.A.shape[1]):
                A_new[i, j] = hull.A[i, j]
            b_new[i] = hull.b[i]
        A_new[hull.A.shape[0], 2] = 1
        b_new[hull.A.shape[0]] = b_inter_right
        A_new[hull.A.shape[0] + 1, 2] = -1
        b_new[hull.A.shape[0] + 1] = b_inter_left
        result = pc.Polytope(A_new, b_new)
    return result


def transform_poly_to_abstract_frames(concrete_poly, frames_rect, over_approximate=False, project_to_pos=False,
                                      check_convex=False):
    if project_to_pos:
        concrete_poly_new = project_region_to_position_coordinates(copy.deepcopy(concrete_poly))
    else:
        concrete_poly_new = copy.deepcopy(concrete_poly)
    '''
    ang = (frames_rect[0, 2] + frames_rect[1, 2]) / 2
    source_full_low_temp = copy.deepcopy(frames_rect[0, :])
    source_full_low_temp[2] = ang
    source_full_up_temp = copy.deepcopy(frames_rect[1, :])
    source_full_up_temp[2] = ang
    '''
    poly_1 = transform_poly_to_abstract(concrete_poly_new, frames_rect[0, :], project_to_pos)
    poly_2 = transform_poly_to_abstract(concrete_poly_new, frames_rect[1, :], project_to_pos)
    poly_3 = transform_poly_to_abstract(concrete_poly_new, np.array([frames_rect[0, 0], frames_rect[0, 1],
                                                                     frames_rect[1, 2]]), project_to_pos)
    poly_4 = transform_poly_to_abstract(concrete_poly_new, np.array([frames_rect[1, 0], frames_rect[1, 1],
                                                                     frames_rect[0, 2]]), project_to_pos)

    if over_approximate:
        result = get_poly_list_convex_hull([poly_1, poly_2, poly_3, poly_4])
        # result = get_poly_list_convex_hull([poly_1, poly_2])
    else:
        result = get_poly_intersection(poly_1, poly_2, project_to_pos, check_convex)  # pc.intersect(poly_1, poly_2)
        result = get_poly_intersection(result, poly_3, project_to_pos, check_convex)
        result = get_poly_intersection(result, poly_4, project_to_pos, check_convex)
    return result


def check_rect_empty(rect, allow_zero_dim):
    if allow_zero_dim and np.all(rect[0, :] <= rect[1, :]) and np.any(rect[0, :] < rect[1, :]):
        return False
    elif (not allow_zero_dim) and np.all(rect[0, :] < rect[1, :]):
        return False
    else:
        return True


def do_rects_inter(rect1, rect2):
    assert len(rect1.shape) == 2 and len(rect2.shape) == 2, print("Dimensions of,", rect1, " or ", rect2,
                                                                  " do not match!")
    '''
    if check_rect_empty(rect1, 1) or check_rect_empty(rect2, 1):
        rect1
        rect2
        print("Do not pass empty rectangles to intersect function")
        return False;
    else:
    '''
    for i in range(rect1.shape[1] - 1):
        if rect1[0, i] > rect2[1, i] + 0.01 or rect1[1, i] + 0.01 < rect2[0, i]:
            return False
    if not do_intervals_intersect(rect1[0, 2], rect1[1, 2], rect2[0, 2], rect2[1, 2]):
        return False
    return True


def add_rects_to_solver(rects, var_dict, cur_solver):
    # print("Adding the following rectangles to solver: ", rects.shape)
    for rect_idx in range(rects.shape[0]):
        rect = rects[rect_idx, :, :]
        c = []
        for dim in range(rect.shape[1]):
            c.append(var_dict[dim] < rect[0, dim])
            c.append(var_dict[dim] > rect[1, dim])
        cur_solver.add(Or(c))
    return cur_solver


def do_rects_list_contain_smt(rect1, var_dict, cur_solver):
    # adding the rectangle to the z3 solver
    # print("The rectangles in the solver do not contain ", rect1)
    cur_solver.push()
    for dim in range(rect1.shape[1]):
        cur_solver.add(var_dict[dim] >= rect1[0, dim])
        cur_solver.add(var_dict[dim] <= rect1[1, dim])
    res = cur_solver.check()
    if res == sat:
        uncovered_state = np.average(rect1, axis=0)
        cur_solver.pop()
        return np.array(uncovered_state)
    cur_solver.pop()
    return None


def does_rect_contain(rect1, rect2):  # does rect2 contains rect1
    result = True
    for i in range(rect1.shape[1] - 1):
        if rect1[0, i] + 0.000001 < rect2[0, i] or rect1[1, i] - 0.000001 > rect2[1, i]:
            # print(rect2, " does not contain ", rect1, " since ", rect1[0, i], "<", rect2[0, i], " or ", rect1[1, i],
            #      ">", rect2[1, i])
            result = False
    if result:
        # print("The interval ", rect2[0, 2], rect2[1, 2], " contains ", rect1[0, 2], rect1[1, 2], "? ",
        #      does_interval_contain(rect1[0, 2], rect1[1, 2], rect2[0, 2], rect2[1, 2]))
        return does_interval_contain(rect1[0, 2], rect1[1, 2], rect2[0, 2], rect2[1, 2])
    return False


def get_rect_volume(rect: np.array):
    vol = np.prod(rect[1, :2] - rect[0, :2])
    rect_temp = copy.deepcopy(rect)
    rect_temp = fix_angle_interval_in_rect(rect_temp)
    dist = rect_temp[1, 2] - rect_temp[0, 2]
    if dist < 0:
        raise "why fix_rect_angles_to_linear_order is not working?"
    dist = dist % (2 * math.pi)
    # vol = vol * dist
    if vol < 0:
        raise "not a valid rectangle"
    return vol


def get_convex_union(list_array: List[np.array], order_matters=False) -> np.array:
    assert len(list_array) > 0, "list array length should be larger than zero"
    result: np.array = np.copy(list_array[0])
    for i in range(1, len(list_array)):
        result[0, :2] = np.minimum(result[0, :2], list_array[i][0, :2])
        result[1, :2] = np.maximum(result[1, :2], list_array[i][1, :2])
        union_interval_list = get_intervals_union(result[0, 2], result[1, 2], list_array[i][0, 2], list_array[i][1, 2],
                                                  order_matters=False)
        union_interval = union_interval_list[0]
        result[0, 2] = union_interval[0]
        result[1, 2] = union_interval[1]
    return result


def get_intersection(list_array: List[np.array]) -> np.array:
    assert len(list_array) > 0, "list array length should be larger than zero"
    result: np.array = np.copy(list_array[0])
    for i in range(len(list_array)):
        if do_rects_inter(list_array[i], result):
            # result = get_rects_inter(result, list_array[i])
            result[0, :2] = np.maximum(result[0, :2], list_array[i][0, :2])
            result[1, :2] = np.minimum(result[1, :2], list_array[i][1, :2])
            inter_interval = get_intervals_intersection(result[0, 2], result[1, 2], list_array[i][0, 2],
                                                        list_array[i][1, 2])
            result[0, 2] = inter_interval[0]
            result[1, 2] = inter_interval[1]
        else:
            return None
    return result


def subtract_rectangles(rect1, rect2):
    """
    Partially Generated using ChatGPT
    Subtract rect2 from rect1 and return the resulting rectangles
    """

    # Find the overlapping region between rect1 and rect2
    min_overlap = np.maximum(rect1[0, :], rect2[0, :])
    max_overlap = np.minimum(rect1[1, :], rect2[1, :])

    # If there is no overlapping region, return rect2
    if np.any(max_overlap <= min_overlap):
        return [rect1]

    # Split rect2 into multiple rectangles based on the overlapping region
    rects = []
    for dim in range(rect1.shape[1]):
        if min_overlap[dim] > rect1[0, dim]:
            rect_left = copy.deepcopy(rect1)
            rect_left[1, dim] = min_overlap[dim]
            rects.append(rect_left)

        if max_overlap[dim] < rect1[1, dim]:
            rect_right = copy.deepcopy(rect1)
            rect_right[0, dim] = max_overlap[dim]
            rects.append(rect_right)

    return rects


def next_quantized_key(curr_key: np.array, quantized_key_range: np.array) -> np.array:
    if len(curr_key.shape) > 1:
        raise ValueError("key must be one dimensional lower left and corner of bounding box")
    next_key = np.copy(curr_key)
    for dim in range(curr_key.shape[0] - 1, -1, -1):
        if curr_key[dim] < quantized_key_range[dim]:
            next_key[dim] += 1
            for reset_dim in range(dim + 1, curr_key.shape[0]):
                next_key[reset_dim] = 0  # quantized_key_range[0, reset_dim]
            return next_key
    raise ValueError("curr_key should not exceed the bounds of the bounding box.")


def rect_to_indices(rect, symbol_step, ref_lower_bound, sym_x, over_approximate=False):

    rect = fix_angle_interval_in_rect(rect)

    if over_approximate:
        low_nd_indices = np.floor((rect[0, :] - ref_lower_bound) / symbol_step)
        up_nd_indices = np.ceil((rect[1, :] - ref_lower_bound) / symbol_step)
    else:
        low_nd_indices = np.ceil((rect[0, :] - ref_lower_bound) / symbol_step)
        up_nd_indices = np.floor((rect[1, :] - ref_lower_bound) / symbol_step)
    low_nd_indices = np.maximum(np.zeros(low_nd_indices.shape), low_nd_indices)
    up_nd_indices = np.minimum(sym_x, up_nd_indices)
    if np.any(low_nd_indices >= up_nd_indices):
        raise "symbol step is too large to the point it is causing empty non-empty rectangles " \
              "having no corresponding indices"
    subscripts = list(itertools.product(range(int(low_nd_indices[0]), int(up_nd_indices[0])),
                                        range(int(low_nd_indices[1]), int(up_nd_indices[1])),
                                        range(int(low_nd_indices[2]), int(up_nd_indices[2]))))
    subscripts = list(np.array([list(idx) for idx in subscripts]).T)
    return np.ravel_multi_index(subscripts, tuple((sym_x).astype(int)))


def nearest_point_to_the_origin(poly):
    """
    Computes the nearest point to the origin in a polytope.
    Args:
        poly (array-like): List of points defining the polytope.
    Returns:
        nearest_point (numpy.ndarray): Coordinates of the nearest point to the origin in the polytope.
        dist (float): Distance to the origin
    """
    x = solve_qp(np.eye(3), np.array([0,0,0]), poly.A, poly.b, solver="clarabel")
    dist = np.linalg.norm(x, ord=2)
    return x, dist


benchmark = False #baseline
strategy_1 = False #polls - all
strategy_2 = False #polls - 400
strategy_3 = False #polls + no closest
strategy_4 = False #polls -full + neighbors
strategy_5 = False #polls -400 + neighbors
strategy_6 = False #polls + no closest + neighbors // was it "polls-full"?
strategy_list = [strategy_1, strategy_2, strategy_3, strategy_4, strategy_5, strategy_6, benchmark]

def create_symmetry_abstract_states_threaded(lock_one, lock_two, symbols_to_explore, symbol_step, targets, obstacles, sym_x, X_low, X_up,
                                    reachability_rtree_idx3d, abstract_reachable_sets, symmetry_transformed_targets_and_obstacles, concrete_to_abstract,
                                    abstract_to_concrete, symmetry_abstract_states, u_idx_to_abstract_states_indices, nearest_target_of_concrete, valid_hit_idx_of_concrete,
                                    next_abstract_state_id, threshold_num_results, Q, thread_index, manager, stolen_work, steal_send_lock, steal_receive_lock, stealQueue, sendQueue,
                                    obstacles_rects, obstacle_indices, targets_rects, target_indices, concrete_edges, neighbor_map):

    #each new execution requires new opening of the rtree files
    p = index.Property()
    p.dimension = 3
    p.dat_extension = 'data'
    p.idx_extension = 'index'
    reachability_rtree_idx3d = index.Index('3d_index_abstract',
                                           properties=p)

    get_concrete_transition_calls = 0

    #grab indices for work
    first_index = int(len(symbols_to_explore)/cpu_count) * thread_index
    second_index = (int(len(symbols_to_explore)/cpu_count) * (thread_index+1)) - 1 if thread_index+1 != cpu_count else int(len(symbols_to_explore))

    #keep track of how much work we should be processing
    work_processed = 0
    
    #check if we are stealing work or assigned work
    if (stolen_work):

        #grab work from other tasks
        waiting_at_lock = time.time()
        with steal_send_lock:

            #if we have waited 50 seconds at the lock, die
            if time.time() - waiting_at_lock > 50:
                exit(0)

            stealQueue.put(1)

            #if we spin for 50 seconds, give up
            start_spin_timer = time.time()
            while(sendQueue.empty()):
                time.sleep(1) #trade off of cpu usage vs time jump
                if (time.time() - start_spin_timer > 50):
                    exit(0)
                pass

            #grab our new indices
            assignment = sendQueue.get()
            first_index = assignment[0]
            second_index = assignment[1]


    #keep track of position
    current_index = first_index
    
    #split task
    for s in symbols_to_explore[first_index : ]:

        if work_processed == (second_index + 1) - first_index:
            break

        current_index += 1

        #see if we should allow someone to steal work
        if second_index - current_index > 10:
            with steal_receive_lock:
                if not stealQueue.empty():

                    #take message
                    stealQueue.get()

                    #determine partition
                    remaining_work = second_index - current_index 
                    held = int(remaining_work/2)
                    stolen = remaining_work - held

                    #update our work and the new task's work
                    stolen_second_index = second_index
                    second_index = current_index + held

                    #adjust first index based on rounding
                    stolen_first_index = current_index + stolen
                    if (remaining_work % 2) == 0:
                        stolen_first_index += 1

                    #print("thread: ", thread_index, "giving work: ", stolen_first_index, "-",stolen_second_index, " | keeping ", current_index, "-", second_index)

                    sendQueue.put((stolen_first_index, stolen_second_index))

        s_subscript = np.array(np.unravel_index(s, tuple((sym_x[0, :]).astype(int))))
        s_rect: np.array = np.row_stack((s_subscript * symbol_step + X_low,
                                         s_subscript * symbol_step + symbol_step + X_low))
        s_rect[0, :] = np.maximum(X_low, s_rect[0, :])
        s_rect[1, :] = np.minimum(X_up, s_rect[1, :])

        # transforming the targets and obstacles to a new coordinate system relative to the states in s.
        abstract_targets_polys = []
        abstract_targets_rects = []
        abstract_targets_polys_over_approx = []
        abstract_targets_rects_over_approx = []
        abstract_pos_targets_polys = []

        for target_idx, target_poly in enumerate(targets):
            abstract_target_poly = transform_poly_to_abstract_frames(target_poly, s_rect, over_approximate=False)
            abstract_pos_target_poly = transform_poly_to_abstract_frames(target_poly, s_rect, over_approximate=False, project_to_pos=True)
            abstract_target_poly_over_approx = transform_poly_to_abstract_frames(
                target_poly, s_rect, over_approximate=True)  # project_to_pos=True
            if not pc.is_empty(abstract_target_poly):
                rc, x1 = pc.cheby_ball(abstract_target_poly)
                abstract_target_rect = np.array([x1 - rc, x1 + rc])
            elif not pc.is_empty(abstract_pos_target_poly):
                # pdb.set_trace()
                raise "abstract target is empty for a concrete state"

            else:
                # pdb.set_trace()
                print("empty abstract_target_poly: ", abstract_target_poly)
                raise "empty abstract_target_poly error, grid must be refined, it's too far to see the position of " \
                      "the target similarly even within the same grid cell! "
            abstract_target_rect_over_approx = np.column_stack(pc.bounding_box(abstract_target_poly_over_approx)).T
            abstract_targets_rects.append(abstract_target_rect)
            abstract_targets_polys.append(abstract_target_poly)
            abstract_targets_rects_over_approx.append(abstract_target_rect_over_approx)
            abstract_targets_polys_over_approx.append(abstract_target_poly_over_approx)
            abstract_pos_targets_polys.append(abstract_pos_target_poly)

        if len(abstract_targets_polys) == 0:
            # pdb.set_trace()
            raise "Abstract target is empty"

        abstract_obstacles = pc.Region(list_poly=[])
        for obstacle_poly in obstacles:
            abstract_obstacle = transform_poly_to_abstract_frames(obstacle_poly, s_rect,
                                                                  over_approximate=True)  # project_to_pos=True
            abstract_obstacles = get_poly_union(abstract_obstacles, abstract_obstacle)

        with lock_one:
            symmetry_transformed_targets_and_obstacles[s] = RelCoordState(s, abstract_targets_polys,
                                                                        abstract_obstacles)
                

        min_dist = np.inf
        for curr_target_idx, curr_target_poly in enumerate(abstract_targets_polys):

            # CHANGE: compute the closest point of the target polytope to the origin
            curr_nearest_point, curr_dist = nearest_point_to_the_origin(curr_target_poly)
            if min_dist > curr_dist:
                nearest_point = curr_nearest_point
                min_dist = curr_dist

        with lock_one:
            nearest_target_of_concrete[s] = nearest_point
        
        curr_num_results = 3
        is_obstructed_u_idx = {}
        added_to_existing_state = False
        while curr_num_results < threshold_num_results:

            if strategy_3 or strategy_6 or curr_num_results == len(abstract_reachable_sets):
                hits = list(range(len(abstract_reachable_sets)))
            else:
                rtree_hits = list(reachability_rtree_idx3d.nearest(
                    (nearest_point[0], nearest_point[1], nearest_point[2],
                    nearest_point[0]+0.001, nearest_point[1]+0.001, nearest_point[2]+0.001),
                    num_results=curr_num_results, objects=True))
                hits = [hit.object for hit in rtree_hits]
                #bbox_hits = [hit.bbox for hit in rtree_hits]
        
            if len(hits):
                for idx, hit_object in enumerate(hits):
                    if not hit_object in is_obstructed_u_idx:
                        next_concrete_state_indices, _ = get_concrete_transition(s, hit_object, concrete_edges, neighbor_map,
                                                                sym_x, symbol_step, abstract_reachable_sets,
                                                                obstacles_rects, obstacle_indices, targets_rects,
                                                                target_indices, X_low, X_up, benchmark)
                        get_concrete_transition_calls += 1

                        is_obstructed_u_idx[hit_object] = (next_concrete_state_indices == [-2])
                        
                    if not is_obstructed_u_idx[hit_object]:
                        with lock_one:
                            if not hit_object in u_idx_to_abstract_states_indices:
                                rect = get_bounding_box(abstract_reachable_sets[hit_object][-1])
                                new_abstract_state = ThreadedAbstractState(next_abstract_state_id['next_abstract_state_id'],
                                                        np.average(rect, axis=0),
                                                        hit_object,
                                                        copy.deepcopy(symmetry_transformed_targets_and_obstacles[s].abstract_obstacles),
                                                        [s],
                                                        set([k for k, v in is_obstructed_u_idx.items() if v == True]), 
                                                        manager)
                                symmetry_abstract_states.append(new_abstract_state)
                                concrete_to_abstract[s] = next_abstract_state_id['next_abstract_state_id']
                                u_idx_to_abstract_states_indices[hit_object] = manager.list([next_abstract_state_id['next_abstract_state_id']])
                                abstract_to_concrete[next_abstract_state_id['next_abstract_state_id']] = manager.list([s])
                                next_abstract_state_id['next_abstract_state_id'] += 1
                                added_to_existing_state = True
                                valid_hit_idx_of_concrete[s] = idx
                                break
                            else:
                                if len(u_idx_to_abstract_states_indices[hit_object]):
                                    add_concrete_state_to_symmetry_abstract_state(s, u_idx_to_abstract_states_indices[hit_object][0],
                                        symmetry_transformed_targets_and_obstacles[s].abstract_obstacles, symmetry_abstract_states,
                                        concrete_to_abstract, abstract_to_concrete, is_obstructed_u_idx)
                                    added_to_existing_state = True
                                    valid_hit_idx_of_concrete[s] = idx
                                    break
                                else:
                                    raise "No abstract states for u_idx when one was expected"
                if added_to_existing_state:
                    break
            else:
                raise "No hits but rtree's nearest should always return a result"
            if added_to_existing_state:
                break
            else:
                if curr_num_results == threshold_num_results - 1:
                    break
                else:
                    curr_num_results = min(5 * curr_num_results, threshold_num_results - 1)
        if not added_to_existing_state:

            with lock_one:
                add_concrete_state_to_symmetry_abstract_state(s, 0, pc.Region(list_poly=[]),
                    symmetry_abstract_states, concrete_to_abstract, abstract_to_concrete, {})

                valid_hit_idx_of_concrete[s] = len(abstract_reachable_sets)

        work_processed += 1
        
    Q.put([symmetry_transformed_targets_and_obstacles, work_processed, concrete_edges, get_concrete_transition_calls])
    exit(0)

def create_symmetry_abstract_states_parallel(symbols_to_explore, symbol_step, targets, targets_rects, target_indices, obstacles,  obstacles_rects, obstacle_indices,
                                    sym_x, X_low, X_up, reachability_rtree_idx3d, abstract_reachable_sets):
    t_start = time.time()
    print('\n%s\tStart of the symmetry abstraction \n', time.time() - t_start)

    #make manaing object
    manager = Manager()

    #make all managed dictionaries
    symmetry_transformed_targets_and_obstacles = {}
    concrete_to_abstract = manager.dict()
    abstract_to_concrete = manager.dict()
    symmetry_abstract_states = manager.list()
    u_idx_to_abstract_states_indices = manager.dict()
    nearest_target_of_concrete = manager.dict()
    valid_hit_idx_of_concrete = manager.dict()
    next_abstract_state_id = manager.dict()

    #we now pickel the edges dict
    concrete_edges = {}
    neighbor_map = {}

    obstacle_state = ThreadedAbstractState(0, None, None, [], [], set(), manager)
    symmetry_abstract_states.append(obstacle_state)
    abstract_to_concrete[0] = manager.list()
    get_concrete_transition_calls = 0

    next_abstract_state_id['next_abstract_state_id'] = 1

    if strategy_5 or strategy_2:
        threshold_num_results = 376
    elif strategy_1 or strategy_4:
        threshold_num_results = len(abstract_reachable_sets) + 1
    else:
        threshold_num_results = 4

    #process locks (incase I need them)
    lock_one = multiprocess.Lock()
    lock_two = multiprocess.Lock()
    steal_receive_lock = multiprocess.Lock()
    steal_send_lock = multiprocess.Lock()

    #close file
    reachability_rtree_idx3d.close()

    #queue for communication
    Q = Queue()
    sendQueue = Queue()
    stealQueue = Queue()

    #spawn up threadpool and submit tasks
    max_assignment = len(symbols_to_explore)
    process_count = cpu_count

    #only assign as many threads as we have work for
    if max_assignment < cpu_count:
        process_count = max_assignment

    #create our pool
    for i in range(process_count):
        future_pool[i] = Process(target=create_symmetry_abstract_states_threaded, args=(lock_one, lock_two, list(symbols_to_explore), symbol_step, targets, obstacles, sym_x, X_low, X_up,
                                    reachability_rtree_idx3d, abstract_reachable_sets, symmetry_transformed_targets_and_obstacles, concrete_to_abstract,
                                    abstract_to_concrete, symmetry_abstract_states, u_idx_to_abstract_states_indices, nearest_target_of_concrete, valid_hit_idx_of_concrete,
                                    next_abstract_state_id, threshold_num_results, Q, i, manager, False, steal_send_lock, steal_receive_lock, stealQueue, sendQueue,
                                    obstacles_rects, obstacle_indices, targets_rects, target_indices, concrete_edges, neighbor_map))
    #start them
    for i in range(process_count):
        future_pool[i].start()
    
    #get results from each process
    counter_threads = 0
    current_returns = 0
    current_thread_index_counter = process_count

    while current_returns != len(symbols_to_explore):
        print("Awaiting Processes: " + str(int((current_returns/len(symbols_to_explore))*100)) + "%", end="\r")  

        if (int((current_returns/len(symbols_to_explore))*100) == 100):
            print(current_returns)

        result = Q.get()
        current_returns += result[1]
        counter_threads += 1
        symmetry_transformed_targets_and_obstacles.update(result[0])
        concrete_edges.update(result[2])
        get_concrete_transition_calls += result[3]

        
        #spawn new thread again
        future_pool[current_thread_index_counter] = Process(target=create_symmetry_abstract_states_threaded, args=(lock_one, lock_two, list(symbols_to_explore), symbol_step, targets, obstacles, sym_x, X_low, X_up,
                                    reachability_rtree_idx3d, abstract_reachable_sets, symmetry_transformed_targets_and_obstacles, concrete_to_abstract,
                                    abstract_to_concrete, symmetry_abstract_states, u_idx_to_abstract_states_indices, nearest_target_of_concrete, valid_hit_idx_of_concrete,
                                    next_abstract_state_id, threshold_num_results, Q, current_thread_index_counter, manager, True, steal_send_lock, steal_receive_lock, stealQueue, sendQueue,
                                    obstacles_rects, obstacle_indices, targets_rects, target_indices, concrete_edges, neighbor_map))

        future_pool[current_thread_index_counter].start()

        current_thread_index_counter += 1

        
    #kill any waiting theif processes
    for i in future_pool:
        if i != None:
            try:
                os.kill(i.pid, signal.SIGTERM)
            except OSError:
                pass

    print("I counted: ", current_returns, " states returned out of ", len(symbols_to_explore), " symbols to explore")
    print(['Done creation of symmetry abstract states in: ', time.time() - t_start, ' seconds'])
    print("concrete_to_abstract: ", len(concrete_to_abstract))
    print("abstract_to_concrete: ", len(abstract_to_concrete))
    print("concrete states deemed 'obstacle': ", len(symmetry_abstract_states[0].concrete_state_indices))
    print("symmetry abstract states found: ", len(symmetry_abstract_states))

    #convert the ThreadedAbstractStates to AbstractStates
    symmetry_abstract_states_single = dict()
    for idx in range(len(symmetry_abstract_states)):

        #make new key
        symmetry_abstract_states_single[idx] =  AbstractState(symmetry_abstract_states[idx].id,
                                                symmetry_abstract_states[idx].quantized_abstract_target,
                                                symmetry_abstract_states[idx].u_idx,
                                                symmetry_abstract_states[idx].abstract_obstacles[:],
                                                symmetry_abstract_states[idx].concrete_state_indices[:],
                                                set(list(symmetry_abstract_states[idx].obstructed_u_idx_set)[:])) # copy to list to force shallow copy
                                                
    #overwrite
    symmetry_abstract_states = symmetry_abstract_states_single

    #grab all values from abstract_to_concrete
    abstract_to_concrete_single = dict()
    for key, value in abstract_to_concrete.items():
        hold_array = []
        for i in abstract_to_concrete[key]:
            hold_array.append(i)
        abstract_to_concrete_single[key] = copy.deepcopy(hold_array)

    #overwrite
    abstract_to_concrete = copy.deepcopy(abstract_to_concrete_single)

    return symmetry_transformed_targets_and_obstacles, concrete_to_abstract, abstract_to_concrete, symmetry_abstract_states, nearest_target_of_concrete, valid_hit_idx_of_concrete, concrete_edges, neighbor_map, get_concrete_transition_calls


def add_concrete_state_to_symmetry_abstract_state(curr_concrete_state_idx, abstract_state_id, symmetry_transformed_obstacles_curr,
                                                  symmetry_abstract_states, concrete_to_abstract, abstract_to_concrete, is_obstructed_u_idx):

    if abstract_state_id > 0:
        union_poly_obstacles = get_poly_union(symmetry_transformed_obstacles_curr,
                                            symmetry_abstract_states[abstract_state_id].abstract_obstacles, check_convex=False)  # pc.union
        symmetry_abstract_states[abstract_state_id].abstract_obstacles = union_poly_obstacles
        symmetry_abstract_states[abstract_state_id].obstructed_u_idx_set = \
            (symmetry_abstract_states[abstract_state_id].obstructed_u_idx_set).union(set([k for k, v in is_obstructed_u_idx.items() if v == True]))
    
    symmetry_abstract_states[abstract_state_id].concrete_state_indices.append(curr_concrete_state_idx)
    concrete_to_abstract[curr_concrete_state_idx] = abstract_state_id
    abstract_to_concrete[abstract_state_id].append(curr_concrete_state_idx)
        
    return


def get_concrete_transition(s_idx, u_idx, concrete_edges, neighbor_map, #concrete_to_abstract,
                            sym_x, symbol_step, abstract_reachable_sets,
                            obstacles_rects, obstacle_indices, targets_rects, target_indices, X_low, X_up, benchmark):
    if benchmark and (s_idx, u_idx) in concrete_edges:
        return [-2], False
    elif (s_idx, u_idx) in concrete_edges:
        neighbors = concrete_edges[(s_idx, u_idx)]
        indices_to_delete = []
        for idx, succ_idx in enumerate(neighbors):
            if succ_idx == -1 or succ_idx in target_indices:
                indices_to_delete.append(idx)
            
        if len(indices_to_delete) == len(neighbors):
            concrete_edges[(s_idx, u_idx)] = [-1]
            return [-1], False

        if indices_to_delete:
            neighbors = np.delete(np.array(neighbors), np.array(indices_to_delete).astype(int)).tolist()
            neighbors.append(-1)

        concrete_edges[(s_idx, u_idx)] = copy.deepcopy(neighbors)
        return set(neighbors), False

    s_subscript = np.array(np.unravel_index(s_idx, tuple((sym_x[0, :]).astype(int))))
    s_rect: np.array = np.row_stack((s_subscript * symbol_step + X_low,
                                     s_subscript * symbol_step + symbol_step + X_low))
    s_rect[0, :] = np.maximum(X_low, s_rect[0, :])
    s_rect[1, :] = np.minimum(X_up, s_rect[1, :])
    for t_idx in range(len(abstract_reachable_sets[u_idx])):
        reachable_rect = np.column_stack(pc.bounding_box(abstract_reachable_sets[u_idx][t_idx])).T
        concrete_succ = transform_to_frames(reachable_rect[0, :],
                                            reachable_rect[1, :],
                                            s_rect[0, :], s_rect[1, :])
        if np.any(concrete_succ[1, :2] > X_up[:2]) or np.any(concrete_succ[0, :2] < X_low[:2]):
                #or np.any(concrete_succ[0, :] == concrete_succ[1, :]):
            concrete_edges[(s_idx, u_idx)] = [-2]
            return [-2], True  # unsafe
        for obstacle_rect in obstacles_rects:
            if do_rects_inter(obstacle_rect, concrete_succ):
                concrete_edges[(s_idx, u_idx)] = [-2]
                return [-2], True  # unsafe
    reachable_rect = np.column_stack(pc.bounding_box(abstract_reachable_sets[u_idx][-1])).T
    concrete_succ = transform_to_frames(reachable_rect[0, :],
                                        reachable_rect[1, :],
                                        s_rect[0, :], s_rect[1, :])
    '''for rect in targets_rects:
        if does_rect_contain(concrete_succ, rect):
            concrete_edges[(s_idx, u_idx)] = [-1]
            return [-1]  # reached target'''
    neighbors = rect_to_indices(concrete_succ, symbol_step, X_low, sym_x[0, :],
                                over_approximate=True).tolist()
    
    #neighbor_map[(s_idx, u_idx)] = neighbors

    indices_to_delete = []
    for idx, succ_idx in enumerate(neighbors):
        if succ_idx in obstacle_indices:
            concrete_edges[(s_idx, u_idx)] = [-2]
            return [-2], True
        if succ_idx in target_indices:
            indices_to_delete.append(idx)
        

    if len(indices_to_delete) == len(neighbors):
        # if not benchmark:
        concrete_edges[(s_idx, u_idx)] = [-1]
        return [-1], True

    if indices_to_delete:
        neighbors = np.delete(np.array(neighbors), np.array(indices_to_delete).astype(int)).tolist()
        neighbors.append(-1)

    # if not benchmark:
    concrete_edges[(s_idx, u_idx)] = copy.deepcopy(neighbors)
    return set(neighbors), True


# add quantized target and reachable set of u_idx
def plot_abstract_states(symmetry_abstract_states, deleted_abstract_states,
                         abstract_reachable_sets, state_to_paths_idx, abstract_to_concrete):
    obstacle_color = 'r'
    # target_color = 'g'
    reach_color = 'b'
    quantized_target_color = 'g'
    indices_to_plot = np.array(range(len(symmetry_abstract_states)))
    indices_to_plot = np.setdiff1d(indices_to_plot, np.array(deleted_abstract_states)).tolist()
    # indices_to_plot = state_to_paths_ind.keys()
    for idx in indices_to_plot:  # enumerate(symmetry_abstract_states) 
        abstract_state = symmetry_abstract_states[idx]
        if abstract_state.id != 0:
            plt.figure("Abstract state: " + str(idx))
            currentAxis = plt.gca()
            obstructed_u_indices = abstract_state.obstructed_u_idx_set
            # abstract_targets = abstract_state.abstract_targets

            # plot obstructed u_idx
            for obs_u_idx in obstructed_u_indices:
                for ind, region in enumerate(abstract_reachable_sets[obs_u_idx]):
                    if ind > 0:
                        if isinstance(region, pc.Region):
                            poly_list = region.list_poly
                        else:
                            poly_list = [region]
                        for poly in poly_list:
                            points = pc.extreme(poly)
                            points = points[:, :2]
                            hull = ConvexHull(points)
                            poly_patch = Polygon(points[hull.vertices, :], alpha=.5, color=obstacle_color, fill=True)
                            currentAxis.add_patch(poly_patch)
            
            
            point = abstract_state.quantized_abstract_target
            poly_patch = Rectangle((point[0]-.05, point[1]-.05), .1, .1, alpha=.3, color=quantized_target_color)
            currentAxis.add_patch(poly_patch)

            
            
            for ind, region in enumerate(abstract_reachable_sets[abstract_state.u_idx]):
                if ind > 0:
                    if isinstance(region, pc.Region):
                        poly_list = region.list_poly
                    else:
                        poly_list = [region]
                    for poly in poly_list:
                        points = pc.extreme(poly)
                        points = points[:, :2]
                        hull = ConvexHull(points)
                        poly_patch = Polygon(points[hull.vertices, :], alpha=.5, color=reach_color, fill=True)
                        currentAxis.add_patch(poly_patch)
            # -1,1 or -0.5,0.5
            plt.ylim([-.7, .7])
            plt.xlim([-.35, .35])
            plt.savefig("Abstract state: " + str(idx))
            # plt.show()
            plt.cla()
            plt.close()

    plt.figure("Abstract reachable sets after synthesis")
    color = 'orange'
    currentAxis_2 = plt.gca()
    for path_idx in range(len(abstract_reachable_sets)):
        for ind, region in enumerate(abstract_reachable_sets[path_idx]):
            if ind > 0:
                if isinstance(region, pc.Region):
                    poly_list = region.list_poly
                else:
                    poly_list = [region]
                for poly in poly_list:
                    points = pc.extreme(poly)
                    points = points[:, :2]
                    hull = ConvexHull(points)
                    poly_patch = Polygon(points[hull.vertices, :], alpha=.5, color=color, fill=True)
                    currentAxis_2.add_patch(poly_patch)
    plt.ylim([-3, 3])
    plt.xlim([-3, 3])
    plt.savefig("Abstract reachable sets after synthesis")


def plot_concrete_states(controllable_concrete_states, targets_rects, obstacles_rects,
                         state_to_paths_idx, sym_x, symbol_step, X_low, X_up):
    obstacle_color = 'r'
    target_color = 'g'
    reach_color = 'b'
    plt.figure("Controllable states")
    currentAxis = plt.gca()
    for rect in obstacles_rects:
        rect_patch = Rectangle((rect[0, 0], rect[0, 1]), rect[1, 0] - rect[0, 0], rect[1, 1] - rect[0, 1],
                               alpha=.5, color=obstacle_color, fill=True)
        currentAxis.add_patch(rect_patch)

    for rect in targets_rects:
        rect_patch = Rectangle((rect[0, 0], rect[0, 1]), rect[1, 0] - rect[0, 0], rect[1, 1] - rect[0, 1],
                               alpha=.5, color=target_color, fill=True)
        currentAxis.add_patch(rect_patch)

    for concrete_state_idx in controllable_concrete_states:  # equivalent to it being controllable
        rect = concrete_index_to_rect(concrete_state_idx, sym_x, symbol_step, X_low, X_up)
        rect_patch = Rectangle((rect[0, 0], rect[0, 1]), rect[1, 0] - rect[0, 0], rect[1, 1] - rect[0, 1],
                               alpha=.5, color=reach_color, fill=True)
        currentAxis.add_patch(rect_patch)
    plt.ylim([X_low[1] - 1, X_up[1] + 1])
    plt.xlim([X_low[0] - 1, X_up[0] + 1])
    plt.savefig("Controllable concrete space")
    # plt.show()
    plt.cla()
    plt.close()

def plot_controller(concrete_controller,
                    controllable_concrete_states,
                    U_discrete, sym_x, symbol_step, X_low, X_up):
    
    discrete_state = np.random.choice(controllable_concrete_states)
    rect_state = concrete_index_to_rect(discrete_state, sym_x, symbol_step, X_low, X_up)
    continuous_state = "random point in discrete_state"
    cos, sin = np.cos, np.sin

    for _ in range(0, 100):
        u_idx = concrete_controller[discrete_state]
        u = U_discrete[:, u_idx].reshape((1, 3)).T

        #"apply ODE solver to continuous_state until tau=3s"
        for _ in range(0,2):

            dx = [u[0]*cos(x(3)) - u[1]*sin(x(3)) + u(4), u[0]*sin(x(3)) + u[1]*cos(x(3))+u(5), u[2]+u(6)]

        #get last state

        #find grid cell with symbol step
    
        #grid cell to concrete_state

        #if in target then break

        #else continue

    pass

def create_symmetry_abstract_reachable_sets(Symbolic_reduced, n, reachability_rtree_idx2d, reachability_rtree_idx3d):
    reachable_rect_global_cntr = 0
    abstract_reachable_sets = []
    intersection_radius_threshold = None
    per_dim_max_travelled_distance = [0] * n
    for abs_s_idx in range(Symbolic_reduced.shape[0]):
        for u_idx in range(Symbolic_reduced.shape[1]):
            abstract_rect_low = Symbolic_reduced[abs_s_idx, u_idx, np.arange(n), -1]
            abstract_rect_up = Symbolic_reduced[abs_s_idx, u_idx, n + np.arange(n), -1]
            rect = np.array([abstract_rect_low, abstract_rect_up])
            rect = fix_angle_interval_in_rect(rect)
            curr_max_reachable_rect_radius = np.linalg.norm(rect[1, :] - rect[0, :]) / 2
            if intersection_radius_threshold is None or curr_max_reachable_rect_radius > intersection_radius_threshold:
                intersection_radius_threshold = curr_max_reachable_rect_radius

            reachability_rtree_idx3d.insert(reachable_rect_global_cntr, (rect[0, 0], rect[0, 1],
                                                                         rect[0, 2], rect[1, 0],
                                                                         rect[1, 1], rect[1, 2]),
                                            obj=u_idx)
            reachability_rtree_idx2d.insert(reachable_rect_global_cntr, (rect[0, 0], rect[0, 1], rect[1, 0],
                                                                         rect[1, 1]),
                                            obj=u_idx)
            abstract_initial_rect_low = Symbolic_reduced[abs_s_idx, u_idx, np.arange(n), 0]
            abstract_initial_rect_up = Symbolic_reduced[abs_s_idx, u_idx, n + np.arange(n), 0]
            initial_rect = np.array([abstract_initial_rect_low, abstract_initial_rect_up])
            initial_rect = fix_angle_interval_in_rect(initial_rect)
            for dim in range(n):
                per_dim_max_travelled_distance[dim] = max(per_dim_max_travelled_distance[dim],
                                                          abs(rect[0, dim] + rect[1, dim] -
                                                              initial_rect[0, dim] - initial_rect[1, dim]) / 2)
            reachable_rect_global_cntr += 1
            original_abstract_path = []
            for t_idx in range(Symbolic_reduced.shape[3]):
                rect = np.array([Symbolic_reduced[abs_s_idx, u_idx, np.arange(n), t_idx],
                                 Symbolic_reduced[abs_s_idx, u_idx, n + np.arange(n), t_idx]])
                rect = fix_angle_interval_in_rect(rect)
                poly = pc.box2poly(rect.T)
                original_abstract_path.append(poly)
            abstract_reachable_sets.append(original_abstract_path)
    return abstract_reachable_sets, reachable_rect_global_cntr, intersection_radius_threshold, \
        np.array(per_dim_max_travelled_distance)


def create_targets_and_obstacles(Target_low, Target_up, Obstacle_low, Obstacle_up, symbol_step, sym_x, X_low):
    obstacles = []
    targets = []
    targets_rects = []
    obstacles_rects = []
    obstacle_indices = set()
    target_indices = set()
    for obstacle_idx in range(Obstacle_low.shape[0]):
        obstacle_rect = np.array([Obstacle_low[obstacle_idx, :], Obstacle_up[obstacle_idx, :]])
        # obstacle_rect = fix_rect_angles(obstacle_rect)
        obstacles_rects.append(obstacle_rect)
        obstacle_poly = pc.box2poly(obstacle_rect.T)
        obstacles.append(obstacle_poly)
        temp_obstacle_indices = rect_to_indices(obstacle_rect, symbol_step, X_low,
                                                sym_x[0, :], over_approximate=True)
        for idx in temp_obstacle_indices:
            obstacle_indices.add(idx)

    for target_idx in range(Target_low.shape[0]):
        target_rect = np.array([Target_low[target_idx, :], Target_up[target_idx, :]])
        # target_rect = fix_rect_angles(target_rect)
        target_poly = pc.box2poly(target_rect.T)
        targets.append(target_poly)
        targets_rects.append(target_rect)
        temp_target_indices = rect_to_indices(target_rect, symbol_step, X_low,
                                              sym_x[0, :], over_approximate=False)
        for idx in temp_target_indices:
            target_indices.add(idx)

    return targets, targets_rects, target_indices, obstacles, obstacles_rects, obstacle_indices


def set_in_target(last_set: np.array, target: pc.Polytope):
    new_target = get_poly_list_with_decomposed_angle_intervals(target)
    uncovered_region = pc.mldivide(new_target, pc.box2poly(last_set.T))
    if pc.is_empty(uncovered_region):
        return 2  # contained in target
    return 0


def successor_in_or_intersects_target_smt(concrete_initial_set, u_idx, abstract_reachable_sets,
                                          cur_solver, var_dict,
                                          extended_target_rtree_idx3d, n, eng,
                                          time_step, W_low, W_up, U_discrete, symmetry_based=False):
    if symmetry_based:
        concrete_reachable_poly = transform_poly_to_frames(abstract_reachable_sets[u_idx][-1], concrete_initial_set[0, :],
                                                           concrete_initial_set[1, :])
        reachable_set = [np.column_stack(pc.bounding_box(concrete_reachable_poly)).T]
    else:
        rect_low = concrete_initial_set[0, :].reshape((1, 3)).T
        rect_up = concrete_initial_set[1, :].reshape((1, 3)).T
        u = U_discrete[:, u_idx].reshape((1, 3)).T
        reachtube = compute_reachable_set_tira(eng, time_step, rect_low, rect_up, u, W_low, W_up)
        reachable_set = [reachtube[-1]]
    hits = list(
        extended_target_rtree_idx3d.intersection(
            (reachable_set[-1][0, 0], reachable_set[-1][0, 1], reachable_set[-1][0, 2],
             reachable_set[-1][1, 0], reachable_set[-1][1, 1],
             reachable_set[-1][1, 2]),
            objects=True))
    inter_num = len(hits)
    if inter_num > 0:
        hits_rects = np.array([np.array([hit.bbox[:n], hit.bbox[n:]]) for hit in hits])
        cur_solver.reset()
        cur_solver = add_rects_to_solver(hits_rects, var_dict, cur_solver)
        uncovered_state = do_rects_list_contain_smt(reachable_set[-1], var_dict, cur_solver)
        if uncovered_state is None:
            return 2
        return 1
    return 0


def which_obstacle_successor_intersects(abstract_state_idx, u_idx, abstract_reachable_sets,
                                        abstract_obstacles_polytopes: List[pc.Polytope]):
    for t_idx in range(len(abstract_reachable_sets[u_idx])):
        for idx, poly in enumerate(abstract_obstacles_polytopes):
            if not pc.is_empty(get_poly_intersection(abstract_reachable_sets[u_idx][t_idx],
                                                     poly,
                                                     check_convex=False)):
                return idx
    return -1


def concrete_index_to_rect(concrete_state_idx, sym_x, symbol_step, X_low, X_up):
    concrete_subscript = np.array(
        np.unravel_index(concrete_state_idx, tuple((sym_x[0, :]).astype(int))))
    concrete_rect: np.array = np.row_stack(
        (concrete_subscript * symbol_step + X_low,
         concrete_subscript * symbol_step + symbol_step + X_low))
    concrete_rect[0, :] = np.maximum(X_low, concrete_rect[0, :])
    concrete_rect[1, :] = np.minimum(X_up, concrete_rect[1, :])
    return concrete_rect


def quantize(grid_rtree, point, cell_size_per_dim):
    hits = []
    hits.extend(list(grid_rtree.nearest(
        (point[0], point[1], point[2],
         point[0] + 0.001, point[1] + 0.001, point[2] + 0.001),
        num_results=1,
        objects=True)))
    grid_point = tuple(hits[0].bbox[:3])
    point_in_cell = True
    for dim in point.shape[0]:
            if abs(point[dim] - grid_point[0]) > cell_size_per_dim[dim]:
                point_in_cell = False
                break
    return grid_point, point_in_cell


def symmetry_abstract_synthesis_helper(concrete_states_to_explore,
                                       concrete_edges,
                                       neighbor_map,
                                       abstract_to_concrete,
                                       concrete_to_abstract,
                                       symmetry_transformed_targets_and_obstacles,
                                       nearest_abstract_target_of_concrete,
                                       valid_hit_idx_of_concrete,
                                       abstract_reachable_sets,
                                       symmetry_abstract_states,
                                       refinement_candidates,
                                       controllable_concrete_states,
                                       concrete_controller,
                                       reachability_rtree_idx3d,
                                       per_dim_max_travelled_distance,
                                       obstacles_rects, obstacle_indices,
                                       targets_rects, target_indices,
                                       X_low, X_up, sym_x, symbol_step):
    t_start = time.time()
    num_controllable_states = 0
    n = X_up.shape[0]
    # visited_concrete_states = {}
    abstract_state_to_u_idx_poll = {} #initialize on the spot
    abstract_state_to_u_idx_set = {}

    total_nb_explore = len(concrete_states_to_explore)

    if strategy_5 or strategy_2:
        threshold_num_results = 376
    elif strategy_1 or strategy_4:
        threshold_num_results = len(abstract_reachable_sets) + 1
    else:
        threshold_num_results = 1

    
    nb_iterations = 0
    running_sum_path_lengths = 0
    sum_ratios_neighbor_to_total = 0
    exploration_record = []

    total_state_u_pairs_explored = 0
    unique_state_u_pairs_explored = 0
    total_states_explored = 0

    while True: # one iteration of this loop will try current abstraction to find controllable states
        
        temp_controllable_concrete_states = set()

        num_new_symbols = 0

        #debug_status = [0,0,0]
        
        for concrete_state_idx in concrete_states_to_explore:
            
            

            '''if concrete_state_idx in obstacle_indices \
                    or concrete_state_idx in target_indices \
                    or concrete_state_idx in controllable_concrete_states \
                    or (concrete_state_idx in concrete_to_abstract \
                    and concrete_to_abstract[concrete_state_idx] == 0):
                    #or concrete_state_idx in visited_concrete_states:
                #debug_status[0] += 1
                continue'''
            if (not concrete_state_idx in concrete_to_abstract) or concrete_to_abstract[concrete_state_idx] == 0:
                continue

            total_states_explored += 1

            abstract_state_idx = concrete_to_abstract[concrete_state_idx]


            abstract_state = symmetry_abstract_states[abstract_state_idx]
            #quantized_target = abstract_state.quantized_abstract_target

            '''rect: np.array = concrete_index_to_rect(concrete_state_idx,
                                                    sym_x, symbol_step, X_low, X_up)
            rect_center = np.average(rect, axis=0)
            angle_interval = [rect[0, 2], rect[1, 2]]
            angle_interval_center = (angle_interval[0] + angle_interval[1]) / 2'''
            # decomposed_angle_intervals = get_decomposed_angle_intervals(original_angle_interval)
            # target_hits = []
            #quantized_target_centers = []

            #obstacle_hits = []
            #quantized_obstacle_centers = []
            

            if not abstract_state_idx in abstract_state_to_u_idx_poll:
                if abstract_state_idx:
                    abstract_state_to_u_idx_poll[abstract_state_idx] = [(0, abstract_state.u_idx)]
                    abstract_state_to_u_idx_set[abstract_state_idx] = set([abstract_state.u_idx])
                else:
                    abstract_state_to_u_idx_poll[abstract_state_idx] = []
                    abstract_state_to_u_idx_set[abstract_state_idx] = set([])



            valid_vote = None
            for v, u_idx in abstract_state_to_u_idx_poll[abstract_state_idx]: #enumerate

                next_concrete_state_indices, is_new_entry = get_concrete_transition(concrete_state_idx, u_idx, concrete_edges, neighbor_map,
                                                                    sym_x, symbol_step, abstract_reachable_sets,
                                                                    obstacles_rects, obstacle_indices, targets_rects,
                                                                    controllable_concrete_states, X_low, X_up, benchmark)
                if is_new_entry:
                    unique_state_u_pairs_explored += 1
                total_state_u_pairs_explored += 1
                
                is_controlled = (next_concrete_state_indices == [-1])
                '''for next_concrete_state_idx in next_concrete_state_indices:
                    if not (next_concrete_state_idx == -1 or
                            (next_concrete_state_idx >= 0 and
                            next_concrete_state_idx in controllable_concrete_states)):
                        is_controlled = False
                        break'''
                if is_controlled:
                    abstract_state_to_u_idx_poll[abstract_state_idx].remove((v, u_idx)) #linked list later
                    controllable_concrete_states.add(concrete_state_idx)

                    temp_controllable_concrete_states.add(concrete_state_idx)

                    valid_vote = (v+1, u_idx)
                    bisect.insort(abstract_state_to_u_idx_poll[abstract_state_idx], valid_vote, key=lambda x: -x[0])
                    concrete_controller[concrete_state_idx] = valid_vote[1]
                    num_new_symbols +=1
                    break

            

            if valid_vote is None:
                visited_u_idx = abstract_state_to_u_idx_set[abstract_state_idx].copy()

                
                if strategy_3 or strategy_6:
                    curr_num_results = threshold_num_results - 1
                else:
                    curr_num_results = min((valid_hit_idx_of_concrete[concrete_state_idx] + 1) * 5, threshold_num_results - 1)
                

                nearest_point = nearest_abstract_target_of_concrete[concrete_state_idx]
                
                #is_obstructed_u_idx = {}
                
                new_u_idx_found = False
                while curr_num_results < threshold_num_results:
                    # hits = list(range(len(abstract_reachable_sets)))
                    
                    if strategy_3 or strategy_6 or curr_num_results == len(abstract_reachable_sets):
                        hits = list(range(len(abstract_reachable_sets)))
                    elif (strategy_2 or strategy_5) and curr_num_results == threshold_num_results - 1:
                        hit_count = 0
                        random_hits = []
                        while hit_count < 75:
                            hit_candidate = np.random.randint(len(abstract_reachable_sets))
                            if hit_candidate in visited_u_idx or hit_candidate in random_hits:
                                continue
                            random_hits.append(hit_candidate)
                            hit_count += 1
                        hits = random_hits
                    else:
                        hits = [hit.object for hit in list(reachability_rtree_idx3d.nearest(
                            (nearest_point[0], nearest_point[1], nearest_point[2],
                            nearest_point[0]+0.001, nearest_point[1]+0.001, nearest_point[2]+0.001),
                            num_results=curr_num_results, objects=True))
                        ]
                    
                    if len(hits):
                        for hit_object in hits:

                            if not hit_object in visited_u_idx:

                                '''if not hit_object in is_obstructed_u_idx:
                                    for p_idx in range(len(abstract_reachable_sets[hit_object]), 0, -1):
                                        if type(symmetry_transformed_targets_and_obstacles[s].abstract_obstacles) == pc.Region:
                                            list_obstacles = symmetry_transformed_targets_and_obstacles[s].abstract_obstacles.list_poly
                                        else:
                                            list_obstacles = [symmetry_transformed_targets_and_obstacles[s].abstract_obstacles]
                                        for obstacle in list_obstacles:
                                            if not pc.is_empty(pc.intersect(abstract_reachable_sets[hit_object][p_idx-1], obstacle)):
                                                is_obstructed_u_idx[hit_object] = True
                                                break
                                        if hit_object in is_obstructed_u_idx:
                                            break
                                    if not hit_object in is_obstructed_u_idx:
                                        is_obstructed_u_idx[hit_object] = False
                                if not is_obstructed_u_idx[hit_object]:'''
                
                                next_concrete_state_indices, is_new_entry = \
                                    get_concrete_transition(concrete_state_idx, hit_object, concrete_edges, neighbor_map,
                                                                sym_x, symbol_step, abstract_reachable_sets,
                                                                obstacles_rects, obstacle_indices, targets_rects,
                                                                controllable_concrete_states, X_low, X_up, benchmark)
                                if is_new_entry:
                                    unique_state_u_pairs_explored += 1
                                total_state_u_pairs_explored += 1
            
                                is_controlled = (next_concrete_state_indices == [-1])
                                '''for next_concrete_state_idx in next_concrete_state_indices:
                                    if not (next_concrete_state_idx == -1 or
                                            (next_concrete_state_idx >= 0 and
                                            next_concrete_state_idx in controllable_concrete_states)):
                                        is_controlled = False
                                        break'''
                                if is_controlled:
                                    controllable_concrete_states.add(concrete_state_idx)

                                    temp_controllable_concrete_states.add(concrete_state_idx)
                                    
                                    valid_vote = (1, hit_object)
                                    bisect.insort(abstract_state_to_u_idx_poll[abstract_state_idx], valid_vote, key=lambda x: -x[0])
                                    (abstract_state_to_u_idx_set[abstract_state_idx]).add(hit_object)
                                    concrete_controller[concrete_state_idx] = valid_vote[1]
                                    new_u_idx_found = True
                                    num_new_symbols +=1
                                    break
                                visited_u_idx.add(hit_object)

                        if new_u_idx_found:
                            break
                    else:
                        raise "No hits but rtree's nearest should always return a result"
                    if curr_num_results == threshold_num_results - 1:
                        break
                    else:
                        curr_num_results = min(5 * curr_num_results, threshold_num_results - 1)
                if not new_u_idx_found:
                    #abstract_state.concrete_state_indices.remove(concrete_state_idx)
                    #abstract_to_concrete[abstract_state_idx].remove(concrete_state_idx)

                    #add_concrete_state_to_symmetry_abstract_state(s, 0, pc.Region(list_poly=[]), symmetry_abstract_states, concrete_to_abstract, abstract_to_concrete, {})
                    #debug_status[1] += 1
                    pass

        #print(f"{debug_status[0]} states not analyzed for synthesis\n{debug_status[1]} states reached the control threshold")


        if num_new_symbols:
            print(time.time() - t_start, " ", num_new_symbols,
                  " new controllable states have been found in this synthesis iteration\n")
            num_controllable_states += num_new_symbols
            # candidate_initial_set_rect = None
            if strategy_4 or strategy_5 or strategy_6:
                rects = []
                
                for concrete_state_idx in temp_controllable_concrete_states:
                    # print which abstract states got right controls this iteration
                    # print("The abstract symbol ", abstract_state_idx, " is controllable using path indices ", abstract_state_to_u_idx_poll[abstract_state_idx])
                    s_rect: np.array = concrete_index_to_rect(concrete_state_idx, sym_x, symbol_step, X_low, X_up)
                    max_distance = 2 * (per_dim_max_travelled_distance + symbol_step)
                    bloated_rect = np.array([np.maximum(np.add(s_rect[0, :], -max_distance), X_low),
                                            np.minimum(np.add(s_rect[1, :], max_distance), X_up)])
                    temp_rects = [bloated_rect]
                    for obstacle_rect in obstacles_rects:
                        per_obstacle_temp_rects = []
                        for temp_rect in temp_rects:
                            per_obstacle_temp_rects.extend(subtract_rectangles(temp_rect, obstacle_rect))
                        temp_rects = copy.deepcopy(per_obstacle_temp_rects)
                    rects.extend(temp_rects)
                
                concrete_states_to_explore = set()
                for neighborhood_rect in rects:
                    concrete_states_to_explore = concrete_states_to_explore.union(
                        rect_to_indices(neighborhood_rect, symbol_step, X_low,
                                        sym_x[0, :], over_approximate=True))
                concrete_states_to_explore = concrete_states_to_explore.difference(controllable_concrete_states)
                concrete_states_to_explore = concrete_states_to_explore.difference(obstacle_indices)
            else:
                concrete_states_to_explore = concrete_states_to_explore.difference(temp_controllable_concrete_states)

            exploration_record.append((len(concrete_states_to_explore), total_nb_explore - num_controllable_states))
            remaining_to_explore = total_nb_explore - num_controllable_states
            if remaining_to_explore:
                ratio_neighbor_to_total = len(concrete_states_to_explore) / remaining_to_explore
            else:
                ratio_neighbor_to_total = 1
            print("Ratio of neighbors over total explored", ratio_neighbor_to_total)
            sum_ratios_neighbor_to_total += ratio_neighbor_to_total
            nb_iterations += 1
            running_sum_path_lengths += num_new_symbols * nb_iterations

            print(num_controllable_states, ' symbols are controllable to satisfy the reach-avoid specification\n')
        else:
            print('No new controllable state has been found in this synthesis iteration\n', time.time() - t_start)
            break
            
    with open('control_polls.csv', 'w', newline='') as csvfile:
        table = csv.writer(csvfile, delimiter=';', quotechar='|', quoting=csv.QUOTE_MINIMAL)
        table.writerow(['Abstract State', 'First Control', 'Second Control', 'Third Control', 'Nth Control'])
        for abstract_idx in range(1, len(abstract_to_concrete)):
            poll = abstract_state_to_u_idx_poll[abstract_idx]
            table.writerow([str(abstract_idx)] + [str(u)+':'+str(v) for (v,u) in poll])

    poll_lengths = [len(poll) for _, poll in abstract_state_to_u_idx_set.items()]
    if nb_iterations:
        average_ratio_neighbor_to_total = sum_ratios_neighbor_to_total / nb_iterations
    else:
        average_ratio_neighbor_to_total = 'No controllable state found'


    np.save('exploration_record.npy', exploration_record)

    if concrete_controller:
        average_path_length = running_sum_path_lengths / len(concrete_controller)
    else:
        average_path_length = 'No controllable state found'

    return concrete_controller, refinement_candidates, poll_lengths, average_ratio_neighbor_to_total, neighbor_map, unique_state_u_pairs_explored, total_state_u_pairs_explored, total_states_explored, average_path_length, nb_iterations

def symmetry_synthesis_helper(concrete_states_to_explore,
                              concrete_edges,
                              neighbor_map,
                              abstract_reachable_sets,
                              controllable_concrete_states,
                              concrete_controller,
                              per_dim_max_travelled_distance,
                              obstacles_rects, obstacle_indices,
                              targets_rects, X_low, X_up, sym_x, symbol_step):
    t_start = time.time()
    num_controllable_states = 0

    nb_iterations = 0
    running_sum_path_lengths = 0

    neighbor_map = {}

    total_state_u_pairs_explored = 0
    unique_state_u_pairs_explored = 0
    total_states_explored = 0

    while True:
        temp_controllable_concrete_states = set()
        num_new_symbols = 0
        
        for concrete_state_idx in concrete_states_to_explore:
            '''
            hit_count = 0
            random_hits = set()
            while hit_count <= 400:
                hit_candidate = np.random.randint(len(abstract_reachable_sets))
                if hit_candidate in random_hits:
                    continue
                random_hits.add(hit_candidate)
                hit_count += 1
            hits = random_hits
            '''
            
            hits = list(range(len(abstract_reachable_sets)))
            total_states_explored += 1
            
            if len(hits):
                for hit in hits:
        
                    next_concrete_state_indices, is_new_entry = get_concrete_transition(concrete_state_idx, hit,
                                                    concrete_edges, neighbor_map,
                                                    sym_x, symbol_step, abstract_reachable_sets,
                                                    obstacles_rects, obstacle_indices, targets_rects,
                                                    controllable_concrete_states, X_low, X_up, benchmark)
                    if is_new_entry:
                        unique_state_u_pairs_explored += 1
                    total_state_u_pairs_explored += 1

                    is_controlled = (next_concrete_state_indices == [-1])

                    if is_controlled:
                        controllable_concrete_states.add(concrete_state_idx)

                        temp_controllable_concrete_states.add(concrete_state_idx)
                        
                        concrete_controller[concrete_state_idx] = hit
                        num_new_symbols +=1
                        break
            else:
                raise "No hits but abstract_reachable_sets should be non empty"
            

        if num_new_symbols:
            print(time.time() - t_start, " ", num_new_symbols,
                  " new controllable states have been found in this synthesis iteration\n")
            
            num_controllable_states += num_new_symbols

            concrete_states_to_explore = concrete_states_to_explore.difference(temp_controllable_concrete_states)

            nb_iterations += 1
            running_sum_path_lengths += num_new_symbols * nb_iterations

            print(num_controllable_states, ' symbols are controllable to satisfy the reach-avoid specification\n')
        else:
            print('No new controllable state has been found in this synthesis iteration\n', time.time() - t_start)
            break

    if concrete_controller:
        average_path_length = running_sum_path_lengths / len(concrete_controller)
    else:
        average_path_length = 'No controllable state found'

    return concrete_controller, neighbor_map, unique_state_u_pairs_explored, total_state_u_pairs_explored, total_states_explored, average_path_length, nb_iterations


def get_decomposed_angle_intervals(original_angle_interval):
    decomposed_angle_intervals = []
    # original_angle_interval[0] = fix_angle_to_positive_value(original_angle_interval[0])
    # original_angle_interval[1] = fix_angle_to_positive_value(original_angle_interval[1])
    original_angle_interval[0], original_angle_interval[1] = \
        fix_angle_interval(original_angle_interval[0], original_angle_interval[1])
    if original_angle_interval[0] > original_angle_interval[1]:
        raise "how did an angle interval ended up with flipped order"
    while original_angle_interval[0] > 2 * math.pi and original_angle_interval[1] > 2 * math.pi:
        original_angle_interval[0] -= 2 * math.pi
        original_angle_interval[1] -= 2 * math.pi
    if original_angle_interval[0] < 2 * math.pi < original_angle_interval[1]:
        decomposed_angle_intervals.append([0, original_angle_interval[1] - 2 * math.pi])
        decomposed_angle_intervals.append([original_angle_interval[0], original_angle_interval[1]])  # 2 * math.pi
        # decomposed_angle_intervals.append([original_angle_interval[0] + 2 * math.pi, 4 * math.pi])
    else:
        decomposed_angle_intervals.append(original_angle_interval)
    return decomposed_angle_intervals


def split_abstract_state(abstract_state_idx, concrete_indices,
                         abstract_to_concrete, concrete_to_abstract, target_parents,
                         symmetry_transformed_targets_and_obstacles, symmetry_abstract_states):
    abstract_state_1 = None
    abstract_state_2 = None
    if len(concrete_indices) >= len(abstract_to_concrete[abstract_state_idx]):
        print("The concrete indices provided are all that ", abstract_state_idx, " represents, so no need to split.")
        return concrete_to_abstract, abstract_to_concrete, target_parents
    rest_of_concrete_indices = np.setdiff1d(np.array(abstract_to_concrete[abstract_state_idx]), concrete_indices)
    for concrete_state_idx in concrete_indices:
        if pc.is_empty(get_poly_intersection(
                symmetry_transformed_targets_and_obstacles[concrete_state_idx].abstract_targets[0],
                symmetry_abstract_states[abstract_state_idx].abstract_targets[0])):
            print("A concrete state has a relative target that is no longer "
                  "intersecting the relative target of the abstract state it belongs to. This shouldn't happen.")
        abstract_state_1 = add_concrete_state_to_symmetry_abstract_state(concrete_state_idx,
                                                                         copy.deepcopy(abstract_state_1),
                                                                         symmetry_transformed_targets_and_obstacles)
    for concrete_state_idx in rest_of_concrete_indices:
        if pc.is_empty(get_poly_intersection(
                symmetry_transformed_targets_and_obstacles[concrete_state_idx].abstract_targets[0],
                symmetry_abstract_states[abstract_state_idx].abstract_targets[0])):
            print("A concrete state has a relative target that is no longer "
                  "intersecting the relative target of the abstract state it belongs to. This shouldn't happen.")
        abstract_state_2 = add_concrete_state_to_symmetry_abstract_state(concrete_state_idx,
                                                                         copy.deepcopy(abstract_state_2),
                                                                         symmetry_transformed_targets_and_obstacles)

    symmetry_abstract_states.append(abstract_state_1)
    abstract_to_concrete.append(abstract_state_1.concrete_state_idx)
    for idx in abstract_state_1.concrete_state_idx:
        concrete_to_abstract[idx] = len(abstract_to_concrete) - 1
    symmetry_abstract_states.append(abstract_state_2)
    abstract_to_concrete.append(abstract_state_2.concrete_state_idx)
    for idx in abstract_state_2.concrete_state_idx:
        concrete_to_abstract[idx] = len(abstract_to_concrete) - 1

    abstract_to_concrete[abstract_state_idx] = []

    return concrete_to_abstract, abstract_to_concrete, target_parents


def refine(concrete_to_abstract, abstract_to_concrete, symmetry_abstract_states,
           remaining_abstract_states, refinement_candidates, target_parents, inverse_abstract_transitions,
           local_abstract_states_to_explore,
           abstract_states_to_explore,
           controllable_abstract_states, symmetry_transformed_targets_and_obstacles,
           abstract_reachable_sets, concrete_edges, sym_x, symbol_step,
           obstacles_rects, obstacle_indices, targets_rects, target_indices, X_low, X_up):
    itr = 0
    progress = False
    num_new_abstract_states = 0
    deleted_abstract_states = []
    max_num_of_refinements = 10  # len(refinement_candidates)
    local_abstract_states_to_explore = set()
    controllable_abstract_states_temp = set()
    while itr < max_num_of_refinements and len(refinement_candidates):
        itr += 1
        abstract_state_idx = random.choice(tuple(refinement_candidates))
        concrete_indices_len = len(abstract_to_concrete[abstract_state_idx])
        if concrete_indices_len > 1:
            concrete_indices = set()
            if abstract_state_idx in target_parents:
                _, u_idx = random.choice(tuple(target_parents[abstract_state_idx]))
                for concrete_state_idx in abstract_to_concrete[abstract_state_idx]:
                    next_concrete_state_indices = get_concrete_transition(concrete_state_idx, u_idx, concrete_edges,
                                                                          sym_x, symbol_step, abstract_reachable_sets,
                                                                          obstacles_rects, obstacle_indices,
                                                                          targets_rects, target_indices, X_low, X_up)
                    for next_concrete_state_idx in next_concrete_state_indices:
                        if not (next_concrete_state_idx == -1 or
                                (next_concrete_state_idx >= 0 and
                                 concrete_to_abstract[next_concrete_state_idx] in controllable_abstract_states)):
                            concrete_indices.add(concrete_state_idx)
                            break
                if not concrete_indices:
                    controllable_abstract_states.add(abstract_state_idx)
                    controllable_abstract_states_temp.add(abstract_state_idx)
                    abstract_states_to_explore.remove(abstract_state_idx)
                    # np.setdiff1d(np.array(abstract_states_to_explore), [abstract_state_idx]).tolist()
                    del target_parents[abstract_state_idx]
                    if abstract_state_idx in inverse_abstract_transitions:
                        for parent_abstract_state_idx, parent_u_idx in inverse_abstract_transitions[abstract_state_idx]:
                            if parent_abstract_state_idx != abstract_state_idx \
                                    and parent_abstract_state_idx in abstract_states_to_explore:
                                if parent_abstract_state_idx in target_parents:
                                    target_parents[parent_abstract_state_idx].add((abstract_state_idx, parent_u_idx))
                                else:
                                    target_parents[parent_abstract_state_idx] = {(abstract_state_idx, parent_u_idx)}
                                if len(abstract_to_concrete[parent_abstract_state_idx]) > 1:
                                    refinement_candidates.add(parent_abstract_state_idx)
                    refinement_candidates.remove(abstract_state_idx)
                    # local_abstract_states_to_explore = copy.deepcopy(abstract_states_to_explore)
                    # set(target_parents.keys())
                    abstract_states_to_explore = abstract_states_to_explore.difference(
                        controllable_abstract_states_temp)
                    # np.setdiff1d(abstract_states_to_explore, controllable_abstract_states_temp).tolist()
                    continue
            if abstract_state_idx not in target_parents \
                    or len(concrete_indices) >= len(abstract_to_concrete[abstract_state_idx]):
                concrete_indices = random.choices(abstract_to_concrete[abstract_state_idx],
                                                  k=int(concrete_indices_len / 2))
            if concrete_indices:
                concrete_indices = list(set(concrete_indices))
            progress = True
            if len(concrete_indices) < len(abstract_to_concrete[abstract_state_idx]):
                concrete_to_abstract, abstract_to_concrete, target_parents = \
                    split_abstract_state(abstract_state_idx, concrete_indices,
                                         abstract_to_concrete, concrete_to_abstract, target_parents,
                                         symmetry_transformed_targets_and_obstacles,
                                         symmetry_abstract_states)
                # remaining_abstract_states.remove(abstract_state_idx) # np.setdiff1d(, np.array([abstract_state_idx]))
                # remaining_abstract_states.add(len(abstract_to_concrete) - 1)
                # remaining_abstract_states.add(len(abstract_to_concrete) - 2)
                if len(abstract_to_concrete[-1]) > 1:
                    refinement_candidates.add(len(abstract_to_concrete) - 1)
                if len(abstract_to_concrete[-2]) > 1:
                    refinement_candidates.add(len(abstract_to_concrete) - 2)
                abstract_states_to_explore.remove(abstract_state_idx)
                # np.setdiff1d(np.array(abstract_states_to_explore),  [abstract_state_idx]).tolist()
                abstract_states_to_explore.add(len(abstract_to_concrete) - 1)  # append
                abstract_states_to_explore.add(len(abstract_to_concrete) - 2)
                if abstract_state_idx in target_parents:
                    del target_parents[abstract_state_idx]
                    # target_parents.remove(abstract_state_idx)
                    # target_parents[len(abstract_to_concrete) - 1] = set()
                    # target_parents[len(abstract_to_concrete) - 2] = set()
                # target_parents.add(len(abstract_to_concrete) - 1)
                # target_parents.add(len(abstract_to_concrete) - 2)
                refinement_candidates.remove(abstract_state_idx)
                deleted_abstract_states.append(abstract_state_idx)
                if abstract_state_idx in local_abstract_states_to_explore:
                    local_abstract_states_to_explore.remove(abstract_state_idx)
                local_abstract_states_to_explore.add(len(abstract_to_concrete) - 1)
                local_abstract_states_to_explore.add(len(abstract_to_concrete) - 2)
                num_new_abstract_states += 1
        else:
            refinement_candidates.remove(abstract_state_idx)

    return progress, num_new_abstract_states, concrete_to_abstract, abstract_to_concrete, \
        refinement_candidates, target_parents, local_abstract_states_to_explore, \
        abstract_states_to_explore, \
        remaining_abstract_states, deleted_abstract_states, controllable_abstract_states, \
        controllable_abstract_states_temp


def compute_reachable_set_tira(eng, time_step, rect_low, rect_up, u, W_low, W_up):
    rect_low_matlab = matlab.double(rect_low.tolist())
    rect_up_matlab = matlab.double(rect_up.tolist())
    input_low = np.concatenate((u, W_low), axis=0)
    input_up = np.concatenate((u, W_up), axis=0)
    J_x_low, J_x_up, J_p_low, J_p_up = eng.UP_Jacobian_Bounds(0.0, 3.0, rect_low_matlab, rect_up_matlab,
                                                              input_low, input_up, nargout=4)
    time_step_matlab = matlab.double(time_step.tolist())
    succ_low, succ_up = eng.OA_3_CT_Mixed_Monotonicity2_full_reach(time_step_matlab, rect_low_matlab, rect_up_matlab,
                                                                   input_low, input_up,
                                                                   J_x_low, J_x_up, J_p_low, J_p_up,
                                                                   nargout=2)
    succ_low = np.array(succ_low).T
    succ_up = np.array(succ_up).T
    reachtube = []
    for idx in range(succ_low.shape[0]):
        reachtube.append(np.array([succ_low[idx, :], succ_up[idx, :]]))
    return reachtube  # np.concatenate((succ_low, succ_up), axis=0)


def save_abstraction(symbols_to_explore, symbol_step, targets, targets_rects, target_indices, 
                    obstacles, obstacles_rects, obstacle_indices, sym_x, X_low, X_up,
                    reachability_rtree_idx3d, abstract_reachable_sets):

    t_start = time.time()

    symmetry_transformed_targets_and_obstacles, concrete_to_abstract, abstract_to_concrete, \
    symmetry_abstract_states, nearest_target_of_concrete, valid_hit_idx_of_concrete, \
    concrete_edges, neighbor_map, get_concrete_transition_calls = \
        create_symmetry_abstract_states(
            symbols_to_explore,
            symbol_step, targets, targets_rects, target_indices, 
            obstacles, obstacles_rects, obstacle_indices, sym_x, X_low, X_up,
            reachability_rtree_idx3d, abstract_reachable_sets)


    t_abstraction = time.time() - t_start
    print(['Construction of symmetry-based abstraction took: ', t_abstraction, ' seconds'])

    
    '''save_file = {
        "symmetry_transformed_targets_and_obstacles": symmetry_transformed_targets_and_obstacles, 
        "concrete_to_abstract": concrete_to_abstract, 
        "abstract_to_concrete": abstract_to_concrete,
        "symmetry_abstract_states": symmetry_abstract_states, 
        "nearest_target_of_concrete": nearest_target_of_concrete,
        "valid_hit_idx_of_concrete": valid_hit_idx_of_concrete,
        "t_abstraction": t_abstraction
    }'''

    '''np.savez("abstraction_data", 
             symmetry_transformed_targets_and_obstacles=symmetry_transformed_targets_and_obstacles,
             concrete_to_abstract=concrete_to_abstract, 
            abstract_to_concrete=abstract_to_concrete,
            symmetry_abstract_states=symmetry_abstract_states, 
            nearest_target_of_concrete=nearest_target_of_concrete,
            valid_hit_idx_of_concrete=valid_hit_idx_of_concrete,
            t_abstraction=[t_abstraction]
             )'''

    return symmetry_transformed_targets_and_obstacles, concrete_to_abstract, abstract_to_concrete, \
        symmetry_abstract_states, nearest_target_of_concrete, valid_hit_idx_of_concrete, concrete_edges, neighbor_map, t_abstraction, get_concrete_transition_calls

    

def abstract_synthesis(U_discrete, time_step, W_low, W_up,
                       Symbolic_reduced, sym_x, sym_u, state_dimensions,
                       Target_low, Target_up, Obstacle_low, Obstacle_up, X_low, X_up, eng, test_to_run, parallel,
                       abstraction_data=None):


    #handle tests to run
    global strategy_1
    global strategy_2
    global strategy_3
    global strategy_4
    global strategy_5
    global strategy_6
    global benchmark
    global strategy_list
    global create_symmetry_abstract_states

    #set desired test to true
    match(int(test_to_run)):
        case 1:
            strategy_1 = True
        case 2:
            strategy_2 = True
        case 3:
            strategy_3 = True
        case 4:
            strategy_4 = True
        case 5:
            strategy_5 = True
        case 6:
            strategy_6 = True
        
    strategy_list = [strategy_1, strategy_2, strategy_3, strategy_4, strategy_5, strategy_6, benchmark]

    xor_strategy = (sum([ int(strategy) for strategy in strategy_list]) == 1)
    if not xor_strategy:
        raise("Zero or multiple strategies were selected, please only select one")

    #handle parallelism 
    if parallel:
        create_symmetry_abstract_states = create_symmetry_abstract_states_parallel

    t_start = time.time()
    n = state_dimensions.shape[1]
    reachable_target_region = pc.Region(list_poly=[])
    abstraction_level = 1
    # same number of concrete states, but smartly exploring control actions
    # 0 exploring all states along with all actions, as in Meyer et al.
    # 2 merging symmetric states together.
    p = index.Property()
    p.dimension = 3
    p.dat_extension = 'data'
    p.idx_extension = 'index'
    reachability_rtree_idx3d = index.Index('3d_index_abstract',
                                           properties=p)
    
    p2 = index.Property()
    p2.dimension = 2
    p2.dat_extension = 'data'
    p2.idx_extension = 'index'
    reachability_rtree_idx2d = index.Index('2d_index_abstract',
                                           properties=p2)

    symmetry_under_approx_abstract_targets_rtree_idx3d = index.Index(
        '3d_index_under_approx_abstract_targets',
        properties=p)
    extended_target_rtree_idx3d = index.Index(
        '3d_index_extended_target',
        properties=p)

    symbol_step = (X_up - X_low) / sym_x[0, :]

    state_to_paths_idx = {}

    # defining the z3 solver that we'll use to check if a rectangle is in a set of rectangles
    '''cur_solver = Solver()
    var_dict = []
    for dim in range(n):
        var_dict.append(Real("x" + str(dim)))'''

    targets, targets_rects, target_indices, obstacles, obstacles_rects, obstacle_indices = \
        create_targets_and_obstacles(Target_low, Target_up, Obstacle_low, Obstacle_up, symbol_step, sym_x, X_low)

    target_rtree_idx = 0
    for target in targets_rects:
        extended_target_rtree_idx3d.insert(-1, (
            target[0, 0], target[0, 1],
            target[0, 2], target[1, 0],
            target[1, 1], target[1, 2]))
        # rect_center = np.average(target, axis=0)
        # extended_target_rtree_idx3d.insert(-1,
        #                                   (rect_center[0], rect_center[1], rect_center[2],
        #                                    rect_center[0] + 0.001, rect_center[1] + 0.001,
        #                                    rect_center[2] + 0.001),
        #                                   obj=tuple(map(tuple, target)))
    # cur_solver = add_rects_to_solver(np.array(targets_rects), var_dict, cur_solver)

    
    abstract_reachable_sets, reachable_rect_global_cntr, intersection_radius_threshold, per_dim_max_travelled_distance = \
        create_symmetry_abstract_reachable_sets(Symbolic_reduced, n, reachability_rtree_idx2d, reachability_rtree_idx3d)
    
    

    matrix_dim_full = [np.prod(sym_x[0, :]), np.prod(sym_u), 2 * n]
    symbols_to_explore = set(
        range(int(matrix_dim_full[0])))  # np.setdiff1d(np.array(range(int(matrix_dim_full[0]))), target_indices)
    candidate_initial_set_rect = None
    for target in targets_rects:
        bloated_rect = np.array([np.minimum(np.add(target[0, :],
                                                   -1 * per_dim_max_travelled_distance),
                                            X_low),
                                 np.maximum(np.add(target[1, :], per_dim_max_travelled_distance),
                                            X_up)])
        if candidate_initial_set_rect is None:
            candidate_initial_set_rect = bloated_rect
        else:
            candidate_initial_set_rect = get_convex_union([bloated_rect, candidate_initial_set_rect])
    concrete_states_to_explore = rect_to_indices(candidate_initial_set_rect, symbol_step, X_low,
                                                 sym_x[0, :], over_approximate=True)
    concrete_states_to_explore = set(concrete_states_to_explore.flatten())
    nb_concrete = len(concrete_states_to_explore)
    concrete_states_to_explore = concrete_states_to_explore.difference(target_indices)
    concrete_states_to_explore = concrete_states_to_explore.difference(obstacle_indices)
    symbols_to_explore = symbols_to_explore.difference(target_indices)
    symbols_to_explore = symbols_to_explore.difference(obstacle_indices)

    # intersection_radius_threshold = intersection_radius_threshold * 10

    if abstraction_data is None:
        if not benchmark:
            symmetry_transformed_targets_and_obstacles, concrete_to_abstract, abstract_to_concrete, \
            symmetry_abstract_states, nearest_target_of_concrete, valid_hit_idx_of_concrete, \
                concrete_edges, neighbor_map, t_abstraction, get_concrete_transition_calls = \
                save_abstraction(symbols_to_explore, symbol_step, targets, targets_rects, target_indices, 
                                obstacles, obstacles_rects, obstacle_indices, sym_x, X_low, X_up,
                                reachability_rtree_idx3d, abstract_reachable_sets)
            
            nb_abstract_obstacle = len(abstract_to_concrete[0])
        else:
            abstract_to_concrete = []
            concrete_edges = {}
            neighbor_map = {}
            get_concrete_transition_calls = 0
            nb_abstract_obstacle = 0
            t_abstraction = time.time() - t_start
        
    else:
        saved_data = np.load(abstraction_data, allow_pickle=True)

        symmetry_transformed_targets_and_obstacles, concrete_to_abstract, abstract_to_concrete, \
        symmetry_abstract_states, nearest_target_of_concrete, valid_hit_idx_of_concrete, concrete_edges, neighbor_map, t_abstraction = \
            saved_data['symmetry_transformed_targets_and_obstacles'], saved_data['concrete_to_abstract'], saved_data['abstract_to_concrete'], \
            saved_data['symmetry_abstract_states'], saved_data['nearest_target_of_concrete'], saved_data['valid_hit_idx_of_concrete'], \
                saved_data['concrete_edges'], saved_data['neighbor_map'],  saved_data['t_abstraction'][0]
        
        print(['Construction of symmetry-based abstraction took: ', t_abstraction, ' seconds'])


    nb_abstract = len(abstract_to_concrete)

    p = index.Property()
    p.dimension = 3
    p.dat_extension = 'data'
    p.idx_extension = 'index'
    reachability_rtree_idx3d = index.Index('3d_index_abstract',
                                           properties=p)
    
    #===================================================
    #if we are meant to run clustered, wait here
    #===================================================

    #signal completion
    wait_cond[os.getpid()] = 0

    #keep track how long we have been here and spin
    time_spinning = time.time()
    while (wait_cond[-1] == 0):
        time.sleep(100)
    time_spinning = time.time() - time_spinning

    controller = {}
    t_synthesis_start = time.time()
    t_refine = 0
    t_synthesis = 0
    print('\n%s\tStart of the control synthesis\n', time.time() - t_start)
    refinement_itr = 0
    if abstraction_level < 2:
        max_num_refinement_steps = 1
    else:
        max_num_refinement_steps = 10000
    remaining_abstract_states = set(list(range(len(abstract_to_concrete))))
    abstract_states_to_explore = set(range(len(abstract_to_concrete)))
    local_abstract_states_to_explore = set(range(len(abstract_to_concrete)))
    #controllable_abstract_states = set()
    controllable_concrete_states = target_indices.copy()
    refinement_candidates = set()
    abstract_transitions = {}  # [None] * len(abstract_to_concrete)
    inverse_abstract_transitions = {}
    concrete_transitions = {}
    continuous_failure_counter_max = 10
    continuous_failure_counter = 0
    potential_new_target_parents = False
    deleted_abstract_states = []
    W_up = W_up.reshape((1, 3)).T
    W_low = W_low.reshape((1, 3)).T
    time_step = time_step.reshape((1, 3))
    temp_t_synthesis = time.time()

    nb_explore = len(concrete_states_to_explore)

    if benchmark:
        concrete_controller, neighbor_map, unique_state_u_pairs_explored, \
            total_state_u_pairs_explored, total_states_explored, average_path_length, nb_synthesis = symmetry_synthesis_helper(
            concrete_states_to_explore,
            concrete_edges,
            neighbor_map,
            abstract_reachable_sets,
            controllable_concrete_states,
            {}, # concrete_controller
            per_dim_max_travelled_distance,
            obstacles_rects, obstacle_indices,
            targets_rects, X_low, X_up, sym_x, symbol_step)
        
    else:
        obstacle_indices = obstacle_indices.union(set(abstract_to_concrete[0]))

        concrete_controller, refinement_candidates, poll_lengths, average_ratio_neighbor_to_total, \
            neighbor_map, unique_state_u_pairs_explored, total_state_u_pairs_explored, total_states_explored, \
                average_path_length, nb_synthesis = symmetry_abstract_synthesis_helper(
            concrete_states_to_explore,
            concrete_edges,
            neighbor_map,
            abstract_to_concrete,
            concrete_to_abstract,
            symmetry_transformed_targets_and_obstacles,
            nearest_target_of_concrete,
            valid_hit_idx_of_concrete,
            abstract_reachable_sets,
            symmetry_abstract_states,
            refinement_candidates,
            controllable_concrete_states,
            {}, # concrete_controller
            reachability_rtree_idx3d, per_dim_max_travelled_distance, obstacles_rects, obstacle_indices,
            targets_rects, target_indices, X_low, X_up, sym_x, symbol_step)
        
        plot_abstract_states(symmetry_abstract_states, [], abstract_reachable_sets, state_to_paths_idx, abstract_to_concrete)

    t_synthesis += time.time() - temp_t_synthesis

    #save concrete controler to file
    np.save('concrete_controller.npy', concrete_controller)
    np.save('neighbor_map.npy', neighbor_map)

    #stats
    if strategy_1:
        print("Strategy: polls - all")
    elif strategy_2:
        print("Strategy: polls - 400")
    elif strategy_3:
        print("Strategy: polls + no closest")
    elif strategy_4:
        print("Strategy: polls -full + neighbors")
    elif strategy_5:
        print("Strategy: polls -400 + neighbors")
    elif strategy_6:
        print("Strategy: polls + no closest + neighbors") # was it "polls-full"?
    else:
        print("Strategy: baseline")


    print('Total number of concrete states: ', nb_concrete)
    print('Concrete states to explore: ', nb_explore)
    print('Number of abstract states: ', nb_abstract)
    print('States in the abstract obstacle: ', nb_abstract_obstacle)
    print('Number of controllable states: ', len(concrete_controller))
    if benchmark:
        print('No poll stats')
        print('No neighbor/total exploration ratio')
    elif poll_lengths:
        poll_lengths.sort()
        min_len = poll_lengths[0]
        max_len = poll_lengths[-1]
        if nb_abstract % 2 == 0:
            median_len = poll_lengths[nb_abstract // 2 - 1]
        else:
            median_len = (poll_lengths[nb_abstract // 2 - 1] + poll_lengths[nb_abstract // 2]) / 2
        average_len = sum(poll_lengths) / (nb_abstract - 1)
        print('Poll stats (min/average/median/max): ', min_len, '/', average_len, '/', median_len, '/', max_len)
        print('Average neighbor/total exploration ratio', average_ratio_neighbor_to_total)
    else:
        print('Poll stats (min/average/median/max): Zero polls')
        print('Average neighbor/total exploration ratio', average_ratio_neighbor_to_total)
    print('Unique (s,u) explored: ', get_concrete_transition_calls + unique_state_u_pairs_explored)
    print('Total (s,u) explored: ', get_concrete_transition_calls + total_state_u_pairs_explored)
    print('Total s explored: ', total_states_explored)
    print('Average path length: ', average_path_length)
    print('Number of synthesis iterations: ', nb_synthesis)
    print('Abstraction time: ', t_abstraction)
    print('Synthesis time: ', t_synthesis)
    print('Total time: ', (time.time() - t_start) - time_spinning)
    
    #write out to csv file
    csvOut = open("results.csv", "w")
    csvOut.write(str(nb_concrete) + "," + str(nb_explore) + "," + str(nb_abstract) + "," + str(nb_abstract_obstacle) + "," + str(len(concrete_controller)) + "," + str(min_len) + '/' + str(average_len) + '/' + str(median_len) + '/' + str(max_len) + "," + str(average_ratio_neighbor_to_total) + "," + str(get_concrete_transition_calls + unique_state_u_pairs_explored) + "," + str(get_concrete_transition_calls + total_state_u_pairs_explored) + "," + str(average_path_length) + "," + str(nb_synthesis) + "," + str(t_abstraction) + "," + str(t_synthesis) + "," + str((time.time() - t_start) - time_spinning))
    csvOut.close()
    
    #signal full program run
    wait_cond[os.getpid()] = 1

    exit(0)



    while refinement_itr < max_num_refinement_steps:
        temp_t_synthesis = time.time()
        controller, controllable_abstract_states_temp, unsafe_abstract_states, local_abstract_states_to_explore, \
            abstract_states_to_explore, reachable_rect_global_cntr, \
            target_parents, refinement_candidates, target_rtree_idx = symmetry_abstract_synthesis_helper(
            local_abstract_states_to_explore,
            concrete_states_to_explore,  # abstract_states_to_explore,
            abstract_to_concrete,
            concrete_to_abstract,
            symmetry_transformed_targets_and_obstacles,
            nearest_target_of_concrete,
            valid_hit_idx_of_concrete,
            Symbolic_reduced,
            abstract_reachable_sets,
            symmetry_abstract_states, target_parents,
            refinement_candidates,
            controllable_abstract_states,
            reachable_target_region,
            abstract_transitions,
            inverse_abstract_transitions,
            concrete_transitions,
            controller, reachability_rtree_idx3d,
            reachable_rect_global_cntr,
            sym_x, symbol_step,
            obstacles_rects, obstacle_indices, targets_rects, target_indices, extended_target_rtree_idx3d,
            target_rtree_idx, cur_solver, var_dict, state_to_paths_idx, per_dim_max_travelled_distance, X_low, X_up,
            eng, time_step, W_low, W_up, U_discrete)
        '''
            symmetry_abstract_synthesis_helper(local_abstract_states_to_explore,
                                               abstract_states_to_explore,
                                               abstract_to_concrete,
                                               abstract_reachable_sets,
                                               symmetry_abstract_states, target_parents,
                                               refinement_candidates,
                                               controllable_abstract_states,
                                               reachable_target_region,
                                               abstract_transitions,
                                               concrete_transitions,
                                               controller, reachability_rtree_idx3d,
                                               reachable_rect_global_cntr)
        '''
        t_synthesis += time.time() - temp_t_synthesis

        controllable_abstract_states = controllable_abstract_states.union(controllable_abstract_states_temp)

        if abstraction_level >= 2:
            refinement_candidates = refinement_candidates.difference(controllable_abstract_states)
            if controllable_abstract_states_temp:  # or not refinement_candidates:
                # update target parents only if there is a significant number of newly added controllable states.
                continuous_failure_counter = 0
                potential_new_target_parents = True
                # new_controllable_concrete_states = 0
                for abstract_state_idx in controllable_abstract_states_temp:
                    if not abstract_to_concrete[abstract_state_idx]:
                        print("Why this controllable state has been refined?")
                    controllable_concrete_states = controllable_concrete_states.union(
                        abstract_to_concrete[abstract_state_idx])

                # local_abstract_states_to_explore = copy.deepcopy(abstract_states_to_explore)
                # new_controllable_concrete_states += len(abstract_to_concrete[abstract_state_idx])
                '''
                if new_controllable_concrete_states > 10 or not refinement_candidates:
                    for abstract_state_idx in abstract_states_to_explore:
                        if abstract_state_idx not in target_parents:
                            is_target_parent = False
                            rect = symmetry_abstract_states[abstract_state_idx].rtree_target_rect_under_approx
                            # rtree_target_rect_over_approx
                            original_angle_interval = [rect[0, 2], rect[1, 2]]
                            decomposed_angle_intervals = get_decomposed_angle_intervals(original_angle_interval)
                            for interval in decomposed_angle_intervals:
                                is_target_parent = reachability_rtree_idx3d.count(
                                    (rect[0, 0], rect[0, 1], interval[0], rect[1, 0], rect[1, 1], interval[1]))
                                if is_target_parent:
                                    break
                            if is_target_parent:
                                target_parents.add(abstract_state_idx)
                                if not abstract_to_concrete[abstract_state_idx]:
                                    print("Why ", abstract_state_idx,
                                          " is being added to target_parents and it has been refined?")
                '''
            else:
                continuous_failure_counter += 1
            # print("target parents: ", target_parents)
            if not refinement_candidates:
                if not potential_new_target_parents:
                    if continuous_failure_counter >= continuous_failure_counter_max:
                        print("No states to refine anymore.")
                        break
                else:
                    local_abstract_states_to_explore = copy.deepcopy(abstract_states_to_explore)
                    potential_new_target_parents = False
                    continuous_failure_counter = 0
            else:
                temp_t_refine = time.time()
                progress, num_new_abstract_states, concrete_to_abstract, abstract_to_concrete, \
                    refinement_candidates, target_parents, local_abstract_states_to_explore, \
                    abstract_states_to_explore, \
                    remaining_abstract_states, deleted_abstract_states, controllable_abstract_states, \
                    controllable_abstract_states_temp = \
                    refine(concrete_to_abstract, abstract_to_concrete, symmetry_abstract_states,
                           remaining_abstract_states, refinement_candidates, target_parents,
                           inverse_abstract_transitions,
                           local_abstract_states_to_explore,
                           abstract_states_to_explore,
                           controllable_abstract_states, symmetry_transformed_targets_and_obstacles,
                           abstract_reachable_sets, concrete_transitions, sym_x, symbol_step,
                           obstacles_rects, obstacle_indices, targets_rects, target_indices, X_low, X_up)

                if controllable_abstract_states_temp:  # or not refinement_candidates:
                    # update target parents only if there is a significant number of newly added controllable states.
                    print(len(controllable_abstract_states_temp),
                          " new controllable states have been found in this refinement iteration\n")
                    continuous_failure_counter = 0
                    potential_new_target_parents = True
                    # new_controllable_concrete_states = 0

                    # new_controllable_concrete_states += len(abstract_to_concrete[abstract_state_idx])

                    # local_abstract_states_to_explore = copy.deepcopy(abstract_states_to_explore)

                t_refine += time.time() - temp_t_refine
                '''
                if progress:
                    abstract_states_to_explore = np.setdiff1d(np.array(abstract_states_to_explore),
                                                              deleted_abstract_states).tolist()
                    target_parents = target_parents.difference(set(deleted_abstract_states))
                    for i in range(0, 2 * num_new_abstract_states, 2):
                        if not abstract_to_concrete[len(abstract_to_concrete) - 1 - i] or not abstract_to_concrete[len(abstract_to_concrete) - 1 - (i + 1)]:
                            print("How?")
                        abstract_states_to_explore.append(len(abstract_to_concrete) - 1 - i)
                        abstract_states_to_explore.append(len(abstract_to_concrete) - 1 - (i + 1))
                        target_parents.add(len(abstract_to_concrete) - 1 - i)
                        target_parents.add(len(abstract_to_concrete) - 1 - (i + 1))
                '''

                # target_parents = np.setdiff1d(np.array(target_parents), controllable_abstract_states).tolist()
                # target_parents = np.setdiff1d(np.array(target_parents), deleted_abstract_states).tolist()
                refinement_itr += num_new_abstract_states
                print("progress: ", progress)
                if not progress and potential_new_target_parents:  # not controllable_abstract_states_temp:
                    local_abstract_states_to_explore = copy.deepcopy(abstract_states_to_explore)
                    potential_new_target_parents = False
            # print("adjacency_list after refinement: ", adjacency_list)
            # print("concrete target parents: ", concrete_target_parents)
        else:
            break

    for abstract_state_idx in controllable_abstract_states:
        if not abstract_to_concrete[abstract_state_idx]:
            print("Why this controllable state has been refined?")
        controllable_concrete_states = controllable_concrete_states.union(
            abstract_to_concrete[abstract_state_idx])

    print(['Construction of symmetry-based abstraction took: ', t_abstraction, ' seconds'])
    print(
        ['Controller synthesis along with refinement for reach-avoid specification: ', time.time() - t_synthesis_start,
         ' seconds'])
    print(['Total time for symmetry abstraction-refinement-based controller synthesis'
           ' for reach-avoid specification: ', time.time() - t_synthesis_start + t_abstraction, ' seconds'])
    print(['Pure refinement took a total of: ', t_refine, ' seconds'])
    print(['Pure synthesis took a total of: ', t_synthesis, ' seconds'])
    print(['Number of splits of abstract states: ', refinement_itr])
    print(['Number of abstract states before refinement is: ', num_abstract_states_before_refinement]) #nb_abstract
    num_abstract_states = 0
    for abstract_s_idx in range(len(abstract_to_concrete)):
        if abstract_to_concrete[abstract_s_idx]:
            num_abstract_states += 1
    print(['Number of abstract states after refinement is: ', num_abstract_states])
    print(['Number of concrete states is: ', len(concrete_to_abstract)])
    if len(controllable_abstract_states):
        print(len(controllable_abstract_states), 'abstract symbols are controllable to satisfy the reach-avoid '
                                                 'specification\n')
        for abstract_s in controllable_abstract_states:
            if not abstract_to_concrete[abstract_s]:
                print(abstract_s, " does not represent any concrete state, why is it controllable?")
            print("Controllable abstract state ", abstract_s, " represents the following concrete states: ",
                  abstract_to_concrete[abstract_s])
        print(len(controllable_concrete_states), 'concrete symbols are controllable to satisfy the reach-avoid '
                                                 'specification\n')
    else:
        print('The reach-avoid specification cannot be satisfied from any initial state\n')
    plot_abstract_states(symmetry_abstract_states, deleted_abstract_states, abstract_reachable_sets,
                         state_to_paths_idx, abstract_to_concrete)
    plot_concrete_states(controllable_concrete_states, targets_rects, obstacles_rects,
                         state_to_paths_idx, sym_x, symbol_step, X_low, X_up)
