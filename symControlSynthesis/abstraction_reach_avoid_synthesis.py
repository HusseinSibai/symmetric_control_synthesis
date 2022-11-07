import pdb

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
import os
import itertools
from scipy.spatial import ConvexHull

import matplotlib

matplotlib.use("macOSX")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, Polygon


class AbstractState:

    def __init__(self, abstract_targets, abstract_targets_without_angles,
                 abstract_obstacles, concrete_state_idx, rtree_target_rect_over_approx, empty_abstract_target):
        self.abstract_targets = abstract_targets
        self.abstract_targets_without_angles = abstract_targets_without_angles
        self.abstract_obstacles = abstract_obstacles
        self.concrete_state_idx = concrete_state_idx
        rc, x1 = pc.cheby_ball(abstract_targets[0])
        self.rtree_target_rect_under_approx = np.array([x1 - rc, x1 + rc])
        self.rtree_target_rect_over_approx = rtree_target_rect_over_approx
        # abstract_targets_rects_over_approx[0]
        # np.column_stack(pc.bounding_box(abstract_targets[0])).T
        self.set_of_allowed_controls = None
        self.abstract_targets_over_approximation = copy.deepcopy(abstract_targets)
        self.empty_abstract_target = empty_abstract_target


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


'''
def fix_rect_angles(rect: np.array):
    rect[0, 2] = fix_angle(rect[0, 2])
    rect[1, 2] = fix_angle(rect[1, 2])
    return rect
'''


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
                    b_inter_left = -1 * inter_interval[0]
                    b_inter_right = inter_interval[1]
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
    if True:  # project_to_pos:
        return pc.union(poly_1, poly_2, check_convex=check_convex)
    result = pc.Region(list_poly=[])
    if pc.is_empty(poly_1):
        return poly_2
    if pc.is_empty(poly_2):
        return poly_1
    # reg_pos_1 = project_region_to_position_coordinates(poly_1)
    # reg_pos_2 = project_region_to_position_coordinates(poly_2)
    # reg_pos_union = pc.union(reg_pos_1, reg_pos_2)
    # for poly in reg_pos_union:
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
            p_union_pos = pc.union(p1_pos, p2_pos)
            # print("p_inter_pos: ", p_inter_pos)
            # if not pc.is_empty(p_union_pos):
            for poly_union_pos in p_union_pos.list_poly:
                union_interval = get_intervals_union(-1 * b1[i_low_1], b1[i_up_1], -1 * b2[i_low_2], b2[i_up_2],
                                                     order_matters)
                if union_interval is not None:
                    b_inter_left = -1 * union_interval[0]
                    b_inter_right = union_interval[1]
                    A_new = np.zeros((poly_union_pos.A.shape[0] + 2, A1.shape[1]))
                    b_new = np.zeros((poly_union_pos.A.shape[0] + 2,))
                    for i in range(poly_union_pos.A.shape[0]):
                        for j in range(poly_union_pos.A.shape[1]):
                            A_new[i, j] = poly_union_pos.A[i, j]
                        b_new[i] = poly_union_pos.b[i]
                    A_new[poly_union_pos.A.shape[0], 2] = 1
                    b_new[poly_union_pos.A.shape[0]] = b_inter_right
                    A_new[poly_union_pos.A.shape[0] + 1, 2] = -1
                    b_new[poly_union_pos.A.shape[0] + 1] = b_inter_left
                    result = pc.union(result, pc.Polytope(A_new, b_new), check_convex=check_convex)
    return result


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
    '''
    a -= angle
    b -= angle
    a = fix_angle(a)
    b = fix_angle(b)
    if a * b >= 0:
        return False
    return abs(a - b) < math.pi
    '''


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
        return 0, 2 * math.pi
    if does_interval_contain(a_s, b_s, a_l, b_l):
        return a_s, b_s
    if does_interval_contain(a_l, b_l, a_s, b_s):
        return a_l, b_l
    if is_within_range(a_s, a_l, b_l):
        while a_s > b_l:
            b_l += 2 * math.pi
        result = [a_s, b_l]
    elif is_within_range(b_s, a_l, b_l):
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
    if b_s - a_s >= 2 * math.pi - 0.01 or b_l - a_l >= 2 * math.pi - 0.01:  # to not lose over-approximation of reachability analysis
        return 0, 2 * math.pi
    if does_interval_contain(a_s, b_s, a_l, b_l):
        return a_l, b_l
    if does_interval_contain(a_l, b_l, a_s, b_s):
        return a_s, b_s
    if is_within_range(a_s, a_l, b_l):
        while a_l > b_s:
            b_s += 2 * math.pi
        result = [a_l, b_s]
    elif is_within_range(b_s, a_l, b_l):
        while a_s > b_l:
            b_l += 2 * math.pi
        result = [a_s, b_l]
    else:
        while a_s > b_l:
            b_l += 2 * math.pi
        a_s, b_l = fix_angle_interval(a_s, b_l)
        if order_matters or not b_l - a_s > 2 * math.pi:
            result = [a_s, b_l]
        else:
            while a_l > b_s:
                b_s += 2 * math.pi
            result = [a_l, b_s]
    # if disjoint, choose a direction to join them
    result[0], result[1] = fix_angle_interval(result[0], result[1])
    return result


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
    box_1 = transform_to_frame(np.array([low_red, up_red]), source_full_low, overapproximate=True)
    box_2 = transform_to_frame(np.array([low_red, up_red]), source_full_up, overapproximate=True)
    box_3 = transform_to_frame(np.array([low_red, up_red]), np.array([source_full_low[0], source_full_low[1],
                                                                      source_full_up[2]]), overapproximate=True)
    box_4 = transform_to_frame(np.array([low_red, up_red]), np.array([source_full_up[0], source_full_up[1],
                                                                      source_full_low[2]]), overapproximate=True)
    result = get_convex_union([box_1, box_2, box_3, box_4])
    return result  # np.array([result_low, result_up]);


def transform_rect_to_abstract_frames(concrete_rect, frames_rect, over_approximate=False, project_to_pos=False):
    box_1 = transform_rect_to_abstract(concrete_rect, frames_rect[0, :], overapproximate=over_approximate)
    box_2 = transform_rect_to_abstract(concrete_rect, frames_rect[1, :], overapproximate=over_approximate)
    box_3 = transform_rect_to_abstract(concrete_rect, np.array([frames_rect[0, 0], frames_rect[0, 1],
                                                                frames_rect[1, 2]]), overapproximate=over_approximate)
    box_4 = transform_rect_to_abstract(concrete_rect, np.array([frames_rect[1, 0], frames_rect[1, 1],
                                                                frames_rect[0, 2]]), overapproximate=over_approximate)
    if over_approximate:
        result = get_convex_union([box_1, box_4, box_3, box_2],
                                  order_matters=True)  # Hussein: order in the list matters!!!!
        # it determines the direction of the union of the angles
    else:
        result = get_intersection([box_1, box_2, box_3, box_4])
    if project_to_pos:
        result = result[:, :2]
    return result  # np.array([result_low, result_up]);


def transform_poly_to_abstract_frames(concrete_poly, frames_rect, over_approximate=False, project_to_pos=False,
                                      check_convex=False):
    if project_to_pos:
        concrete_poly_new = project_region_to_position_coordinates(copy.deepcopy(concrete_poly))
    else:
        concrete_poly_new = copy.deepcopy(concrete_poly)
    poly_1 = transform_poly_to_abstract(concrete_poly_new, frames_rect[0, :], project_to_pos)
    poly_2 = transform_poly_to_abstract(concrete_poly_new, frames_rect[1, :], project_to_pos)
    poly_3 = transform_poly_to_abstract(concrete_poly_new, np.array([frames_rect[0, 0], frames_rect[0, 1],
                                                                     frames_rect[1, 2]]), project_to_pos)
    poly_4 = transform_poly_to_abstract(concrete_poly_new, np.array([frames_rect[1, 0], frames_rect[1, 1],
                                                                     frames_rect[0, 2]]), project_to_pos)

    if over_approximate:
        result = get_poly_union(poly_1, poly_4, project_to_pos, order_matters=True)  # pc.union(poly_1, poly_2)
        result = get_poly_union(result, poly_2, project_to_pos, order_matters=True)  # pc.union(result, poly_3)
        result = get_poly_union(result, poly_3, project_to_pos, order_matters=True)  # pc.union(result, poly_4)
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
        union_interval = get_intervals_union(result[0, 2], result[1, 2], list_array[i][0, 2], list_array[i][1, 2],
                                             order_matters=order_matters)
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


def create_symmetry_abstract_states(symbols_to_explore, symbol_step, targets, obstacles, sym_x, X_low, X_up,
                                    intersection_radius_threshold, symmetry_under_approx_abstract_targets_rtree_idx3d,
                                    symmetry_over_approx_abstract_targets_rtree_idx3d):
    t_start = time.time()
    print('\n%s\tStart of the symmetry abstraction \n', time.time() - t_start)
    symmetry_transformed_targets_and_obstacles = {}  # [None] * int(matrix_dim_full[0])
    concrete_to_abstract = {}  # [None] * int(matrix_dim_full[0])
    abstract_to_concrete = []
    symmetry_abstract_states = []
    abstract_states_to_rtree_ids = {}
    rtree_ids_to_abstract_states = {}
    next_rtree_id_candidate = 0
    for s in symbols_to_explore:
        s_subscript = np.array(np.unravel_index(s, tuple((sym_x[0, :]).astype(int))))
        s_rect: np.array = np.row_stack((s_subscript * symbol_step + X_low,
                                         s_subscript * symbol_step + symbol_step + X_low))
        s_rect[0, :] = np.maximum(X_low, s_rect[0, :])
        s_rect[1, :] = np.minimum(X_up, s_rect[1, :])
        # print("s_rect: ", s_rect)

        # transforming the targets and obstacles to a new coordinate system relative to the states in s.

        abstract_targets_polys = []
        abstract_targets_rects = []
        abstract_targets_polys_over_approx = []
        abstract_targets_rects_over_approx = []
        abstract_pos_targets_polys = []
        empty_abstract_target = False
        for target_idx, target_poly in enumerate(targets):
            abstract_target_poly = transform_poly_to_abstract_frames(target_poly, s_rect, over_approximate=False)
            abstract_pos_target_poly = transform_poly_to_abstract_frames(target_poly, s_rect, over_approximate=False,
                                                                         project_to_pos=True)
            abstract_target_poly_over_approx = transform_poly_to_abstract_frames(
                target_poly, s_rect, over_approximate=True)  # project_to_pos=True
            if not pc.is_empty(abstract_target_poly):
                rc, x1 = pc.cheby_ball(abstract_target_poly)
                abstract_target_rect = np.array([x1 - rc, x1 + rc])
            elif not pc.is_empty(abstract_pos_target_poly):
                pdb.set_trace()
                raise "abstract target is empty for a concrete state"
                empty_abstract_target = True
                rc_pos, x1_pos = pc.cheby_ball(abstract_pos_target_poly)
                abstract_target_rect_pos = np.array([x1_pos - rc_pos, x1_pos + rc_pos])
                abstract_target_rect = np.array([[abstract_target_rect_pos[0, 0], abstract_target_rect_pos[0, 1], 0],
                                                 [abstract_target_rect_pos[1, 0], abstract_target_rect_pos[1, 1],
                                                  2 * math.pi]])
            else:
                pdb.set_trace()
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
            pdb.set_trace()
            raise "Abstract target is empty"

        abstract_obstacles = pc.Region(list_poly=[])
        for obstacle_poly in obstacles:
            abstract_obstacle = transform_poly_to_abstract_frames(obstacle_poly, s_rect,
                                                                  over_approximate=True)  # project_to_pos=True
            abstract_obstacles = pc.union(abstract_obstacles, abstract_obstacle)  # get_poly_union
            # abstract_obstacles.append(abstract_obstacle)

        symmetry_transformed_targets_and_obstacles[s] = AbstractState(abstract_targets_polys,
                                                                      abstract_pos_targets_polys,
                                                                      abstract_obstacles, s,
                                                                      abstract_targets_rects_over_approx[0],
                                                                      empty_abstract_target)

        # Now adding the abstract state to a cluster --> combining abstract states with overlapping (abstract) targets
        added_to_existing_state = False
        for curr_target_idx, curr_target_rect in enumerate(abstract_targets_rects):
            hits = list(symmetry_under_approx_abstract_targets_rtree_idx3d.intersection(
                (curr_target_rect[0, 0], curr_target_rect[0, 1], curr_target_rect[0, 2],
                 curr_target_rect[1, 0], curr_target_rect[1, 1], curr_target_rect[1, 2]),
                objects=True))
            if len(hits):
                max_rad = 0
                max_rad_idx = None
                has_empty_angle_interval = pc.is_empty(abstract_targets_polys[0])
                max_intersection_rect = None
                # max_union_rect = None
                for idx, hit in enumerate(hits):
                    if hit.id in abstract_states_to_rtree_ids.values():
                        abstract_state = hit.object
                        for target_idx, abstract_target_poly in enumerate(abstract_state.abstract_targets):
                            if True:  # not abstract_state.empty_abstract_target and not empty_abstract_target:
                                intersection_poly = get_poly_intersection(
                                    copy.deepcopy(abstract_targets_polys[curr_target_idx]),
                                    copy.deepcopy(abstract_target_poly), check_convex=False)  # pc.intersect
                            else:
                                intersection_poly = get_poly_intersection(
                                    copy.deepcopy(abstract_pos_targets_polys[curr_target_idx]),
                                    copy.deepcopy(
                                        abstract_state.abstract_targets_without_angles[target_idx]))  # pc.intersect
                            if not pc.is_empty(intersection_poly):
                                rc, x1 = pc.cheby_ball(intersection_poly)
                                if np.linalg.norm(rc) > np.linalg.norm(max_rad):
                                    max_rad = rc
                                    max_rad_idx = idx
                                    max_intersection_rect = np.array([x1 - rc, x1 + rc])  # intersection_rect
                                    # max_union_rect = np.column_stack(
                                    #    pc.bounding_box(abstract_targets_polys[curr_target_idx])).T
                                    # max_union_rect = get_convex_union([max_union_rect,
                                    #                                   abstract_state.rtree_target_rect_over_approx])

                # Now we want to make sure that the intersection is large enough to be useful in synthesis later
                # if np.all(max_intersection_rect[1, :] - max_intersection_rect[0, :] >= 2 * symbol_step):
                if max_rad >= intersection_radius_threshold:  # 2 * symbol_step:
                    abstract_state = hits[max_rad_idx].object
                    new_abstract_state = add_concrete_state_to_symmetry_abstract_state(s, abstract_state,
                                                                                       symmetry_transformed_targets_and_obstacles)
                    inter_poly_3 = new_abstract_state.abstract_targets[0]
                    ##########################
                    rtree_target_rect_under_approx = abstract_state.rtree_target_rect_under_approx
                    original_angle_interval = [rtree_target_rect_under_approx[0, 2],
                                               rtree_target_rect_under_approx[1, 2]]
                    decomposed_angle_intervals = get_decomposed_angle_intervals(original_angle_interval)
                    for interval in decomposed_angle_intervals:
                        rtree_target_rect_under_approx_temp = copy.deepcopy(rtree_target_rect_under_approx)
                        rtree_target_rect_under_approx_temp[0, 2] = interval[0]
                        rtree_target_rect_under_approx_temp[1, 2] = interval[1]
                        symmetry_under_approx_abstract_targets_rtree_idx3d.delete(hits[max_rad_idx].id,
                                                                                  (rtree_target_rect_under_approx_temp[
                                                                                       0, 0],
                                                                                   rtree_target_rect_under_approx_temp[
                                                                                       0, 1],
                                                                                   rtree_target_rect_under_approx_temp[
                                                                                       0, 2],
                                                                                   rtree_target_rect_under_approx_temp[
                                                                                       1, 0],
                                                                                   rtree_target_rect_under_approx_temp[
                                                                                       1, 1],
                                                                                   rtree_target_rect_under_approx_temp[
                                                                                       1, 2]))
                    original_angle_interval_new = [new_abstract_state.rtree_target_rect_under_approx[0, 2],
                                                   new_abstract_state.rtree_target_rect_under_approx[1, 2]]
                    decomposed_angle_intervals_new = get_decomposed_angle_intervals(original_angle_interval_new)
                    for interval in decomposed_angle_intervals_new:
                        rtree_target_rect_under_approx_temp = copy.deepcopy(
                            new_abstract_state.rtree_target_rect_under_approx)
                        rtree_target_rect_under_approx_temp[0, 2] = interval[0]
                        rtree_target_rect_under_approx_temp[1, 2] = interval[1]
                        symmetry_under_approx_abstract_targets_rtree_idx3d.insert(next_rtree_id_candidate, (
                            rtree_target_rect_under_approx_temp[0, 0], rtree_target_rect_under_approx_temp[0, 1],
                            rtree_target_rect_under_approx_temp[0, 2], rtree_target_rect_under_approx_temp[1, 0],
                            rtree_target_rect_under_approx_temp[1, 1], rtree_target_rect_under_approx_temp[1, 2]),
                                                                                  obj=new_abstract_state)
                    ##########################
                    '''
                    symmetry_under_approx_abstract_targets_rtree_idx3d.delete(hits[max_rad_idx].id,
                                                                              hits[max_rad_idx].bbox)
                    symmetry_under_approx_abstract_targets_rtree_idx3d.insert(hits[max_rad_idx].id, (
                        max_intersection_rect[0, 0], max_intersection_rect[0, 1], max_intersection_rect[0, 2],
                        max_intersection_rect[1, 0], max_intersection_rect[1, 1], max_intersection_rect[1, 2]),
                                                                              obj=new_abstract_state)
                    '''
                    # symmetry_abstract_states[hits[max_rad_idx].id] = new_abstract_state
                    symmetry_abstract_states[rtree_ids_to_abstract_states[hits[max_rad_idx].id]] = new_abstract_state
                    concrete_to_abstract[s] = rtree_ids_to_abstract_states[hits[max_rad_idx].id]  # hits[max_rad_idx].id
                    abstract_states_to_rtree_ids[rtree_ids_to_abstract_states[hits[max_rad_idx].id]] = \
                        next_rtree_id_candidate
                    abstract_to_concrete[rtree_ids_to_abstract_states[hits[max_rad_idx].id]].append(
                        s)  # hits[max_rad_idx].id
                    rtree_ids_to_abstract_states[next_rtree_id_candidate] = \
                        copy.deepcopy(rtree_ids_to_abstract_states[hits[max_rad_idx].id])
                    del rtree_ids_to_abstract_states[hits[max_rad_idx].id]
                    next_rtree_id_candidate += 1
                    added_to_existing_state = True
                    break
        if not added_to_existing_state:  # concrete_to_abstract[s] is None:
            # create a new abstract state since there isn't a current one suitable for s.
            new_abstract_state = AbstractState(abstract_targets_polys, abstract_pos_targets_polys,
                                               abstract_obstacles, [s],
                                               abstract_targets_rects_over_approx[0], empty_abstract_target)
            # for target_idx in range(len(new_abstract_state.abstract_targets)):
            #    rect = abstract_targets_rects[target_idx]
            original_angle_interval = [new_abstract_state.rtree_target_rect_under_approx[0, 2],
                                       new_abstract_state.rtree_target_rect_under_approx[1, 2]]
            decomposed_angle_intervals = get_decomposed_angle_intervals(original_angle_interval)
            for interval in decomposed_angle_intervals:
                rtree_target_rect_under_approx_temp = copy.deepcopy(
                    new_abstract_state.rtree_target_rect_under_approx)
                rtree_target_rect_under_approx_temp[0, 2] = interval[0]
                rtree_target_rect_under_approx_temp[1, 2] = interval[1]
                # id was len(symmetry_abstract_states) instead of s in rtree, but to solve the non-determinism
                # of deletion problem, changed it to the concrete value. Still attempting to delete
                # though to avoid the increasing size of the rtree.
                symmetry_under_approx_abstract_targets_rtree_idx3d.insert(next_rtree_id_candidate, (
                    rtree_target_rect_under_approx_temp[0, 0], rtree_target_rect_under_approx_temp[0, 1],
                    rtree_target_rect_under_approx_temp[0, 2], rtree_target_rect_under_approx_temp[1, 0],
                    rtree_target_rect_under_approx_temp[1, 1], rtree_target_rect_under_approx_temp[1, 2]),
                                                                          obj=new_abstract_state)
            # symmetry_under_approx_abstract_targets_rtree_idx3d.insert(len(symmetry_abstract_states), (
            #    rect[0, 0], rect[0, 1], rect[0, 2], rect[1, 0], rect[1, 1], rect[1, 2]), obj=new_abstract_state)
            concrete_to_abstract[s] = len(symmetry_abstract_states)
            abstract_states_to_rtree_ids[len(symmetry_abstract_states)] = next_rtree_id_candidate
            rtree_ids_to_abstract_states[next_rtree_id_candidate] = len(symmetry_abstract_states)
            abstract_to_concrete.append([s])
            symmetry_abstract_states.append(new_abstract_state)
            next_rtree_id_candidate += 1

    # over_rect_to_be_deleted = hits[max_rad_idx].object.rtree_target_rect_over_approx
    # over_rect_in_tuple_format = (
    #    over_rect_to_be_deleted[0, 0], over_rect_to_be_deleted[0, 1], over_rect_to_be_deleted[0, 2],
    #    over_rect_to_be_deleted[1, 0], over_rect_to_be_deleted[1, 1], over_rect_to_be_deleted[1, 2])
    # symmetry_over_approx_abstract_targets_rtree_idx3d.delete(hits[max_rad_idx].id,
    #                                                         over_rect_in_tuple_format)
    '''
    for idx, abstract_state in enumerate(symmetry_abstract_states):
        max_union_rect = abstract_state.rtree_target_rect_over_approx
        symmetry_over_approx_abstract_targets_rtree_idx3d.insert(idx, (
            max_union_rect[0, 0], max_union_rect[0, 1], max_union_rect[0, 2],
            max_union_rect[1, 0], max_union_rect[1, 1], max_union_rect[1, 2]), obj=abstract_state)
    '''
    print(['Done creation of symmetry abstract states in: ', time.time() - t_start, ' seconds'])
    print("concrete_to_abstract: ", len(concrete_to_abstract))
    print("abstract_to_concrete: ", len(abstract_to_concrete))
    return symmetry_transformed_targets_and_obstacles, concrete_to_abstract, abstract_to_concrete, \
           symmetry_abstract_states, abstract_states_to_rtree_ids, rtree_ids_to_abstract_states, next_rtree_id_candidate


def add_concrete_state_to_symmetry_abstract_state(curr_concrete_state_idx, abstract_state,
                                                  symmetry_transformed_targets_and_obstacles):
    concrete_state = symmetry_transformed_targets_and_obstacles[curr_concrete_state_idx]
    if abstract_state is None:
        return AbstractState(copy.deepcopy(concrete_state.abstract_targets),
                             copy.deepcopy(concrete_state.abstract_targets_without_angles),
                             copy.deepcopy(concrete_state.abstract_obstacles),
                             [curr_concrete_state_idx], concrete_state.rtree_target_rect_over_approx,
                             concrete_state.empty_abstract_target)
    concrete_state = symmetry_transformed_targets_and_obstacles[curr_concrete_state_idx]
    for target_idx in range(len(abstract_state.abstract_targets)):
        if not (abstract_state.empty_abstract_target or concrete_state.empty_abstract_target):
            inter_poly_1 = copy.deepcopy(concrete_state.abstract_targets[target_idx])
            inter_poly_2 = copy.deepcopy(abstract_state.abstract_targets[target_idx])
        else:
            pdb.set_trace()
            raise "empty intersection between the relative target of the abstract state" \
                  " and that of the concrete state it is being added to!"
            # inter_poly_1 = copy.deepcopy(concrete_state.abstract_targets_without_angles[target_idx])
            # inter_poly_2 = copy.deepcopy(abstract_state.abstract_targets_without_angles[target_idx])
        intersection_poly = get_poly_intersection(inter_poly_1, inter_poly_2, check_convex=False)  # pc.intersect
        while pc.is_empty(intersection_poly):
            pdb.set_trace()
            if True:  # abstract_state.empty_abstract_target or concrete_state.empty_abstract_target:
                raise "empty abstract_target_poly error, grid must be refined, it's too far to see the position of " \
                      "the target similarly even within the same grid cell! "
            else:
                abstract_state.empty_abstract_target = True
                inter_poly_1 = copy.deepcopy(concrete_state.abstract_targets_without_angles[target_idx])
                inter_poly_2 = copy.deepcopy(abstract_state.abstract_targets_without_angles[target_idx])
                intersection_poly = get_poly_intersection(inter_poly_1, inter_poly_2, project_to_pos=True)
        # if not (abstract_state.empty_abstract_target or concrete_state.empty_abstract_target):
        u_poly_1 = copy.deepcopy(concrete_state.abstract_targets_over_approximation[target_idx])
        u_poly_2 = copy.deepcopy(abstract_state.abstract_targets_over_approximation[target_idx])
        union_poly = get_poly_union(u_poly_1, u_poly_2, check_convex=False)  # pc.union
        abstract_state.abstract_targets[target_idx] = copy.deepcopy(intersection_poly)
        abstract_state.abstract_targets_over_approximation[target_idx] = copy.deepcopy(union_poly)

    rc, x1 = pc.cheby_ball(copy.deepcopy(abstract_state.abstract_targets[0]))
    rtree_target_rect_under_approx = np.array([x1 - rc, x1 + rc])
    abstract_state.rtree_target_rect_under_approx = \
        fix_angle_interval_in_rect(rtree_target_rect_under_approx)
    abstract_state.rtree_target_rect_over_approx = get_convex_union([concrete_state.rtree_target_rect_over_approx,
                                                                     abstract_state.rtree_target_rect_over_approx])
    # np.column_stack(
    # pc.bounding_box(abstract_state.abstract_targets_over_approximation[0])).T
    union_poly_obstacles = get_poly_union(concrete_state.abstract_obstacles,
                                          abstract_state.abstract_obstacles, check_convex=False)  # pc.union
    abstract_state.abstract_obstacles = union_poly_obstacles
    abstract_state.concrete_state_idx.append(curr_concrete_state_idx)
    return abstract_state


def create_symmetry_abstract_transitions(Symbolic_reduced, abstract_paths, abstract_to_concrete, concrete_to_abstract,
                                         symmetry_abstract_states, symmetry_under_approx_abstract_targets_rtree_idx3d,
                                         abstract_states_to_rtree_ids, rtree_ids_to_abstract_states,
                                         symbol_step, targets_rects, target_indices,
                                         obstacles_rects, obstacle_indices,
                                         sym_x, X_low, X_up):
    adjacency_list = []
    for abstract_s in range(len(abstract_to_concrete)):
        adjacency_list.append([])
        for u_ind in range(Symbolic_reduced.shape[1]):
            adjacency_list[abstract_s].append([])
    inverse_adjacency_list = []
    for abstract_s in range(len(abstract_to_concrete)):
        inverse_adjacency_list.append([])
        for u_ind in range(Symbolic_reduced.shape[1]):
            inverse_adjacency_list[abstract_s].append([])
    abstract_to_concrete_edges = None  # [[[]] * Symbolic_reduced.shape[1]] * len(abstract_to_concrete)
    target_parents = []
    concrete_target_parents = []
    concrete_edges = {}
    inverse_concrete_edges = {}
    for concrete_state_ind in concrete_to_abstract:
        concrete_edges[concrete_state_ind] = []
        inverse_concrete_edges[concrete_state_ind] = []
        for u_ind in range(Symbolic_reduced.shape[1]):
            concrete_edges[concrete_state_ind].append([])
            inverse_concrete_edges[concrete_state_ind].append([])
    for abstract_state_ind in range(len(abstract_to_concrete)):
        for u_ind in range(Symbolic_reduced.shape[1]):
            '''
            neighbors = get_abstract_transition(concrete_to_abstract,
                                                abstract_to_concrete,
                                                abstract_to_concrete_edges,
                                                concrete_edges,
                                                inverse_concrete_edges,
                                                concrete_target_parents,
                                                abstract_state_ind,
                                                u_ind,
                                                sym_x, symbol_step, X_low,
                                                X_up, abstract_paths,
                                                obstacles_rects,
                                                obstacle_indices,
                                                targets_rects, target_indices)
            '''
            neighbors = list(get_abstract_transition_without_concrete(abstract_state_ind, u_ind,
                                                                      symmetry_abstract_states,
                                                                      [],
                                                                      symmetry_under_approx_abstract_targets_rtree_idx3d,
                                                                      abstract_states_to_rtree_ids,
                                                                      rtree_ids_to_abstract_states,
                                                                      abstract_paths))
            adjacency_list[abstract_state_ind][u_ind] = copy.deepcopy(neighbors)
            for next_abstract_state_ind in adjacency_list[abstract_state_ind][u_ind]:
                if next_abstract_state_ind >= 0 and \
                        abstract_state_ind not in inverse_adjacency_list[next_abstract_state_ind][u_ind]:
                    inverse_adjacency_list[next_abstract_state_ind][u_ind].append(abstract_state_ind)
                elif next_abstract_state_ind == -1 and abstract_state_ind not in target_parents:
                    target_parents.append(abstract_state_ind)
        '''
        for concrete_state_ind in abstract_to_concrete[abstract_state_ind]:
            for u_ind in range(Symbolic_reduced.shape[1]):
                if not concrete_edges[concrete_state_ind][u_ind]:
                    raise "Why is this empty?"
        '''
        '''
        for s_ind in symbols_to_explore:
            for u_ind in range(Symbolic_reduced.shape[1]):
                if -2 in adjacency_list[concrete_to_abstract[s_ind]][u_ind]:
                    continue
                concrete_neighbors = get_concrete_transition(s_ind, u_ind, sym_x, symbol_step, X_low, X_up,
                                                             abstract_paths,
                                                             obstacles_rects, obstacle_indices, targets_rects,
                                                             target_indices)
                if -2 in concrete_neighbors:
                    adjacency_list[concrete_to_abstract[s_ind]][u_ind] = [-2]
                    continue
                for concrete_neighbor in concrete_neighbors:
                    if concrete_to_abstract[concrete_neighbor] not in adjacency_list[concrete_to_abstract[s_ind]][
                        u_ind]:
                        adjacency_list[concrete_to_abstract[s_ind]][u_ind].append(
                            concrete_to_abstract[concrete_neighbor])
        '''

        '''
        s_subscript = np.array(np.unravel_index(s, tuple((sym_x[0, :]).astype(int))))
        # here you should transform s to the abstract coordinates, but in our case, it is just the origin.
        s_rect: np.array = np.row_stack((s_subscript * symbol_step + X_low,
                                         s_subscript * symbol_step + symbol_step + X_low))
        s_rect[0, :] = np.maximum(X_low, s_rect[0, :])
        s_rect[1, :] = np.minimum(X_up, s_rect[1, :])
        # TODO: check if all transitions of the abstract state representing this state are unsafe,
        for u_ind in range(Symbolic_reduced.shape[1]):
            # TODO: check if the abstract transition representing this transition has already been marked unsafe
            if np.any(np.array(adjacency_list[concrete_to_abstract[s]][u_ind]) == -2):
                continue
            succ_intersects_obstacle = False
            succ_in_target = False
            for t_ind in range(len(abstract_paths[u_ind])):
                reachable_rect = np.column_stack(pc.bounding_box(abstract_paths[u_ind][t_ind])).T
                concrete_succ = transform_to_frames(reachable_rect[0, :],
                                                    reachable_rect[1, :],
                                                    s_rect[0, :], s_rect[1, :])
                if np.any(concrete_succ[1, :] > X_up) or np.any(concrete_succ[0, :] < X_low) \
                        or np.any(concrete_succ[0, :] == concrete_succ[1, :]):
                    adjacency_list[concrete_to_abstract[s]][u_ind] = [-2]  # unsafe if it goes out of boundary
                    succ_intersects_obstacle = True
                    break
                for obstacle_rect in obstacles_rects:
                    if do_rects_inter(obstacle_rect, concrete_succ):
                        adjacency_list[concrete_to_abstract[s]][u_ind] = [-2]  # unsafe if it goes out of boundary
                        succ_intersects_obstacle = True
                        break
                if succ_intersects_obstacle:
                    break
            if not succ_intersects_obstacle:
                reachable_rect = np.column_stack(pc.bounding_box(abstract_paths[u_ind][-1])).T
                for rect in targets_rects:
                    if does_rect_contain(reachable_rect, rect):
                        adjacency_list[concrete_to_abstract[s]][u_ind] = [-1]
                        # TODO: fix this as all concrete states should result in the target, not just this one
                        succ_in_target = True
                        break
            if (not succ_intersects_obstacle) and (not succ_in_target):
                reachable_rect = np.column_stack(pc.bounding_box(abstract_paths[u_ind][-1])).T
                concrete_succ = transform_to_frames(reachable_rect[0, :],
                                                    reachable_rect[1, :],
                                                    s_rect[0, :], s_rect[1, :])
                concrete_succ_indices = rect_to_indices(concrete_succ, symbol_step, X_low, sym_x[0, :],
                                                        over_approximate=True)
                for succ_idx in concrete_succ_indices:
                    if succ_idx in obstacle_indices:  # or succ_idx in symbols_to_discard:
                        adjacency_list[concrete_to_abstract[s]][u_ind] = [-2]
                        succ_intersects_obstacle = True
                        break
                if not succ_intersects_obstacle:
                    indices_to_delete = []
                    for idx, succ_idx in enumerate(concrete_succ_indices):
                        if succ_idx in target_indices:
                            if np.all(np.array(adjacency_list[concrete_to_abstract[s]][u_ind][-1]) != -1):
                                adjacency_list[concrete_to_abstract[s]][u_ind].append(-1)
                            indices_to_delete.append(idx)

                    concrete_succ_indices = np.delete(concrete_succ_indices, np.array(indices_to_delete).astype(int))
                    if len(concrete_succ_indices) == 0:
                        break
                    for idx in concrete_succ_indices:
                        if idx not in concrete_to_abstract:
                            raise "idx outside range of concrete_to_abstract, a problem with obstacles and targets"
                        adjacency_list[concrete_to_abstract[s]][u_ind].append(concrete_to_abstract[idx])
        '''
    return adjacency_list, inverse_adjacency_list, target_parents, concrete_edges, concrete_target_parents, \
           inverse_concrete_edges, abstract_to_concrete_edges


def get_abstract_transition_without_concrete(abstract_state_ind, u_ind,
                                             symmetry_abstract_states,
                                             controllable_abstract_states,
                                             symmetry_under_approx_abstract_targets_rtree_idx3d,
                                             abstract_states_to_rtree_ids,
                                             rtree_ids_to_abstract_states,
                                             abstract_paths):
    neighbors = set()
    # rc, x1 = pc.cheby_ball(symmetry_abstract_states[abstract_state_ind].abstract_obstacles)
    # obstacle_rect = np.array([x1 - rc, x1 + rc])
    reachable_rect = np.column_stack(pc.bounding_box(abstract_paths[u_ind][-1])).T
    # if does_rect_contain(reachable_rect, obstacle_rect):
    #    return [-2]
    # rc, x1 = pc.cheby_ball(symmetry_abstract_states[abstract_state_ind].abstract_targets[0])
    # target_rect = np.array([x1 - rc, x1 + rc])
    # if does_rect_contain(reachable_rect, target_rect):
    #    return [-1]
    for t_ind in range(len(abstract_paths[u_ind])):
        if not pc.is_empty(get_poly_intersection(abstract_paths[u_ind][t_ind],
                                                 symmetry_abstract_states[abstract_state_ind].abstract_obstacles,
                                                 check_convex=False)):
            neighbors.add(-2)
            break
    # not pc.is_empty(get_poly_intersection(abstract_paths[u_ind][-1],
    #                                      symmetry_abstract_states[abstract_state_ind].abstract_targets_over_approximation[0],
    #                                      check_convex=False)):
    if not symmetry_abstract_states[abstract_state_ind].empty_abstract_target and \
            do_rects_inter(reachable_rect,
                           symmetry_abstract_states[abstract_state_ind].rtree_target_rect_over_approx):
        neighbors.add(-1)
        # if -2 not in neighbors:
        # target_poly_after_transition = transform_poly_to_abstract_frames(
        #    symmetry_abstract_states[abstract_state_ind].abstract_targets[0],
        #    reachable_rect,
        #    over_approximate=False, check_convex=True)
        # target_rect_after_transition = transform_rect_to_abstract_frames()
        # if np.all(target_poly_after_transition.contains(np.zeros((3,1)))):
        #    return [-1]

    # abstract_targets_over_approximation = symmetry_abstract_states[abstract_state_ind].abstract_targets_over_approximation
    # print("abstract_targets[0] bounding box:", pc.bounding_box(abstract_targets_over_approximation[0]))
    # target_poly_after_transition = transform_poly_to_abstract_frames(abstract_targets_over_approximation[0], reachable_rect,
    #                                                                 over_approximate=True)
    # rc, x1 = pc.cheby_ball(target_poly_after_transition)
    # target_rect_after_transition = np.array([x1 - rc, x1 + rc])
    # rect = pc.bounding_box(target_poly_after_transition)
    # target_rect_after_transition = get_region_bounding_box(target_poly_after_transition)
    # target_rect_after_transition = fix_rect_angles(target_rect_after_transition) # np.column_stack(rect).T
    # TODO: keep track of the over-approximation of the target for each abstract state
    target_rect_after_transition = transform_rect_to_abstract_frames(
        symmetry_abstract_states[abstract_state_ind].rtree_target_rect_over_approx,
        # abstract_targets_over_approximation[0],
        reachable_rect,
        over_approximate=True)

    target_poly_after_transition_under_approximation = transform_poly_to_abstract_frames(
        symmetry_abstract_states[abstract_state_ind].abstract_targets[0],
        reachable_rect,
        over_approximate=False)

    origin = np.zeros((3, 1))
    another_origin = np.zeros((3, 1))
    another_origin[2] = 2 * math.pi
    if (not pc.is_empty(target_poly_after_transition_under_approximation)) and -1 in neighbors and -2 not in neighbors \
            and (target_poly_after_transition_under_approximation.contains(origin)
                 or target_poly_after_transition_under_approximation.contains(another_origin)):
        return {-1}

    '''
    if target_rect_after_transition_under_approximation is not None and -1 in neighbors and -2 not in neighbors and \
            np.all(target_rect_after_transition_under_approximation[0, :2] <= np.array([0, 0])) and \
            np.all(np.array([0, 0]) <= target_rect_after_transition_under_approximation[1, :2]) and \
            (target_rect_after_transition_under_approximation[0, 2] <= 0 <=
             target_rect_after_transition_under_approximation[1, 2] or
             target_rect_after_transition_under_approximation[0, 2] <= 2 * math.pi <=
             target_rect_after_transition_under_approximation[1, 2]):
        return [-1]
    '''

    original_angle_interval = [target_rect_after_transition[0, 2], target_rect_after_transition[1, 2]]
    decomposed_angle_intervals = get_decomposed_angle_intervals(original_angle_interval)
    for interval in decomposed_angle_intervals:
        target_rect_after_transition_temp = copy.deepcopy(target_rect_after_transition)
        target_rect_after_transition_temp[0, 2] = interval[0]
        target_rect_after_transition_temp[1, 2] = interval[1]
        hits = list(
            symmetry_under_approx_abstract_targets_rtree_idx3d.intersection((target_rect_after_transition_temp[0, 0],
                                                                             target_rect_after_transition_temp[0, 1],
                                                                             target_rect_after_transition_temp[0, 2],
                                                                             target_rect_after_transition_temp[
                                                                                 1, 0],
                                                                             target_rect_after_transition_temp[
                                                                                 1, 1],
                                                                             target_rect_after_transition_temp[
                                                                                 1, 2]),
                                                                            objects=True))
        # hits = np.setdiff1d(hits, np.array(covered_hits)).tolist()
        target_poly_after_transition = pc.box2poly(target_rect_after_transition_temp.T)
        # for idx, abstract_state in enumerate(symmetry_abstract_states):
        for hit in hits:
            # if abstract_to_concrete[idx] \
            #        and does_rect_contain(abstract_state.rtree_target_rect_under_approx, target_rect_after_transition):
            #    neighbors.append(idx)
            # if not pc.is_empty(pc.intersect(hit.object.abstract_targets[0], target_poly_after_transition)):
            # if do_rects_inter(hit.object.rtree_target_rect_under_approx, target_rect_after_transition) \
            # hit.id in abstract_states_to_rtree_ids.values() and \
            # and rtree_ids_to_abstract_states[hit.id] not in neighbors
            if hit.id in rtree_ids_to_abstract_states and \
                    not pc.is_empty(
                        get_poly_intersection(hit.object.abstract_targets[0], target_poly_after_transition)):
                if rtree_ids_to_abstract_states[hit.id] in controllable_abstract_states:
                    if -1 not in neighbors:
                        neighbors.add(-1)
                else:
                    neighbors.add(rtree_ids_to_abstract_states[hit.id])
    # abstract_to_concrete_edges[abstract_state_ind][u_ind] = neighbors
    if not neighbors:
        pdb.set_trace()
        print("No neighbors for abstract_state_ind ", abstract_state_ind, " with control ", u_ind)
        # print("No neighbors!")
        # raise "No neighbors!"
    return neighbors


def update_parent_after_split(parent_abstract_state_ind, new_child_abstract_state_ind_1,
                              new_child_abstract_state_ind_2, u_ind,
                              symmetry_abstract_states,
                              abstract_paths):
    neighbors = []
    reachable_rect = np.column_stack(pc.bounding_box(abstract_paths[u_ind][-1])).T
    # TODO: keep track of the over-approximation of the target for each abstract state
    target_rect_after_transition = transform_rect_to_abstract_frames(
        symmetry_abstract_states[parent_abstract_state_ind].rtree_target_rect_over_approx,
        # abstract_targets_over_approximation[0],
        reachable_rect,
        over_approximate=True)

    #   if do_rects_inter(symmetry_abstract_states[new_child_abstract_state_ind_1].rtree_target_rect_under_approx,
    #                  target_rect_after_transition):
    target_poly_after_transition = pc.box2poly(target_rect_after_transition.T)
    if not pc.is_empty(
            get_poly_intersection(symmetry_abstract_states[new_child_abstract_state_ind_1].abstract_targets[0],
                                  target_poly_after_transition)):
        neighbors.append(new_child_abstract_state_ind_1)

    # if do_rects_inter(symmetry_abstract_states[new_child_abstract_state_ind_2].rtree_target_rect_under_approx,
    #                  target_rect_after_transition):
    if not pc.is_empty(
            get_poly_intersection(symmetry_abstract_states[new_child_abstract_state_ind_2].abstract_targets[0],
                                  target_poly_after_transition)):
        neighbors.append(new_child_abstract_state_ind_2)
    # abstract_to_concrete_edges[abstract_state_ind][u_ind] = neighbors
    if not neighbors:
        print("Warning: No neighbors for parent ", parent_abstract_state_ind)
        # pdb.set_trace()
    return neighbors


def get_abstract_transition(concrete_to_abstract, abstract_to_concrete, abstract_to_concrete_edges,
                            concrete_edges, inverse_concrete_edges, concrete_target_parents,
                            abstract_state_ind, u_ind,
                            sym_x, symbol_step, X_low, X_up, abstract_paths,
                            obstacles_rects, obstacle_indices, targets_rects, target_indices):
    neighbors = []
    # unsafe_transition = False
    for concrete_state_ind in abstract_to_concrete[abstract_state_ind]:
        concrete_neighbors = get_concrete_transition(concrete_state_ind, u_ind, concrete_edges, inverse_concrete_edges,
                                                     sym_x, symbol_step, X_low, X_up, abstract_paths,
                                                     obstacles_rects, obstacle_indices, targets_rects, target_indices)
        if not concrete_neighbors:
            raise "Why this concrete state " + str(concrete_state_ind) + " has no neighbors?"
        # if -2 in concrete_neighbors: # not unsafe_transition and
        # neighbors = [-2]
        # abstract_to_concrete_edges[abstract_state_ind][u_ind] = [-2]
        #    unsafe_transition = True
        #    neighbors = [-2]
        # return [-2]
        # if not unsafe_transition:
        for concrete_neighbor in concrete_neighbors:
            if concrete_neighbor == -1:
                if concrete_state_ind not in concrete_target_parents:
                    concrete_target_parents.append(concrete_state_ind)
                if -1 not in neighbors:
                    neighbors.append(-1)
            elif concrete_neighbor == -2:
                if -2 not in neighbors:
                    neighbors.append(-2)
            elif concrete_neighbor not in concrete_to_abstract:
                raise "where did this concrete_neighbor come from? " + str(concrete_neighbor)
            elif concrete_to_abstract[concrete_neighbor] not in neighbors:
                neighbors.append(concrete_to_abstract[concrete_neighbor])
    # abstract_to_concrete_edges[abstract_state_ind][u_ind] = neighbors
    if not neighbors:
        raise "No neighbors!"
    return neighbors


def get_concrete_transition(s_ind, u_ind, concrete_edges, inverse_concrete_edges,
                            sym_x, symbol_step, X_low, X_up,
                            abstract_paths, obstacles_rects,
                            obstacle_indices, targets_rects, target_indices):
    if concrete_edges[s_ind][u_ind]:
        return concrete_edges[s_ind][u_ind]
    s_subscript = np.array(np.unravel_index(s_ind, tuple((sym_x[0, :]).astype(int))))
    s_rect: np.array = np.row_stack((s_subscript * symbol_step + X_low,
                                     s_subscript * symbol_step + symbol_step + X_low))
    s_rect[0, :] = np.maximum(X_low, s_rect[0, :])
    s_rect[1, :] = np.minimum(X_up, s_rect[1, :])
    for t_ind in range(len(abstract_paths[u_ind])):
        reachable_rect = np.column_stack(pc.bounding_box(abstract_paths[u_ind][t_ind])).T
        concrete_succ = transform_to_frames(reachable_rect[0, :],
                                            reachable_rect[1, :],
                                            s_rect[0, :], s_rect[1, :])
        if np.any(concrete_succ[1, :] > X_up) or np.any(concrete_succ[0, :] < X_low) \
                or np.any(concrete_succ[0, :] == concrete_succ[1, :]):
            concrete_edges[s_ind][u_ind] = [-2]
            return [-2]  # unsafe
        for obstacle_rect in obstacles_rects:
            if do_rects_inter(obstacle_rect, concrete_succ):
                concrete_edges[s_ind][u_ind] = [-2]
                return [-2]  # unsafe
    reachable_rect = np.column_stack(pc.bounding_box(abstract_paths[u_ind][-1])).T
    concrete_succ = transform_to_frames(reachable_rect[0, :],
                                        reachable_rect[1, :],
                                        s_rect[0, :], s_rect[1, :])
    for rect in targets_rects:
        if does_rect_contain(concrete_succ, rect):
            concrete_edges[s_ind][u_ind] = [-1]
            return [-1]  # reached target
    neighbors = rect_to_indices(concrete_succ, symbol_step, X_low, sym_x[0, :],
                                over_approximate=True).tolist()
    for succ_idx in neighbors:
        if succ_idx in obstacle_indices:  # or succ_idx in symbols_to_discard:
            concrete_edges[s_ind][u_ind] = [-2]
            return [-2]  # unsafe
    indices_to_delete = []
    for idx, succ_idx in enumerate(neighbors):
        if succ_idx in target_indices:
            indices_to_delete.append(idx)

    if len(indices_to_delete) == len(neighbors):
        concrete_edges[s_ind][u_ind] = [-1]
        return [-1]

    if indices_to_delete:
        neighbors = np.delete(np.array(neighbors), np.array(indices_to_delete).astype(int)).tolist()
        neighbors.append(-1)
    concrete_edges[s_ind][u_ind] = copy.deepcopy(neighbors)
    for neighbor in neighbors:
        if neighbor >= 0:
            if s_ind not in inverse_concrete_edges[neighbor][u_ind]:
                inverse_concrete_edges[neighbor][u_ind].append(s_ind)
    return neighbors


def plot_abstract_states(symmetry_abstract_states, deleted_abstract_states):
    obstacle_color = 'r'
    target_color = 'g'
    indices_to_plot = np.array(range(len(symmetry_abstract_states)))
    indices_to_plot = np.setdiff1d(indices_to_plot, np.array(deleted_abstract_states)).tolist()
    for idx in indices_to_plot:  # enumerate(symmetry_abstract_states)
        # print("Plotting abstract state: ", idx)
        abstract_state = symmetry_abstract_states[idx]
        plt.figure("Abstract state: " + str(idx))
        currentAxis = plt.gca()
        abstract_obstacles = abstract_state.abstract_obstacles
        abstract_targets = abstract_state.abstract_targets
        for region in abstract_obstacles:
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

        for region in abstract_targets:
            if isinstance(region, pc.Region):
                poly_list = region.list_poly
            else:
                poly_list = [region]
            for poly in poly_list:
                points = pc.extreme(poly)
                points = points[:, :2]
                hull = ConvexHull(points)
                poly_patch = Polygon(points[hull.vertices, :], alpha=.5, color=target_color, fill=True)
                currentAxis.add_patch(poly_patch)
        plt.ylim([-10, 10])
        plt.xlim([-10, 10])
        plt.savefig("Abstract state: " + str(idx))
        # plt.show()
        plt.cla()
        plt.close()


def create_symmetry_abstract_reachable_sets(Symbolic_reduced, n, reachability_rtree_idx3d):
    reachable_rect_global_cntr = 0
    abstract_paths = []
    intersection_radius_threshold = None
    for s_ind in range(Symbolic_reduced.shape[0]):
        for u_ind in range(Symbolic_reduced.shape[1]):
            abstract_rect_low = Symbolic_reduced[s_ind, u_ind, np.arange(n), -1]
            abstract_rect_up = Symbolic_reduced[s_ind, u_ind, n + np.arange(n), -1]
            rect = np.array([abstract_rect_low, abstract_rect_up])
            rect = fix_angle_interval_in_rect(rect)
            curr_max_reachable_rect_radius = np.linalg.norm(rect[1, :] - rect[0, :]) / 2
            if intersection_radius_threshold is None or curr_max_reachable_rect_radius > intersection_radius_threshold:
                intersection_radius_threshold = curr_max_reachable_rect_radius
            reachability_rtree_idx3d.insert(reachable_rect_global_cntr, (rect[0, 0], rect[0, 1],
                                                                         rect[0, 2], rect[1, 0],
                                                                         rect[1, 1], rect[1, 2]),
                                            obj=(s_ind, u_ind))
            reachable_rect_global_cntr += 1
            original_abstract_path = []
            for t_ind in range(Symbolic_reduced.shape[3]):
                rect = np.array([Symbolic_reduced[s_ind, u_ind, np.arange(n), t_ind],
                                 Symbolic_reduced[s_ind, u_ind, n + np.arange(n), t_ind]])
                rect = fix_angle_interval_in_rect(rect)
                poly = pc.box2poly(rect.T)
                original_abstract_path.append(poly)
            abstract_paths.append(original_abstract_path)
    return abstract_paths, reachable_rect_global_cntr, intersection_radius_threshold


def create_targets_and_obstacles(Target_low, Target_up, Obstacle_low, Obstacle_up, symbol_step, sym_x, X_low):
    obstacles = []
    targets = []
    targets_rects = []
    obstacles_rects = []
    obstacle_indices = []
    target_indices = []
    for obstacle_idx in range(Obstacle_low.shape[0]):
        obstacle_rect = np.array([Obstacle_low[obstacle_idx, :], Obstacle_up[obstacle_idx, :]])
        # obstacle_rect = fix_rect_angles(obstacle_rect)
        obstacles_rects.append(obstacle_rect)
        obstacle_poly = pc.box2poly(obstacle_rect.T)
        obstacles.append(obstacle_poly)
        indices = rect_to_indices(obstacle_rect, symbol_step, X_low,
                                  sym_x[0, :], over_approximate=True)
        obstacle_indices.extend(indices)

    for target_idx in range(Target_low.shape[0]):
        target_rect = np.array([Target_low[target_idx, :], Target_up[target_idx, :]])
        # target_rect = fix_rect_angles(target_rect)
        target_poly = pc.box2poly(target_rect.T)
        targets.append(target_poly)
        targets_rects.append(target_rect)
        target_indices.extend(rect_to_indices(target_rect, symbol_step, X_low,
                                              sym_x[0, :], over_approximate=False))

    return targets, targets_rects, target_indices, obstacles, obstacles_rects, obstacle_indices


def symmetry_abstract_synthesis_helper(abstract_to_concrete, remaining_abstract_states, adjacency_list,
                                       inverse_adjacency_list, target_parents, refinement_candidates,
                                       global_controllable_abstract_states,
                                       controller):
    t_start = time.time()
    num_controllable_states = 0
    controllable_abstract_states = []
    unsafe_abstract_states = []
    abstract_states_to_explore = copy.deepcopy(target_parents)
    # copy.deepcopy(remaining_abstract_states)  # list(range(len(abstract_to_concrete)))
    while True:
        num_new_symbols = 0
        temp_controllable_abstract_states = []
        temp_unsafe_abstract_states = []
        for abstract_s in abstract_states_to_explore:
            unsafe_state = True
            for u_ind in range(len(adjacency_list[abstract_s])):
                result_reach = True
                if not adjacency_list[abstract_s][u_ind]:
                    raise "how did this happen?"
                if -2 in adjacency_list[abstract_s][u_ind]:
                    continue
                unsafe_state = False
                for next_abstract_s in adjacency_list[abstract_s][u_ind]:
                    if not (next_abstract_s == -1 or next_abstract_s in temp_controllable_abstract_states
                            or next_abstract_s in controllable_abstract_states
                            or next_abstract_s in global_controllable_abstract_states):
                        result_reach = False
                        break
                if result_reach:
                    controller[abstract_s] = u_ind
                    temp_controllable_abstract_states.append(abstract_s)
                    num_new_symbols += 1
                    unsafe_state = False
                    break
            if unsafe_state:
                temp_unsafe_abstract_states.append(abstract_s)

        # unsafe_abstract_states.extend(temp_unsafe_abstract_states)
        # abstract_states_to_explore = np.setdiff1d(abstract_states_to_explore, temp_unsafe_abstract_states)
        if num_new_symbols:
            print(time.time() - t_start, " ", num_new_symbols,
                  " new controllable states have been found in this synthesis iteration\n")
            controllable_abstract_states.extend(temp_controllable_abstract_states)
            num_controllable_states += num_new_symbols
            for abstract_s in temp_controllable_abstract_states:
                for u_ind in range(len(adjacency_list[abstract_s])):
                    for parent in inverse_adjacency_list[abstract_s][u_ind]:
                        if parent not in global_controllable_abstract_states and \
                                parent not in controllable_abstract_states and \
                                parent not in abstract_states_to_explore:  # and parent not in
                            abstract_states_to_explore.append(parent)
                            if len(abstract_to_concrete[parent]) > 1 and parent not in refinement_candidates:
                                refinement_candidates.append(parent)
            abstract_states_to_explore = np.setdiff1d(abstract_states_to_explore,
                                                      temp_controllable_abstract_states).tolist()
            refinement_candidates = np.setdiff1d(refinement_candidates, temp_controllable_abstract_states).tolist()
            print(num_controllable_states, ' symbols are controllable to satisfy the reach-avoid specification\n')
        else:
            print('No new controllable state has been found in this synthesis iteration\n', time.time() - t_start)
            break

    return controller, controllable_abstract_states, unsafe_abstract_states, abstract_states_to_explore

    '''
    plt.figure("Original coordinates")
    currentAxis = plt.gca()
    color = 'r'
    edge_color = 'k'
    for i in range(Obstacle_up.shape[0]):  # and np.any(rect_curr):
        rect = np.array([Obstacle_low[i, :], Obstacle_up[i, :]])
        rect_patch = Rectangle(rect[0, [0, 1]], rect[1, 0] - rect[0, 0],
                               rect[1, 1] - rect[0, 1], linewidth=1,
                               edgecolor=color, facecolor=color)
        currentAxis.add_patch(rect_patch)

    color = 'g'
    for target_idx in range(Target_low.shape[0]):
        rect = np.array([Target_low[target_idx, :], Target_up[target_idx, :]])
        rect_patch = Rectangle(rect[0, [0, 1]], rect[1, 0] - rect[0, 0],
                               rect[1, 1] - rect[0, 1], linewidth=1,
                               edgecolor=color, facecolor=color)
        currentAxis.add_patch(rect_patch)

    color = 'b'
    if len(result_abstract_paths) > 0:
        for initial_set, path in result_abstract_paths:
            for rect in path:
                rect = transform_to_frames(rect[0, :], rect[1, :], initial_set[0, :], initial_set[1, :])
                rect_patch = Rectangle(rect[0, [0, 1]], rect[1, 0] - rect[0, 0],
                                       rect[1, 1] - rect[0, 1], linewidth=1,
                                       edgecolor=edge_color, facecolor=color)
                currentAxis.add_patch(rect_patch)
            rect = transform_to_frames(path[-1][0, :], path[-1][1, :], initial_set[0, :],
                                       initial_set[1, :])
            print("last rect: ", rect)
            print("targets[-1]: ", targets[-1])
            print("Does the end of the path belong to the target? ", does_rect_contain(rect, targets[-1]))
            rect_patch = Rectangle(rect[0, [0, 1]], rect[1, 0] - rect[0, 0],
                                   rect[1, 1] - rect[0, 1], linewidth=1,
                                   edgecolor=edge_color, facecolor='m')
            currentAxis.add_patch(rect_patch)
    else:
        for path in abstract_paths:
            for rect in path:
                # rect = transform_to_frames(rect[0, :], rect[1, :], initial_set[0, :], initial_set[1, :])
                rect_patch = Rectangle(rect[0, [0, 1]], rect[1, 0] - rect[0, 0],
                                       rect[1, 1] - rect[0, 1], linewidth=1,
                                       edgecolor=edge_color, facecolor=color)
                currentAxis.add_patch(rect_patch)

    plt.ylim([X_low[1], X_up[1]])
    plt.xlim([X_low[0], X_up[0]])

    plt.figure("Resulting reachable sets in reduced coordinates")
    color = 'orange'
    currentAxis_1 = plt.gca()
    for path in abstract_paths:
        for rect in path:
            rect_patch = Rectangle(rect[0, [0, 1]], rect[1, 0] - rect[0, 0],
                                   rect[1, 1] - rect[0, 1], linewidth=1,
                                   edgecolor=edge_color, facecolor=color)
            currentAxis_1.add_patch(rect_patch)
    plt.ylim([-3, 3])
    plt.xlim([-3, 3])

    plt.show()
    '''


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


def split_abstract_state(abstract_state_ind, concrete_indices,
                         abstract_to_concrete, concrete_to_abstract, abstract_edges,
                         concrete_edges, inverse_concrete_edges, concrete_target_parents, adjacency_list,
                         inverse_adjacency_list, target_parents, controllable_abstract_states, abstract_paths,
                         symmetry_transformed_targets_and_obstacles, symmetry_abstract_states,
                         symmetry_under_approx_abstract_targets_rtree_idx3d, abstract_states_to_rtree_ids,
                         rtree_ids_to_abstract_states, next_rtree_id_candidate,
                         symmetry_over_approx_abstract_targets_rtree_idx3d,
                         sym_x, symbol_step, X_low, X_up,
                         obstacles_rects, obstacle_indices, targets_rects, target_indices):
    abstract_state_1 = None
    abstract_state_2 = None
    if len(concrete_indices) >= len(abstract_to_concrete[abstract_state_ind]):
        print("The concrete indices provided are all that ", abstract_state_ind, " represents, so no need to split.")
        return concrete_to_abstract, abstract_to_concrete, symmetry_abstract_states, \
               adjacency_list, inverse_adjacency_list, target_parents
    rest_of_concrete_indices = np.setdiff1d(np.array(abstract_to_concrete[abstract_state_ind]), concrete_indices)
    for concrete_state_idx in concrete_indices:
        abstract_state_1 = add_concrete_state_to_symmetry_abstract_state(concrete_state_idx,
                                                                         abstract_state_1,
                                                                         symmetry_transformed_targets_and_obstacles)
    for concrete_state_idx in rest_of_concrete_indices:
        abstract_state_2 = add_concrete_state_to_symmetry_abstract_state(concrete_state_idx,
                                                                         abstract_state_2,
                                                                         symmetry_transformed_targets_and_obstacles)

    symmetry_abstract_states.append(abstract_state_1)
    abstract_to_concrete.append(abstract_state_1.concrete_state_idx)
    for idx in abstract_state_1.concrete_state_idx:
        concrete_to_abstract[idx] = len(abstract_to_concrete) - 1
    symmetry_abstract_states.append(abstract_state_2)
    abstract_to_concrete.append(abstract_state_2.concrete_state_idx)
    for idx in abstract_state_2.concrete_state_idx:
        concrete_to_abstract[idx] = len(abstract_to_concrete) - 1

    rtree_target_rect_under_approx = symmetry_abstract_states[abstract_state_ind].rtree_target_rect_under_approx
    original_angle_interval = [rtree_target_rect_under_approx[0, 2], rtree_target_rect_under_approx[1, 2]]
    decomposed_angle_intervals = get_decomposed_angle_intervals(original_angle_interval)
    for interval in decomposed_angle_intervals:
        rtree_target_rect_under_approx_temp = copy.deepcopy(rtree_target_rect_under_approx)
        rtree_target_rect_under_approx_temp[0, 2] = interval[0]
        rtree_target_rect_under_approx_temp[1, 2] = interval[1]
        symmetry_under_approx_abstract_targets_rtree_idx3d.delete(abstract_states_to_rtree_ids[abstract_state_ind],
                                                                  (rtree_target_rect_under_approx_temp[0, 0],
                                                                   rtree_target_rect_under_approx_temp[0, 1],
                                                                   rtree_target_rect_under_approx_temp[0, 2],
                                                                   rtree_target_rect_under_approx_temp[1, 0],
                                                                   rtree_target_rect_under_approx_temp[1, 1],
                                                                   rtree_target_rect_under_approx_temp[1, 2]))
    del rtree_ids_to_abstract_states[abstract_states_to_rtree_ids[abstract_state_ind]]
    del abstract_states_to_rtree_ids[abstract_state_ind]
    '''
    rtree_target_rect_over_approx = symmetry_abstract_states[abstract_state_ind].rtree_target_rect_over_approx
    symmetry_over_approx_abstract_targets_rtree_idx3d.delete(abstract_state_ind,
                                                             (rtree_target_rect_over_approx[0, 0],
                                                              rtree_target_rect_over_approx[0, 1],
                                                              rtree_target_rect_over_approx[0, 2],
                                                              rtree_target_rect_over_approx[1, 0],
                                                              rtree_target_rect_over_approx[1, 1],
                                                              rtree_target_rect_over_approx[1, 2]))
    '''
    original_angle_interval_1 = [abstract_state_1.rtree_target_rect_under_approx[0, 2],
                                 abstract_state_1.rtree_target_rect_under_approx[1, 2]]
    decomposed_angle_intervals_1 = get_decomposed_angle_intervals(original_angle_interval_1)
    for interval in decomposed_angle_intervals_1:
        rtree_target_rect_under_approx_temp = copy.deepcopy(abstract_state_1.rtree_target_rect_under_approx)
        rtree_target_rect_under_approx_temp[0, 2] = interval[0]
        rtree_target_rect_under_approx_temp[1, 2] = interval[1]
        symmetry_under_approx_abstract_targets_rtree_idx3d.insert(next_rtree_id_candidate, (
            rtree_target_rect_under_approx_temp[0, 0], rtree_target_rect_under_approx_temp[0, 1],
            rtree_target_rect_under_approx_temp[0, 2], rtree_target_rect_under_approx_temp[1, 0],
            rtree_target_rect_under_approx_temp[1, 1], rtree_target_rect_under_approx_temp[1, 2]),
                                                                  obj=symmetry_abstract_states[-2])
    abstract_states_to_rtree_ids[len(symmetry_abstract_states) - 2] = next_rtree_id_candidate
    rtree_ids_to_abstract_states[next_rtree_id_candidate] = len(symmetry_abstract_states) - 2
    next_rtree_id_candidate += 1
    original_angle_interval_2 = [abstract_state_2.rtree_target_rect_under_approx[0, 2],
                                 abstract_state_2.rtree_target_rect_under_approx[1, 2]]
    decomposed_angle_intervals_2 = get_decomposed_angle_intervals(original_angle_interval_2)
    for interval in decomposed_angle_intervals_2:
        rtree_target_rect_under_approx_temp = copy.deepcopy(abstract_state_2.rtree_target_rect_under_approx)
        rtree_target_rect_under_approx_temp[0, 2] = interval[0]
        rtree_target_rect_under_approx_temp[1, 2] = interval[1]
        symmetry_under_approx_abstract_targets_rtree_idx3d.insert(next_rtree_id_candidate, (
            rtree_target_rect_under_approx_temp[0, 0], rtree_target_rect_under_approx_temp[0, 1],
            rtree_target_rect_under_approx_temp[0, 2], rtree_target_rect_under_approx_temp[1, 0],
            rtree_target_rect_under_approx_temp[1, 1], rtree_target_rect_under_approx_temp[1, 2]),
                                                                  obj=symmetry_abstract_states[-1])

    abstract_states_to_rtree_ids[len(symmetry_abstract_states) - 1] = next_rtree_id_candidate
    rtree_ids_to_abstract_states[next_rtree_id_candidate] = len(symmetry_abstract_states) - 1
    next_rtree_id_candidate += 1
    '''
    symmetry_over_approx_abstract_targets_rtree_idx3d.insert(len(abstract_to_concrete) - 2, (
        abstract_state_1.rtree_target_rect_over_approx[0, 0], abstract_state_1.rtree_target_rect_over_approx[0, 1],
        abstract_state_1.rtree_target_rect_over_approx[0, 2], abstract_state_1.rtree_target_rect_over_approx[1, 0],
        abstract_state_1.rtree_target_rect_over_approx[1, 1], abstract_state_1.rtree_target_rect_over_approx[1, 2]),
                                                             obj=symmetry_abstract_states[-2])
    symmetry_over_approx_abstract_targets_rtree_idx3d.insert(len(abstract_to_concrete) - 1, (
        abstract_state_2.rtree_target_rect_over_approx[0, 0], abstract_state_2.rtree_target_rect_over_approx[0, 1],
        abstract_state_2.rtree_target_rect_over_approx[0, 2], abstract_state_2.rtree_target_rect_over_approx[1, 0],
        abstract_state_2.rtree_target_rect_over_approx[1, 1], abstract_state_2.rtree_target_rect_over_approx[1, 2]),
                                                             obj=symmetry_abstract_states[-1])
    '''

    original_neighbors = copy.deepcopy(adjacency_list[abstract_state_ind])
    original_parents = copy.deepcopy(inverse_adjacency_list[abstract_state_ind])
    # original_concrete = copy.deepcopy(abstract_to_concrete[abstract_state_ind])
    adjacency_list[abstract_state_ind] = None  # [[] * len(abstract_paths)]  # cut the original abstract state from
    # all connections
    inverse_adjacency_list[abstract_state_ind] = None  # [[] * len(abstract_paths)]
    abstract_to_concrete[abstract_state_ind] = []
    adjacency_list.append([])
    for u_ind in range(len(abstract_paths)):
        adjacency_list[-1].append([])
    adjacency_list.append([])
    for u_ind in range(len(abstract_paths)):
        adjacency_list[-1].append([])
    # adjacency_list.append([[]] * len(abstract_paths))
    inverse_adjacency_list.append([])
    for u_ind in range(len(abstract_paths)):
        inverse_adjacency_list[-1].append([])
    inverse_adjacency_list.append([])
    for u_ind in range(len(abstract_paths)):
        inverse_adjacency_list[-1].append([])
    # inverse_adjacency_list.append([[]] * len(abstract_paths))
    # inverse_adjacency_list.append([[]] * len(abstract_paths))
    for u_ind in range(len(abstract_paths)):
        # parents_to_explore = np.setdiff1d(inverse_adjacency_list[abstract_state_ind][u_ind],
        #                                  np.array(updated_parents)).tolist()
        updated_parents = []
        for parent in original_parents[u_ind]:  # inverse_adjacency_list[abstract_state_ind]
            if parent != abstract_state_ind:  # and parent not in controllable_abstract_states:
                # and parent < len(abstract_to_concrete) - 2:
                # and parent not in controllable_abstract_states:  # and parent not in
                # updated_parents:
                if adjacency_list[parent] is None or not adjacency_list[parent][u_ind]:
                    print("Warning: parent ", parent, " was split before and the adjacency list of ",
                          abstract_state_ind, " was not updated")
                    # pdb.set_trace()
                else:
                    # raise "Where did " + str(parent) + " come from?"
                    '''
                    parent_neighbors = get_abstract_transition(concrete_to_abstract,
                                                               abstract_to_concrete,
                                                               abstract_edges,
                                                               concrete_edges,
                                                               inverse_concrete_edges,
                                                               concrete_target_parents,
                                                               parent, u_ind,
                                                               sym_x,
                                                               symbol_step, X_low,
                                                               X_up, abstract_paths,
                                                               obstacles_rects,
                                                               obstacle_indices,
                                                               targets_rects,
                                                               target_indices)
                    '''
                    # parent_neighbors = get_abstract_transition_without_concrete(parent, u_ind,
                    #                                                            symmetry_abstract_states,
                    #                                                            symmetry_under_approx_abstract_targets_rtree_idx3d,
                    #                                                            abstract_paths)
                    parent_children = update_parent_after_split(parent, len(abstract_to_concrete) - 1,
                                                                len(abstract_to_concrete) - 2, u_ind,
                                                                symmetry_abstract_states,
                                                                abstract_paths)
                    child_idx_to_delete = None
                    for idx, child in enumerate(adjacency_list[parent][u_ind]):
                        if child == abstract_state_ind:
                            child_idx_to_delete = idx
                            break
                    if child_idx_to_delete is not None:
                        adjacency_list[parent][u_ind].pop(child_idx_to_delete)
                        updated_parents.append(parent)
                    for child in parent_children:
                        if child not in adjacency_list[parent][u_ind]:
                            adjacency_list[parent][u_ind].append(child)
                        if parent not in inverse_adjacency_list[child][u_ind]:
                            inverse_adjacency_list[child][u_ind].append(parent)
                # if abstract_state_ind in parent_neighbors:
                #    print("how did this happen?")
                # adjacency_list[parent][u_ind] = copy.deepcopy(parent_neighbors)
                '''
                for next_abstract_state_ind in adjacency_list[parent][u_ind]:
                    if next_abstract_state_ind >= len(abstract_to_concrete) - 2 and \
                            parent not in inverse_adjacency_list[next_abstract_state_ind][u_ind]:
                        inverse_adjacency_list[next_abstract_state_ind][u_ind].append(parent)
                    # elif (next_abstract_state_ind == -1 or next_abstract_state_ind in controllable_abstract_states) \
                    #        and parent not in target_parents and parent not in controllable_abstract_states:
                    #    target_parents.append(parent)
                # updated_parents.append(parent)
                '''

        '''
        for original_parent in original_parents[u_ind]:
            if abstract_state_ind != original_parent:
                if u_ind >= len(adjacency_list[original_parent]):
                    raise "how?"
                if abstract_state_ind in adjacency_list[original_parent][u_ind]:
                    raise "how did this stay here?"
        '''

        '''
        neighbors_2 = get_abstract_transition(concrete_to_abstract,
                                              abstract_to_concrete,
                                              abstract_edges,
                                              concrete_edges,
                                              inverse_concrete_edges,
                                              concrete_target_parents,
                                              len(abstract_to_concrete) - 2,
                                              u_ind, sym_x,
                                              symbol_step, X_low,
                                              X_up, abstract_paths,
                                              obstacles_rects,
                                              obstacle_indices,
                                              targets_rects,
                                              target_indices)
        '''
        neighbors_2 = list(get_abstract_transition_without_concrete(len(abstract_to_concrete) - 2, u_ind,
                                                                    symmetry_abstract_states,
                                                                    controllable_abstract_states,
                                                                    symmetry_under_approx_abstract_targets_rtree_idx3d,
                                                                    abstract_states_to_rtree_ids,
                                                                    rtree_ids_to_abstract_states,
                                                                    abstract_paths))
        # if abstract_state_ind in neighbors_2:
        #    print("how did this happen?")
        adjacency_list[len(abstract_to_concrete) - 2][u_ind] = copy.deepcopy(neighbors_2)
        # neighbors_to_update = np.setdiff1d(np.array(neighbors), np.array(updated_neighbors_1)).tolist()
        neighbor_indices_to_delete = []  # these are neighbors resulting from new over-approximation error
        for idx, neighbor in enumerate(neighbors_2):
            # if neighbor not in original_neighbors[u_ind] and neighbor < len(abstract_to_concrete) - 2:
            #    raise "how?"
            if neighbor >= 0:
                if neighbor >= len(abstract_to_concrete) - 2:
                    if len(abstract_to_concrete) - 2 not in inverse_adjacency_list[neighbor][u_ind]:
                        inverse_adjacency_list[neighbor][u_ind].append(len(abstract_to_concrete) - 2)
                else:
                    if neighbor not in original_neighbors[u_ind]:
                        neighbor_indices_to_delete.append(idx)
                        print("Warning: new child", neighbor, " was going to be added as a child of ",
                              len(abstract_to_concrete) - 2,
                              "which was not a child of the original parent ", abstract_state_ind)
                    else:
                        if adjacency_list[neighbor] is None or inverse_adjacency_list[neighbor] is None:
                            print("Warning: get_abstract_transition_without_concrete returned a child ", neighbor,
                                  " which was deleted before! ")
                            # pdb.set_trace()
                        else:
                            parent_idx_to_modify = None
                            for idx, parent in enumerate(inverse_adjacency_list[neighbor][u_ind]):
                                if parent == abstract_state_ind:
                                    parent_idx_to_modify = idx
                                    break
                            if parent_idx_to_modify is not None:
                                inverse_adjacency_list[neighbor][u_ind][parent_idx_to_modify] = len(
                                    abstract_to_concrete) - 2
                            # updated_neighbors_1.append(neighbor)
            # if (neighbor == -1 or neighbor in controllable_abstract_states) and \
            #        len(abstract_to_concrete) - 2 not in target_parents:
            #    target_parents.append(len(abstract_to_concrete) - 2)
        for idx in sorted(neighbor_indices_to_delete, reverse=True):
            del adjacency_list[len(abstract_to_concrete) - 2][u_ind][idx]

        '''
        neighbors = get_abstract_transition(concrete_to_abstract,
                                            abstract_to_concrete, abstract_edges,
                                            concrete_edges,
                                            inverse_concrete_edges,
                                            concrete_target_parents,
                                            len(abstract_to_concrete) - 1,
                                            u_ind, sym_x,
                                            symbol_step, X_low,
                                            X_up, abstract_paths,
                                            obstacles_rects,
                                            obstacle_indices,
                                            targets_rects,
                                            target_indices)
        '''
        neighbors = list(get_abstract_transition_without_concrete(len(abstract_to_concrete) - 1, u_ind,
                                                                  symmetry_abstract_states,
                                                                  controllable_abstract_states,
                                                                  symmetry_under_approx_abstract_targets_rtree_idx3d,
                                                                  abstract_states_to_rtree_ids,
                                                                  rtree_ids_to_abstract_states,
                                                                  abstract_paths))
        # if abstract_state_ind in neighbors:
        #    print("how did this happen?")
        adjacency_list[len(abstract_to_concrete) - 1][u_ind] = copy.deepcopy(neighbors)
        # neighbors_to_update = np.setdiff1d(np.array(neighbors), np.array(updated_neighbors_2)).tolist()
        neighbor_indices_to_delete = []
        for idx, neighbor in enumerate(neighbors):
            # if neighbor not in original_neighbors[u_ind] and neighbor < len(abstract_to_concrete) - 2:
            #    raise "how?"
            if neighbor >= 0:
                if neighbor >= len(abstract_to_concrete) - 2:
                    if len(abstract_to_concrete) - 1 not in inverse_adjacency_list[neighbor][u_ind]:
                        inverse_adjacency_list[neighbor][u_ind].append(len(abstract_to_concrete) - 1)
                else:
                    if neighbor not in original_neighbors[u_ind]:
                        print("Warning: new child", neighbor, " was going to be "
                                                              "added as a child of ", len(abstract_to_concrete) - 1,
                              "which was not a child of the original parent ", abstract_state_ind)
                        neighbor_indices_to_delete.append(idx)
                    else:
                        if adjacency_list[neighbor] is None or inverse_adjacency_list[neighbor] is None:
                            print("Warning: get_abstract_transition_without_concrete returned a child ", neighbor,
                                  " which was deleted before! ")
                            # pdb.set_trace()
                        else:
                            parent_idx_to_modify = None
                            add_new_item = False
                            for idx, parent in enumerate(inverse_adjacency_list[neighbor][u_ind]):
                                if parent == abstract_state_ind:
                                    parent_idx_to_modify = idx
                                    break
                                elif parent == len(abstract_to_concrete) - 2:
                                    add_new_item = True
                                    break
                            if parent_idx_to_modify is not None:
                                inverse_adjacency_list[neighbor][u_ind][parent_idx_to_modify] = len(
                                    abstract_to_concrete) - 1
                            elif add_new_item:
                                inverse_adjacency_list[neighbor][u_ind].append(len(abstract_to_concrete) - 1)
                                # updated_neighbors_2.append(neighbor)
            # if (neighbor == -1 or neighbor in controllable_abstract_states) \
            #        and len(abstract_to_concrete) - 1 not in target_parents:
            #    target_parents.append(len(abstract_to_concrete) - 1)
        for idx in sorted(neighbor_indices_to_delete, reverse=True):
            del adjacency_list[len(abstract_to_concrete) - 1][u_ind][idx]
        original_neighbors[u_ind] = np.setdiff1d(np.array(original_neighbors[u_ind]), np.array(neighbors_2)).tolist()
        original_neighbors[u_ind] = np.setdiff1d(np.array(original_neighbors[u_ind]), np.array(neighbors)).tolist()
        for neighbor in original_neighbors[u_ind]:
            if neighbor != abstract_state_ind:
                if adjacency_list[neighbor] is None or inverse_adjacency_list[neighbor] is None:
                    print("neighbor ", neighbor, " in original_neighbors of ", abstract_state_ind, " at u_ind ", u_ind)
                    # pdb.set_trace()
                else:
                    parent_idx_to_delete = None
                    for idx, parent in enumerate(inverse_adjacency_list[neighbor][u_ind]):
                        if parent == abstract_state_ind:
                            parent_idx_to_delete = idx
                            break
                    if parent_idx_to_delete is not None:
                        inverse_adjacency_list[neighbor][u_ind].pop(parent_idx_to_delete)
        original_parents[u_ind] = np.setdiff1d(np.array(original_parents[u_ind]), np.array(updated_parents)).tolist()
        for parent in original_parents[u_ind]:
            if parent != abstract_state_ind:
                if adjacency_list[parent] is None or inverse_adjacency_list[parent] is None:
                    print("parent ", parent, " in original_parents of ", abstract_state_ind, " at u_ind ", u_ind)
                    # pdb.set_trace()
                else:
                    child_idx_to_delete = None
                    for idx, child in enumerate(adjacency_list[parent][u_ind]):
                        if child == abstract_state_ind:
                            child_idx_to_delete = idx
                            break
                    if child_idx_to_delete is not None:
                        adjacency_list[parent][u_ind].pop(child_idx_to_delete)

        '''
        for original_neighbor in original_neighbors[u_ind]:
            if abstract_state_ind != original_neighbor:
                if u_ind >= len(inverse_adjacency_list[original_neighbor]):
                    raise "how?"
                for parent in inverse_adjacency_list[original_neighbor][u_ind]:
                    if parent == abstract_state_ind and abstract_state_ind != original_neighbor:
                        raise "how did this stay here?"
        '''
    '''
    for u_ind in range(len(abstract_paths)):
        for original_parent in original_parents[u_ind]:
            if abstract_state_ind != original_parent:
                if u_ind >= len(adjacency_list[original_parent]):
                    raise "how?"
                if abstract_state_ind in adjacency_list[original_parent][u_ind]:
                    raise "how did this stay here?"

    for u_ind in range(len(abstract_paths)):
        for original_neighbor in original_neighbors[u_ind]:
            if abstract_state_ind != original_neighbor:
                if u_ind >= len(adjacency_list[original_neighbor]):
                    raise "how?"
                if abstract_state_ind in inverse_adjacency_list[original_neighbor][u_ind]:
                    raise "how did this stay here?"
    '''
    for idx, parent in enumerate(target_parents):
        if parent == abstract_state_ind:
            target_parents.pop(idx)
            target_parents.append(len(abstract_to_concrete) - 1)
            target_parents.append(len(abstract_to_concrete) - 2)
            break

    '''
    for u_ind in range(len(abstract_paths)):
        for abstract_s in range(len(abstract_to_concrete)):
            if adjacency_list[abstract_s] is not None:
                if abstract_state_ind in adjacency_list[abstract_s][u_ind]:
                    raise "how did this stay here?"
                if abstract_state_ind in inverse_adjacency_list[abstract_s][u_ind]:
                    raise "how did this stay here?"
    '''

    return concrete_to_abstract, abstract_to_concrete, symmetry_abstract_states, \
           adjacency_list, inverse_adjacency_list, target_parents, abstract_states_to_rtree_ids, \
           rtree_ids_to_abstract_states, next_rtree_id_candidate


def refine(concrete_to_abstract, abstract_to_concrete, abstract_edges,
           concrete_edges, inverse_concrete_edges, concrete_target_parents, symmetry_abstract_states,
           remaining_abstract_states,
           adjacency_list, inverse_adjacency_list, refinement_candidates, target_parents, controllable_abstract_states,
           unsafe_abstract_states,
           symmetry_transformed_targets_and_obstacles, symmetry_under_approx_abstract_targets_rtree_idx3d,
           abstract_states_to_rtree_ids,
           rtree_ids_to_abstract_states, next_rtree_id_candidate,
           symmetry_over_approx_abstract_targets_rtree_idx3d,
           abstract_paths, reachability_rtree_idx3d, sym_x,
           symbol_step, X_low,
           X_up,
           obstacles_rects,
           obstacle_indices,
           targets_rects,
           target_indices):
    # for abstract_state_ind in unsafe_abstract_states:
    itr = 0
    progress = False
    num_new_abstract_states = 0
    deleted_abstract_states = []
    # refinement_candidates_temp = copy.deepcopy(refinement_candidates)
    # len_refinement_candidates = len(refinement_candidates)
    max_num_of_refinements = 3  # len(refinement_candidates)
    while itr < max_num_of_refinements and len(refinement_candidates):  # remaining_abstract_states.shape[0] / 10):
        # for abstract_state_ind in target_parents:
        itr += 1
        # abstract_state = symmetry_abstract_states[abstract_state_ind]
        # target_rect = abstract_state.rtree_target_rect
        # nearest_control_hits = list(reachability_rtree_idx3d.nearest((target_rect[0, 0], target_rect[0, 1],
        #                                                              target_rect[0, 2], target_rect[1, 0] + 0.01,
        #                                                             target_rect[1, 1] + 0.01, target_rect[1, 2]
        #                                                              + 0.01), num_results=1, objects=True))
        # concrete_indices = []
        random_ind = np.random.randint(0, len(refinement_candidates))  # remaining_abstract_states)
        abstract_state_ind = refinement_candidates[random_ind]
        # itr_u = 0
        # progress_u = False
        concrete_indices_len = len(abstract_to_concrete[abstract_state_ind])
        if concrete_indices_len > 1:
            concrete_indices = random.choices(abstract_to_concrete[abstract_state_ind],
                                              k=int(concrete_indices_len / 2))
            concrete_indices = list(dict.fromkeys(concrete_indices))  # removing duplicates
            # while not progress_u and itr_u < len(abstract_paths):
            #    itr_u += 1
            # u_ind = np.random.randint(0, len(abstract_paths))
            # s_ind = nearest_control_hits[0].object[0]
            # u_ind = nearest_control_hits[0].object[1]
            # rc, x1 = pc.cheby_ball(abstract_paths[u_ind][-1])
            # reachable_rect = np.column_stack(abstract_paths[u_ind][-1].bounding_box).T  # np.array([x1 - rc, x1 + rc])
            '''
            for concrete_state_idx in abstract_to_concrete[abstract_state_ind]:
                # concrete_state = symmetry_transformed_targets_and_obstacles[concrete_state_idx]
                # if pc.is_empty(pc.intersect(abstract_paths[u_ind][-1], concrete_state.abstract_obstacles)) and \
                #        does_rect_contain(reachable_rect, concrete_state.rtree_target_rect):
                if concrete_edges[s_ind][u_ind][0] == -1:
                    concrete_indices.append(concrete_state_idx)
                    progress_u = True
                    # if len(concrete_indices) >= len(abstract_to_concrete[abstract_state_ind]) / 2:
                    #    break  # we want a balanced split
            '''
            # if progress_u:
            progress = True
            if len(concrete_indices) < len(abstract_to_concrete[abstract_state_ind]):
                concrete_to_abstract, abstract_to_concrete, symmetry_abstract_states, \
                adjacency_list, inverse_adjacency_list, \
                target_parents, abstract_states_to_rtree_ids, \
                rtree_ids_to_abstract_states, \
                next_rtree_id_candidate = split_abstract_state(abstract_state_ind,
                                                               concrete_indices,
                                                               abstract_to_concrete,
                                                               concrete_to_abstract,
                                                               abstract_edges,
                                                               concrete_edges,
                                                               inverse_concrete_edges,
                                                               concrete_target_parents,
                                                               adjacency_list,
                                                               inverse_adjacency_list,
                                                               target_parents,
                                                               controllable_abstract_states,
                                                               abstract_paths,
                                                               symmetry_transformed_targets_and_obstacles,
                                                               symmetry_abstract_states,
                                                               symmetry_under_approx_abstract_targets_rtree_idx3d,
                                                               abstract_states_to_rtree_ids,
                                                               rtree_ids_to_abstract_states,
                                                               next_rtree_id_candidate,
                                                               symmetry_over_approx_abstract_targets_rtree_idx3d,
                                                               sym_x, symbol_step, X_low,
                                                               X_up,
                                                               obstacles_rects,
                                                               obstacle_indices,
                                                               targets_rects,
                                                               target_indices)
                remaining_abstract_states = np.setdiff1d(remaining_abstract_states, abstract_state_ind)
                remaining_abstract_states = np.append(remaining_abstract_states, len(abstract_to_concrete) - 1)
                remaining_abstract_states = np.append(remaining_abstract_states, len(abstract_to_concrete) - 2)
                if len(abstract_to_concrete[-1]) > 1:
                    refinement_candidates.append(len(abstract_to_concrete) - 1)
                if len(abstract_to_concrete[-2]) > 1:
                    refinement_candidates.append(len(abstract_to_concrete) - 2)
                refinement_candidates.pop(random_ind)
                '''
                for idx, abstract_s in enumerate(refinement_candidates):
                    if abstract_s == abstract_state_ind:
                        refinement_candidates.pop(idx)
                        break     
                '''
                deleted_abstract_states.append(abstract_state_ind)
                num_new_abstract_states += 1
        else:
            refinement_candidates.pop(random_ind)

        # cut the original abstract state from all connections
    return progress, num_new_abstract_states, concrete_to_abstract, abstract_to_concrete, symmetry_abstract_states, \
           adjacency_list, inverse_adjacency_list, refinement_candidates, \
           target_parents, remaining_abstract_states, deleted_abstract_states, abstract_states_to_rtree_ids, \
           rtree_ids_to_abstract_states, \
           next_rtree_id_candidate


# def simulate_control(abstract_controller, )

def abstract_synthesis(Symbolic_reduced, sym_x, sym_u, state_dimensions, Target_low, Target_up,
                       Obstacle_low, Obstacle_up, X_low, X_up):
    t_start = time.time()
    n = state_dimensions.shape[1]
    p = index.Property()
    p.dimension = 3
    p.dat_extension = 'data'
    p.idx_extension = 'index'
    reachability_rtree_idx3d = index.Index('3d_index_abstract',
                                           properties=p)
    symmetry_under_approx_abstract_targets_rtree_idx3d = index.Index('3d_index_under_approx_abstract_targets',
                                                                     properties=p)
    symmetry_over_approx_abstract_targets_rtree_idx3d = index.Index('3d_index_over_approx_abstract_targets',
                                                                    properties=p)

    symbol_step = (X_up - X_low) / sym_x[0, :]

    targets, targets_rects, target_indices, obstacles, obstacles_rects, obstacle_indices = \
        create_targets_and_obstacles(Target_low, Target_up, Obstacle_low, Obstacle_up, symbol_step, sym_x, X_low)

    abstract_paths, reachable_rect_global_cntr, intersection_radius_threshold = \
        create_symmetry_abstract_reachable_sets(Symbolic_reduced, n, reachability_rtree_idx3d)

    matrix_dim_full = [np.prod(sym_x[0, :]), np.prod(sym_u), 2 * n]
    symbols_to_explore = np.setdiff1d(np.array(range(int(matrix_dim_full[0]))), target_indices)
    symbols_to_explore = np.setdiff1d(symbols_to_explore, obstacle_indices)

    # intersection_radius_threshold = intersection_radius_threshold * 10

    symmetry_transformed_targets_and_obstacles, \
    concrete_to_abstract, abstract_to_concrete, \
    symmetry_abstract_states, abstract_states_to_rtree_ids, \
    rtree_ids_to_abstract_states, next_rtree_id_candidate = create_symmetry_abstract_states(
        symbols_to_explore,
        symbol_step, targets,
        obstacles, sym_x, X_low,
        X_up,
        intersection_radius_threshold,
        symmetry_under_approx_abstract_targets_rtree_idx3d,
        symmetry_over_approx_abstract_targets_rtree_idx3d)

    # Now, create the edges in the discrete model
    # We add two to the dimensions of the adjacency matrix: one for the unsafe state and one for the target.
    adjacency_list, inverse_adjacency_list, target_parents, \
    concrete_edges, concrete_target_parents, inverse_concrete_edges, \
    abstract_edges = create_symmetry_abstract_transitions(
        Symbolic_reduced,
        abstract_paths,
        abstract_to_concrete,
        concrete_to_abstract,
        symmetry_abstract_states, symmetry_under_approx_abstract_targets_rtree_idx3d,
        abstract_states_to_rtree_ids,
        rtree_ids_to_abstract_states,
        symbol_step,
        targets_rects,
        target_indices,
        obstacles_rects,
        obstacle_indices,
        sym_x, X_low, X_up)

    t_abstraction = time.time() - t_start
    print(['Construction of symmetry-based abstraction took: ', t_abstraction, ' seconds'])
    num_abstract_states_before_refinement = len(abstract_to_concrete)

    '''
    grid_size_is_not_small_enough = True
    for candidate_controllable_state in concrete_target_parents:
        for u_ind in range(Symbolic_reduced.shape[1]):
            if concrete_edges[candidate_controllable_state][u_ind] and \
                    np.all(np.array(concrete_edges[candidate_controllable_state][u_ind]) == -1):
                grid_size_is_not_small_enough = False
                print("State ", candidate_controllable_state, " can reach the target with control ", u_ind)
                # break
        # if grid_size_is_not_small_enough:
        #    break
    if grid_size_is_not_small_enough:
        raise "Decrease grid size, no concrete state can be controlled to reach the origin."
    else:
        print("Grid size is good enough, for now.")
    '''
    controller = {}  # [-1] * len(abstract_to_concrete)
    t_synthesis_start = time.time()
    t_refine = 0
    t_synthesis = 0
    print('\n%s\tStart of the control synthesis\n', time.time() - t_start)
    refinement_itr = 0
    max_num_refinement_steps = 10000
    remaining_abstract_states = np.array(range(len(abstract_to_concrete)))
    controllable_abstract_states = []
    refinement_candidates = copy.deepcopy(target_parents)
    while refinement_itr < max_num_refinement_steps:
        temp_t_synthesis = time.time()
        controller, controllable_abstract_states_temp, unsafe_abstract_states, abstract_states_to_explore = \
            symmetry_abstract_synthesis_helper(abstract_to_concrete, remaining_abstract_states, adjacency_list,
                                               inverse_adjacency_list, target_parents, refinement_candidates,
                                               controllable_abstract_states,
                                               controller)
        t_synthesis += time.time() - temp_t_synthesis

        remaining_abstract_states = np.setdiff1d(remaining_abstract_states,
                                                 controllable_abstract_states_temp)
        target_parents = copy.deepcopy(abstract_states_to_explore)  # np.setdiff1d(np.array(target_parents),
        # controllable_abstract_states_temp).tolist()
        controllable_abstract_states.extend(controllable_abstract_states_temp)
        # for parent in target_parents:
        #    if len(abstract_to_concrete[parent]) > 1:
        refinement_candidates = np.setdiff1d(np.array(refinement_candidates), controllable_abstract_states).tolist()

        # refinement_candidates = copy.deepcopy(target_parents)
        # np.setdiff1d(np.array(copy.deepcopy(target_parents)), controllable_abstract_states_temp).tolist()
        temp_t_refine = time.time()
        progress, num_new_abstract_states, concrete_to_abstract, abstract_to_concrete, symmetry_abstract_states, adjacency_list, \
        inverse_adjacency_list, refinement_candidates, target_parents, remaining_abstract_states, deleted_abstract_states, \
        abstract_states_to_rtree_ids, \
        rtree_ids_to_abstract_states, \
        next_rtree_id_candidate = refine(
            concrete_to_abstract,
            abstract_to_concrete,
            abstract_edges,
            concrete_edges,
            inverse_concrete_edges,
            concrete_target_parents,
            symmetry_abstract_states,
            remaining_abstract_states,
            adjacency_list,
            inverse_adjacency_list,
            refinement_candidates,
            target_parents,
            controllable_abstract_states,
            unsafe_abstract_states,
            symmetry_transformed_targets_and_obstacles,
            symmetry_under_approx_abstract_targets_rtree_idx3d,
            abstract_states_to_rtree_ids,
            rtree_ids_to_abstract_states, next_rtree_id_candidate,
            symmetry_over_approx_abstract_targets_rtree_idx3d,
            abstract_paths,
            reachability_rtree_idx3d,
            sym_x,
            symbol_step, X_low,
            X_up,
            obstacles_rects,
            obstacle_indices,
            targets_rects,
            target_indices)
        t_refine += time.time() - temp_t_refine
        target_parents = np.setdiff1d(np.array(target_parents), controllable_abstract_states).tolist()
        target_parents = np.setdiff1d(np.array(target_parents), deleted_abstract_states).tolist()
        # target_parents = copy.deepcopy(target_parents)
        refinement_itr += num_new_abstract_states
        if not refinement_candidates:
            print("No states to refine anymore.")
            break
        print("progress: ", progress)
        # print("adjacency_list after refinement: ", adjacency_list)
        print("target parents: ", target_parents)
        # print("concrete target parents: ", concrete_target_parents)

    print(['Construction of symmetry-based abstraction took: ', t_abstraction, ' seconds'])
    print(
        ['Controller synthesis along with refinement for reach-avoid specification: ', time.time() - t_synthesis_start,
         ' seconds'])
    print(['Total time for symmetry abstraction-refinement-based controller synthesis'
           ' for reach-avoid specification: ', time.time() - t_synthesis_start + t_abstraction, ' seconds'])
    print(['Pure refinement took a total of: ', t_refine, ' seconds'])
    print(['Pure synthesis took a total of: ', t_synthesis, ' seconds'])
    print(['Number of splits of abstract states: ', refinement_itr])
    print(['Number of abstract states before refinement is: ', num_abstract_states_before_refinement])
    num_abstract_states = 0
    for abstract_s_idx in range(len(abstract_to_concrete)):
        if abstract_to_concrete[abstract_s_idx]:
            num_abstract_states += 1
    print(['Number of abstract states after refinement is: ', num_abstract_states])
    print(['Number of concrete states is: ', len(concrete_to_abstract)])
    if len(controllable_abstract_states):
        print(len(controllable_abstract_states), 'abstract symbols are controllable to satisfy the reach-avoid '
                                                 'specification\n')
        controllable_concrete_states = []
        for abstract_s in controllable_abstract_states:
            if not abstract_to_concrete[abstract_s]:
                print(abstract_s, " does not represent any concrete state, why is it controllable?")
            controllable_concrete_states.extend(abstract_to_concrete[abstract_s])
            print("Controllable abstract state ", abstract_s, " represents the following concrete states: ",
                  abstract_to_concrete[abstract_s])
        print(len(controllable_concrete_states), 'concrete symbols are controllable to satisfy the reach-avoid '
                                                 'specification\n')
    else:
        print('The reach-avoid specification cannot be satisfied from any initial state\n')
    plot_abstract_states(symmetry_abstract_states, deleted_abstract_states)

    # TODO: implement the refinement subroutine. Choose an action, check its reachable set, split the corresponding
    #  concrete states into two sets: one with this action being unsafe, the others are the rest. fix the adjacency
    #  list accordingly

    '''
    print('\n%s\tStart of the control synthesis\n', time.time() - t_start)
    controller = [-1] * len(abstract_to_concrete) # int(matrix_dim_full[0])
    u_ind = 0
    num_controllable_states = 0
    cur_target = targets[0]
    while True:  # len(traversal_stack) and fail_itr < M:
        num_new_symbols = 0
        temp_target_indices = []
        print("Checking which states can use the rectangle ", u_ind, " in the abstract rtree to reach the target")
        fewer_symbols_to_explore_hits = list(concrete_rtree_idx3d.nearest((cur_target[0, 0], cur_target[0, 1],
                                                                           cur_target[0, 2], cur_target[1, 0] + 0.01,
                                                                           cur_target[1, 1] + 0.01, cur_target[1, 2]
                                                                           + 0.01), num_results=100, objects=True))
        for hit in fewer_symbols_to_explore_hits:  # symbols_to_explore:
            s = hit.object
            result_avoid = False
            result_reach = False
            s_subscript = np.array(np.unravel_index(s, tuple((sym_x[0, :] + 1).astype(int))))
            curr_initset: np.array = np.row_stack((s_subscript * symbol_step + X_low,
                                                   s_subscript * symbol_step + symbol_step + X_low))
            if u_ind < len(abstract_paths):
                result_avoid, result_reach = check_one_step_abstract_reach_avoid(targets, obstacles,
                                                                                 abstract_targets_and_obstacles,
                                                                                 abstract_paths[u_ind], curr_initset, s)
            else:
                for curr_u_ind in range(matrix_dim_full[2]):
                    new_curr_initset = transform_to_frames(abstract_paths[curr_u_ind][-1][0, :],
                                                           abstract_paths[curr_u_ind][-1][1, :],
                                                           curr_initset[0, :], curr_initset[1, :])
                    new_indices = rect_to_indices(new_curr_initset, symbol_step, X_low,
                                                  sym_x[0, :], over_approximate=True)
                    if np.all(np.isin(new_indices, target_indices)):
                        abstract_rect_global_cntr = add_new_paths(controller, abstract_paths, new_indices, s,
                                                                  curr_u_ind, abstract_rtree_idx3d,
                                                                  abstract_rect_global_cntr)
                        break

            if result_avoid and result_reach:
                cur_target = get_convex_union([cur_target, curr_initset])
                controller[s] = u_ind
                temp_target_indices.append(s)
                concrete_rtree_idx3d.delete(s, (curr_initset[0, 0], curr_initset[0, 1], curr_initset[0, 2],
                                                curr_initset[1, 0], curr_initset[1, 1], curr_initset[1, 2]))
                num_new_symbols += 1

        if num_new_symbols:
            print(time.time() - t_start, " ", num_new_symbols,
                  " new controllable states have been found in this synthesis iteration\n")
            target_indices.extend(temp_target_indices)
            num_controllable_states += num_new_symbols
            symbols_to_explore = np.setdiff1d(symbols_to_explore, temp_target_indices)
            print(num_controllable_states, ' symbols are controllable to satisfy the reach-avoid specification\n')
        else:
            print('No new controllable state has been found in this synthesis iteration\n', time.time() - t_start)
            break

        u_ind += 1
        itr += 1

    print(['Controller synthesis for reach-avoid specification: ', time.time() - t_start, ' seconds'])
    if abstract_rect_global_cntr:
        print(len(abstract_paths), ' symbols are controllable to satisfy the reach-avoid specification\n')
    else:
        print('The reach-avoid specification cannot be satisfied from any initial state\n')

    plt.figure("Original coordinates")
    currentAxis = plt.gca()
    color = 'r'
    edge_color = 'k'
    for i in range(Obstacle_up.shape[0]):  # and np.any(rect_curr):
        rect = np.array([Obstacle_low[i, :], Obstacle_up[i, :]])
        rect_patch = Rectangle(rect[0, [0, 1]], rect[1, 0] - rect[0, 0],
                               rect[1, 1] - rect[0, 1], linewidth=1,
                               edgecolor=color, facecolor=color)
        currentAxis.add_patch(rect_patch)

    color = 'g'
    for target_idx in range(Target_low.shape[0]):
        rect = np.array([Target_low[target_idx, :], Target_up[target_idx, :]])
        rect_patch = Rectangle(rect[0, [0, 1]], rect[1, 0] - rect[0, 0],
                               rect[1, 1] - rect[0, 1], linewidth=1,
                               edgecolor=color, facecolor=color)
        currentAxis.add_patch(rect_patch)

    color = 'b'
    if len(result_abstract_paths) > 0:
        for initial_set, path in result_abstract_paths:
            for rect in path:
                rect = transform_to_frames(rect[0, :], rect[1, :], initial_set[0, :], initial_set[1, :])
                rect_patch = Rectangle(rect[0, [0, 1]], rect[1, 0] - rect[0, 0],
                                       rect[1, 1] - rect[0, 1], linewidth=1,
                                       edgecolor=edge_color, facecolor=color)
                currentAxis.add_patch(rect_patch)
            rect = transform_to_frames(path[-1][0, :], path[-1][1, :], initial_set[0, :],
                                       initial_set[1, :])
            print("last rect: ", rect)
            print("targets[-1]: ", targets[-1])
            print("Does the end of the path belong to the target? ", does_rect_contain(rect, targets[-1]))
            rect_patch = Rectangle(rect[0, [0, 1]], rect[1, 0] - rect[0, 0],
                                   rect[1, 1] - rect[0, 1], linewidth=1,
                                   edgecolor=edge_color, facecolor='m')
            currentAxis.add_patch(rect_patch)
    else:
        for path in abstract_paths:
            for rect in path:
                # rect = transform_to_frames(rect[0, :], rect[1, :], initial_set[0, :], initial_set[1, :])
                rect_patch = Rectangle(rect[0, [0, 1]], rect[1, 0] - rect[0, 0],
                                       rect[1, 1] - rect[0, 1], linewidth=1,
                                       edgecolor=edge_color, facecolor=color)
                currentAxis.add_patch(rect_patch)

    plt.ylim([X_low[1], X_up[1]])
    plt.xlim([X_low[0], X_up[0]])

    plt.figure("Resulting reachable sets in reduced coordinates")
    color = 'orange'
    currentAxis_1 = plt.gca()
    for path in abstract_paths:
        for rect in path:
            rect_patch = Rectangle(rect[0, [0, 1]], rect[1, 0] - rect[0, 0],
                                   rect[1, 1] - rect[0, 1], linewidth=1,
                                   edgecolor=edge_color, facecolor=color)
            currentAxis_1.add_patch(rect_patch)
    plt.ylim([-3, 3])
    plt.xlim([-3, 3])

    plt.show()
    '''


def synthesize(Symbolic_reduced, sym_x, sym_u, state_dimensions, Target_low, Target_up,
               Obstacle_low, Obstacle_up, X_low, X_up, U_low, U_up, N, M):
    n = state_dimensions.shape[1]
    itr = 0
    fail_itr = 0
    succ_itr = 0
    t_start = time.time()

    print('\n%s\tStart of the control synthesis\n', time.time() - t_start)

    color_initial = 'y'
    color_reach = 'b'
    color_target = 'm'
    p = index.Property()
    p.dimension = 3
    p.dat_extension = 'data'
    p.idx_extension = 'index'

    symbol_step = (X_up - X_low) / sym_x[0, :]
    quantized_key_range: np.array = np.floor(np.array([X_low, X_up]) / symbol_step)

    obstacles = []
    targets = []
    obstacle_indices = []
    target_indices = []
    for obstacle_idx in range(Obstacle_low.shape[0]):
        obstacle_rect = np.array([Obstacle_low[obstacle_idx, :], Obstacle_up[obstacle_idx, :]])
        obstacle_rect = fix_rect_angles(obstacle_rect)
        obstacle_poly = pc.box2poly(obstacle_rect.T)
        obstacles.append(obstacle_poly)
        obstacle_indices.extend(rect_to_indices(obstacle_rect, symbol_step, X_low,
                                                sym_x[0, :], over_approximate=True))

    for target_idx in range(Target_low.shape[0]):
        target_rect = np.array([Target_low[target_idx, :], Target_up[target_idx, :]])
        target_rect = fix_rect_angles(target_rect)
        # target_poly = pc.box2poly(target_rect.T)
        targets.append(target_rect)
        target_indices.extend(rect_to_indices(target_rect, symbol_step, X_low,
                                              sym_x[0, :], over_approximate=False))

    original_abstract_paths = []
    abstract_path_last_set_parts = []
    abstract_rtree_idx3d = index.Index('3d_index_abstract',
                                       properties=p)
    concrete_rtree_idx3d = index.Index('3d_index_concrete',
                                       properties=p)
    abstract_rect_global_cntr = 0
    abstract_paths = []
    for s_ind in range(Symbolic_reduced.shape[0]):
        abstract_path_last_set_parts.append([])
        original_abstract_paths.append([])
        for u_ind in range(Symbolic_reduced.shape[1]):
            abstract_path_last_set_parts[-1].append([])
            abstract_rect_low = Symbolic_reduced[s_ind, u_ind, np.arange(n), -1]
            abstract_rect_up = Symbolic_reduced[s_ind, u_ind, n + np.arange(n), -1]
            rect = np.array([abstract_rect_low, abstract_rect_up])
            rect = fix_rect_angles(rect)
            abstract_path_last_set_parts[-1][-1].append(rect)
            abstract_rtree_idx3d.insert(abstract_rect_global_cntr, (rect[0, 0], rect[0, 1],
                                                                    rect[0, 2], rect[1, 0],
                                                                    rect[1, 1], rect[1, 2]),
                                        obj=(s_ind, u_ind))
            abstract_rect_global_cntr += 1
            original_abstract_path = []
            for t_ind in range(Symbolic_reduced.shape[3]):
                rect = np.array([Symbolic_reduced[s_ind, u_ind, np.arange(n), t_ind],
                                 Symbolic_reduced[s_ind, u_ind, n + np.arange(n), t_ind]])
                rect = fix_rect_angles(rect)
                original_abstract_path.append(rect)

            original_abstract_paths[-1].append(copy.deepcopy(original_abstract_path))
            abstract_paths.append(original_abstract_path)

    # abstract_paths = copy.deepcopy(original_abstract_paths)
    # Target_up = np.array([[10, 6.5, 2 * math.pi / 3]])
    # Target_low = np.array([[7, 0, math.pi / 3]])
    # initial_set = np.array([[6, 2, math.pi / 2], [6.1, 2.1, math.pi / 2 + 0.01]])
    # traversal_stack = [copy.deepcopy(initial_set)]
    result_abstract_paths = []
    matrix_dim_full = [np.prod(sym_x[0, :]), np.prod(sym_u), 2 * n]
    # initial_set = traversal_stack.pop()
    print("matrix_dim_full: ", matrix_dim_full)
    symbols_to_explore = np.setdiff1d(np.array(range(int(matrix_dim_full[0]))), target_indices)
    symbols_to_explore = np.setdiff1d(symbols_to_explore, obstacle_indices)

    concrete_rect_global_cntr = 0
    for s in symbols_to_explore:
        s_subscript = np.array(np.unravel_index(s, tuple((sym_x[0, :] + 1).astype(int))))
        rect: np.array = np.row_stack((s_subscript * symbol_step + X_low,
                                       s_subscript * symbol_step + symbol_step + X_low))
        concrete_rtree_idx3d.insert(abstract_rect_global_cntr, (rect[0, 0], rect[0, 1],
                                                                rect[0, 2], rect[1, 0],
                                                                rect[1, 1], rect[1, 2]),
                                    obj=s)
        concrete_rect_global_cntr += 1

    abstract_targets_and_obstacles = []
    for i in range(int(matrix_dim_full[0])):
        abstract_targets_and_obstacles.append(None)
    controller = []
    for i in range(int(matrix_dim_full[0])):
        controller.append(-1)  # [-1] * int(matrix_dim_full[0])
    u_ind = 0
    num_controllable_states = 0
    cur_target = targets[0]
    while True:  # len(traversal_stack) and fail_itr < M:
        num_new_symbols = 0
        temp_target_indices = []
        print("Checking which states can use the rectangle ", u_ind, " in the abstract rtree to reach the target")
        fewer_symbols_to_explore_hits = list(concrete_rtree_idx3d.nearest((cur_target[0, 0], cur_target[0, 1],
                                                                           cur_target[0, 2], cur_target[1, 0] + 0.01,
                                                                           cur_target[1, 1] + 0.01, cur_target[1, 2]
                                                                           + 0.01), num_results=100, objects=True))
        for hit in fewer_symbols_to_explore_hits:  # symbols_to_explore:
            s = hit.object
            result_avoid = False
            result_reach = False
            s_subscript = np.array(np.unravel_index(s, tuple((sym_x[0, :] + 1).astype(int))))
            curr_initset: np.array = np.row_stack((s_subscript * symbol_step + X_low,
                                                   s_subscript * symbol_step + symbol_step + X_low))
            '''
            result, result_abstract_path = synthesize_helper(Symbolic_reduced, sym_x, sym_u,
                                                                           state_dimensions,
                                                                           targets, obstacles, X_low, X_up,
                                                                           abstract_rtree_idx3d,
                                                                           abstract_rect_global_cntr, abstract_paths,
                                                                           # abstract_path_last_set_parts,
                                                                           curr_initset, N)
            '''
            if u_ind < len(abstract_paths):
                result_avoid, result_reach = check_one_step_abstract_reach_avoid(targets, obstacles,
                                                                                 abstract_targets_and_obstacles,
                                                                                 abstract_paths[u_ind], curr_initset, s)
            else:
                for curr_u_ind in range(matrix_dim_full[2]):
                    new_curr_initset = transform_to_frames(abstract_paths[curr_u_ind][-1][0, :],
                                                           abstract_paths[curr_u_ind][-1][1, :],
                                                           curr_initset[0, :], curr_initset[1, :])
                    new_indices = rect_to_indices(new_curr_initset, symbol_step, X_low,
                                                  sym_x[0, :], over_approximate=True)
                    if np.all(np.isin(new_indices, target_indices)):
                        abstract_rect_global_cntr = add_new_paths(controller, abstract_paths, new_indices, s,
                                                                  curr_u_ind, abstract_rtree_idx3d,
                                                                  abstract_rect_global_cntr)
                        break

            if result_avoid and result_reach:
                cur_target = get_convex_union([cur_target, curr_initset])
                controller[s] = u_ind
                temp_target_indices.append(s)
                concrete_rtree_idx3d.delete(s, (curr_initset[0, 0], curr_initset[0, 1], curr_initset[0, 2],
                                                curr_initset[1, 0], curr_initset[1, 1], curr_initset[1, 2]))
                num_new_symbols += 1

        if num_new_symbols:
            print(time.time() - t_start, " ", num_new_symbols,
                  " new controllable states have been found in this synthesis iteration\n")
            target_indices.extend(temp_target_indices)
            num_controllable_states += num_new_symbols
            symbols_to_explore = np.setdiff1d(symbols_to_explore, temp_target_indices)
            print(num_controllable_states, ' symbols are controllable to satisfy the reach-avoid specification\n')
        else:
            print('No new controllable state has been found in this synthesis iteration\n', time.time() - t_start)
            break

        u_ind += 1

        '''
                dim = np.argmax(initial_set[1, :] - initial_set[0, :])
                temp = np.zeros(initial_set[0, :].shape)
                temp[dim] = (initial_set[1, dim] - initial_set[0, dim]) / 2
                new_initial_set_1 = np.array([initial_set[0, :], initial_set[1, :] - temp])
                new_initial_set_2 = np.array([initial_set[0, :] + temp, initial_set[1, :]])
                traversal_stack.append(new_initial_set_1)
                traversal_stack.append(new_initial_set_2)
                abstract_paths = []
                os.remove("3d_index_abstract.data")
                os.remove("3d_index_abstract.index")
                abstract_rtree_idx3d = index.Index('3d_index_abstract',
                                                   properties=p)
                print("abstract_rtree_idx3d: ", abstract_rtree_idx3d.leaves())
                abstract_rect_global_cntr = 0
                for s_ind in range(Symbolic_reduced.shape[0]):
                    # abstract_paths.append([])
                    for u_ind in range(Symbolic_reduced.shape[1]):
                        new_last_sets = []
                        while len(abstract_path_last_set_parts[s_ind][u_ind]):
                            old_last_set = abstract_path_last_set_parts[s_ind][u_ind].pop()
                            dim = np.argmax(old_last_set[1, :] - old_last_set[0, :])
                            if old_last_set[1, dim] - old_last_set[0, dim] <= partition_threshold:
                                new_last_sets.append(old_last_set)
                                temp_path = copy.deepcopy(original_abstract_paths[s_ind][u_ind])
                                temp_path.append(old_last_set)
                                abstract_paths.append(temp_path)
                                abstract_rtree_idx3d.insert(abstract_rect_global_cntr,
                                                            (old_last_set[0, 0], old_last_set[0, 1],
                                                             old_last_set[0, 2], old_last_set[1, 0],
                                                             old_last_set[1, 1], old_last_set[1, 2]),
                                                            obj=(s_ind, u_ind))
                                abstract_rect_global_cntr += 1
                                continue
                            temp = np.zeros(old_last_set[0, :].shape)
                            temp[dim] = (old_last_set[1, dim] - old_last_set[0, dim]) / 2
                            new_last_set_1 = np.array([old_last_set[0, :], old_last_set[1, :] - temp])
                            new_last_set_2 = np.array([old_last_set[0, :] + temp, old_last_set[1, :]])
                            new_last_sets.append(new_last_set_1)
                            new_last_sets.append(new_last_set_2)
                            # print("old_last_set: ", old_last_set)
                            # print("new_last_set_1: ", new_last_set_1)
                            # print("new_last_set_2: ", new_last_set_2)
                            abstract_rtree_idx3d.insert(abstract_rect_global_cntr,
                                                        (new_last_set_1[0, 0], new_last_set_1[0, 1],
                                                         new_last_set_1[0, 2], new_last_set_1[1, 0],
                                                         new_last_set_1[1, 1], new_last_set_1[1, 2]),
                                                        obj=(s_ind, u_ind))
                            new_abstract_path = copy.deepcopy(original_abstract_paths[s_ind][u_ind])
                            new_abstract_path.append(new_last_set_1)
                            abstract_paths.append(new_abstract_path)
                            abstract_rect_global_cntr += 1
                            abstract_rtree_idx3d.insert(abstract_rect_global_cntr,
                                                        (new_last_set_2[0, 0], new_last_set_2[0, 1],
                                                         new_last_set_2[0, 2], new_last_set_2[1, 0],
                                                         new_last_set_2[1, 1], new_last_set_2[1, 2]),
                                                        obj=(s_ind, u_ind))
                            new_abstract_path = copy.deepcopy(original_abstract_paths[s_ind][u_ind])
                            new_abstract_path.append(new_last_set_2)
                            abstract_paths.append(new_abstract_path)
                            abstract_rect_global_cntr += 1
    
                        # abstract_path_last_set_parts[s_ind][u_ind] = copy.deepcopy(new_last_sets)
    
                # break;
            print("# of RRT iterations so far: ", itr)
            '''
        itr += 1

    print(['Controller synthesis for reach-avoid specification: ', time.time() - t_start, ' seconds'])
    if abstract_rect_global_cntr:
        print(len(abstract_paths), ' symbols are controllable to satisfy the reach-avoid specification\n')
    else:
        print('The reach-avoid specification cannot be satisfied from any initial state\n')

    plt.figure("Original coordinates")
    currentAxis = plt.gca()
    color = 'r'
    edge_color = 'k'
    for i in range(Obstacle_up.shape[0]):  # and np.any(rect_curr):
        rect = np.array([Obstacle_low[i, :], Obstacle_up[i, :]])
        rect_patch = Rectangle(rect[0, [0, 1]], rect[1, 0] - rect[0, 0],
                               rect[1, 1] - rect[0, 1], linewidth=1,
                               edgecolor=color, facecolor=color)
        currentAxis.add_patch(rect_patch)

    color = 'g'
    for target_idx in range(Target_low.shape[0]):
        rect = np.array([Target_low[target_idx, :], Target_up[target_idx, :]])
        rect_patch = Rectangle(rect[0, [0, 1]], rect[1, 0] - rect[0, 0],
                               rect[1, 1] - rect[0, 1], linewidth=1,
                               edgecolor=color, facecolor=color)
        currentAxis.add_patch(rect_patch)

    color = 'b'
    if len(result_abstract_paths) > 0:
        for initial_set, path in result_abstract_paths:
            for rect in path:
                rect = transform_to_frames(rect[0, :], rect[1, :], initial_set[0, :], initial_set[1, :])
                rect_patch = Rectangle(rect[0, [0, 1]], rect[1, 0] - rect[0, 0],
                                       rect[1, 1] - rect[0, 1], linewidth=1,
                                       edgecolor=edge_color, facecolor=color)
                currentAxis.add_patch(rect_patch)
            rect = transform_to_frames(path[-1][0, :], path[-1][1, :], initial_set[0, :],
                                       initial_set[1, :])
            print("last rect: ", rect)
            print("targets[-1]: ", targets[-1])
            print("Does the end of the path belong to the target? ", does_rect_contain(rect, targets[-1]))
            rect_patch = Rectangle(rect[0, [0, 1]], rect[1, 0] - rect[0, 0],
                                   rect[1, 1] - rect[0, 1], linewidth=1,
                                   edgecolor=edge_color, facecolor='m')
            currentAxis.add_patch(rect_patch)
    else:
        for path in abstract_paths:
            for rect in path:
                # rect = transform_to_frames(rect[0, :], rect[1, :], initial_set[0, :], initial_set[1, :])
                rect_patch = Rectangle(rect[0, [0, 1]], rect[1, 0] - rect[0, 0],
                                       rect[1, 1] - rect[0, 1], linewidth=1,
                                       edgecolor=edge_color, facecolor=color)
                currentAxis.add_patch(rect_patch)
    '''
    for rect in discovered_rect:
        rect_patch = Rectangle(rect[0, [0, 1]], rect[1, 0] - rect[0, 0],
                               rect[1, 1] - rect[0, 1], linewidth=1,
                               edgecolor=edge_color, facecolor=color_reach)
        currentAxis.add_patch(rect_patch)

    for rect in initial_discovered_rect:
        rect_patch = Rectangle(rect[0, [0, 1]], rect[1, 0] - rect[0, 0],
                               rect[1, 1] - rect[0, 1], linewidth=1,
                               edgecolor=edge_color, facecolor=color_initial)
        currentAxis.add_patch(rect_patch)

    for rect in target_discovered_rect:
        rect_patch = Rectangle(rect[0, [0, 1]], rect[1, 0] - rect[0, 0],
                               rect[1, 1] - rect[0, 1], linewidth=1,
                               edgecolor=edge_color, facecolor=color_target)
        currentAxis.add_patch(rect_patch)

    color = 'y'
    print("obstacles_intersecting_rect: ", obstacles_intersecting_rect)
    for rect in obstacles_intersecting_rect:
        rect_patch = Rectangle(rect[0, [0, 1]], rect[1, 0] - rect[0, 0],
                               rect[1, 1] - rect[0, 1], linewidth=1,
                               edgecolor=edge_color, facecolor=color)
        # currentAxis.add_patch(rect_patch)
    '''

    plt.ylim([X_low[1], X_up[1]])
    plt.xlim([X_low[0], X_up[0]])

    plt.figure("Resulting reachable sets in reduced coordinates")
    color = 'orange'
    currentAxis_1 = plt.gca()
    for path in abstract_paths:
        for rect in path:
            rect_patch = Rectangle(rect[0, [0, 1]], rect[1, 0] - rect[0, 0],
                                   rect[1, 1] - rect[0, 1], linewidth=1,
                                   edgecolor=edge_color, facecolor=color)
            currentAxis_1.add_patch(rect_patch)
    '''
    for s_ind in range(Symbolic_reduced.shape[0]):
        for u_ind in range(Symbolic_reduced.shape[1]):
            for idx in range(Symbolic_reduced.shape[3]):
                abstract_rect_low = Symbolic_reduced[s_ind, u_ind, np.arange(n), idx]
                abstract_rect_up = Symbolic_reduced[s_ind, u_ind, n + np.arange(n), idx]
                rect_patch = Rectangle(abstract_rect_low[[0, 1]], abstract_rect_up[0] - abstract_rect_low[0],
                                       abstract_rect_up[1] - abstract_rect_low[1], linewidth=1,
                                       edgecolor=edge_color, facecolor=color)
                currentAxis_1.add_patch(rect_patch)
    '''
    plt.ylim([-3, 3])
    plt.xlim([-3, 3])

    plt.show()


def check_one_step_abstract_reach_avoid(targets, obstacles, abstract_targets_and_obstacles,
                                        abstract_path, curr_initset, s):
    debugging = False
    good_hit = False
    intersects_obstacle = False
    if abstract_targets_and_obstacles[s] is None:
        abstract_targets = []
        for target_rect in targets:
            abstract_target = transform_rect_to_abstract_frames(target_rect, curr_initset, over_approximate=False)
            abstract_target_over_approx = transform_rect_to_abstract_frames(target_rect, curr_initset,
                                                                            over_approximate=True)
            if abstract_target is not None:
                abstract_targets.append(abstract_target)
        if len(abstract_targets) == 0:
            if debugging:
                raise "Abstract target is empty"
        abstract_obstacles = []
        for obstacle_poly in obstacles:
            abstract_obstacle = transform_poly_to_abstract_frames(obstacle_poly, curr_initset, over_approximate=True)
            abstract_obstacles.append(abstract_obstacle)

        abstract_targets_and_obstacles[s] = AbstractState(abstract_targets, abstract_obstacles, s,
                                                          abstract_target_over_approx)

    abstract_state = abstract_targets_and_obstacles[s]
    for abstract_target in abstract_state.abstract_targets:
        if does_rect_contain(abstract_path[-1], abstract_target):
            good_hit = True
            break
    for rect in abstract_path:
        poly = pc.box2poly(rect.T)
        for abstract_obstacle in abstract_state.abstract_obstacles:
            if not pc.is_empty(pc.intersect(poly, abstract_obstacle)):
                intersects_obstacle = True
                break
    # if good_hit and not intersects_obstacle:
    return not intersects_obstacle, good_hit


def add_new_paths(controller, abstract_paths, new_indices, s, curr_u_ind, abstract_rtree_idx3d,
                  abstract_rect_global_cntr):
    new_path_prefix = copy.deepcopy(abstract_paths[curr_u_ind])
    new_path_suffix = []
    for new_s in range(new_indices):
        for idx, rect in enumerate(abstract_paths[controller[new_s]]):
            new_rect = transform_to_frames(rect[0, :], rect[1, :],
                                           new_path_prefix[-1][0, :],
                                           new_path_prefix[-1][1, :])
            if len(new_path_suffix) <= idx:
                new_path_suffix.append(new_rect)
            else:
                new_path_suffix[idx] = get_convex_union([new_path_suffix[idx], new_rect])
    new_path_prefix.extend(new_path_suffix)
    abstract_paths.append(new_path_prefix)
    result_rect = new_path_prefix[-1]
    abstract_rtree_idx3d.insert(abstract_rect_global_cntr, (result_rect[0, 0], result_rect[0, 1],
                                                            result_rect[0, 2], result_rect[1, 0],
                                                            result_rect[1, 1], result_rect[1, 2]),
                                obj=None)
    abstract_rect_global_cntr += 1
    return abstract_rect_global_cntr


'''
def synthesize_helper(Symbolic_reduced, sym_x, sym_u, state_dimensions,
                      X_low, X_up, abstract_rtree_idx3d, abstract_rect_global_cntr, targets,
                      obstacles, abstract_targets_and_obstacles,
                      abstract_paths, curr_initset, s):
    n = state_dimensions.shape[1]
    debugging = False
    print("new synthesize_helper call ")
    result_list = []
    if abstract_targets_and_obstacles[s] is None:
        abstract_targets = []
        for target_rect in targets:
            abstract_target = transform_rect_to_abstract_frames(target_rect, curr_initset, over_approximate=False)
            if abstract_target is not None:
                abstract_targets.append(abstract_target)

        if len(abstract_targets) == 0:
            if debugging:
                raise "Abstract target is empty"
            return False
        abstract_obstacles = []
        for obstacle_poly in obstacles:
            abstract_obstacle = transform_poly_to_abstract_frames(obstacle_poly, curr_initset, over_approximate=True)
            abstract_obstacles.append(abstract_obstacle)

        abstract_targets_and_obstacles[s] = AbstractState(abstract_targets, abstract_obstacles, s)

    abstract_state = abstract_targets_and_obstacles[s]
    hits_not_intersecting_obstacles = []
    good_hit = False
    intersects_obstacle = True
    for abstract_target_rect in abstract_state.abstract_targets:
        abstract_target_rect = fix_rect_angles(abstract_target_rect)
        target_hits = list(abstract_rtree_idx3d.nearest(
            (abstract_target_rect[0, 0], abstract_target_rect[0, 1], abstract_target_rect[0, 2],
             abstract_target_rect[1, 0] + 0.01, abstract_target_rect[1, 1] + 0.01,
             abstract_target_rect[1, 2]
             + 0.01), num_results=100, objects=True))
        closest_hit_idx = 0
        closest_hit_distance = 100
        for idx, hit in enumerate(target_hits):
            hit_bbox = np.array([hit.bbox[:n], hit.bbox[n:]])
            distance = np.linalg.norm(np.average(hit_bbox, axis=0) - np.average(abstract_target_rect), axis=0)
            if distance < closest_hit_distance:
                closest_hit_idx = idx
                closest_hit_distance = distance
        hit = target_hits[closest_hit_idx]
        good_hit = True
        intersects_obstacle = False
        new_curr_initset = transform_to_frames(abstract_paths[hit.id][-2][0, :],
                                               abstract_paths[hit.id][-2][1, :],
                                               curr_initset[0, :], curr_initset[1, :])
        if not does_rect_contain(abstract_paths[hit.id][-1], abstract_target_rect):
            good_hit = False
        path = abstract_paths[hit.id]
        for rect in path:
            poly = pc.box2poly(rect.T)
            for abstract_obstacle in abstract_state.abstract_obstacles:
                if not pc.is_empty(pc.intersect(poly, abstract_obstacle)):
                    # might be slow
                    # do_rects_inter(rect, abstract_obstacle):
                    intersects_obstacle = True
                    break
            if intersects_obstacle:
                break
        if good_hit and not intersects_obstacle:
            result_list.append((curr_initset, hit.id, abstract_paths[hit.id]))
            break
        if not intersects_obstacle:
            hits_not_intersecting_obstacles.append(hit)

    if good_hit and not intersects_obstacle:
        continue

    if len(hits_not_intersecting_obstacles) == 0 or max_path_length == 0:
        if debugging:
            print("All sampled paths intersect obstacles")
        return -1, None, []  # Failure

    if debugging:
        print("hits_not_intersecting_obstacles: ", len(hits_not_intersecting_obstacles))
    success = False
    for hit in hits_not_intersecting_obstacles:
        s_ind, u_ind = hit.object
        hit_id = hit.id
        # if len(abstract_path_last_set_parts[s_ind][u_ind]):
        # for last_set in abstract_path_last_set_parts[s_ind][u_ind]: TODO: this needs to be fixed, last_sets
        #  should be defined for all rectangles in abstract_rtree, not just the original ones temp,
        #  try this over-approximation fix:
        new_curr_initset = transform_to_frames(abstract_paths[hit.id][-1][0, :],
                                               abstract_paths[hit.id][-1][1, :],
                                               curr_initset[0, :], curr_initset[1, :])
        # new_curr_initset = transform_to_frames(last_set[0, :], last_set[1, :], curr_initset[0, :], curr_initset[1,
        # :])
        # result, result_abstract_path
        curr_result_list = synthesize_helper(Symbolic_reduced, sym_x, sym_u, state_dimensions,
                                             targets,
                                             obstacles, X_low, X_up, abstract_rtree_idx3d,
                                             abstract_rect_global_cntr, abstract_paths,
                                             # abstract_path_last_set_parts,
                                             new_curr_initset, max_path_length - 1)

        if curr_result_list:  # in range(len(abstract_paths))
            new_path_prefix = copy.deepcopy(abstract_paths[hit_id])
            new_path_suffix = []
            success = True
            for result_initset, hit_id, result_new_path in curr_result_list:
                # iterate over results and take the union of the paths.
                # print("Length of previous path: ", len(new_path))
                for idx, rect in enumerate(result_new_path):  # abstract_paths[result]:
                    # also this should be fixed, changed last_set to abstract_paths[hit.id][-1] temporarily
                    new_rect = transform_to_frames(rect[0, :], rect[1, :], abstract_paths[hit.id][-1][0, :],
                                                   abstract_paths[hit.id][-1][1, :])
                    if len(new_path_suffix) <= idx:
                        new_path_suffix.append(new_rect)
                    else:
                        new_path_suffix[idx] = get_convex_union([new_path_suffix[idx], new_rect])
            new_path_prefix.extend(new_path_suffix)
            if debugging:
                print("new path: ", new_path_prefix)
            abstract_paths.append(new_path_prefix)
            result_rect = new_path_prefix[-1]
            abstract_rtree_idx3d.insert(abstract_rect_global_cntr, (result_rect[0, 0], result_rect[0, 1],
                                                                    result_rect[0, 2], result_rect[1, 0],
                                                                    result_rect[1, 1], result_rect[1, 2]),
                                        obj=(s_ind, u_ind))
            abstract_rect_global_cntr += 1
        else:
            success = False
        if success:
            # print("length of abstract_paths[hid.id]: ", len(new_path))
            result_list.append((curr_initset, hit.id, new_path_prefix))
            break
            # return hit.id, new_path
    if not success:  # this part of the initial set cannot reach the target, then the whole initial set fails.
        return -1, None, []
    if np.all(curr_key == quantized_key_range[1, :]):
        break
    curr_key = next_quantized_key(curr_key, quantized_key_range)

    print("All sampled paths do not reach target")
    return -1, None, []
'''

'''
def synthesize_helper(Symbolic_reduced, sym_x, sym_u, state_dimensions, targets,
                      obstacles, X_low, X_up, abstract_rtree_idx3d, abstract_rect_global_cntr,
                      abstract_paths,  # abstract_path_last_set_parts,
                      initial_set, max_path_length):
    n = state_dimensions.shape[1]
    num_nearest_controls = int(Symbolic_reduced.shape[1] / 2)
    debugging = False
    target_aiming_prob = 0.8
    symbol_step = (X_up - X_low) / sym_x
    print("new synthesize_helper call ")
    print("initial_set: ", initial_set)
    print("max_path_length: ", max_path_length)
    abstract_targets = []

    result_list = []
    quantized_key_range: np.array = np.floor(initial_set / symbol_step)
    curr_key: np.array = quantized_key_range[0, :]
    while True:
        curr_initset: np.array = np.row_stack((curr_key * symbol_step, curr_key * symbol_step + symbol_step))
        for target_idx in range(len(targets)):
            target_rect = targets[target_idx]  # np.array([targets[target_idx][0, :], targets[target_idx][1, :]])
            # abstract_target_1 = transform_poly_to_abstract(target_poly, initial_set[0, :])
            # abstract_target_2 = transform_poly_to_abstract(target_poly, initial_set[1, :])
            # abstract_target = pc.intersect(abstract_target_1, abstract_target_2)
            # if not pc.is_empty(abstract_target):
            #    abstract_targets.append(abstract_target)
            #    print("abstract target: ", np.column_stack(abstract_target.bounding_box).T)
            # abstract_target_1 = transform_rect_to_abstract(target_rect, initial_set[0, :])
            # abstract_target_2 = transform_rect_to_abstract(target_rect, initial_set[1, :])
            # if do_rects_inter(abstract_target_1, abstract_target_2):
            #    abstract_target_up = np.minimum(abstract_target_1[1, :], abstract_target_2[1, :])
            #    abstract_target_low = np.maximum(abstract_target_1[0, :], abstract_target_2[0, :])
            #    abstract_targets.append(np.array([abstract_target_low, abstract_target_up]))
            abstract_target = transform_rect_to_abstract_frames(target_rect, curr_initset, over_approximate=False)
            if abstract_target is not None:
                abstract_targets.append(abstract_target)  # np.array([abstract_target_low, abstract_target_up])
                if debugging:
                    concrete_target = transform_to_frame(abstract_targets[-1], curr_initset[0, :],
                                                         overapproximate=False)
                    if not does_rect_contain(concrete_target, target_rect):
                        print("concrete_target: ", concrete_target)
                        print("original_target: ", target_rect)
                        print("Some error in transformation from and to abstract coordinates!!!")
                    concrete_target = transform_to_frame(abstract_targets[-1], curr_initset[1, :],
                                                         overapproximate=False)
                    if not does_rect_contain(concrete_target, target_rect):
                        print("concrete_target: ", concrete_target)
                        print("original_target: ", target_rect)
                        print("Some error in transformation from and to abstract coordinates!!!")

        if len(abstract_targets) == 0:
            if debugging:
                print("Abstract target is empty")
            return -1, None

        abstract_obstacles = []
        for obstacle_idx in range(len(obstacles)):
            obstacle_poly = obstacles[obstacle_idx]
            # np.array([Obstacle_low[obstacle_idx, :], Obstacle_up[obstacle_idx, :]])
            # abstract_obstacle_1 = transform_rect_to_abstract(obstacle_rect, curr_initset[0, :])
            # abstract_obstacle_2 = transform_rect_to_abstract(obstacle_rect, curr_initset[1, :])
            # abstract_obstacle_1 = transform_poly_to_abstract(obstacle_poly, curr_initset[0, :])
            # abstract_obstacle_2 = transform_poly_to_abstract(obstacle_poly, curr_initset[1, :])
            # obstacle_rect = np.column_stack(obstacle_poly.bounding_box).T
            abstract_obstacle = transform_poly_to_abstract_frames(obstacle_poly, curr_initset)
            # abstract_obstacles.append(pc.box2poly(abstract_obstacle.T))
            abstract_obstacles.append(abstract_obstacle)
            # abstract_obstacles.append(pc.union(abstract_obstacle_1, abstract_obstacle_2))
            if debugging:
                print("bounding box of abstract obstacle: ", np.column_stack(abstract_obstacles[-1].bounding_box).T)
            # print("abstract obstacle: ", abstract_obstacle)
            # abstract_obstacle_low = np.minimum(abstract_obstacle_1[1, :], abstract_obstacle_2[1, :])
            # abstract_obstacle_up = np.maximum(abstract_obstacle_1[0, :], abstract_obstacle_2[0, :])
            # abstract_obstacles.append(np.array([abstract_obstacle_low, abstract_obstacle_up]))

        hits_not_intersecting_obstacles = []
        good_hit = False
        intersects_obstacle = True
        for target_idx in range(len(abstract_targets)):
            # abstract_target_poly = abstract_targets[target_idx]
            # print("abstract_target_poly: ", abstract_target_poly)
            # abstract_target_rect = np.column_stack(abstract_target_poly.bounding_box).T
            # print("abstract_target_rect: ", abstract_target_rect)
            # if random.random() < target_aiming_prob:  # self.goal_sample_rate:
            #    target_idx = random.randint(0, len(targets))
            # else:
            #    sampled_state = np.zeros([]) + np.array([random.random() * ub for ub in
            #                                                         sampling_rectangle[1, :].tolist()])
            abstract_target_rect = abstract_targets[target_idx]
            abstract_target_rect = fix_rect_angles(abstract_target_rect)
            target_hits = list(abstract_rtree_idx3d.nearest(
                (abstract_target_rect[0, 0], abstract_target_rect[0, 1], abstract_target_rect[0, 2],
                 abstract_target_rect[1, 0] + 0.01, abstract_target_rect[1, 1] + 0.01,
                 abstract_target_rect[1, 2]
                 + 0.01), num_results=100, objects=True))
            closest_hit_idx = 0
            closest_hit_distance = 100
            for idx, hit in enumerate(target_hits):
                hit_bbox = np.array([hit.bbox[:n], hit.bbox[n:]])
                distance = np.linalg.norm(np.average(hit_bbox, axis=0) - np.average(abstract_target_rect), axis=0)
                if distance < closest_hit_distance:
                    closest_hit_idx = idx
                    closest_hit_distance = distance
            hit = target_hits[closest_hit_idx]
            if debugging:
                print("Abstract target: ", abstract_target_rect)
                print("Length of target_hits: ", len(target_hits))
                for hit in target_hits:
                    hit_bbox = np.array([hit.bbox[:n], hit.bbox[n:]])
                    print("Distance from abstract_tree to abstract target: ",
                          np.linalg.norm(np.average(hit_bbox, axis=0) - np.average(abstract_target_rect), axis=0))

                print("Number of abstract paths: ", len(abstract_paths))
                for path in abstract_paths:
                    print("Distance from an abstract_path to abstract target: ",
                          np.linalg.norm(np.average(path[-1], axis=0) - np.average(abstract_target_rect), axis=0))
            # hit = random.choice(target_hits)  # random.randint(0, len(target_hits))
            # hits = [target_hits[hit_idx]]
            # for hit in hits:
            good_hit = True
            intersects_obstacle = False
            new_curr_initset = transform_to_frames(abstract_paths[hit.id][-2][0, :],
                                                   abstract_paths[hit.id][-2][1, :],
                                                   curr_initset[0, :], curr_initset[1, :])
            # if not does_rect_contain(abstract_paths[hit.id][-2], abstract_target_rect):
            if not does_rect_contain(new_curr_initset, targets[target_idx]):
                good_hit = False
                if debugging:
                    print("target_rect ", targets[target_idx], " does not contain ", new_curr_initset)
            # else:
            # print("abstract_target_rect ", abstract_target_rect, " contains ",
            #      abstract_paths[hit.id][-2])
            # np.array([hit.bbox[:n], hit.bbox[n:]]))
            path = abstract_paths[hit.id]
            for rect in path:
                poly = pc.box2poly(rect.T)
                for abstract_obstacle in abstract_obstacles:
                    if not pc.is_empty(pc.intersect(poly, abstract_obstacle)):
                        # might be slow
                        # do_rects_inter(rect, abstract_obstacle):
                        intersects_obstacle = True
                        break
                if intersects_obstacle:
                    break
            if good_hit and not intersects_obstacle:
                if debugging:
                    print("reached target at depth ", max_path_length)
                    print("the last reachable set is: ", new_curr_initset)
                # abstract_paths[hit.id].append(abstract_target_rect)
                result_list.append((curr_initset, hit.id, abstract_paths[hit.id]))
                break
                # return hit.id, abstract_paths[hit.id]
            if not intersects_obstacle:
                # print("abstract_target_rect that is not reached: ", abstract_target_rect)
                hits_not_intersecting_obstacles.append(hit)

        if good_hit and not intersects_obstacle:
            continue

        if len(hits_not_intersecting_obstacles) == 0 or max_path_length == 0:
            if debugging:
                print("All sampled paths intersect obstacles")
            return -1, None, []  # Failure

        if debugging:
            print("hits_not_intersecting_obstacles: ", len(hits_not_intersecting_obstacles))
        success = False
        for hit in hits_not_intersecting_obstacles:
            s_ind, u_ind = hit.object
            hit_id = hit.id
            # if len(abstract_path_last_set_parts[s_ind][u_ind]):
            # for last_set in abstract_path_last_set_parts[s_ind][u_ind]: TODO: this needs to be fixed, last_sets
            #  should be defined for all rectangles in abstract_rtree, not just the original ones temp,
            #  try this over-approximation fix:
            new_curr_initset = transform_to_frames(abstract_paths[hit.id][-1][0, :],
                                                   abstract_paths[hit.id][-1][1, :],
                                                   curr_initset[0, :], curr_initset[1, :])
            # new_curr_initset = transform_to_frames(last_set[0, :], last_set[1, :], curr_initset[0, :], curr_initset[1,
            # :])
            # result, result_abstract_path
            curr_result_list = synthesize_helper(Symbolic_reduced, sym_x, sym_u, state_dimensions,
                                                 targets,
                                                 obstacles, X_low, X_up, abstract_rtree_idx3d,
                                                 abstract_rect_global_cntr, abstract_paths,
                                                 # abstract_path_last_set_parts,
                                                 new_curr_initset, max_path_length - 1)

            if curr_result_list:  # in range(len(abstract_paths))
                new_path_prefix = copy.deepcopy(abstract_paths[hit_id])
                new_path_suffix = []
                success = True
                for result_initset, hit_id, result_new_path in curr_result_list:
                    # iterate over results and take the union of the paths.
                    # print("Length of previous path: ", len(new_path))
                    for idx, rect in enumerate(result_new_path):  # abstract_paths[result]:
                        # also this should be fixed, changed last_set to abstract_paths[hit.id][-1] temporarily
                        new_rect = transform_to_frames(rect[0, :], rect[1, :], abstract_paths[hit.id][-1][0, :],
                                                       abstract_paths[hit.id][-1][1, :])
                        if len(new_path_suffix) <= idx:
                            new_path_suffix.append(new_rect)
                        else:
                            new_path_suffix[idx] = get_convex_union([new_path_suffix[idx], new_rect])
                new_path_prefix.extend(new_path_suffix)
                if debugging:
                    print("new path: ", new_path_prefix)
                abstract_paths.append(new_path_prefix)
                result_rect = new_path_prefix[-1]
                abstract_rtree_idx3d.insert(abstract_rect_global_cntr, (result_rect[0, 0], result_rect[0, 1],
                                                                        result_rect[0, 2], result_rect[1, 0],
                                                                        result_rect[1, 1], result_rect[1, 2]),
                                            obj=(s_ind, u_ind))
                abstract_rect_global_cntr += 1
            else:
                success = False
            if success:
                # print("length of abstract_paths[hid.id]: ", len(new_path))
                result_list.append((curr_initset, hit.id, new_path_prefix))
                break
                # return hit.id, new_path
        if not success:  # this part of the initial set cannot reach the target, then the whole initial set fails.
            return -1, None, []
        if np.all(curr_key == quantized_key_range[1, :]):
            break
        curr_key = next_quantized_key(curr_key, quantized_key_range)

    print("All sampled paths do not reach target")
    return -1, None, []
'''
