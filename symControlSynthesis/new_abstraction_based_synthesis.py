import numpy as np
import time
import math
from rtree import index
from typing import List

from z3 import *
import polytope as pc

import copy

import itertools
from scipy.spatial import ConvexHull

from qpsolvers import solve_qp

import matplotlib

matplotlib.use("macOSX")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, Polygon

import bisect

import csv

class AbstractState:

    def __init__(self, state_id, quantized_abstract_target, u_idx,
                 abstract_obstacles, concrete_state_indices, obstructed_u_idx_set):
        self.id = state_id
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
        raise TypeError("this function only takes polytopes")
    poly.bbox = None
    if verbose:
        print("working")
    return np.column_stack(poly.bounding_box).T


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
    rect[0, 2], rect[1, 2] = fix_angle_interval(rect[0, 2], rect[1, 2])
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
        if A[i, 2] == -1:
            i_low = i
    b_i_low_neg, b_i_up = fix_angle_interval(-1 * b[i_low], b[i_up])
    b[i_low] = -1 * b_i_low_neg
    b[i_up] = b_i_up
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
            if not pc.is_empty(p_inter_pos):
                inter_interval = get_intervals_intersection(-1 * b1[i_low_1], b1[i_up_1], -1 * b2[i_low_2], b2[i_up_2])
                if inter_interval is not None:
                    intervals = [inter_interval]
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


def get_poly_union(poly_1: pc.Region, poly_2: pc.Region, check_convex=False):
    return pc.union(poly_1, poly_2, check_convex=check_convex)


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
    if not do_intervals_intersect(a_s, b_s, a_l, b_l):
        return None
    a_s, b_s = fix_angle_interval(a_s, b_s)
    a_l, b_l = fix_angle_interval(a_l, b_l)

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
    a_s, b_s = fix_angle_interval(a_s, b_s)
    a_l, b_l = fix_angle_interval(a_l, b_l)
    if b_s - a_s >= 2 * math.pi - 0.001 or b_l - a_l >= 2 * math.pi - 0.001:
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
                
    result_list = []
    if order_matters:
        for interval in result:
            new_interval_l, new_interval_u = fix_angle_interval(interval[0], interval[1])
            new_intervals = get_decomposed_angle_intervals([new_interval_l, new_interval_u])
            for new_interval in new_intervals:
                result_list.append(new_interval)
                
    else:
        result_l, result_u = fix_angle_interval(result[0][0], result[0][1])
        result_list.append([result_l, result_u])
    return result_list


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

    sin_a, cos_a = abs(math.sin(angle)), abs(math.cos(angle))
    if side_short <= 2. * sin_a * cos_a * side_long or abs(sin_a - cos_a) < 1e-10:
        x = 0.5 * side_short
        wr, hr = (x / sin_a, x / cos_a) if width_is_longer else (x / cos_a, x / sin_a)
    else:
        cos_2a = cos_a * cos_a - sin_a * sin_a
        wr, hr = (w * cos_a - h * sin_a) / cos_2a, (h * cos_a - w * sin_a) / cos_2a

    return wr, hr


def transform_to_frame(rect: np.array, state: np.array, overapproximate=True):
    ang = state[2] 

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

    return result


def get_poly_list_convex_hull(poly_list: List[pc.Region]):
    result = poly_list[0]
    for ind in range(1, len(poly_list)):
        result = get_poly_union(result, poly_list[ind])
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
        interval = interval_list[0]
        for interval_idx in range(1, len(interval_list)):
            temp_interval_list = get_intervals_union(interval[0], interval[1],
                                                     interval_list[interval_idx][0],
                                                     interval_list[interval_idx][1])
            interval = temp_interval_list[0]
            
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

    poly_1 = transform_poly_to_abstract(concrete_poly_new, frames_rect[0, :], project_to_pos)
    poly_2 = transform_poly_to_abstract(concrete_poly_new, frames_rect[1, :], project_to_pos)
    poly_3 = transform_poly_to_abstract(concrete_poly_new, np.array([frames_rect[0, 0], frames_rect[0, 1],
                                                                     frames_rect[1, 2]]), project_to_pos)
    poly_4 = transform_poly_to_abstract(concrete_poly_new, np.array([frames_rect[1, 0], frames_rect[1, 1],
                                                                     frames_rect[0, 2]]), project_to_pos)

    if over_approximate:
        result = get_poly_list_convex_hull([poly_1, poly_2, poly_3, poly_4])
    else:
        result = get_poly_intersection(poly_1, poly_2, project_to_pos, check_convex) 
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

    for i in range(rect1.shape[1] - 1):
        if rect1[0, i] > rect2[1, i] + 0.01 or rect1[1, i] + 0.01 < rect2[0, i]:
            return False
    if not do_intervals_intersect(rect1[0, 2], rect1[1, 2], rect2[0, 2], rect2[1, 2]):
        return False
    return True


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


def subtract_rectangles(rect1, rect2):
    """
    Partially Generated using ChatGPT
    Subtract rect2 from rect1 and return the resulting rectangles
    """
    min_overlap = np.maximum(rect1[0, :], rect2[0, :])
    max_overlap = np.minimum(rect1[1, :], rect2[1, :])

    if np.any(max_overlap <= min_overlap):
        return [rect1]

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
strategy_2 = True #polls - 400
strategy_3 = False #polls + no closest
strategy_4 = False #polls -full + neighbors
strategy_5 = False #polls -400 + neighbors
strategy_6 = False #polls + no closest + neighbors // was it "polls-full"?
strategy_list = [strategy_1, strategy_2, strategy_3, strategy_4, strategy_5, strategy_6, benchmark]

def create_symmetry_abstract_states(symbols_to_explore, symbol_step, targets, targets_rects, target_indices, 
                                    obstacles, obstacles_rects, obstacle_indices, sym_x, X_low, X_up,
                                    reachability_rtree_idx3d, abstract_reachable_sets):
    t_start = time.time()
    print('\n%s\tStart of the symmetry abstraction \n', time.time() - t_start)
    symmetry_transformed_targets_and_obstacles = {}
    concrete_to_abstract = {}
    abstract_to_concrete = {}
    symmetry_abstract_states = []
    
    
    u_idx_to_abstract_states_indices = {}

    obstacle_state = AbstractState(0, None, None, [], [], set())
    symmetry_abstract_states.append(obstacle_state)
    abstract_to_concrete[0] = []

    next_abstract_state_id = 1
    if strategy_5 or strategy_2:
        threshold_num_results = 376
    elif strategy_1 or strategy_4:
        threshold_num_results = len(abstract_reachable_sets) + 1
    else:
        threshold_num_results = 4

    nearest_target_of_concrete = {}
    valid_hit_idx_of_concrete = {}

    concrete_edges = {}
    neighbor_map = {}

    get_concrete_transition_calls = 0

    for s in symbols_to_explore:
        s_subscript = np.array(np.unravel_index(s, tuple((sym_x[0, :]).astype(int))))
        s_rect: np.array = np.row_stack((s_subscript * symbol_step + X_low,
                                         s_subscript * symbol_step + symbol_step + X_low))
        s_rect[0, :] = np.maximum(X_low, s_rect[0, :])
        s_rect[1, :] = np.minimum(X_up, s_rect[1, :])

        abstract_targets_polys = []
        abstract_targets_rects = []
        abstract_targets_polys_over_approx = []
        abstract_targets_rects_over_approx = []
        abstract_pos_targets_polys = []

        for target_idx, target_poly in enumerate(targets):
            abstract_target_poly = transform_poly_to_abstract_frames(target_poly, s_rect, over_approximate=False)
            abstract_pos_target_poly = transform_poly_to_abstract_frames(target_poly, s_rect, over_approximate=False, project_to_pos=True)
            abstract_target_poly_over_approx = transform_poly_to_abstract_frames(
                target_poly, s_rect, over_approximate=True)
            if not pc.is_empty(abstract_target_poly):
                rc, x1 = pc.cheby_ball(abstract_target_poly)
                abstract_target_rect = np.array([x1 - rc, x1 + rc])
            elif not pc.is_empty(abstract_pos_target_poly):
                raise "abstract target is empty for a concrete state"

            else:
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
            raise "Abstract target is empty"

        abstract_obstacles = pc.Region(list_poly=[])
        for obstacle_poly in obstacles:
            abstract_obstacle = transform_poly_to_abstract_frames(obstacle_poly, s_rect,
                                                                  over_approximate=True) 
            abstract_obstacles = get_poly_union(abstract_obstacles, abstract_obstacle)

        symmetry_transformed_targets_and_obstacles[s] = RelCoordState(s, abstract_targets_polys,
                                                                      abstract_obstacles)
                

        min_dist = np.inf
        for curr_target_idx, curr_target_poly in enumerate(abstract_targets_polys):
            curr_nearest_point, curr_dist = nearest_point_to_the_origin(curr_target_poly)
            if min_dist > curr_dist:
                nearest_point = curr_nearest_point
                min_dist = curr_dist

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
                        if not hit_object in u_idx_to_abstract_states_indices:
                            rect = get_bounding_box(abstract_reachable_sets[hit_object][-1])
                            new_abstract_state = AbstractState(next_abstract_state_id,
                                                               np.average(rect , axis=0),
                                                                hit_object,
                                                                copy.deepcopy(symmetry_transformed_targets_and_obstacles[s].abstract_obstacles),
                                                                [s],
                                                                set([k for k, v in is_obstructed_u_idx.items() if v == True]))
                            symmetry_abstract_states.append(new_abstract_state)
                            concrete_to_abstract[s] = next_abstract_state_id
                            u_idx_to_abstract_states_indices[hit_object] = [next_abstract_state_id]
                            abstract_to_concrete[next_abstract_state_id] = [s]
                            next_abstract_state_id += 1
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
            add_concrete_state_to_symmetry_abstract_state(s, 0, pc.Region(list_poly=[]),
                symmetry_abstract_states, concrete_to_abstract, abstract_to_concrete, {})
            valid_hit_idx_of_concrete[s] = len(abstract_reachable_sets)
                
    print(['Done creation of symmetry abstract states in: ', time.time() - t_start, ' seconds'])
    print("concrete_to_abstract: ", len(concrete_to_abstract))
    print("abstract_to_concrete: ", len(abstract_to_concrete))
    print("concrete states deemed 'obstacle': ", len(symmetry_abstract_states[0].concrete_state_indices))
    return symmetry_transformed_targets_and_obstacles, concrete_to_abstract, abstract_to_concrete, symmetry_abstract_states, nearest_target_of_concrete, valid_hit_idx_of_concrete, concrete_edges, neighbor_map, get_concrete_transition_calls


def add_concrete_state_to_symmetry_abstract_state(curr_concrete_state_idx, abstract_state_id, symmetry_transformed_obstacles_curr,
                                                  symmetry_abstract_states, concrete_to_abstract, abstract_to_concrete, is_obstructed_u_idx):

    if abstract_state_id > 0:
        union_poly_obstacles = get_poly_union(symmetry_transformed_obstacles_curr,
                                            symmetry_abstract_states[abstract_state_id].abstract_obstacles, check_convex=False)
        symmetry_abstract_states[abstract_state_id].abstract_obstacles = union_poly_obstacles
        symmetry_abstract_states[abstract_state_id].obstructed_u_idx_set = \
            (symmetry_abstract_states[abstract_state_id].obstructed_u_idx_set).union(set([k for k, v in is_obstructed_u_idx.items() if v == True]))
    
    symmetry_abstract_states[abstract_state_id].concrete_state_indices.append(curr_concrete_state_idx)
    concrete_to_abstract[curr_concrete_state_idx] = abstract_state_id
    abstract_to_concrete[abstract_state_id].append(curr_concrete_state_idx)
        
    return


def get_concrete_transition(s_idx, u_idx, concrete_edges, neighbor_map,
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
            concrete_edges[(s_idx, u_idx)] = [-2]
            return [-2], True  
        for obstacle_rect in obstacles_rects:
            if do_rects_inter(obstacle_rect, concrete_succ):
                concrete_edges[(s_idx, u_idx)] = [-2]
                return [-2], True 
    reachable_rect = np.column_stack(pc.bounding_box(abstract_reachable_sets[u_idx][-1])).T
    concrete_succ = transform_to_frames(reachable_rect[0, :],
                                        reachable_rect[1, :],
                                        s_rect[0, :], s_rect[1, :])
    
    neighbors = rect_to_indices(concrete_succ, symbol_step, X_low, sym_x[0, :],
                                over_approximate=True).tolist()

    indices_to_delete = []
    for idx, succ_idx in enumerate(neighbors):
        if succ_idx in obstacle_indices:
            concrete_edges[(s_idx, u_idx)] = [-2]
            return [-2], True
        if succ_idx in target_indices:
            indices_to_delete.append(idx)
        

    if len(indices_to_delete) == len(neighbors):
        if not benchmark:
            concrete_edges[(s_idx, u_idx)] = [-1]
        return [-1], True

    if indices_to_delete:
        neighbors = np.delete(np.array(neighbors), np.array(indices_to_delete).astype(int)).tolist()
        neighbors.append(-1)

    if not benchmark:
        concrete_edges[(s_idx, u_idx)] = copy.deepcopy(neighbors)
    return set(neighbors), True


def plot_abstract_states(symmetry_abstract_states, deleted_abstract_states,
                         abstract_reachable_sets, state_to_paths_idx, abstract_to_concrete):
    obstacle_color = 'r'

    reach_color = 'b'
    quantized_target_color = 'g'
    indices_to_plot = np.array(range(len(symmetry_abstract_states)))
    indices_to_plot = np.setdiff1d(indices_to_plot, np.array(deleted_abstract_states)).tolist()

    for idx in indices_to_plot: 
        abstract_state = symmetry_abstract_states[idx]
        if abstract_state.id != 0:
            plt.figure("Abstract state: " + str(idx))
            currentAxis = plt.gca()
            obstructed_u_indices = abstract_state.obstructed_u_idx_set
            
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

            plt.ylim([-.7, .7])
            plt.xlim([-.35, .35])
            plt.savefig("Abstract state: " + str(idx))

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
        obstacles_rects.append(obstacle_rect)
        obstacle_poly = pc.box2poly(obstacle_rect.T)
        obstacles.append(obstacle_poly)
        temp_obstacle_indices = rect_to_indices(obstacle_rect, symbol_step, X_low,
                                                sym_x[0, :], over_approximate=True)
        for idx in temp_obstacle_indices:
            obstacle_indices.add(idx)

    for target_idx in range(Target_low.shape[0]):
        target_rect = np.array([Target_low[target_idx, :], Target_up[target_idx, :]])
        target_poly = pc.box2poly(target_rect.T)
        targets.append(target_poly)
        targets_rects.append(target_rect)
        temp_target_indices = rect_to_indices(target_rect, symbol_step, X_low,
                                              sym_x[0, :], over_approximate=False)
        for idx in temp_target_indices:
            target_indices.add(idx)

    return targets, targets_rects, target_indices, obstacles, obstacles_rects, obstacle_indices


def concrete_index_to_rect(concrete_state_idx, sym_x, symbol_step, X_low, X_up):
    concrete_subscript = np.array(
        np.unravel_index(concrete_state_idx, tuple((sym_x[0, :]).astype(int))))
    concrete_rect: np.array = np.row_stack(
        (concrete_subscript * symbol_step + X_low,
         concrete_subscript * symbol_step + symbol_step + X_low))
    concrete_rect[0, :] = np.maximum(X_low, concrete_rect[0, :])
    concrete_rect[1, :] = np.minimum(X_up, concrete_rect[1, :])
    return concrete_rect


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
    abstract_state_to_u_idx_poll = {} 
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

    while True:
        
        temp_controllable_concrete_states = set()

        num_new_symbols = 0
        
        for concrete_state_idx in concrete_states_to_explore:
            
            if (not concrete_state_idx in concrete_to_abstract) or concrete_to_abstract[concrete_state_idx] == 0:
                continue

            abstract_state_idx = concrete_to_abstract[concrete_state_idx]

            abstract_state = symmetry_abstract_states[abstract_state_idx]

            if not abstract_state_idx in abstract_state_to_u_idx_poll:
                if abstract_state_idx:
                    abstract_state_to_u_idx_poll[abstract_state_idx] = [(0, abstract_state.u_idx)]
                    abstract_state_to_u_idx_set[abstract_state_idx] = set([abstract_state.u_idx])
                else:
                    abstract_state_to_u_idx_poll[abstract_state_idx] = []
                    abstract_state_to_u_idx_set[abstract_state_idx] = set([])

            valid_vote = None
            for v, u_idx in abstract_state_to_u_idx_poll[abstract_state_idx]:

                next_concrete_state_indices, is_new_entry = get_concrete_transition(concrete_state_idx, u_idx, concrete_edges, neighbor_map,
                                                                    sym_x, symbol_step, abstract_reachable_sets,
                                                                    obstacles_rects, obstacle_indices, targets_rects,
                                                                    controllable_concrete_states, X_low, X_up, benchmark)
                if is_new_entry:
                    unique_state_u_pairs_explored += 1
                total_state_u_pairs_explored += 1
                
                is_controlled = (next_concrete_state_indices == [-1])

                if is_controlled:
                    abstract_state_to_u_idx_poll[abstract_state_idx].remove((v, u_idx))
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

                new_u_idx_found = False
                while curr_num_results < threshold_num_results:
                    
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
                
                                next_concrete_state_indices, is_new_entry = \
                                    get_concrete_transition(concrete_state_idx, hit_object, concrete_edges, neighbor_map,
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

        if num_new_symbols:
            print(time.time() - t_start, " ", num_new_symbols,
                  " new controllable states have been found in this synthesis iteration\n")
            num_controllable_states += num_new_symbols
            
            if strategy_4 or strategy_5 or strategy_6:
                rects = []
                
                for concrete_state_idx in temp_controllable_concrete_states:
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

    return concrete_controller, refinement_candidates, poll_lengths, average_ratio_neighbor_to_total, neighbor_map, unique_state_u_pairs_explored, total_state_u_pairs_explored, average_path_length, nb_iterations


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

    while True:
        temp_controllable_concrete_states = set()
        num_new_symbols = 0
        
        for concrete_state_idx in concrete_states_to_explore:

                    
            hits = list(range(len(abstract_reachable_sets)))
            
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

    return concrete_controller, neighbor_map, unique_state_u_pairs_explored, total_state_u_pairs_explored, average_path_length, nb_iterations


def get_decomposed_angle_intervals(original_angle_interval):
    decomposed_angle_intervals = []
    original_angle_interval[0], original_angle_interval[1] = \
        fix_angle_interval(original_angle_interval[0], original_angle_interval[1])
    if original_angle_interval[0] > original_angle_interval[1]:
        raise "how did an angle interval ended up with flipped order"
    while original_angle_interval[0] > 2 * math.pi and original_angle_interval[1] > 2 * math.pi:
        original_angle_interval[0] -= 2 * math.pi
        original_angle_interval[1] -= 2 * math.pi
    if original_angle_interval[0] < 2 * math.pi < original_angle_interval[1]:
        decomposed_angle_intervals.append([0, original_angle_interval[1] - 2 * math.pi])
        decomposed_angle_intervals.append([original_angle_interval[0], original_angle_interval[1]]) 
    else:
        decomposed_angle_intervals.append(original_angle_interval)
    return decomposed_angle_intervals


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

    return symmetry_transformed_targets_and_obstacles, concrete_to_abstract, abstract_to_concrete, \
        symmetry_abstract_states, nearest_target_of_concrete, valid_hit_idx_of_concrete, concrete_edges, neighbor_map, t_abstraction, get_concrete_transition_calls


def abstract_synthesis(U_discrete, time_step, W_low, W_up,
                       Symbolic_reduced, sym_x, sym_u, state_dimensions,
                       Target_low, Target_up, Obstacle_low, Obstacle_up, X_low, X_up, eng,
                       abstraction_data=None):

    xor_strategy = (sum([ int(strategy) for strategy in strategy_list]) == 1)
    if not xor_strategy:
        raise("Zero or multiple strategies were selected, please only select one")

    t_start = time.time()
    n = state_dimensions.shape[1]

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


    extended_target_rtree_idx3d = index.Index(
        '3d_index_extended_target',
        properties=p)

    symbol_step = (X_up - X_low) / sym_x[0, :]

    state_to_paths_idx = {}

    targets, targets_rects, target_indices, obstacles, obstacles_rects, obstacle_indices = \
        create_targets_and_obstacles(Target_low, Target_up, Obstacle_low, Obstacle_up, symbol_step, sym_x, X_low)

    for target in targets_rects:
        extended_target_rtree_idx3d.insert(-1, (
            target[0, 0], target[0, 1],
            target[0, 2], target[1, 0],
            target[1, 1], target[1, 2]))

    
    abstract_reachable_sets, reachable_rect_global_cntr, intersection_radius_threshold, per_dim_max_travelled_distance = \
        create_symmetry_abstract_reachable_sets(Symbolic_reduced, n, reachability_rtree_idx2d, reachability_rtree_idx3d)
    
    

    matrix_dim_full = [np.prod(sym_x[0, :]), np.prod(sym_u), 2 * n]
    symbols_to_explore = set(
        range(int(matrix_dim_full[0]))) 
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

    


    t_synthesis = 0
    print('\n%s\tStart of the control synthesis\n', time.time() - t_start)

    controllable_concrete_states = target_indices.copy()
    refinement_candidates = set()
    
    W_up = W_up.reshape((1, 3)).T
    W_low = W_low.reshape((1, 3)).T
    time_step = time_step.reshape((1, 3))
    temp_t_synthesis = time.time()
    concrete_controller = {}

    nb_explore = len(concrete_states_to_explore)

    if benchmark:
        concrete_controller, neighbor_map, unique_state_u_pairs_explored, \
            total_state_u_pairs_explored, average_path_length, nb_synthesis = symmetry_synthesis_helper(
            concrete_states_to_explore,
            concrete_edges,
            neighbor_map,
            abstract_reachable_sets,
            controllable_concrete_states,
            concrete_controller,
            per_dim_max_travelled_distance,
            obstacles_rects, obstacle_indices,
            targets_rects, X_low, X_up, sym_x, symbol_step)
        
    else:
        obstacle_indices = obstacle_indices.union(set(abstract_to_concrete[0]))

        concrete_controller, refinement_candidates, poll_lengths, average_ratio_neighbor_to_total, \
            neighbor_map, unique_state_u_pairs_explored, total_state_u_pairs_explored, \
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
            concrete_controller,
            reachability_rtree_idx3d, per_dim_max_travelled_distance, obstacles_rects, obstacle_indices,
            targets_rects, target_indices, X_low, X_up, sym_x, symbol_step)
        
        plot_abstract_states(symmetry_abstract_states, [], abstract_reachable_sets, state_to_paths_idx, abstract_to_concrete)

    t_synthesis += time.time() - temp_t_synthesis

    np.save('concrete_controller.npy', concrete_controller)
    np.save('neighbor_map.npy', neighbor_map)

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
        print("Strategy: polls + no closest + neighbors")
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
    print('Average path length: ', average_path_length)
    print('Number of synthesis iterations: ', nb_synthesis)
    print('Abstraction time: ', t_abstraction)
    print('Synthesis time: ', t_synthesis)
    print('Total time: ', time.time() - t_start)