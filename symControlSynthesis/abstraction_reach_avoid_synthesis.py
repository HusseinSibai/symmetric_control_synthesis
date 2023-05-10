import pdb

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
        self.abstract_targets_over_approximation = copy.deepcopy(abstract_targets)
        self.empty_abstract_target = empty_abstract_target
        self.set_of_abstract_target_directions = [copy.deepcopy(self.rtree_target_rect_under_approx)]
        # self.uncontrolled_region = copy.deepcopy(self.abstract_targets_over_approximation)


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
    if True:  # project_to_pos:
        return pc.union(poly_1, poly_2, check_convex=check_convex)
    else:
        result = pc.Region(list_poly=[])
        temp_result = pc.union(poly_1, poly_2, check_convex=check_convex)
        if type(temp_result) == pc.Polytope:
            temp_result = pc.Region(list_poly=[temp_result])
        for poly in temp_result.list_poly:
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
            original_interval = fix_angle_interval(-1 * b[i_low], b[i_up])
            interval_list = get_decomposed_angle_intervals(list(original_interval))
            # union_interval_list = get_intervals_union(-1 * b[i_low_1], b[i_up_1], -1 * b[i_low_2], b[i_up_2],
            #                                         order_matters=order_matters)
            for interval in interval_list:
                b_inter_left = -1 * interval[0]
                b_inter_right = interval[1]
                A_new = np.zeros((p_pos.A.shape[0] + 2, A.shape[1]))
                b_new = np.zeros((p_pos.A.shape[0] + 2,))
                for i in range(p_pos.A.shape[0]):
                    for j in range(p_pos.A.shape[1]):
                        A_new[i, j] = p_pos.A[i, j]
                    b_new[i] = p_pos.b[i]
                A_new[p_pos.A.shape[0], 2] = 1
                b_new[p_pos.A.shape[0]] = b_inter_right
                A_new[p_pos.A.shape[0] + 1, 2] = -1
                b_new[p_pos.A.shape[0] + 1] = b_inter_left
                result = pc.union(result, pc.Polytope(A_new, b_new), check_convex=check_convex)
    '''
    result = pc.Region(list_poly=[])
    if pc.is_empty(poly_1):
        return poly_2
    if pc.is_empty(poly_2):
        return poly_1
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
                union_interval_list = get_intervals_union(-1 * b1[i_low_1], b1[i_up_1], -1 * b2[i_low_2], b2[i_up_2],
                                                     order_matters=order_matters)
                if union_interval_list is not None:
                    for union_interval in union_interval_list:
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
    '''

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
        for interval_ind in range(1, len(interval_list)):
            # b_inter_left = -1 * interval[0]
            # b_inter_right = interval[1]
            # A_new = np.zeros((p_pos.A.shape[0] + 2, A.shape[1]))
            # b_new = np.zeros((p_pos.A.shape[0] + 2,))
            temp_interval_list = get_intervals_union(interval[0], interval[1],
                                                     interval_list[interval_ind][0],
                                                     interval_list[interval_ind][1])
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
    for rect_ind in range(rects.shape[0]):
        rect = rects[rect_ind, :, :]
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
                                    intersection_radius_threshold,
                                    symmetry_under_approx_abstract_targets_rtree_idx3d,
                                    reachability_rtree_idx3d):
    t_start = time.time()
    print('\n%s\tStart of the symmetry abstraction \n', time.time() - t_start)
    symmetry_transformed_targets_and_obstacles = {}  # [None] * int(matrix_dim_full[0])
    concrete_to_abstract = {}  # [None] * int(matrix_dim_full[0])
    abstract_to_concrete = []
    symmetry_abstract_states = []
    abstract_states_to_rtree_ids = {}
    rtree_ids_to_abstract_states = {}
    target_parents = {}  # set()
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
                # pdb.set_trace()
                raise "abstract target is empty for a concrete state"
                empty_abstract_target = True
                rc_pos, x1_pos = pc.cheby_ball(abstract_pos_target_poly)
                abstract_target_rect_pos = np.array([x1_pos - rc_pos, x1_pos + rc_pos])
                abstract_target_rect = np.array([[abstract_target_rect_pos[0, 0], abstract_target_rect_pos[0, 1], 0],
                                                 [abstract_target_rect_pos[1, 0], abstract_target_rect_pos[1, 1],
                                                  2 * math.pi]])
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
            abstract_obstacles = pc.union(abstract_obstacles, abstract_obstacle)  # get_poly_union
            # abstract_obstacles.append(abstract_obstacle)

        symmetry_transformed_targets_and_obstacles[s] = AbstractState(abstract_targets_polys,
                                                                      abstract_pos_targets_polys,
                                                                      abstract_obstacles, s,
                                                                      abstract_targets_rects_over_approx[0],
                                                                      empty_abstract_target)

        # Now adding the abstract state to a cluster --> combining abstract states with overlapping (abstract) targets
        added_to_existing_state = False
        if False:
            for curr_target_idx, curr_target_rect in enumerate(abstract_targets_rects):
                hits = list(symmetry_under_approx_abstract_targets_rtree_idx3d.intersection(
                    (curr_target_rect[0, 0], curr_target_rect[0, 1], curr_target_rect[0, 2],
                     curr_target_rect[1, 0], curr_target_rect[1, 1], curr_target_rect[1, 2]),
                    objects=True))
                if len(hits) >= 1:
                    hits = list(random.choices(hits, k=1))
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
                        new_abstract_state = add_concrete_state_to_symmetry_abstract_state(s, copy.deepcopy(
                            abstract_state),
                                                                                           symmetry_transformed_targets_and_obstacles)
                        '''
                        if pc.is_empty(get_poly_intersection(
                                symmetry_transformed_targets_and_obstacles[s].abstract_targets[0],
                                new_abstract_state.abstract_targets[0])):
                            print("A concrete state has a relative target that is not "
                                  "intersecting the relative target of the abstract state it was added to."
                                  " This shouldn't happen.")
                        '''
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
                                                                                      (
                                                                                          rtree_target_rect_under_approx_temp[
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
                        if rtree_ids_to_abstract_states[hits[max_rad_idx].id] not in target_parents:
                            original_concrete_angle_interval = [rtree_target_rect_under_approx[0, 2],
                                                                rtree_target_rect_under_approx[1, 2]]
                            decomposed_concrete_angle_intervals = get_decomposed_angle_intervals(
                                original_concrete_angle_interval)
                            for interval in decomposed_concrete_angle_intervals:
                                # abstract_targets_rects_over_approx[0]
                                hits_control = list(reachability_rtree_idx3d.intersection(
                                    (
                                        rtree_target_rect_under_approx[0, 0], rtree_target_rect_under_approx[0, 1],
                                        interval[0],
                                        rtree_target_rect_under_approx[1, 0], rtree_target_rect_under_approx[1, 1],
                                        interval[1])))
                                if len(hits_control):
                                    target_parents[rtree_ids_to_abstract_states[hits[max_rad_idx].id]] = {
                                        (None, hits_control[0].object)}
                                    break
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
                        symmetry_abstract_states[
                            rtree_ids_to_abstract_states[hits[max_rad_idx].id]] = new_abstract_state
                        concrete_to_abstract[s] = rtree_ids_to_abstract_states[
                            hits[max_rad_idx].id]  # hits[max_rad_idx].id
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
        if True:  # not added_to_existing_state:  # concrete_to_abstract[s] is None:
            # create a new abstract state since there isn't a current one suitable for s.
            new_abstract_state = AbstractState(copy.deepcopy(abstract_targets_polys),
                                               copy.deepcopy(abstract_pos_targets_polys),
                                               copy.deepcopy(abstract_obstacles), [s],
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
            ######################
            original_concrete_angle_interval = [new_abstract_state.rtree_target_rect_under_approx[0, 2],
                                                new_abstract_state.rtree_target_rect_under_approx[1, 2]]
            decomposed_concrete_angle_intervals = get_decomposed_angle_intervals(original_concrete_angle_interval)
            for interval in decomposed_concrete_angle_intervals:
                # abstract_targets_rects_over_approx[0]
                hits_control = list(reachability_rtree_idx3d.intersection(
                    (new_abstract_state.rtree_target_rect_under_approx[0, 0],
                     new_abstract_state.rtree_target_rect_under_approx[0, 1], interval[0],
                     new_abstract_state.rtree_target_rect_under_approx[1, 0],
                     new_abstract_state.rtree_target_rect_under_approx[1, 1], interval[1]), objects=True))
                if len(hits_control):
                    target_parents[len(symmetry_abstract_states)] = {(None, hits_control[0].object)}
                    break
            ######################
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
    '''
    for abstract_state_ind in range(len(abstract_to_concrete)):
        for concrete_state_idx in abstract_to_concrete[abstract_state_ind]:
            if pc.is_empty(get_poly_intersection(symmetry_transformed_targets_and_obstacles[concrete_state_idx].abstract_targets[0],
                                                 symmetry_abstract_states[abstract_state_ind].abstract_targets[0])):
                print("Intersection is wrong.")
    '''
    return symmetry_transformed_targets_and_obstacles, concrete_to_abstract, abstract_to_concrete, \
        symmetry_abstract_states, abstract_states_to_rtree_ids, rtree_ids_to_abstract_states, next_rtree_id_candidate, target_parents


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
            # pdb.set_trace()
            raise "empty intersection between the relative target of the abstract state" \
                  " and that of the concrete state it is being added to!"
            # inter_poly_1 = copy.deepcopy(concrete_state.abstract_targets_without_angles[target_idx])
            # inter_poly_2 = copy.deepcopy(abstract_state.abstract_targets_without_angles[target_idx])
        intersection_poly = get_poly_intersection(inter_poly_1, inter_poly_2, check_convex=False)  # pc.intersect
        while pc.is_empty(intersection_poly):
            # pdb.set_trace()
            if True:  # abstract_state.empty_abstract_target or concrete_state.empty_abstract_target:
                raise "empty abstract_target_poly error! Adding a concrete state" \
                      " to an abstract state without relative target intersection "
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
                                         abstract_states_to_rtree_ids, rtree_ids_to_abstract_states):
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
    abstract_to_concrete_edges = None
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
        # pdb.set_trace()
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


def get_abstract_transition(abstract_state_ind, u_ind,
                            concrete_to_abstract, abstract_to_concrete,
                            controllable_abstract_states,
                            concrete_edges,
                            sym_x, symbol_step, abstract_paths,
                            obstacles_rects, obstacle_indices, targets_rects, target_indices, X_low, X_up):
    neighbors = set()
    for concrete_state_ind in abstract_to_concrete[abstract_state_ind]:
        concrete_neighbors = get_concrete_transition(concrete_state_ind, u_ind, concrete_edges,
                                                     sym_x, symbol_step, abstract_paths,
                                                     obstacles_rects, obstacle_indices, targets_rects, target_indices,
                                                     X_low, X_up)
        if not concrete_neighbors:
            raise "Why this concrete state " + str(concrete_state_ind) + " has no neighbors?"
        for concrete_neighbor in concrete_neighbors:
            if concrete_neighbor == -1:  # or (concrete_neighbor >= 0 and concrete_to_abstract[concrete_neighbor] in
                # controllable_abstract_states):
                if -1 not in neighbors:
                    neighbors.add(-1)
            elif concrete_neighbor == -2:
                if -2 not in neighbors:
                    neighbors.add(-2)
            else:
                # elif concrete_to_abstract[concrete_neighbor] not in neighbors:
                neighbors.add(concrete_to_abstract[concrete_neighbor])
    if not neighbors:
        raise "No neighbors!"
    return neighbors


def get_concrete_transition(s_ind, u_ind, concrete_edges,
                            sym_x, symbol_step, abstract_paths,
                            obstacles_rects, obstacle_indices, targets_rects, target_indices, X_low, X_up):
    if (s_ind, u_ind) in concrete_edges:  # and concrete_edges[(s_ind, u_ind)]:  # concrete_edges[s_ind][u_ind]:
        return concrete_edges[(s_ind, u_ind)]
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
            concrete_edges[(s_ind, u_ind)] = [-2]
            return [-2]  # unsafe
        for obstacle_rect in obstacles_rects:
            if do_rects_inter(obstacle_rect, concrete_succ):
                concrete_edges[(s_ind, u_ind)] = [-2]
                return [-2]  # unsafe
    reachable_rect = np.column_stack(pc.bounding_box(abstract_paths[u_ind][-1])).T
    concrete_succ = transform_to_frames(reachable_rect[0, :],
                                        reachable_rect[1, :],
                                        s_rect[0, :], s_rect[1, :])
    for rect in targets_rects:
        if does_rect_contain(concrete_succ, rect):
            concrete_edges[(s_ind, u_ind)] = [-1]
            return [-1]  # reached target
    neighbors = rect_to_indices(concrete_succ, symbol_step, X_low, sym_x[0, :],
                                over_approximate=True).tolist()
    indices_to_delete = []
    for idx, succ_idx in enumerate(neighbors):
        if succ_idx in obstacle_indices:
            concrete_edges[(s_ind, u_ind)] = [-2]
            return [-2]
        if succ_idx in target_indices:
            indices_to_delete.append(idx)

    if len(indices_to_delete) == len(neighbors):
        concrete_edges[(s_ind, u_ind)] = [-1]
        return [-1]

    if indices_to_delete:
        neighbors = np.delete(np.array(neighbors), np.array(indices_to_delete).astype(int)).tolist()
        neighbors.append(-1)

    concrete_edges[(s_ind, u_ind)] = copy.deepcopy(neighbors)
    return set(neighbors)


def plot_abstract_states(symmetry_abstract_states, deleted_abstract_states,
                         abstract_paths, state_to_paths_ind, abstract_to_concrete):
    obstacle_color = 'r'
    target_color = 'g'
    reach_color = 'b'
    indices_to_plot = np.array(range(len(symmetry_abstract_states)))
    indices_to_plot = np.setdiff1d(indices_to_plot, np.array(deleted_abstract_states)).tolist()
    # indices_to_plot = state_to_paths_ind.keys()
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

        concrete_state_ind = random.choice(abstract_to_concrete[idx])
        if concrete_state_ind in state_to_paths_ind:  # equivalent to it being controllable
            for ind, region in enumerate(abstract_paths[state_to_paths_ind[concrete_state_ind]]):
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
        plt.ylim([-10, 10])
        plt.xlim([-10, 10])
        plt.savefig("Abstract state: " + str(idx))
        # plt.show()
        plt.cla()
        plt.close()


def plot_concrete_states(controllable_concrete_states, targets_rects, obstacles_rects,
                         state_to_paths_ind, sym_x, symbol_step, X_low, X_up):
    obstacle_color = 'r'
    target_color = 'g'
    reach_color = 'b'
    plt.figure("Controllable states")
    currentAxis = plt.gca()
    for idx in controllable_concrete_states:  # enumerate(symmetry_abstract_states)
        # print("Plotting abstract state: ", idx)
        for rect in obstacles_rects:
            rect_patch = Rectangle((rect[0, 0], rect[0, 1]), rect[1, 0] - rect[0, 0], rect[1, 1] - rect[0, 1],
                                   alpha=.5, color=obstacle_color, fill=True)
            currentAxis.add_patch(rect_patch)

        for rect in targets_rects:
            rect_patch = Rectangle((rect[0, 0], rect[0, 1]), rect[1, 0] - rect[0, 0], rect[1, 1] - rect[0, 1],
                                   alpha=.5, color=target_color, fill=True)
            currentAxis.add_patch(rect_patch)

        for concrete_state_ind in controllable_concrete_states:  # equivalent to it being controllable
            rect = concrete_index_to_rect(concrete_state_ind, sym_x, symbol_step, X_low, X_up)
            rect_patch = Rectangle((rect[0, 0], rect[0, 1]), rect[1, 0] - rect[0, 0], rect[1, 1] - rect[0, 1],
                                   alpha=.5, color=reach_color, fill=True)
            currentAxis.add_patch(rect_patch)
    plt.ylim([X_low[1] - 1, X_up[1] + 1])
    plt.xlim([X_low[0] - 1, X_up[0] + 1])
    plt.savefig("Controllable concrete space")
    # plt.show()
    plt.cla()
    plt.close()


def create_symmetry_abstract_reachable_sets(Symbolic_reduced, n, reachability_rtree_idx3d, controllable_region):
    reachable_rect_global_cntr = 0
    abstract_paths = []
    intersection_radius_threshold = None
    per_dim_max_travelled_distance = [0] * n
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
                                            obj=u_ind)
            abstract_initial_rect_low = Symbolic_reduced[s_ind, u_ind, np.arange(n), 0]
            abstract_initial_rect_up = Symbolic_reduced[s_ind, u_ind, n + np.arange(n), 0]
            initial_rect = np.array([abstract_initial_rect_low, abstract_initial_rect_up])
            initial_rect = fix_angle_interval_in_rect(initial_rect)
            for dim in range(n):
                per_dim_max_travelled_distance[dim] = max(per_dim_max_travelled_distance[dim],
                                                          abs(rect[0, dim] + rect[1, dim] -
                                                              initial_rect[0, dim] - initial_rect[1, dim]) / 2)
            reachable_rect_global_cntr += 1
            original_abstract_path = []
            for t_ind in range(Symbolic_reduced.shape[3]):
                rect = np.array([Symbolic_reduced[s_ind, u_ind, np.arange(n), t_ind],
                                 Symbolic_reduced[s_ind, u_ind, n + np.arange(n), t_ind]])
                rect = fix_angle_interval_in_rect(rect)
                poly = pc.box2poly(rect.T)
                original_abstract_path.append(poly)
            abstract_paths.append(original_abstract_path)
            reg = get_poly_list_with_decomposed_angle_intervals(original_abstract_path[-1])
            controllable_region = get_poly_union(controllable_region, reg)
    return abstract_paths, reachable_rect_global_cntr, intersection_radius_threshold, \
        np.array(per_dim_max_travelled_distance), controllable_region


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
    # target_poly_after_transition_under_approximation = transform_poly_to_abstract_frames(
    #    target, last_set, over_approximate=False)

    # origin = np.zeros((3, 1))
    # another_origin = np.zeros((3, 1))
    # another_origin[2] = 2 * math.pi
    # if (not pc.is_empty(target_poly_after_transition_under_approximation)) \
    #        and (target_poly_after_transition_under_approximation.contains(origin)
    #             or target_poly_after_transition_under_approximation.contains(another_origin)):
    #    return 2  # contained in target
    # return 0


def successor_in_or_intersects_target(abstract_state_ind, u_ind, abstract_paths,
                                      symmetry_abstract_states, concrete_to_abstract,
                                      abstract_to_concrete, controllable_abstract_states,
                                      concrete_transitions, abstract_transitions, inverse_abstract_transitions,
                                      sym_x, symbol_step,
                                      obstacles_rects, obstacle_indices, targets_rects, target_indices, X_low, X_up):
    if (abstract_state_ind, u_ind) not in abstract_transitions:  # or abstract_transitions[abstract_state_ind] is None:
        reachable_rect = np.column_stack(pc.bounding_box(abstract_paths[u_ind][-1])).T
        '''
        if symmetry_abstract_states[abstract_state_ind].empty_abstract_target or \
                not do_rects_inter(reachable_rect,
                                   symmetry_abstract_states[abstract_state_ind].rtree_target_rect_over_approx):
            return 0, abstract_transitions  # does not intersect target
        '''
        res = set_in_target(reachable_rect, symmetry_abstract_states[abstract_state_ind].abstract_targets[0])
        if res == 2:
            abstract_transitions[(abstract_state_ind, u_ind)] = [-1]
            return 2, abstract_transitions, inverse_abstract_transitions

        # if abstract_transitions[abstract_state_ind] is None:
        abstract_transitions[(abstract_state_ind, u_ind)] = get_abstract_transition(abstract_state_ind, u_ind,
                                                                                    concrete_to_abstract,
                                                                                    abstract_to_concrete,
                                                                                    controllable_abstract_states,
                                                                                    concrete_transitions,
                                                                                    sym_x, symbol_step, abstract_paths,
                                                                                    obstacles_rects, obstacle_indices,
                                                                                    targets_rects,
                                                                                    target_indices, X_low, X_up)

    else:
        for next_abstract_state_ind in abstract_transitions[(abstract_state_ind, u_ind)]:
            if not abstract_to_concrete[next_abstract_state_ind]:
                abstract_transitions[(abstract_state_ind, u_ind)] = get_abstract_transition(abstract_state_ind, u_ind,
                                                                                            concrete_to_abstract,
                                                                                            abstract_to_concrete,
                                                                                            controllable_abstract_states,
                                                                                            concrete_transitions,
                                                                                            sym_x, symbol_step,
                                                                                            abstract_paths,
                                                                                            obstacles_rects,
                                                                                            obstacle_indices,
                                                                                            targets_rects,
                                                                                            target_indices, X_low, X_up)
                break

    intersects_target = False
    contained_in_target = True
    for next_abstract_state_ind in abstract_transitions[(abstract_state_ind, u_ind)]:
        if next_abstract_state_ind not in inverse_abstract_transitions:
            inverse_abstract_transitions[next_abstract_state_ind] = {(abstract_state_ind, u_ind)}
        else:
            inverse_abstract_transitions[next_abstract_state_ind].add((abstract_state_ind, u_ind))
        if next_abstract_state_ind != -1 and next_abstract_state_ind not in controllable_abstract_states:
            contained_in_target = False
        else:
            intersects_target = True

    if contained_in_target:
        return 2, abstract_transitions, inverse_abstract_transitions
    if intersects_target:
        return 1, abstract_transitions, inverse_abstract_transitions
    '''
    print("abstract_state_ind: ", abstract_state_ind)
    target_poly_after_transition_over_approximation = transform_poly_to_abstract_frames(
        symmetry_abstract_states[abstract_state_ind].abstract_targets[0],
        reachable_rect,
        over_approximate=True)
    print("target_poly_after_transition_over_approximation: ", target_poly_after_transition_over_approximation)
    '''
    # for reg in controllable_regions_list:
    # if not pc.is_empty(target_poly_after_transition_over_approximation):
    # reg = get_poly_list_with_decomposed_angle_intervals(target_poly_after_transition_under_approximation)

    # uncontrolled_region = \
    #    pc.mldivide(target_poly_after_transition_over_approximation, controllable_region)
    # print("uncontrolled_region: ", uncontrolled_region)
    # if pc.is_empty(uncontrolled_region):
    #    return 2  # contained in target

    return 0, abstract_transitions, inverse_abstract_transitions


def successor_in_or_intersects_target_smt(concrete_initial_set, s_ind, path_ind, abstract_paths,
                                          state_to_paths_ind, cur_solver, var_dict,
                                          abstract_transitions, inverse_abstract_transitions,
                                          reachability_rtree_idx3d,
                                          extended_target_rtree_idx3d, n):
    '''
    reachable_set = [transform_to_frames(Symbolic_reduced[s_ind, u_ind, np.arange(n), -1],
                                         Symbolic_reduced[
                                             s_ind, u_ind, n + np.arange(n), -1],
                                         concrete_initial_set[0, :],
                                         concrete_initial_set[1, :])]
    '''
    # for t_ind in range(Symbolic_reduced.shape[3] - 1):  # the last reachable_rect
    #    # we get from the unified one to make sure all transformed reachable sets end up in the target.
    #    reachable_set.append(transform_to_frames(Symbolic_reduced[s_ind, u_ind, np.arange(n), t_ind],
    #                                             Symbolic_reduced[
    #                                                 s_ind, u_ind, n + np.arange(n), t_ind],
    #                                             concrete_initial_set[0, :], concrete_initial_set[1, :]))
    # discovered_rect.append(reachable_set[-1])
    concrete_reachable_poly = transform_poly_to_frames(abstract_paths[path_ind][-1], concrete_initial_set[0, :],
                                                       concrete_initial_set[1, :])
    '''
    abstract_reachable_rect = np.column_stack(pc.bounding_box(abstract_paths[path_ind][-1])).T
    reachable_set = [transform_to_frames(abstract_reachable_rect[0, :],
                                         abstract_reachable_rect[1, :],
                                         concrete_initial_set[0, :],
                                         concrete_initial_set[1, :])]
    '''
    reachable_set = [np.column_stack(pc.bounding_box(concrete_reachable_poly)).T]
    hits = list(
        extended_target_rtree_idx3d.intersection(
            (reachable_set[-1][0, 0], reachable_set[-1][0, 1], reachable_set[-1][0, 2],
             reachable_set[-1][1, 0], reachable_set[-1][1, 1],
             reachable_set[-1][1, 2]),
            objects=True))
    inter_num = len(hits)
    new_path_ind = path_ind
    if inter_num > 0:
        hits_rects = np.array([np.array([hit.bbox[:n], hit.bbox[n:]]) for hit in hits])
        cur_solver.reset()
        cur_solver = add_rects_to_solver(hits_rects, var_dict, cur_solver)
        uncovered_state = do_rects_list_contain_smt(reachable_set[-1], var_dict, cur_solver)
        if uncovered_state is None:
            new_abstract_path = copy.deepcopy(abstract_paths[path_ind])
            t_ind = 1  # the zero index is the origin and when transformed to the concrete coordinates
            # would lead to the last poly in abstract_paths[path_ind]
            t_limit_reached = False
            while True:  # for t_ind in range(len(abstract_paths[path_ind])):
                concrete_reachable_poly_t_ind_total = None
                for hit_ind, hit in enumerate(hits):
                    if hit.id >= 0:
                        hit_path_ind = state_to_paths_ind[hit.id]
                        if t_ind >= len(abstract_paths[hit_path_ind]):
                            t_limit_reached = True
                            break
                        concrete_reachable_poly_t_ind = transform_poly_to_frames(abstract_paths[path_ind][-1],
                                                                                 hits_rects[hit_ind][0, :],
                                                                                 hits_rects[hit_ind][1, :])
                        '''
                        abstract_reachable_rect = np.column_stack(
                            pc.bounding_box(abstract_paths[hit_path_ind][t_ind])).T
                        
                        hit_concrete_reachable_rect = transform_to_frames(abstract_reachable_rect[0, :],
                                                                          abstract_reachable_rect[1, :],
                                                                          hits_rects[hit_ind][0, :],
                                                                          hits_rects[hit_ind][1, :])
                        '''
                        if concrete_reachable_poly_t_ind_total is None:
                            concrete_reachable_poly_t_ind_total = concrete_reachable_poly_t_ind
                        else:
                            concrete_reachable_poly_t_ind_total = get_poly_list_convex_hull(
                                [concrete_reachable_poly_t_ind_total, concrete_reachable_poly_t_ind])
                if t_limit_reached or concrete_reachable_poly_t_ind_total is None:
                    break
                # concrete_reachable_rect_t_ind = np.column_stack(
                #    pc.bounding_box(concrete_reachable_poly_t_ind)).T
                # rect = fix_angle_interval_in_rect(concrete_reachable_rect)
                # poly = pc.box2poly(rect.T)
                abstract_poly = transform_poly_to_abstract_frames(concrete_reachable_poly_t_ind_total,
                                                                  concrete_initial_set, over_approximate=True)
                new_abstract_path.append(abstract_poly)
                t_ind += 1
            if t_ind > 1:
                abstract_paths.append(new_abstract_path)
                new_path_ind = len(abstract_paths) - 1
                last_rect = np.column_stack(pc.bounding_box(new_abstract_path[-1])).T
                reachability_rtree_idx3d.insert(new_path_ind, (last_rect[0, 0], last_rect[0, 1],
                                                               last_rect[0, 2], last_rect[1, 0],
                                                               last_rect[1, 1], last_rect[1, 2]),
                                                obj=new_path_ind)  # obj should be the control associated with
                # path_ind instead
            else:
                new_path_ind = path_ind
            return 2, new_path_ind, abstract_transitions, inverse_abstract_transitions
        return 1, new_path_ind, abstract_transitions, inverse_abstract_transitions
    return 0, new_path_ind, abstract_transitions, inverse_abstract_transitions


def successor_avoids_obstacles(abstract_state_ind, u_ind, abstract_paths, symmetry_abstract_states):
    for t_ind in range(len(abstract_paths[u_ind])):
        if not pc.is_empty(get_poly_intersection(abstract_paths[u_ind][t_ind],
                                                 symmetry_abstract_states[abstract_state_ind].abstract_obstacles,
                                                 check_convex=False)):
            return False
    return True


def concrete_index_to_rect(concrete_state_ind, sym_x, symbol_step, X_low, X_up):
    concrete_subscript = np.array(
        np.unravel_index(concrete_state_ind, tuple((sym_x[0, :]).astype(int))))
    concrete_rect: np.array = np.row_stack(
        (concrete_subscript * symbol_step + X_low,
         concrete_subscript * symbol_step + symbol_step + X_low))
    concrete_rect[0, :] = np.maximum(X_low, concrete_rect[0, :])
    concrete_rect[1, :] = np.minimum(X_up, concrete_rect[1, :])
    return concrete_rect


def symmetry_abstract_synthesis_helper_old(local_abstract_states_to_explore,
                                           abstract_states_to_explore,
                                           abstract_to_concrete,
                                           concrete_to_abstract,
                                           abstract_paths,
                                           symmetry_abstract_states, target_parents,
                                           refinement_candidates,
                                           global_controllable_abstract_states,
                                           reachable_target_region,
                                           abstract_transitions,
                                           inverse_abstract_transitions,
                                           concrete_transitions,
                                           controller, reachability_rtree_idx3d,
                                           reachable_rect_global_cntr,
                                           sym_x, symbol_step,
                                           obstacles_rects, obstacle_indices,
                                           targets_rects, target_indices, X_low, X_up):
    t_start = time.time()
    num_controllable_states = len(global_controllable_abstract_states)
    controllable_abstract_states = set()
    unsafe_abstract_states = []
    # newly_added_rects_lower_bound = 0
    # temp_controller = {}
    n = X_up.shape[0]
    while True:
        num_new_symbols = 0
        temp_controllable_abstract_states = set()
        # newly_added_rects_lower_bound_temp = 0
        for abstract_state_ind in local_abstract_states_to_explore:  # target_parents: # abstract_states_to_explore
            if not abstract_to_concrete[abstract_state_ind]:
                continue
            # for u_ind in range(len(abstract_paths)):
            '''
            if abstract_state_ind in temp_controller and not abstract_transitions[abstract_state_ind][temp_controller[abstract_state_ind]] is None:
                for next_abstract_state_ind in abstract_transitions[abstract_state_ind][temp_controller[abstract_state_ind]]:
                    if next_abstract_state_ind not in controllable_abstract_states:
            '''
            # if abstract_state_ind not in temp_controller:
            # rect = symmetry_abstract_states[abstract_state_ind].rtree_target_rect_under_approx
            hits = []
            for rect in symmetry_abstract_states[abstract_state_ind].set_of_abstract_target_directions:
                original_angle_interval = [rect[0, 2], rect[1, 2]]
                decomposed_angle_intervals = get_decomposed_angle_intervals(original_angle_interval)
                for interval in decomposed_angle_intervals:
                    # change this to nearest to the under approximation of the target, much faster
                    hits.extend(list(reachability_rtree_idx3d.nearest(
                        (rect[0, 0], rect[0, 1], interval[0], rect[1, 0], rect[1, 1], interval[1]), num_results=3,
                        objects=True)))
            unique_controls = set()
            for hit in hits:
                # if hit.id >= newly_added_rects_lower_bound and do_rects_inter(np.array([hit.bbox[:n],
                #                                                                        hit.bbox[n:]]), ):
                unique_controls.add(hit.object)
            if unique_controls:
                unique_controls = set(random.choices(list(unique_controls), k=3))
            # else:
            #    unique_controls = set()
            #    unique_controls.add(temp_controller[abstract_state_ind])
            if unique_controls:
                for u_ind in unique_controls:
                    reach_res, abstract_transitions, inverse_abstract_transitions = successor_in_or_intersects_target(
                        abstract_state_ind, u_ind,
                        abstract_paths,
                        symmetry_abstract_states,
                        concrete_to_abstract,
                        abstract_to_concrete,
                        global_controllable_abstract_states,
                        concrete_transitions,
                        abstract_transitions,
                        inverse_abstract_transitions,
                        sym_x, symbol_step,
                        obstacles_rects,
                        obstacle_indices,
                        targets_rects,
                        target_indices, X_low, X_up)

                    # avoid_res = successor_avoids_obstacles(abstract_state_ind, u_ind, abstract_paths,
                    #                                       symmetry_abstract_states)

                    if reach_res == 2:  # and avoid_res:
                        controller[abstract_state_ind] = u_ind
                        temp_controllable_abstract_states.add(abstract_state_ind)
                        global_controllable_abstract_states.add(abstract_state_ind)
                        reg = get_poly_list_with_decomposed_angle_intervals(
                            symmetry_abstract_states[abstract_state_ind].abstract_targets[0])
                        reachable_target_region = get_poly_union(reachable_target_region, reg)
                        if abstract_state_ind in inverse_abstract_transitions:
                            for parent_abstract_state_ind, parent_u_ind in inverse_abstract_transitions[
                                abstract_state_ind]:
                                if parent_abstract_state_ind != abstract_state_ind \
                                        and parent_abstract_state_ind in abstract_states_to_explore:
                                    if parent_abstract_state_ind in target_parents:
                                        target_parents[parent_abstract_state_ind].add(
                                            (abstract_state_ind, parent_u_ind))
                                    else:
                                        target_parents[parent_abstract_state_ind] = {(abstract_state_ind, parent_u_ind)}
                                    if len(abstract_to_concrete[parent_abstract_state_ind]) > 1:
                                        refinement_candidates.add(parent_abstract_state_ind)
                        '''
                        rect_under_approx = symmetry_abstract_states[abstract_state_ind].rtree_target_rect_under_approx
                        reachability_rtree_idx3d.insert(reachable_rect_global_cntr, (
                            rect_under_approx[0, 0], rect_under_approx[0, 1], rect_under_approx[0, 2],
                            rect_under_approx[1, 0], rect_under_approx[1, 1], rect_under_approx[1, 2]), obj=u_ind)
                        '''
                        num_new_symbols += 1
                        # if newly_added_rects_lower_bound_temp == 0:
                        #    newly_added_rects_lower_bound_temp = reachable_rect_global_cntr
                        # reachable_rect_global_cntr += 1
                        if abstract_state_ind in refinement_candidates:
                            refinement_candidates.remove(abstract_state_ind)
                        break
                    elif reach_res == 1:  # and abstract_s not in target_parents:
                        if abstract_state_ind not in target_parents:
                            target_parents[abstract_state_ind] = {(None, u_ind)}
                        if len(abstract_to_concrete[abstract_state_ind]) > 1:
                            refinement_candidates.add(abstract_state_ind)
                        for next_abstract_state in abstract_transitions[(abstract_state_ind, u_ind)]:
                            if next_abstract_state in global_controllable_abstract_states or \
                                    next_abstract_state in controllable_abstract_states or \
                                    next_abstract_state in temp_controllable_abstract_states:
                                target_parents[abstract_state_ind].add((next_abstract_state, u_ind))
                            elif next_abstract_state >= 0 and len(abstract_to_concrete[next_abstract_state]) > 1:
                                refinement_candidates.add(next_abstract_state)
                    if global_controllable_abstract_states:
                        other_abstract_state_ind = random.choice(tuple(global_controllable_abstract_states))
                        concrete_target_index = random.choice(abstract_to_concrete[other_abstract_state_ind])
                        concrete_target_rect: np.array = concrete_index_to_rect(concrete_target_index, sym_x,
                                                                                symbol_step, X_low, X_up)
                        concrete_initial_set_index = random.choice(abstract_to_concrete[abstract_state_ind])
                        concrete_initial_set_rect: np.array = concrete_index_to_rect(concrete_initial_set_index,
                                                                                     sym_x, symbol_step,
                                                                                     X_low, X_up)
                        abstract_target_direction_rect = \
                            transform_rect_to_abstract_frames(concrete_target_rect,
                                                              concrete_initial_set_rect, over_approximate=True)
                        symmetry_abstract_states[abstract_state_ind].set_of_abstract_target_directions.append(
                            abstract_target_direction_rect)

        if num_new_symbols:
            print(time.time() - t_start, " ", num_new_symbols,
                  " new controllable states have been found in this synthesis iteration\n")
            controllable_abstract_states = controllable_abstract_states.union(temp_controllable_abstract_states)
            num_controllable_states += num_new_symbols
            # newly_added_rects_lower_bound = newly_added_rects_lower_bound_temp
            # newly_added_rects_lower_bound_temp = 0
            refinement_candidates = refinement_candidates.difference(temp_controllable_abstract_states)
            # target_parents = target_parents.difference(temp_controllable_abstract_states)
            for abstract_state_ind in temp_controllable_abstract_states:
                if abstract_state_ind in target_parents:
                    del target_parents[abstract_state_ind]
            temp_controllable_abstract_states = list(temp_controllable_abstract_states)
            local_abstract_states_to_explore = copy.deepcopy(set(target_parents.keys()))
            # if first_call:
            #    abstract_states_to_explore = copy.deepcopy(target_parents)
            # else:
            abstract_states_to_explore = abstract_states_to_explore.difference(temp_controllable_abstract_states)
            # np.setdiff1d(abstract_states_to_explore, temp_controllable_abstract_states).tolist()
            # np.setdiff1d(refinement_candidates, temp_controllable_abstract_states).tolist()
            print(num_controllable_states, ' symbols are controllable to satisfy the reach-avoid specification\n')
        else:
            print('No new controllable state has been found in this synthesis iteration\n', time.time() - t_start)
            break

    return controller, controllable_abstract_states, unsafe_abstract_states, local_abstract_states_to_explore, \
        abstract_states_to_explore, reachable_rect_global_cntr, target_parents, refinement_candidates


def symmetry_abstract_synthesis_helper(local_abstract_states_to_explore,
                                       concrete_states_to_explore,
                                       abstract_to_concrete,
                                       concrete_to_abstract,
                                       Symbolic_reduced,
                                       abstract_paths,
                                       symmetry_abstract_states, target_parents,
                                       refinement_candidates,
                                       global_controllable_abstract_states,
                                       reachable_target_region,
                                       abstract_transitions,
                                       inverse_abstract_transitions,
                                       concrete_transitions,
                                       controller, reachability_rtree_idx3d,
                                       reachable_rect_global_cntr,
                                       sym_x, symbol_step,
                                       obstacles_rects, obstacle_indices,
                                       targets_rects, target_indices, extended_target_rtree_idx3d, target_rtree_idx,
                                       cur_solver, var_dict,
                                       state_to_paths_ind, per_dim_max_travelled_distance,
                                       X_low, X_up):
    t_start = time.time()
    num_controllable_states = len(global_controllable_abstract_states)
    controllable_abstract_states = set()
    controllable_concrete_states = set()
    unsafe_abstract_states = []
    num_nearest_targets_to_consider = 1
    num_nearest_reachsets_to_consider = 5
    # newly_added_rects_lower_bound = 0
    # temp_controller = {}
    abstraction_level = 1
    n = X_up.shape[0]
    # targets_queue = targets_rects
    abstract_states_to_explore = {}
    while True:
        num_new_symbols = 0
        temp_controllable_abstract_states = set()
        # for now the folowing comment is an incomplete thought and should be considered useless.
        # here we should only iterate over the symbols that are within
        # "abstract_paths" distances from the newly added targets
        # have to instantiate an rtree that stores the bounding boxes of the relative targets
        # of all concrete states. Then, whenever a set of new concrete states are added to the extended target set,
        # go over all the concrete states again, search for the nearest extended target,
        # transform the target with resp
        for concrete_state_ind in concrete_states_to_explore:
            # abstract_state_ind in abstract_states_to_explore:  # target_parents: # abstract_states_to_explore
            # if not abstract_to_concrete[abstract_state_ind]:
            #    continue
            # concrete_initial_set_index = random.choice(abstract_to_concrete[abstract_state_ind])
            if concrete_state_ind in obstacle_indices \
                    or concrete_state_ind in target_indices or concrete_state_ind in controllable_concrete_states:
                continue
            abstract_state_ind = concrete_to_abstract[concrete_state_ind]
            rect: np.array = concrete_index_to_rect(concrete_state_ind,
                                                    sym_x, symbol_step,
                                                    X_low, X_up)
            # the nearest is in euclidean distance which might be the problem
            hits = list(extended_target_rtree_idx3d.nearest(
                (rect[0, 0], rect[0, 1], rect[0, 2], rect[1, 0], rect[1, 1], rect[1, 2]),
                num_results=num_nearest_targets_to_consider, objects=True))
            # symmetry_abstract_states[abstract_state_ind].set_of_abstract_target_directions = []
            reach_hits = []
            for idx, hit in enumerate(hits):
                concrete_target_rect: np.array = np.array([hit.bbox[:n], hit.bbox[n:]])
                # concrete_index_to_rect(concrete_target_index, sym_x,
                #                                                    symbol_step, X_low, X_up)
                # abstract_target_direction_rect = \
                #    transform_rect_to_abstract_frames(concrete_target_rect,
                #                                      rect, over_approximate=False)
                # if abstract_target_direction_rect is None:
                # abstract_target_direction_rect
                target_rect = \
                    transform_rect_to_abstract_frames(concrete_target_rect,
                                                      rect, over_approximate=True)
                # symmetry_abstract_states[abstract_state_ind].set_of_abstract_target_directions.append(
                #    abstract_target_direction_rect)
                # if idx >= num_nearest_targets_to_consider:
                #    break
                # hits = []
                # for target_rect in symmetry_abstract_states[abstract_state_ind].set_of_abstract_target_directions:
                original_angle_interval = [target_rect[0, 2], target_rect[1, 2]]
                target_rect_center = np.average(target_rect, axis=1)
                decomposed_angle_intervals = get_decomposed_angle_intervals(original_angle_interval)
                for interval in decomposed_angle_intervals:
                    # change this to nearest to the under approximation of the target, much faster
                    # num_results=3,
                    interval_center = (interval[0] + interval[1]) / 2
                    '''
                    hits.extend(list(reachability_rtree_idx3d.intersection(
                        (target_rect[0, 0], target_rect[0, 1], interval[0],
                         target_rect[1, 0], target_rect[1, 1], interval[1]),
                        objects=True)))
                    '''
                    reach_hits.extend(list(reachability_rtree_idx3d.nearest(
                        (target_rect_center[0], target_rect_center[1], interval_center,
                         target_rect_center[0] + 0.001, target_rect_center[1] + 0.001, interval_center + 0.001),
                        num_results=num_nearest_reachsets_to_consider,
                        objects=True)))

                if idx >= num_nearest_reachsets_to_consider:
                    break
            '''
            if not hits:
                for target_rect in symmetry_abstract_states[abstract_state_ind].set_of_abstract_target_directions:
                    original_angle_interval = [target_rect[0, 2], target_rect[1, 2]]
                    decomposed_angle_intervals = get_decomposed_angle_intervals(original_angle_interval)
                    for interval in decomposed_angle_intervals:
                        # change this to nearest to the under approximation of the target, much faster
                        # num_results=3,
                        hits.extend(list(reachability_rtree_idx3d.nearest(
                            (target_rect[0, 0], target_rect[0, 1], interval[0],
                             target_rect[1, 0], target_rect[1, 1], interval[1]),
                            num_results=num_nearest_reachsets_to_consider,
                            objects=True)))
            '''
            unique_paths = set()
            for hit in reach_hits:
                unique_paths.add(hit.id)
                # if len(unique_controls) >= num_nearest_reachsets_to_consider:
                #    break
            if unique_paths:
                # unique_controls = set(random.choices(list(unique_controls), k=1))
                for path_ind in unique_paths:
                    # avoid_res = successor_avoids_obstacles(abstract_state_ind, path_ind, abstract_paths,
                    #                                        symmetry_abstract_states)
                    if True:  # avoid_res:
                        '''
                        if abstraction_level == 0:
                            reach_res, abstract_transitions, inverse_abstract_transitions = \
                                successor_in_or_intersects_target(
                                    abstract_state_ind, path_ind,
                                    abstract_paths,
                                    symmetry_abstract_states,
                                    concrete_to_abstract,
                                    abstract_to_concrete,
                                    global_controllable_abstract_states,
                                    concrete_transitions,
                                    abstract_transitions,
                                    inverse_abstract_transitions,
                                    sym_x, symbol_step,
                                    obstacles_rects,
                                    obstacle_indices,
                                    targets_rects,
                                    target_indices, X_low, X_up)
                        '''
                        # else:
                        reach_res, new_path_ind, abstract_transitions, \
                            inverse_abstract_transitions = successor_in_or_intersects_target_smt(
                            rect, 0, path_ind, abstract_paths,
                            state_to_paths_ind, cur_solver, var_dict,
                            abstract_transitions, inverse_abstract_transitions,
                            reachability_rtree_idx3d,
                            extended_target_rtree_idx3d, n)
                        '''
                            successor_in_or_intersects_target_smt(
                            rect, 0, u_ind, abstract_paths,
                            cur_solver, var_dict,
                            abstract_transitions, inverse_abstract_transitions,
                            extended_target_rtree_idx3d, n)
                        '''

                        if reach_res == 2:
                            # for concrete_state_idx in abstract_to_concrete[abstract_state_ind]:
                            # s_subscript = np.array(
                            #    np.unravel_index(concrete_state_idx, tuple((sym_x[0, :]).astype(int))))
                            # s_rect: np.array = np.row_stack((s_subscript * symbol_step + X_low,
                            #                                 s_subscript * symbol_step + symbol_step + X_low))
                            extended_target_rtree_idx3d.insert(concrete_state_ind, (
                                rect[0, 0], rect[0, 1], rect[0, 2],
                                rect[1, 0], rect[1, 1], rect[1, 2]))
                            state_to_paths_ind[concrete_state_ind] = new_path_ind
                            # cur_solver = add_rects_to_solver(np.array([s_rect]), var_dict, cur_solver)
                            target_rtree_idx += 1
                            controller[abstract_state_ind] = new_path_ind  # should be changed to u_ind
                            temp_controllable_abstract_states.add(abstract_state_ind)
                            global_controllable_abstract_states.add(abstract_state_ind)
                            reg = get_poly_list_with_decomposed_angle_intervals(
                                symmetry_abstract_states[abstract_state_ind].abstract_targets[0])
                            reachable_target_region = get_poly_union(reachable_target_region, reg)
                            if abstraction_level == 2:
                                if abstract_state_ind in inverse_abstract_transitions:
                                    for parent_abstract_state_ind, parent_u_ind in \
                                            inverse_abstract_transitions[abstract_state_ind]:
                                        if parent_abstract_state_ind != abstract_state_ind \
                                                and parent_abstract_state_ind in abstract_states_to_explore:
                                            if parent_abstract_state_ind in target_parents:
                                                target_parents[parent_abstract_state_ind].add(
                                                    (abstract_state_ind, parent_u_ind))
                                            else:
                                                target_parents[parent_abstract_state_ind] = {
                                                    (abstract_state_ind, parent_u_ind)}
                                            if len(abstract_to_concrete[parent_abstract_state_ind]) > 1:
                                                refinement_candidates.add(parent_abstract_state_ind)
                                if abstract_state_ind in refinement_candidates:
                                    refinement_candidates.remove(abstract_state_ind)
                            '''
                            rect_under_approx = symmetry_abstract_states[abstract_state_ind].rtree_target_rect_under_approx
                            reachability_rtree_idx3d.insert(reachable_rect_global_cntr, (
                                rect_under_approx[0, 0], rect_under_approx[0, 1], rect_under_approx[0, 2],
                                rect_under_approx[1, 0], rect_under_approx[1, 1], rect_under_approx[1, 2]), obj=u_ind)
                            '''
                            num_new_symbols += 1
                            # if newly_added_rects_lower_bound_temp == 0:
                            #    newly_added_rects_lower_bound_temp = reachable_rect_global_cntr
                            # reachable_rect_global_cntr += 1
                            break
                        elif reach_res == 1 and abstraction_level == 2:  # and abstract_s not in target_parents:
                            if abstract_state_ind not in target_parents:
                                target_parents[abstract_state_ind] = {(None, path_ind)}
                            if len(abstract_to_concrete[abstract_state_ind]) > 1:
                                refinement_candidates.add(abstract_state_ind)
                            for next_abstract_state in abstract_transitions[(abstract_state_ind, path_ind)]:
                                if next_abstract_state in global_controllable_abstract_states or \
                                        next_abstract_state in controllable_abstract_states or \
                                        next_abstract_state in temp_controllable_abstract_states:
                                    target_parents[abstract_state_ind].add((next_abstract_state, path_ind))
                        #        elif next_abstract_state >= 0 and len(abstract_to_concrete[next_abstract_state]) > 1:
                        #            refinement_candidates.add(next_abstract_state)
                        '''
                        if global_controllable_abstract_states:
                            # other_abstract_state_ind = random.choice(tuple(global_controllable_abstract_states))
                            # concrete_target_index = random.choice(abstract_to_concrete[other_abstract_state_ind])
                            hits = list(extended_target_rtree_idx3d.nearest(
                                (rect[0, 0], rect[0, 1], rect[0, 2], rect[1, 0], rect[1, 1], rect[1, 2]), num_results=3))
                            symmetry_abstract_states[abstract_state_ind].set_of_abstract_target_directions = []
                            for hit in hits:
                                concrete_target_rect: np.array = np.array([hit.bbox[:n], hit.bbox[n:]])
                                #concrete_index_to_rect(concrete_target_index, sym_x,
                                #                                                    symbol_step, X_low, X_up)
                                concrete_initial_set_index = random.choice(abstract_to_concrete[abstract_state_ind])
                                concrete_initial_set_rect: np.array = concrete_index_to_rect(concrete_initial_set_index,
                                                                                             sym_x, symbol_step,
                                                                                             X_low, X_up)
                                abstract_target_direction_rect = \
                                    transform_rect_to_abstract_frames(concrete_target_rect,
                                                                      concrete_initial_set_rect, over_approximate=True)
                                symmetry_abstract_states[abstract_state_ind].set_of_abstract_target_directions.append(
                                    abstract_target_direction_rect)
                        '''

        if num_new_symbols:
            print(time.time() - t_start, " ", num_new_symbols,
                  " new controllable states have been found in this synthesis iteration\n")
            controllable_abstract_states = controllable_abstract_states.union(temp_controllable_abstract_states)
            num_controllable_states += num_new_symbols
            # newly_added_rects_lower_bound = newly_added_rects_lower_bound_temp
            # newly_added_rects_lower_bound_temp = 0
            refinement_candidates = refinement_candidates.difference(temp_controllable_abstract_states)
            # target_parents = target_parents.difference(temp_controllable_abstract_states)
            # for abstract_state_ind in temp_controllable_abstract_states:
            #    if abstract_state_ind in target_parents:
            #        del target_parents[abstract_state_ind]
            temp_controllable_abstract_states = list(temp_controllable_abstract_states)
            candidate_initial_set_rect = None
            for abstract_state_ind in temp_controllable_abstract_states:
                print("The abstract symbol ", abstract_state_ind,
                      " is controllable using path_ind ", controller[abstract_state_ind])
                for concrete_initial_set_index in abstract_to_concrete[abstract_state_ind]:
                    s_rect: np.array = concrete_index_to_rect(concrete_initial_set_index,
                                                              sym_x, symbol_step,
                                                              X_low, X_up)
                    bloated_rect = np.array([np.maximum(np.add(s_rect[0, :],
                                                               -1 * per_dim_max_travelled_distance),
                                                        X_low),
                                             np.minimum(np.add(s_rect[1, :], per_dim_max_travelled_distance),
                                                        X_up)])
                    if candidate_initial_set_rect is None:
                        candidate_initial_set_rect = bloated_rect
                    else:
                        candidate_initial_set_rect = get_convex_union([bloated_rect, candidate_initial_set_rect])
                    controllable_concrete_states.add(concrete_initial_set_index)
            concrete_states_to_explore = rect_to_indices(candidate_initial_set_rect, symbol_step, X_low,
                                                         sym_x[0, :], over_approximate=True)
            # concrete_states_to_explore = np.setdiff1d(np.array(concrete_states_to_explore),
            #                                          list(controllable_concrete_states))
            # concrete_states_to_explore = np.setdiff1d(np.array(concrete_states_to_explore),
            #                                          list(obstacle_indices))
            # concrete_states_to_explore = np.setdiff1d(np.array(concrete_states_to_explore),
            #                                          list(target_indices))
            if abstraction_level == 2:
                local_abstract_states_to_explore = copy.deepcopy(set(target_parents.keys()))
                # if first_call:
                #    abstract_states_to_explore = copy.deepcopy(target_parents)
                # else:
                abstract_states_to_explore = abstract_states_to_explore.difference(temp_controllable_abstract_states)
                # np.setdiff1d(abstract_states_to_explore, temp_controllable_abstract_states).tolist()
                # np.setdiff1d(refinement_candidates, temp_controllable_abstract_states).tolist()
            print(num_controllable_states, ' symbols are controllable to satisfy the reach-avoid specification\n')
        else:
            print('No new controllable state has been found in this synthesis iteration\n', time.time() - t_start)
            break

    return controller, controllable_abstract_states, unsafe_abstract_states, local_abstract_states_to_explore, \
        abstract_states_to_explore, reachable_rect_global_cntr, target_parents, refinement_candidates, target_rtree_idx


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
                         abstract_to_concrete, concrete_to_abstract, target_parents,
                         symmetry_transformed_targets_and_obstacles, symmetry_abstract_states):
    abstract_state_1 = None
    abstract_state_2 = None
    if len(concrete_indices) >= len(abstract_to_concrete[abstract_state_ind]):
        print("The concrete indices provided are all that ", abstract_state_ind, " represents, so no need to split.")
        return concrete_to_abstract, abstract_to_concrete, target_parents
    rest_of_concrete_indices = np.setdiff1d(np.array(abstract_to_concrete[abstract_state_ind]), concrete_indices)
    for concrete_state_idx in concrete_indices:
        if pc.is_empty(get_poly_intersection(
                symmetry_transformed_targets_and_obstacles[concrete_state_idx].abstract_targets[0],
                symmetry_abstract_states[abstract_state_ind].abstract_targets[0])):
            print("A concrete state has a relative target that is no longer "
                  "intersecting the relative target of the abstract state it belongs to. This shouldn't happen.")
        abstract_state_1 = add_concrete_state_to_symmetry_abstract_state(concrete_state_idx,
                                                                         copy.deepcopy(abstract_state_1),
                                                                         symmetry_transformed_targets_and_obstacles)
    for concrete_state_idx in rest_of_concrete_indices:
        if pc.is_empty(get_poly_intersection(
                symmetry_transformed_targets_and_obstacles[concrete_state_idx].abstract_targets[0],
                symmetry_abstract_states[abstract_state_ind].abstract_targets[0])):
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

    abstract_to_concrete[abstract_state_ind] = []

    return concrete_to_abstract, abstract_to_concrete, target_parents


def refine(concrete_to_abstract, abstract_to_concrete, symmetry_abstract_states,
           remaining_abstract_states, refinement_candidates, target_parents, inverse_abstract_transitions,
           local_abstract_states_to_explore,
           abstract_states_to_explore,
           controllable_abstract_states, symmetry_transformed_targets_and_obstacles,
           abstract_paths, concrete_edges, sym_x, symbol_step,
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
        abstract_state_ind = random.choice(tuple(refinement_candidates))
        concrete_indices_len = len(abstract_to_concrete[abstract_state_ind])
        if concrete_indices_len > 1:
            concrete_indices = set()
            if abstract_state_ind in target_parents:
                _, u_ind = random.choice(tuple(target_parents[abstract_state_ind]))
                for concrete_state_ind in abstract_to_concrete[abstract_state_ind]:
                    next_concrete_state_indices = get_concrete_transition(concrete_state_ind, u_ind, concrete_edges,
                                                                          sym_x, symbol_step, abstract_paths,
                                                                          obstacles_rects, obstacle_indices,
                                                                          targets_rects, target_indices, X_low, X_up)
                    for next_concrete_state_ind in next_concrete_state_indices:
                        if not (next_concrete_state_ind == -1 or
                                (next_concrete_state_ind >= 0 and
                                 concrete_to_abstract[next_concrete_state_ind] in controllable_abstract_states)):
                            concrete_indices.add(concrete_state_ind)
                            break
                if not concrete_indices:
                    controllable_abstract_states.add(abstract_state_ind)
                    controllable_abstract_states_temp.add(abstract_state_ind)
                    abstract_states_to_explore.remove(abstract_state_ind)
                    # np.setdiff1d(np.array(abstract_states_to_explore), [abstract_state_ind]).tolist()
                    del target_parents[abstract_state_ind]
                    if abstract_state_ind in inverse_abstract_transitions:
                        for parent_abstract_state_ind, parent_u_ind in inverse_abstract_transitions[abstract_state_ind]:
                            if parent_abstract_state_ind != abstract_state_ind \
                                    and parent_abstract_state_ind in abstract_states_to_explore:
                                if parent_abstract_state_ind in target_parents:
                                    target_parents[parent_abstract_state_ind].add((abstract_state_ind, parent_u_ind))
                                else:
                                    target_parents[parent_abstract_state_ind] = {(abstract_state_ind, parent_u_ind)}
                                if len(abstract_to_concrete[parent_abstract_state_ind]) > 1:
                                    refinement_candidates.add(parent_abstract_state_ind)
                    refinement_candidates.remove(abstract_state_ind)
                    # local_abstract_states_to_explore = copy.deepcopy(abstract_states_to_explore)
                    # set(target_parents.keys())
                    abstract_states_to_explore = abstract_states_to_explore.difference(
                        controllable_abstract_states_temp)
                    # np.setdiff1d(abstract_states_to_explore, controllable_abstract_states_temp).tolist()
                    continue
            if abstract_state_ind not in target_parents \
                    or len(concrete_indices) >= len(abstract_to_concrete[abstract_state_ind]):
                concrete_indices = random.choices(abstract_to_concrete[abstract_state_ind],
                                                  k=int(concrete_indices_len / 2))
            if concrete_indices:
                concrete_indices = list(set(concrete_indices))
            progress = True
            if len(concrete_indices) < len(abstract_to_concrete[abstract_state_ind]):
                concrete_to_abstract, abstract_to_concrete, target_parents = \
                    split_abstract_state(abstract_state_ind, concrete_indices,
                                         abstract_to_concrete, concrete_to_abstract, target_parents,
                                         symmetry_transformed_targets_and_obstacles,
                                         symmetry_abstract_states)
                # remaining_abstract_states.remove(abstract_state_ind) # np.setdiff1d(, np.array([abstract_state_ind]))
                # remaining_abstract_states.add(len(abstract_to_concrete) - 1)
                # remaining_abstract_states.add(len(abstract_to_concrete) - 2)
                if len(abstract_to_concrete[-1]) > 1:
                    refinement_candidates.add(len(abstract_to_concrete) - 1)
                if len(abstract_to_concrete[-2]) > 1:
                    refinement_candidates.add(len(abstract_to_concrete) - 2)
                abstract_states_to_explore.remove(abstract_state_ind)
                # np.setdiff1d(np.array(abstract_states_to_explore),  [abstract_state_ind]).tolist()
                abstract_states_to_explore.add(len(abstract_to_concrete) - 1)  # append
                abstract_states_to_explore.add(len(abstract_to_concrete) - 2)
                if abstract_state_ind in target_parents:
                    del target_parents[abstract_state_ind]
                    # target_parents.remove(abstract_state_ind)
                    # target_parents[len(abstract_to_concrete) - 1] = set()
                    # target_parents[len(abstract_to_concrete) - 2] = set()
                # target_parents.add(len(abstract_to_concrete) - 1)
                # target_parents.add(len(abstract_to_concrete) - 2)
                refinement_candidates.remove(abstract_state_ind)
                deleted_abstract_states.append(abstract_state_ind)
                if abstract_state_ind in local_abstract_states_to_explore:
                    local_abstract_states_to_explore.remove(abstract_state_ind)
                local_abstract_states_to_explore.add(len(abstract_to_concrete) - 1)
                local_abstract_states_to_explore.add(len(abstract_to_concrete) - 2)
                num_new_abstract_states += 1
        else:
            refinement_candidates.remove(abstract_state_ind)

    return progress, num_new_abstract_states, concrete_to_abstract, abstract_to_concrete, \
        refinement_candidates, target_parents, local_abstract_states_to_explore, \
        abstract_states_to_explore, \
        remaining_abstract_states, deleted_abstract_states, controllable_abstract_states, \
        controllable_abstract_states_temp


def abstract_synthesis_old(Symbolic_reduced, sym_x, sym_u, state_dimensions, Target_low,
                           Target_up,
                           Obstacle_low, Obstacle_up, X_low, X_up):
    t_start = time.time()
    n = state_dimensions.shape[1]
    reachable_target_region = pc.Region(list_poly=[])

    p = index.Property()
    p.dimension = 3
    p.dat_extension = 'data'
    p.idx_extension = 'index'
    reachability_rtree_idx3d = index.Index('3d_index_abstract',
                                           properties=p)
    symmetry_under_approx_abstract_targets_rtree_idx3d = index.Index(
        '3d_index_under_approx_abstract_targets',
        properties=p)

    symbol_step = (X_up - X_low) / sym_x[0, :]

    targets, targets_rects, target_indices, obstacles, obstacles_rects, obstacle_indices = \
        create_targets_and_obstacles(Target_low, Target_up, Obstacle_low, Obstacle_up, symbol_step, sym_x, X_low)

    abstract_paths, reachable_rect_global_cntr, intersection_radius_threshold, reachable_target_region = \
        create_symmetry_abstract_reachable_sets(Symbolic_reduced, n, reachability_rtree_idx3d, reachable_target_region)

    matrix_dim_full = [np.prod(sym_x[0, :]), np.prod(sym_u), 2 * n]
    symbols_to_explore = set(
        range(int(matrix_dim_full[0])))  # np.setdiff1d(np.array(range(int(matrix_dim_full[0]))), target_indices)
    symbols_to_explore = symbols_to_explore.difference(target_indices)
    symbols_to_explore = symbols_to_explore.difference(obstacle_indices)

    # intersection_radius_threshold = intersection_radius_threshold * 10
    symmetry_transformed_targets_and_obstacles, \
        concrete_to_abstract, abstract_to_concrete, \
        symmetry_abstract_states, abstract_states_to_rtree_ids, \
        rtree_ids_to_abstract_states, next_rtree_id_candidate, target_parents = create_symmetry_abstract_states(
        symbols_to_explore,
        symbol_step, targets,
        obstacles, sym_x, X_low,
        X_up,
        intersection_radius_threshold, symmetry_under_approx_abstract_targets_rtree_idx3d, reachability_rtree_idx3d)

    t_abstraction = time.time() - t_start
    print(['Construction of symmetry-based abstraction took: ', t_abstraction, ' seconds'])
    num_abstract_states_before_refinement = len(abstract_to_concrete)

    controller = {}  # [-1] * len(abstract_to_concrete)
    t_synthesis_start = time.time()
    t_refine = 0
    t_synthesis = 0
    print('\n%s\tStart of the control synthesis\n', time.time() - t_start)
    refinement_itr = 0
    max_num_refinement_steps = 10000
    remaining_abstract_states = set(list(range(len(abstract_to_concrete))))
    abstract_states_to_explore = set(range(len(abstract_to_concrete)))
    local_abstract_states_to_explore = set(range(len(abstract_to_concrete)))
    controllable_abstract_states = set()
    controllable_concrete_states = set()
    refinement_candidates = set()
    abstract_transitions = {}  # [None] * len(abstract_to_concrete)
    inverse_abstract_transitions = {}
    concrete_transitions = {}
    continuous_failure_counter_max = 10
    continuous_failure_counter = 0
    potential_new_target_parents = False
    while refinement_itr < max_num_refinement_steps:
        temp_t_synthesis = time.time()
        controller, controllable_abstract_states_temp, unsafe_abstract_states, local_abstract_states_to_explore, \
            abstract_states_to_explore, \
            reachable_rect_global_cntr, target_parents, refinement_candidates = symmetry_abstract_synthesis_helper(
            local_abstract_states_to_explore,
            abstract_states_to_explore,
            abstract_to_concrete,
            concrete_to_abstract,
            abstract_paths,
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
            obstacles_rects, obstacle_indices, targets_rects, target_indices, X_low, X_up)
        '''
            symmetry_abstract_synthesis_helper(local_abstract_states_to_explore,
                                               abstract_states_to_explore,
                                               abstract_to_concrete,
                                               abstract_paths,
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
        refinement_candidates = refinement_candidates.difference(controllable_abstract_states)

        if controllable_abstract_states_temp:  # or not refinement_candidates:
            # update target parents only if there is a significant number of newly added controllable states.
            continuous_failure_counter = 0
            potential_new_target_parents = True
            # new_controllable_concrete_states = 0
            for abstract_state_ind in controllable_abstract_states_temp:
                if not abstract_to_concrete[abstract_state_ind]:
                    print("Why this controllable state has been refined?")
                controllable_concrete_states = controllable_concrete_states.union(
                    abstract_to_concrete[abstract_state_ind])

            # local_abstract_states_to_explore = copy.deepcopy(abstract_states_to_explore)
            # new_controllable_concrete_states += len(abstract_to_concrete[abstract_state_ind])
            '''
            if new_controllable_concrete_states > 10 or not refinement_candidates:
                for abstract_state_ind in abstract_states_to_explore:
                    if abstract_state_ind not in target_parents:
                        is_target_parent = False
                        rect = symmetry_abstract_states[abstract_state_ind].rtree_target_rect_under_approx
                        # rtree_target_rect_over_approx
                        original_angle_interval = [rect[0, 2], rect[1, 2]]
                        decomposed_angle_intervals = get_decomposed_angle_intervals(original_angle_interval)
                        for interval in decomposed_angle_intervals:
                            is_target_parent = reachability_rtree_idx3d.count(
                                (rect[0, 0], rect[0, 1], interval[0], rect[1, 0], rect[1, 1], interval[1]))
                            if is_target_parent:
                                break
                        if is_target_parent:
                            target_parents.add(abstract_state_ind)
                            if not abstract_to_concrete[abstract_state_ind]:
                                print("Why ", abstract_state_ind,
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
                refinement_candidates, target_parents, local_abstract_states_to_explore, abstract_states_to_explore, \
                remaining_abstract_states, deleted_abstract_states, controllable_abstract_states, \
                controllable_abstract_states_temp = \
                refine(concrete_to_abstract, abstract_to_concrete, symmetry_abstract_states,
                       remaining_abstract_states, refinement_candidates, target_parents, inverse_abstract_transitions,
                       local_abstract_states_to_explore,
                       abstract_states_to_explore,
                       controllable_abstract_states, symmetry_transformed_targets_and_obstacles,
                       abstract_paths, concrete_transitions, sym_x, symbol_step,
                       obstacles_rects, obstacle_indices, targets_rects, target_indices, X_low, X_up)

            if controllable_abstract_states_temp:  # or not refinement_candidates:
                # update target parents only if there is a significant number of newly added controllable states.
                print(len(controllable_abstract_states_temp),
                      " new controllable states have been found in this refinement iteration\n")
                continuous_failure_counter = 0
                potential_new_target_parents = True
                # new_controllable_concrete_states = 0
                for abstract_state_ind in controllable_abstract_states_temp:
                    if not abstract_to_concrete[abstract_state_ind]:
                        print("Why this controllable state has been refined?")
                    controllable_concrete_states = controllable_concrete_states.union(
                        abstract_to_concrete[abstract_state_ind])
                    # new_controllable_concrete_states += len(abstract_to_concrete[abstract_state_ind])

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
        for abstract_s in controllable_abstract_states:
            if not abstract_to_concrete[abstract_s]:
                print(abstract_s, " does not represent any concrete state, why is it controllable?")
            print("Controllable abstract state ", abstract_s, " represents the following concrete states: ",
                  abstract_to_concrete[abstract_s])
        print(len(controllable_concrete_states), 'concrete symbols are controllable to satisfy the reach-avoid '
                                                 'specification\n')
    else:
        print('The reach-avoid specification cannot be satisfied from any initial state\n')
    plot_abstract_states(symmetry_abstract_states, deleted_abstract_states)


def abstract_synthesis(reachability_abstraction_level, Symbolic_reduced, sym_x, sym_u, state_dimensions, Target_low,
                       Target_up,
                       Obstacle_low, Obstacle_up, X_low, X_up):
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
    symmetry_under_approx_abstract_targets_rtree_idx3d = index.Index(
        '3d_index_under_approx_abstract_targets',
        properties=p)
    extended_target_rtree_idx3d = index.Index(
        '3d_index_extended_target',
        properties=p)

    symbol_step = (X_up - X_low) / sym_x[0, :]

    state_to_paths_ind = {}

    # defining the z3 solver that we'll use to check if a rectangle is in a set of rectangles
    cur_solver = Solver()
    var_dict = []
    for dim in range(n):
        var_dict.append(Real("x" + str(dim)))

    targets, targets_rects, target_indices, obstacles, obstacles_rects, obstacle_indices = \
        create_targets_and_obstacles(Target_low, Target_up, Obstacle_low, Obstacle_up, symbol_step, sym_x, X_low)

    target_rtree_idx = 0
    for target in targets_rects:
        extended_target_rtree_idx3d.insert(-1, (
            target[0, 0], target[0, 1],
            target[0, 2], target[1, 0],
            target[1, 1], target[1, 2]))
    # cur_solver = add_rects_to_solver(np.array(targets_rects), var_dict, cur_solver)

    abstract_paths, reachable_rect_global_cntr, intersection_radius_threshold, per_dim_max_travelled_distance, \
        reachable_target_region = \
        create_symmetry_abstract_reachable_sets(Symbolic_reduced, n, reachability_rtree_idx3d, reachable_target_region)

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
    concrete_states_to_explore = concrete_states_to_explore.difference(target_indices)
    concrete_states_to_explore = concrete_states_to_explore.difference(obstacle_indices)
    symbols_to_explore = symbols_to_explore.difference(target_indices)
    symbols_to_explore = symbols_to_explore.difference(obstacle_indices)

    # intersection_radius_threshold = intersection_radius_threshold * 10
    symmetry_transformed_targets_and_obstacles, \
        concrete_to_abstract, abstract_to_concrete, \
        symmetry_abstract_states, abstract_states_to_rtree_ids, \
        rtree_ids_to_abstract_states, next_rtree_id_candidate, target_parents = create_symmetry_abstract_states(
        symbols_to_explore,
        symbol_step, targets,
        obstacles, sym_x, X_low,
        X_up,
        intersection_radius_threshold, symmetry_under_approx_abstract_targets_rtree_idx3d,
        reachability_rtree_idx3d)

    t_abstraction = time.time() - t_start
    print(['Construction of symmetry-based abstraction took: ', t_abstraction, ' seconds'])
    num_abstract_states_before_refinement = len(abstract_to_concrete)

    controller = {}  # [-1] * len(abstract_to_concrete)
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
    controllable_abstract_states = set()
    controllable_concrete_states = set()
    refinement_candidates = set()
    abstract_transitions = {}  # [None] * len(abstract_to_concrete)
    inverse_abstract_transitions = {}
    concrete_transitions = {}
    continuous_failure_counter_max = 10
    continuous_failure_counter = 0
    potential_new_target_parents = False
    deleted_abstract_states = []
    while refinement_itr < max_num_refinement_steps:
        temp_t_synthesis = time.time()
        controller, controllable_abstract_states_temp, unsafe_abstract_states, local_abstract_states_to_explore, \
            abstract_states_to_explore, reachable_rect_global_cntr, \
            target_parents, refinement_candidates, target_rtree_idx = symmetry_abstract_synthesis_helper(
            local_abstract_states_to_explore,
            concrete_states_to_explore,  # abstract_states_to_explore,
            abstract_to_concrete,
            concrete_to_abstract,
            Symbolic_reduced,
            abstract_paths,
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
            target_rtree_idx, cur_solver, var_dict, state_to_paths_ind, per_dim_max_travelled_distance, X_low, X_up)
        '''
            symmetry_abstract_synthesis_helper(local_abstract_states_to_explore,
                                               abstract_states_to_explore,
                                               abstract_to_concrete,
                                               abstract_paths,
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
                for abstract_state_ind in controllable_abstract_states_temp:
                    if not abstract_to_concrete[abstract_state_ind]:
                        print("Why this controllable state has been refined?")
                    controllable_concrete_states = controllable_concrete_states.union(
                        abstract_to_concrete[abstract_state_ind])

                # local_abstract_states_to_explore = copy.deepcopy(abstract_states_to_explore)
                # new_controllable_concrete_states += len(abstract_to_concrete[abstract_state_ind])
                '''
                if new_controllable_concrete_states > 10 or not refinement_candidates:
                    for abstract_state_ind in abstract_states_to_explore:
                        if abstract_state_ind not in target_parents:
                            is_target_parent = False
                            rect = symmetry_abstract_states[abstract_state_ind].rtree_target_rect_under_approx
                            # rtree_target_rect_over_approx
                            original_angle_interval = [rect[0, 2], rect[1, 2]]
                            decomposed_angle_intervals = get_decomposed_angle_intervals(original_angle_interval)
                            for interval in decomposed_angle_intervals:
                                is_target_parent = reachability_rtree_idx3d.count(
                                    (rect[0, 0], rect[0, 1], interval[0], rect[1, 0], rect[1, 1], interval[1]))
                                if is_target_parent:
                                    break
                            if is_target_parent:
                                target_parents.add(abstract_state_ind)
                                if not abstract_to_concrete[abstract_state_ind]:
                                    print("Why ", abstract_state_ind,
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
                           abstract_paths, concrete_transitions, sym_x, symbol_step,
                           obstacles_rects, obstacle_indices, targets_rects, target_indices, X_low, X_up)

                if controllable_abstract_states_temp:  # or not refinement_candidates:
                    # update target parents only if there is a significant number of newly added controllable states.
                    print(len(controllable_abstract_states_temp),
                          " new controllable states have been found in this refinement iteration\n")
                    continuous_failure_counter = 0
                    potential_new_target_parents = True
                    # new_controllable_concrete_states = 0

                    # new_controllable_concrete_states += len(abstract_to_concrete[abstract_state_ind])

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

    for abstract_state_ind in controllable_abstract_states:
        if not abstract_to_concrete[abstract_state_ind]:
            print("Why this controllable state has been refined?")
        controllable_concrete_states = controllable_concrete_states.union(
            abstract_to_concrete[abstract_state_ind])

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
        for abstract_s in controllable_abstract_states:
            if not abstract_to_concrete[abstract_s]:
                print(abstract_s, " does not represent any concrete state, why is it controllable?")
            print("Controllable abstract state ", abstract_s, " represents the following concrete states: ",
                  abstract_to_concrete[abstract_s])
        print(len(controllable_concrete_states), 'concrete symbols are controllable to satisfy the reach-avoid '
                                                 'specification\n')
    else:
        print('The reach-avoid specification cannot be satisfied from any initial state\n')
    plot_abstract_states(symmetry_abstract_states, deleted_abstract_states, abstract_paths,
                         state_to_paths_ind, abstract_to_concrete)
    plot_concrete_states(controllable_concrete_states, targets_rects, obstacles_rects,
                         state_to_paths_ind, sym_x, symbol_step, X_low, X_up)
