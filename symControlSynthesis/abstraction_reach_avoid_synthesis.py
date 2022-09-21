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

import matplotlib

matplotlib.use("macOSX")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, Polygon


class Node:
    """
    RRT Node
    """

    def __init__(self, reachable_set: List[np.array], s_ind: int, u_ind: int):
        self.reachable_set = reachable_set
        self.s_ind = s_ind
        self.u_ind = u_ind
        self.last_over_approximated_reachable_set = None
        self.parent = None
        self.discovered = 0
        self.id_in_tracking_rtree = -1
        self.uncovered_state = None


def transform_poly_to_abstract(poly: pc.polytope, state: np.array):
    # this function takes a polytope in the state space and transforms it to the abstract coordinates.
    # this should be provided by the user as it depends on the symmetries
    # Hussein: unlike the work in SceneChecker, we here rotate then translate, non-ideal, I prefer translation
    # then rotation but this requires changing find_frame which would take time.
    translation_vector = state;
    rot_angle = state[2];
    poly_out: pc.Polytope = poly.rotation(i=0, j=1, theta=rot_angle);
    return poly_out.translation(translation_vector)


def transform_rect_to_abstract(rect: np.array, state: np.array):
    ang = -1 * state[2];  # psi = 0 is North, psi = pi/2 is east

    while ang < 0:
        ang += 2 * math.pi
    while ang > 2 * math.pi:
        ang -= 2 * math.pi

    low_red = np.array(
        [(rect[0, 0] - state[0]) * math.cos(ang) -
         (rect[0, 1] - state[1]) * math.sin(ang),
         (rect[0, 0] - state[0]) * math.sin(ang) +
         (rect[0, 1] - state[1]) * math.cos(ang),
         rect[0, 2] + state[2]])
    up_red = np.array(
        [(rect[1, 0] - state[0]) * math.cos(ang) -
         (rect[1, 1] - state[1]) * math.sin(ang),
         (rect[1, 0] - state[0]) * math.sin(ang) +
         (rect[1, 1] - state[1]) * math.cos(ang),
         rect[1, 2] + state[2]])

    if 0 <= ang <= math.pi / 2:
        x_bb_up = up_red[0] + (rect[1, 1] - rect[0, 1]) * math.sin(ang)
        y_bb_up = up_red[1]
        x_bb_low = low_red[0] - (rect[1, 1] - rect[0, 1]) * math.sin(ang)
        y_bb_low = low_red[1]
    elif math.pi / 2 <= ang <= math.pi:
        x_bb_up = low_red[0];
        y_bb_up = low_red[1] + (rect[1, 0] - rect[0, 0]) * math.cos(ang - math.pi / 2);
        x_bb_low = up_red[0];
        y_bb_low = up_red[1] - (rect[1, 0] - rect[0, 0]) * math.cos(ang - math.pi / 2);
    elif math.pi <= ang <= 3 * math.pi / 2.0:
        x_bb_up = low_red[0] + (rect[1, 1] - rect[0, 1]) * math.cos(3 * math.pi / 2 - ang);
        y_bb_up = low_red[1];
        x_bb_low = up_red[0] - (rect[1, 1] - rect[0, 1]) * math.cos(3 * math.pi / 2 - ang)
        y_bb_low = up_red[1];
    else:
        x_bb_up = up_red[0];
        y_bb_up = up_red[1] + (rect[1, 0] - rect[0, 0]) * math.cos(ang - 3 * math.pi / 2);
        x_bb_low = low_red[0];
        y_bb_low = low_red[1] - (rect[1, 0] - rect[0, 0]) * math.cos(ang - 3 * math.pi / 2);

    bb = np.array([[x_bb_low, y_bb_low, low_red[2]], [x_bb_up, y_bb_up, up_red[2]]]);
    return bb


def rectangle_to_vertices(rect: np.array):
    points = [];
    for i in range(2):
        for j in range(2):
            for k in range(2):
                points.append([rect[i, 0], rect[j, 1], rect[k, 2]]);
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
    w = bb[1, 0] - bb[0, 0]
    h = bb[1, 1] - bb[0, 1]

    if overapproximate:
        return bb

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


def find_frame(low_red, up_red, target_full_low, target_full_up):
    # reimplement using Z3? does it return a set of states instead of just a counter-example?
    # print(low_red, up_red, target_full_low, target_full_up)
    if up_red[2] - low_red[2] > target_full_up[2] - target_full_low[2] + 0.01:  # The 0.01 is a tolerance value
        print("find_frame that put ", np.array([low_red, up_red]), " in the target ",
              np.array([target_full_low, target_full_up]), " failed.")
        return float('nan')

    angle_step = 2 * (up_red[2] - low_red[2])
    rect_curr = []
    if angle_step > 0:
        # print("int(np.floor(target_full_up[2] - target_full_low[2]) / angle_step): ",
        #      int(np.floor(target_full_up[2] - target_full_low[2]) / angle_step));
        itr_num = int(np.ceil(target_full_up[2] - target_full_low[2]) / angle_step)
    else:
        itr_num = 1
    for idx in range(itr_num):
        low_angle = target_full_low[2] + idx * angle_step
        high_angle = min(target_full_low[2] + (idx + 1) * angle_step, target_full_up[2])
        theta_low_sys = low_angle - low_red[2]
        theta_up_sys = high_angle - up_red[2]

        theta_low = theta_low_sys
        while theta_low < -math.pi:
            theta_low += 2 * math.pi
        while theta_low > math.pi:
            theta_low -= 2 * math.pi
        theta_up = theta_up_sys
        while theta_up < -math.pi:
            theta_up += 2 * math.pi
        while theta_up > math.pi:
            theta_up -= 2 * math.pi

        if 0 <= theta_low <= math.pi / 2:
            x_target_up_1 = target_full_up[0] - (up_red[1] - low_red[1]) * math.cos(math.pi / 2 - theta_low)
            y_target_up_1 = target_full_up[1]
        elif math.pi / 2 <= theta_low <= math.pi:
            x_target_up_1 = target_full_up[0] - (up_red[0] - low_red[0]) * math.sin(theta_low - math.pi / 2) - (
                    up_red[1] - low_red[1]) * math.cos(theta_low - math.pi / 2)
            y_target_up_1 = target_full_up[1] - (up_red[1] - low_red[1]) * math.sin(theta_low - math.pi / 2)
        elif 0 > theta_low >= - math.pi / 2:
            x_target_up_1 = target_full_up[0]
            y_target_up_1 = target_full_up[1] - (up_red[0] - low_red[0]) * math.sin(-1 * theta_low)
        else:
            x_target_up_1 = target_full_up[0]
            y_target_up_1 = target_full_up[1] - (up_red[1] - low_red[1]) * math.sin(-1 * theta_low - math.pi / 2)
            x_target_up_1 = x_target_up_1 - (up_red[0] - low_red[0]) * math.sin(-1 * theta_low - math.pi / 2)
            y_target_up_1 = y_target_up_1 - (up_red[0] - low_red[0]) * math.cos(-1 * theta_low - math.pi / 2)

        if 0 <= theta_low <= math.pi / 2:
            x_target_low_1 = target_full_low[0] + (up_red[1] - low_red[1]) * math.cos(math.pi / 2 - theta_low)
            y_target_low_1 = target_full_low[1]

        elif math.pi / 2 <= theta_low <= math.pi:
            x_target_low_1 = target_full_low[0] + (up_red[0] - low_red[0]) * math.cos(math.pi - theta_low) + (
                    up_red[1] - low_red[1]) * math.cos(theta_low - math.pi / 2)
            y_target_low_1 = target_full_low[1] + (up_red[1] - low_red[1]) * math.sin(theta_low - math.pi / 2)

        elif 0 > theta_low >= -math.pi / 2:
            x_target_low_1 = target_full_low[0]
            y_target_low_1 = target_full_low[1] + (up_red[0] - low_red[0]) * math.cos(math.pi / 2 + theta_low)

        else:
            x_target_low_1 = target_full_low[0] + (up_red[0] - low_red[0]) * math.cos(math.pi + theta_low)
            y_target_low_1 = target_full_low[1] + (up_red[0] - low_red[0]) * math.sin(math.pi + theta_low) + \
                             (up_red[1] - low_red[1]) * math.cos(math.pi + theta_low)

        curr_low_1 = np.array(
            [x_target_low_1 - (low_red[0]) * math.cos(theta_low_sys) + (low_red[1]) * math.sin(theta_low_sys),
             y_target_low_1 - (low_red[0]) * math.sin(theta_low_sys) - (low_red[1]) * math.cos(theta_low_sys),
             theta_low_sys])
        curr_up_1 = np.array(
            [x_target_up_1 - (up_red[0]) * math.cos(theta_low_sys) + (up_red[1]) * math.sin(theta_low_sys),
             y_target_up_1 - (up_red[0]) * math.sin(theta_low_sys) - (up_red[1]) * math.cos(theta_low_sys),
             theta_low_sys])

        #####################################

        if 0 <= theta_up <= math.pi / 2:
            x_target_up_2 = target_full_up[0] - (up_red[1] - low_red[1]) * math.cos(math.pi / 2 - theta_up)
            y_target_up_2 = target_full_up[1]
        elif math.pi / 2 <= theta_up <= math.pi:
            x_target_up_2 = target_full_up[0] - (up_red[0] - low_red[0]) * math.sin(theta_up - math.pi / 2) - (
                    up_red[1] - low_red[1]) * math.cos(theta_up - math.pi / 2)
            y_target_up_2 = target_full_up[1] - (up_red[1] - low_red[1]) * math.sin(theta_up - math.pi / 2)
        elif 0 > theta_up >= - math.pi / 2:
            x_target_up_2 = target_full_up[0]
            y_target_up_2 = target_full_up[1] - (up_red[0] - low_red[0]) * math.sin(-1 * theta_up)
        else:
            x_target_up_2 = target_full_up[0]
            y_target_up_2 = target_full_up[1] - (up_red[1] - low_red[1]) * math.sin(-1 * theta_up - math.pi / 2)
            x_target_up_2 = x_target_up_2 - (up_red[0] - low_red[0]) * math.sin(-1 * theta_up - math.pi / 2)
            y_target_up_2 = y_target_up_2 - (up_red[0] - low_red[0]) * math.cos(-1 * theta_up - math.pi / 2)

        if 0 <= theta_up <= math.pi / 2:
            x_target_low_2 = target_full_low[0] + (up_red[1] - low_red[1]) * math.cos(math.pi / 2 - theta_up)
            y_target_low_2 = target_full_low[1]

        elif math.pi / 2 <= theta_up <= math.pi:
            x_target_low_2 = target_full_low[0] + (up_red[0] - low_red[0]) * math.cos(math.pi - theta_up) + (
                    up_red[1] - low_red[1]) * math.cos(
                theta_up - math.pi / 2)
            y_target_low_2 = target_full_low[1] + (up_red[1] - low_red[1]) * math.sin(theta_up - math.pi / 2)

        elif 0 > theta_up >= - math.pi / 2:
            x_target_low_2 = target_full_low[0]
            y_target_low_2 = target_full_low[1] + (up_red[0] - low_red[0]) * math.cos(math.pi / 2 + theta_up)

        else:
            x_target_low_2 = target_full_low[0] + (up_red[0] - low_red[0]) * math.cos(math.pi + theta_up)
            y_target_low_2 = target_full_low[1] + (up_red[0] - low_red[0]) * math.sin(math.pi + theta_up) + (
                    up_red[1] - low_red[1]) * math.cos(
                math.pi + theta_up)

        curr_low_2 = np.array(
            [x_target_low_2 - (low_red[0]) * math.cos(theta_up_sys) + (low_red[1]) * math.sin(theta_up_sys),
             y_target_low_2 - (low_red[0]) * math.sin(theta_up_sys) - (low_red[1]) * math.cos(theta_up_sys),
             theta_up_sys])
        curr_up_2 = np.array(
            [x_target_up_2 - (up_red[0]) * math.cos(theta_up_sys) + (up_red[1]) * math.sin(theta_up_sys),
             y_target_up_2 - (up_red[0]) * math.sin(theta_up_sys) - (up_red[1]) * math.cos(theta_up_sys),
             theta_up_sys])

        if np.all(curr_low_1 <= curr_up_1) and np.all(curr_low_2 <= curr_up_2) \
                and do_rects_inter(np.array([curr_low_1[:2], curr_up_1[:2]]),
                                   np.array([curr_low_2[:2], curr_up_2[:2]])):
            curr_low_temp = np.maximum(curr_low_1, curr_low_2)
            curr_up_temp = np.minimum(curr_up_1, curr_up_2)
            curr_low = curr_low_temp
            curr_low[2] = curr_low_1[2]
            curr_up = curr_up_temp
            curr_up[2] = curr_up_2[2]
            rect_curr.append([curr_low.tolist(), curr_up.tolist()])

    if len(rect_curr) == 0:
        return float('nan')

    rect_curr = np.array(rect_curr)
    return rect_curr


def gray(xi, n, k):
    # From: Guan, Dah - Jyh(1998). "Generalized Gray Codes with Applications".
    # Proc.Natl.Sci.Counc.Repub.Of China(A) 22: 841???848.
    # http: // nr.stpi.org.tw / ejournal / ProceedingA / v22n6 / 841 - 848.pdf.
    x = np.zeros((n, int(pow(k, n))))  # The matrix with all combinations
    a = np.zeros((n + 1, 1))  # The current combination following(k, n) - Gray code
    b = np.ones((n + 1, 1))  # +1 or -1
    c = k * np.ones((n + 1, 1))  # The maximum for each digit
    j = 0
    while a[n] == 0:
        # Write current combination in the output
        x[:, j] = xi[np.reshape(a[0:n], (n,)).astype(int)]
        j = j + 1

        # Compute the next combination
        i = 0
        l = a[0] + b[0]
        while (l >= c[i]) or (l < 0):
            b[i] = -b[i]
            i = i + 1
            l = a[i] + b[i]
        a[i] = int(l)
    return x


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
    for i in range(rect1.shape[1]):
        if rect1[0, i] > rect2[1, i] + 0.01 or rect1[1, i] + 0.01 < rect2[0, i]:
            return False
    return True


def does_rect_contain(rect1, rect2):  # does rect2 contains rect1
    for i in range(rect1.shape[1]):
        if rect1[0, i] + 0.01 < rect2[0, i] or rect1[1, i] - 0.01 > rect2[1, i]:
            # print(rect2, " does not contain ", rect1, " since ", rect1[0, i], "<", rect2[0, i], " or ", rect1[1, i],
            #      ">", rect2[1, i])
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


def do_rects_list_contain(rect1, list_rects):  # reimplement using z3
    # print("starting do_rects_list_contain")
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

    n = rect1.shape[1]
    list_rects = [np.array([rect[:n], rect[n:]]) for rect in list_rects]
    rect2 = get_convex_union(list_rects)
    if not does_rect_contain(rect1, rect2):
        return False
    if check_rect_empty(rect1, 1) or len(list_rects) == 0:
        print(rect1)
        print("Do not pass empty rectangles to cover function")
        return 0
    else:
        partition_list = []
        for dim in range(n):
            dim_partition_list = [rect1[0, dim], rect1[1, dim]]
            for rect_item in list_rects:
                if rect1[0, dim] <= rect_item[0, dim] <= rect1[1, dim]:
                    dim_partition_list.append(rect_item[0, dim])
                if rect1[0, dim] <= rect_item[1, dim] <= rect1[1, dim]:
                    dim_partition_list.append(rect_item[1, dim])
            dim_partition_list = np.sort(np.array(dim_partition_list))
            partition_list.append(dim_partition_list)

        curr_key = np.zeros((n,))
        quantized_key_range = np.array([len(dim_partition_list) - 2 for dim_partition_list in partition_list])
        while True:
            cell_covered = False
            for rect_item in list_rects:
                if does_rect_contain(np.array([[partition_list[dim][int(curr_key[dim])] for dim in range(n)],
                                               [partition_list[dim][int(curr_key[dim]) + 1] for dim in range(n)]]),
                                     np.array([rect_item[0, :], rect_item[1, :]])):
                    cell_covered = True
                    break
            if not cell_covered:
                # print("ending do_rects_list_contain with cell not covered")
                return False
            if np.all(curr_key == quantized_key_range):
                break
            curr_key = next_quantized_key(curr_key, quantized_key_range).astype(int)
    # print("ending do_rects_list_contain")
    return True


# This function is useless for the meantime (Hussein: June 5th, 2022)
# It is used now (June 7th, 2022)


def get_convex_union(list_array: List[np.array]) -> np.array:
    assert len(list_array) > 0, "list array length should be larger than zero"
    result: np.array = np.copy(list_array[0])
    for i in range(1, len(list_array)):
        result[0, :] = np.minimum(result[0, :], list_array[i][0, :])
        result[1, :] = np.maximum(result[1, :], list_array[i][1, :])
    return result


def build_unified_abstraction(Symbolic_reduced, state_dimensions, s_ind, u_ind, t_ind):
    transformation_list = []
    n = state_dimensions.shape[1]
    unified_reachable_set = np.vstack(
        (Symbolic_reduced[s_ind, u_ind, np.arange(n), t_ind], Symbolic_reduced[s_ind, u_ind, n + np.arange(n), t_ind]))
    unified_center = (Symbolic_reduced[s_ind, u_ind, np.arange(n), t_ind] + Symbolic_reduced[
        s_ind, u_ind, n + np.arange(n), t_ind]) / 2
    for other_u_ind in range(Symbolic_reduced.shape[1]):
        specific_center = (Symbolic_reduced[s_ind, other_u_ind, np.arange(n), t_ind] + Symbolic_reduced[
            s_ind, other_u_ind, n + np.arange(n), t_ind]) / 2
        transformation_vec = find_frame(specific_center, specific_center, unified_center, unified_center)
        transformed_rect = transform_to_frame(np.array([Symbolic_reduced[s_ind, other_u_ind, np.arange(n), t_ind],
                                                        Symbolic_reduced[s_ind, other_u_ind, n + np.arange(n), t_ind]]),
                                              transformation_vec[0, 0, :], overapproximate=True)
        unified_reachable_set = get_convex_union([unified_reachable_set, transformed_rect])
        transformation_list.append(transformation_vec[0, 0, :])
    return unified_reachable_set, transformation_list


def build_global_unified_abstraction(Symbolic_reduced, state_dimensions):
    global_transformation_list = []
    global_unified_reachable_sets = []
    for s_ind in range(Symbolic_reduced.shape[0]):
        global_transformation_list.append([])
        global_unified_reachable_sets.append([])
        for u_ind in range(Symbolic_reduced.shape[1]):
            global_transformation_list[-1].append([])
            global_unified_reachable_sets[-1].append([])
            for t_ind in range(Symbolic_reduced.shape[3]):
                unified_reachable_set, transformation_list = \
                    build_unified_abstraction(Symbolic_reduced, state_dimensions, s_ind, u_ind, t_ind)
                global_transformation_list[-1][-1].append(transformation_list)
                global_unified_reachable_sets[-1][-1].append(unified_reachable_set)
    return global_transformation_list, global_unified_reachable_sets


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

    partition_threshold = 0.01

    original_abstract_paths = []
    abstract_path_last_set_parts = []
    abstract_rtree_idx3d = index.Index('3d_index_abstract',
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
            abstract_path_last_set_parts[-1][-1].append(np.array([abstract_rect_low, abstract_rect_up]))
            abstract_rtree_idx3d.insert(abstract_rect_global_cntr, (abstract_rect_low[0], abstract_rect_low[1],
                                                                    abstract_rect_low[2], abstract_rect_up[0],
                                                                    abstract_rect_up[1], abstract_rect_up[2]),
                                        obj=(s_ind, u_ind))
            abstract_rect_global_cntr += 1
            original_abstract_path = []
            for t_ind in range(Symbolic_reduced.shape[3]):
                original_abstract_path.append(np.array([Symbolic_reduced[s_ind, u_ind, np.arange(n), t_ind],
                                                        Symbolic_reduced[s_ind, u_ind, n + np.arange(n), t_ind]]))

            original_abstract_paths[-1].append(copy.deepcopy(original_abstract_path))
            abstract_paths.append(original_abstract_path)

    # abstract_paths = copy.deepcopy(original_abstract_paths)
    # Target_up = np.array([[10, 6.5, 2 * math.pi / 3]])
    # Target_low = np.array([[7, 0, math.pi / 3]])
    initial_set = np.array([[7, 0, math.pi/3], [7.1, 0.1, math.pi/3 + 0.01]])
    traversal_stack = [copy.deepcopy(initial_set)]
    while len(traversal_stack) and fail_itr < M:
        initial_set = traversal_stack.pop()

        result = synthesize_helper(Symbolic_reduced, sym_x, sym_u, state_dimensions, Target_low, Target_up,
                                   Obstacle_low, Obstacle_up, X_low, X_up, abstract_rtree_idx3d,
                                   abstract_rect_global_cntr, abstract_paths, initial_set, N)

        itr += 1
        if result != -1:
            succ_itr += 1
            print(time.time() - t_start, " ", abstract_rect_global_cntr,
                  " new controllable states have been found in this synthesis iteration\n")
            print(abstract_rect_global_cntr, ' symbols are controllable to satisfy the reach-avoid specification\n')
            print("success iterations: ", succ_itr)
        else:
            fail_itr += 1
            print('No new controllable state has been found in this synthesis iteration\n', time.time() - t_start)
            print("fail iterations: ", fail_itr)
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

                    abstract_path_last_set_parts[s_ind][u_ind] = copy.deepcopy(new_last_sets)

            # break;
        print("# of RRT iterations so far: ", itr)

    print(['Controller synthesis for reach-avoid specification: ', time.time() - t_start, ' seconds'])
    # controllable_states = np.nonzero(Controller);
    if abstract_rect_global_cntr:
        print(abstract_rect_global_cntr, ' symbols are controllable to satisfy the reach-avoid specification\n')
    else:
        print('The reach-avoid specification cannot be satisfied from any initial state\n')

    '''
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

    plt.ylim([X_low[1], X_up[1]])
    plt.xlim([X_low[0], X_up[0]])
    '''

    plt.figure("Reduced coordinates")
    color = 'orange'
    edge_color = 'k'
    currentAxis_1 = plt.gca()
    for s_ind in range(Symbolic_reduced.shape[0]):
        for u_ind in range(Symbolic_reduced.shape[1]):
            for idx in range(Symbolic_reduced.shape[3]):
                abstract_rect_low = Symbolic_reduced[s_ind, u_ind, np.arange(n), idx]
                abstract_rect_up = Symbolic_reduced[s_ind, u_ind, n + np.arange(n), idx]
                rect_patch = Rectangle(abstract_rect_low[[0, 1]], abstract_rect_up[0] - abstract_rect_low[0],
                                       abstract_rect_up[1] - abstract_rect_low[1], linewidth=1,
                                       edgecolor=edge_color, facecolor=color)
                currentAxis_1.add_patch(rect_patch)
    plt.ylim([-1, 1])
    plt.xlim([-1.5, 1.5])

    plt.show()


def synthesize_helper(Symbolic_reduced, sym_x, sym_u, state_dimensions, Target_low, Target_up,
                      Obstacle_low, Obstacle_up, X_low, X_up, abstract_rtree_idx3d, abstract_rect_global_cntr,
                      abstract_paths, initial_set, max_path_length):
    n = state_dimensions.shape[1]
    num_nearest_controls = int(Symbolic_reduced.shape[1] / 2)
    print("new synthesize_helper call ")
    print("initial_set: ", initial_set)
    print("max_path_length: ", max_path_length)
    abstract_targets = []
    for target_idx in range(Target_low.shape[0]):
        target_rect = np.array([Target_low[target_idx, :], Target_up[target_idx, :]])
        abstract_target_1 = transform_rect_to_abstract(target_rect, initial_set[0, :])
        abstract_target_2 = transform_rect_to_abstract(target_rect, initial_set[1, :])
        if do_rects_inter(abstract_target_1, abstract_target_2):
            abstract_target_up = np.minimum(abstract_target_1[1, :], abstract_target_2[1, :])
            abstract_target_low = np.maximum(abstract_target_1[0, :], abstract_target_2[0, :])
            abstract_targets.append(np.array([abstract_target_low, abstract_target_up]))

    if len(abstract_targets) == 0:
        return -1

    abstract_obstacles = []
    for obstacle_idx in range(Obstacle_low.shape[0]):
        obstacle_rect = np.array([Obstacle_low[obstacle_idx, :], Obstacle_up[obstacle_idx, :]])
        abstract_obstacle_1 = transform_rect_to_abstract(obstacle_rect, initial_set[0, :])
        abstract_obstacle_2 = transform_rect_to_abstract(obstacle_rect, initial_set[1, :])
        abstract_obstacle_low = np.minimum(abstract_obstacle_1[1, :], abstract_obstacle_2[1, :])
        abstract_obstacle_up = np.maximum(abstract_obstacle_1[0, :], abstract_obstacle_2[0, :])
        abstract_obstacles.append(np.array([abstract_obstacle_low, abstract_obstacle_up]))

    hits_not_intersecting_obstacles = []
    for target_idx in range(Target_low.shape[0]):
        abstract_target_rect = abstract_targets[target_idx]
        hits = list(abstract_rtree_idx3d.nearest(
            (abstract_target_rect[0, 0], abstract_target_rect[0, 1], abstract_target_rect[0, 2],
             abstract_target_rect[1, 0] + 0.01, abstract_target_rect[1, 1] + 0.01,
             abstract_target_rect[1, 2]
             + 0.01), 2, objects=True))
        for hit in hits:
            good_hit = True
            intersects_obstacle = False
            if not does_rect_contain(np.array([hit.bbox[:n], hit.bbox[n:]]), abstract_target_rect):
                good_hit = False
            path = abstract_paths[hit.id]
            for rect in path:
                for abstract_obstacle in abstract_obstacles:
                    if do_rects_inter(rect, abstract_obstacle):
                        intersects_obstacle = True
                        break
                if intersects_obstacle:
                    break
            if good_hit and not intersects_obstacle:
                return hit.id
            if not intersects_obstacle:
                hits_not_intersecting_obstacles.append(hit)

    if len(hits_not_intersecting_obstacles) == 0 or max_path_length == 0:
        return -1  # Failure

    for hit in hits_not_intersecting_obstacles:
        new_initial_set = transform_to_frames(abstract_paths[hit.id][-1][0, :], abstract_paths[hit.id][-1][1, :],
                                              initial_set[0, :], initial_set[1, :])

        result = synthesize_helper(Symbolic_reduced, sym_x, sym_u, state_dimensions, Target_low, Target_up,
                                   Obstacle_low, Obstacle_up, X_low, X_up, abstract_rtree_idx3d,
                                   abstract_rect_global_cntr, abstract_paths, new_initial_set, max_path_length - 1)

        if result in range(len(abstract_paths)):
            result_rect = abstract_paths[result][-1]
            new_path = []
            for rect in abstract_paths[result]:
                new_path.append(transform_to_frames(rect[0, :], rect[1, :],
                                                    abstract_paths[hit.id][-1][0, :], abstract_paths[hit.id][-1][1, :]))
            abstract_paths.append(new_path)
            abstract_rtree_idx3d.insert(abstract_rect_global_cntr, (result_rect[0, 0], result_rect[0, 1],
                                                                    result_rect[0, 2], result_rect[1, 0],
                                                                    result_rect[1, 1], result_rect[1, 2]),
                                        obj=result)
            abstract_rect_global_cntr += 1
            return hit.id

    return -1
