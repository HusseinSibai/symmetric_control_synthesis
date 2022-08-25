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
    return result


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

        # print("curr_low_1 x: ", (low_red[0]) * math.cos(theta_low_sys) - (low_red[1]) * math.sin(theta_low_sys))
        # print("curr_low_1 y: ", (low_red[0]) * math.sin(theta_low_sys) + (low_red[1]) * math.cos(theta_low_sys))
        # print("curr_up_1 x: ", (up_red[0]) * math.cos(theta_low_sys) - (up_red[1]) * math.sin(theta_low_sys))
        # print("curr_up_1 y: ", (up_red[0]) * math.sin(theta_low_sys) + (up_red[1]) * math.cos(theta_low_sys))
        # print("target_full_low: ", target_full_low)
        # print("target_full_up: ", target_full_up)

        # sanity_check_transformed_rect = transform_to_frames(low_red, up_red, curr_low_1, curr_up_1);
        # print("curr_low_1, curr_up_1: ", curr_low_1, curr_up_1)
        # print("low_red, up_red: ", low_red, up_red)
        # print("sanity_check_transformed_rect: ", sanity_check_transformed_rect)
        # print("target_full_low, target_full_up:", target_full_low, target_full_up)
        # assert does_rect_contain(sanity_check_transformed_rect, np.array([target_full_low, target_full_up])), "find_frame is wrong!"

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
            curr_low[2] = curr_low_1[2]  # np.minimum(curr_low_temp[2], curr_up_temp[2]);
            curr_up = curr_up_temp
            curr_up[2] = curr_up_2[2]  # np.maximum(curr_low_temp[2], curr_up_temp[2]);
            # print("curr_low_1: ", curr_low_1, " curr_low_2: ", curr_low_2, " curr_up_1: ", curr_up_1, "curr_up_2: ",
            #      curr_up_2)
            # if np.all(curr_low <= curr_up):
            rect_curr.append([curr_low.tolist(), curr_up.tolist()])
        # else:
        #    print("find_frame resulted in a rotated rectangle ", np.array([curr_low_1, curr_up_1]),
        #          " that is non-intersecting the rotated rectangle  ", np.array([curr_low_2, curr_up_2]))

    if len(rect_curr) == 0:
        return float('nan')

    rect_curr = np.array(rect_curr)

    # Note: the following block of code was removed since find_frame is now used also for single states instead of boxes.
    # if check_rect_empty(rect_curr[0, :, :], 1):
    #    print('Result is an empty rectangle!!', rect_curr)
    #    return float('nan');
    # rect_curr = np.concatenate((rect_curr, [curr_low, curr_up]), 0);
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


def build_unified_abstraction(Symbolic_reduced, state_dimensions, s_ind, u_ind):
    transformation_list = []
    n = state_dimensions.shape[1]
    unified_reachable_set = np.vstack(
        (Symbolic_reduced[s_ind, u_ind, np.arange(n), -1], Symbolic_reduced[s_ind, u_ind, n + np.arange(n), -1]))
    unified_center = (Symbolic_reduced[s_ind, u_ind, np.arange(n), -1] + Symbolic_reduced[
        s_ind, u_ind, n + np.arange(n), -1]) / 2
    for other_u_ind in range(Symbolic_reduced.shape[1]):
        specific_center = (Symbolic_reduced[s_ind, other_u_ind, np.arange(n), -1] + Symbolic_reduced[
            s_ind, other_u_ind, n + np.arange(n), -1]) / 2
        transformation_vec = find_frame(specific_center, specific_center, unified_center, unified_center)
        transformed_rect = transform_to_frame(np.array([Symbolic_reduced[s_ind, other_u_ind, np.arange(n), -1],
                                                        Symbolic_reduced[s_ind, other_u_ind, n + np.arange(n), -1]]),
                                              transformation_vec[0, 0, :])
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
            unified_reachable_set, transformation_list = \
                build_unified_abstraction(Symbolic_reduced, state_dimensions, s_ind, u_ind)
            global_transformation_list[-1].append(transformation_list)
            global_unified_reachable_sets[-1].append(unified_reachable_set)
    return global_transformation_list, global_unified_reachable_sets


def reach_avoid_synthesis_sets_forward_pass(initial_set,
                                            Symbolic_reduced, sym_x, sym_u, state_dimensions, Target_low, Target_up,
                                            Obstacle_low, Obstacle_up, X_low, X_up, U_low, U_up):
    n = state_dimensions.shape[1]
    init_radius = n * [0.5]  # unified_reachable_sets[-1][1, :] - unified_reachable_sets[-1][0, :];
    itr = 0
    num_trials = 10000
    target_aiming_prob = 0.8

    t_start = time.time()

    global_transformation_list, global_unified_reachable_sets = \
        build_global_unified_abstraction(Symbolic_reduced, state_dimensions)

    # print("global_transformation_list: ", global_transformation_list)
    # print("global_unified_reachable_sets: ", global_unified_reachable_sets)
    color_initial = 'y'
    color_reach = 'b'
    color_target = 'm'
    p = index.Property()
    p.dimension = 3
    p.dat_extension = 'data'
    p.idx_extension = 'index'
    rtree_idx3d = index.Index('3d_index_forward_pass', properties=p)
    discovered_rect = []  # a list version of rtree_idx3d
    initial_discovered_rect = []
    target_discovered_rect = []
    obstacles_intersecting_rect = []

    tracking_rtree_idx3d = index.Index('3d_index_tracking_forward_pass', properties=p)
    tracking_rect_global_cntr_ids = 0
    tracking_rect_global_cntr = 0
    # tracking_rects = [];
    # tracking_abstract_state_control = []  # this tracks the corresponding control and the
    # abstract discrete state of tracking_rects.
    # tracking_rrt_nodes = [] # this should replace the previous two variables.

    # rtree_idx3d_control = index.Index('3d_index_control', properties=p);

    abstract_rect_global_cntr = 0
    abstract_rtree_idx3d = index.Index('3d_index_abstract_forward_pass',
                                       properties=p)  # contains the reachsets of the abstract system.
    max_reachable_distance = 0
    for s_ind in range(Symbolic_reduced.shape[0]):
        for u_ind in range(Symbolic_reduced.shape[1]):
            abstract_rect_low = Symbolic_reduced[s_ind, u_ind, np.arange(n), -1]
            abstract_rect_up = Symbolic_reduced[s_ind, u_ind, n + np.arange(n), -1]
            abstract_rtree_idx3d.insert(abstract_rect_global_cntr, (abstract_rect_low[0], abstract_rect_low[1],
                                                                    abstract_rect_low[2], abstract_rect_up[0],
                                                                    abstract_rect_up[1], abstract_rect_up[2]),
                                        obj=(s_ind, u_ind))
            max_reachable_distance = max(max_reachable_distance,
                                         np.linalg.norm(np.average(np.array([abstract_rect_low.tolist(),
                                                                             abstract_rect_up.tolist()]))))
            abstract_rect_global_cntr = abstract_rect_global_cntr + 1

    transformed_symbolic_rects = []
    initial_transformed_symbolic_rects = []
    target_transformed_symbolic_rects = []
    figure_initial_set = np.array([[1, 1, 1.5], [1.2, 1.2, 1.6]])
    rect = figure_initial_set  # np.array([[1, 1, 1.5],[1.2, 1.2, 1.6]])
    for s_ind in range(Symbolic_reduced.shape[0]):
        for u_ind in range(Symbolic_reduced.shape[1]):
            for t_ind in range(Symbolic_reduced.shape[3]):
                reachable_rect = transform_to_frame(
                    np.array([Symbolic_reduced[s_ind, u_ind, np.arange(n), t_ind],
                              Symbolic_reduced[s_ind, u_ind, n + np.arange(n), t_ind]]),
                    global_transformation_list[s_ind][0][u_ind],
                    overapproximate=False)
                reachable_rect = transform_to_frames(
                    reachable_rect[0, :],
                    reachable_rect[1, :],
                    rect[0, :],
                    rect[1, :])
                if t_ind == 0:
                    initial_transformed_symbolic_rects.append(reachable_rect)
                elif t_ind == Symbolic_reduced.shape[3] - 1:
                    target_transformed_symbolic_rects.append(reachable_rect)
                else:
                    transformed_symbolic_rects.append(reachable_rect)

    plt.figure("Example transformed coordinates")
    currentAxis = plt.gca()
    for rect in transformed_symbolic_rects:
        rect_patch = Rectangle(rect[0, [0, 1]], rect[1, 0] - rect[0, 0],
                               rect[1, 1] - rect[0, 1], linewidth=1,
                               edgecolor='k', facecolor=color_reach)
        currentAxis.add_patch(rect_patch)

    for rect in initial_transformed_symbolic_rects:
        rect_patch = Rectangle(rect[0, [0, 1]], rect[1, 0] - rect[0, 0],
                               rect[1, 1] - rect[0, 1], linewidth=1,
                               edgecolor='k', facecolor=color_initial)
        currentAxis.add_patch(rect_patch)

    for rect in target_transformed_symbolic_rects:
        rect_patch = Rectangle(rect[0, [0, 1]], rect[1, 0] - rect[0, 0],
                               rect[1, 1] - rect[0, 1], linewidth=1,
                               edgecolor='k', facecolor=color_target)
        currentAxis.add_patch(rect_patch)

    plt.ylim([-1, 2])
    plt.xlim([0, 2.5])
    plt.show()

    rect_curr_cntr = 0
    rect_global_cntr = 0
    intersection_time = 0
    contain_time = 0
    insert_time = 0
    nearest_time = 0

    # defining the z3 solver that we'll use to check if a rectangle is in a set of rectangles
    cur_solver = Solver()
    var_dict = []
    for dim in range(n):
        var_dict.append(Real("x" + str(dim)))

    # targets = [];
    for target_idx in range(Target_low.shape[0]):
        # targets.append([Target_low[target_idx, :], Target_up[target_idx, :]]);
        rtree_idx3d.insert(rect_global_cntr, (Target_low[target_idx, 0], Target_low[target_idx, 1],
                                              Target_low[target_idx, 2], Target_up[target_idx, 0],
                                              Target_up[target_idx, 1], Target_up[target_idx, 2]),
                           obj=(-1, -1, 1))

    tracking_rtree_idx3d.insert(rect_global_cntr, (initial_set[0, 0], initial_set[0, 1],
                                                   initial_set[0, 2], initial_set[1, 0],
                                                   initial_set[1, 1], initial_set[1, 2]),
                                obj=(-1, -1, 1))

    # targets = np.array(targets);
    print('\n%s\tStart of the control synthesis\n', time.time() - t_start)

    sampling_rectangle = np.array([X_low, X_up])

    checking_covered_tracking_ctr = 0
    tracking_start_ctr = 0
    new_start_ctr = 0
    # targets_temp = [];
    success = False
    sample_cntr = 0
    while not success and sample_cntr < num_trials:
        # here we use a simple version of RRT to find a path from the sampled state towards rtree_idx3d.
        rrt_done = False
        sample_cntr += 1
        progress_indicator = False
        sampled_state = sampling_rectangle[0, :] + np.array([random.random() * ub for ub in
                                                             sampling_rectangle[1, :].tolist()])

        hits_tracking = list(tracking_rtree_idx3d.nearest(
            (sampled_state[0], sampled_state[1], sampled_state[2],
             sampled_state[0] + 0.01, sampled_state[1] + 0.01,
             sampled_state[2] + 0.01), 1, objects=True))
        nearest_tracking_rect = np.array([hits_tracking[0].bbox[:n], hits_tracking[0].bbox[n:]])
        nearest_node = hits_tracking[0].object
        nearest_tracking_rect_center = \
            nearest_tracking_rect[0, :] + np.array([random.random() * ub for ub in nearest_tracking_rect[1,
                                                                                   :].tolist()])
        nearest_tracking_rect = np.array([np.maximum(nearest_tracking_rect_center - init_radius,
                                                     nearest_tracking_rect[0, :]),
                                          np.minimum(nearest_tracking_rect_center + init_radius,
                                                     nearest_tracking_rect[1, :])])
        tracking_start_ctr += 1

        if random.random() < target_aiming_prob:  # self.goal_sample_rate:
            # hits = list(rtree_idx3d.nearest(
            #    (nearest_tracking_rect_center[0], nearest_tracking_rect_center[1],
            #     nearest_tracking_rect_center[2],
            #     nearest_tracking_rect_center[0] + 0.01, nearest_tracking_rect_center[1] + 0.01,
            #     nearest_tracking_rect_center[2] + 0.01), 1, objects=True))
            # nearest_rect = np.array([hits[0].bbox[:n], hits[0].bbox[n:]])
            temp_rand_int = random.randint(0, Target_low.shape[0] - 1)
            print("temp_rand_int: ", temp_rand_int)
            nearest_rect = np.array([Target_low[temp_rand_int, :], Target_up[temp_rand_int, :]])
        else:
            nearest_rect = np.array([sampled_state - init_radius, sampled_state + init_radius])
        # number of nearest rectangles in the previous command is larger than 1.
        # print("nearest_rect: ", nearest_rect)
        # nearest_poly = pc.box2poly(nearest_rect.T);
        # if rtree_idx3d.count((sampled_state[0], sampled_state[1], sampled_state[2],
        #                      sampled_state[0] + 0.01,  sampled_state[1] + 0.01, sampled_state[2] + 0.01)) > 0:
        #    continue
        # print("path state: ", path_state)
        # abstract_nearest_poly = transform_poly_to_abstract(nearest_poly, np.average(tracking_rects[-1], axis=0));
        abstract_nearest_rect = transform_rect_to_abstract(nearest_rect,
                                                           np.average(nearest_tracking_rect, axis=0))
        # np.column_stack(
        # abstract_nearest_poly.bounding_box).T;
        # print("abstract_nearest_rect: ", abstract_nearest_rect)
        hits = list(abstract_rtree_idx3d.nearest(
            (abstract_nearest_rect[0, 0], abstract_nearest_rect[0, 1], abstract_nearest_rect[0, 2],
             abstract_nearest_rect[1, 0] + 0.01, abstract_nearest_rect[1, 1] + 0.01,
             abstract_nearest_rect[1, 2]
             + 0.01), 1, objects=True))
        # print("nearest abstract reachable set: ", hits[0].bbox)
        s_ind = hits[0].object[0]
        u_ind = hits[0].object[1]
        reachable_set = []
        initial_set = nearest_tracking_rect  # np.array([nearest_tracking_rect_center - init_radius,
        #          nearest_tracking_rect_center + init_radius]);
        for t_ind in range(Symbolic_reduced.shape[3] - 1):  # the last reachable_rect
            # we get from the unified one to make sure all transformed reachable sets end up in the target.
            reachable_set.append(transform_to_frames(Symbolic_reduced[s_ind, u_ind, np.arange(n), t_ind],
                                                     Symbolic_reduced[
                                                         s_ind, u_ind, n + np.arange(n), t_ind],
                                                     initial_set[0, :], initial_set[1, :]))
            # discovered_rect.append(reachable_set[-1])
        reachable_set.append(transform_to_frames(global_unified_reachable_sets[s_ind][u_ind][0, :],
                                                 global_unified_reachable_sets[s_ind][u_ind][1, :],
                                                 initial_set[0, :],
                                                 initial_set[1, :]))
        intersects_obstacle = False
        for reachable_rect in reachable_set:
            i = 0
            # discovered_rect.append(reachable_rect)
            while i < Obstacle_up.shape[0]:  # and np.any(rect_curr):
                rect_obs = np.array([Obstacle_low[i, :], Obstacle_up[i, :]])
                if do_rects_inter(reachable_rect,  # TODO: change it back.
                                  rect_obs):
                    obstacles_intersecting_rect.append(reachable_rect)
                    intersects_obstacle = True
                    break
                i = i + 1
            if not intersects_obstacle and np.any(reachable_rect[0, :] < X_low) or np.any(
                    reachable_rect[0, :] > X_up) \
                    or np.any(reachable_rect[1, :] < X_low) or np.any(reachable_rect[1, :] > X_up):
                # if np.all(rect_curr[1, :] > X_low):
                #    curr_low = np.maximum(rect_curr[0, :], X_low);
                # else:
                intersects_obstacle = True
            if intersects_obstacle:
                # print("The reachable rect ", reachable_rect, " intersects obstacle ", rect_obs, " :/")
                break
            # else:
            #    curr_low = reachable_set[-1][0, :];
            #    curr_up = reachable_set[-1][1, :];
        if not intersects_obstacle:
            hits = list(
                rtree_idx3d.intersection(
                    (reachable_set[-1][0, 0], reachable_set[-1][0, 1], reachable_set[-1][0, 2],
                     reachable_set[-1][1, 0], reachable_set[-1][1, 1],
                     reachable_set[-1][1, 2]),
                    objects=True))
            inter_num = len(hits)
            if inter_num > 0:
                hits = np.array([np.array([hit.bbox[:n], hit.bbox[n:]]) for hit in hits])
                cur_solver.reset()
                cur_solver = add_rects_to_solver(hits, var_dict, cur_solver)
                uncovered_state = do_rects_list_contain_smt(reachable_set[-1], var_dict, cur_solver)
                rrt_done = uncovered_state is None
            else:
                rrt_done = False
                uncovered_state = np.average(reachable_set[-1], axis=0)
            if not rrt_done:  # only add to the RRT if the last reachable set is not in the target.
                new_node = Node(reachable_set=reachable_set, s_ind=s_ind, u_ind=u_ind)
                new_node.parent = nearest_node
                new_node.id_in_tracking_rtree = tracking_rect_global_cntr_ids
                new_node.uncovered_state = copy.deepcopy(uncovered_state)
                reachable_rect = reachable_set[-1]
                tracking_rtree_idx3d.insert(tracking_rect_global_cntr_ids, (
                    reachable_rect[0, 0], reachable_rect[0, 1], reachable_rect[0, 2],
                    reachable_rect[1, 0], reachable_rect[1, 1], reachable_rect[1, 2]),
                                            obj=new_node)
                tracking_rect_global_cntr += 1
                tracking_rect_global_cntr_ids += 1
        if rrt_done:
            reachable_set_list = [reachable_set]
            abstract_state_control_list = [(s_ind, u_ind)]
            curr_node_list = [None]
            while len(reachable_set_list) > 0:
                reachable_set = reachable_set_list.pop()
                s_ind, u_ind = abstract_state_control_list.pop()
                curr_node = curr_node_list.pop()
                initial_rect = reachable_set[0]
                for rect_idx in range(len(reachable_set)):
                    rect = reachable_set[rect_idx]
                    rtree_idx3d.insert(rect_global_cntr, (rect[0, 0], rect[0, 1], rect[0, 2],
                                                          rect[1, 0], rect[1, 1], rect[1, 2]),
                                       obj=(s_ind, u_ind, 1))
                    rect_global_cntr += 1
                    if rect_idx == 0:
                        initial_discovered_rect.append(rect)
                    elif rect_idx == len(reachable_set) - 1:
                        target_discovered_rect.append(rect)
                    else:
                        discovered_rect.append(rect)

                ####################################
                for other_u_ind in range(len(global_transformation_list[s_ind])):
                    intersects_obstacle = False
                    for t_ind in range(Symbolic_reduced.shape[3]):
                        reachable_rect = transform_to_frame(
                            np.array([Symbolic_reduced[s_ind, other_u_ind, np.arange(n), t_ind],
                                      Symbolic_reduced[s_ind, other_u_ind, n + np.arange(n), t_ind]]),
                            global_transformation_list[s_ind][u_ind][other_u_ind],
                            overapproximate=True)
                        reachable_rect = transform_to_frames(
                            reachable_rect[0, :],
                            reachable_rect[1, :],
                            initial_rect[0, :],
                            initial_rect[1, :])
                        if t_ind == 0:
                            full_initial_set = None
                            if curr_node is None:
                                if nearest_node is not None:
                                    full_initial_set = nearest_node.reachable_set[-1]
                            else:
                                full_initial_set = curr_node.parent.reachable_set[-1]
                            if not full_initial_set is None and not do_rects_inter(reachable_rect, full_initial_set):
                                break
                        # discovered_rect.append(reachable_rect)
                        i = 0
                        while i < Obstacle_up.shape[0]:  # and np.any(rect_curr):
                            rect_obs = np.array([Obstacle_low[i, :], Obstacle_up[i, :]])
                            if do_rects_inter(reachable_rect,  # TODO: change it back.
                                              rect_obs):
                                obstacles_intersecting_rect.append(reachable_rect)
                                intersects_obstacle = True
                                break
                            i = i + 1
                        if intersects_obstacle:
                            break
                    if not intersects_obstacle:
                        for t_ind in range(Symbolic_reduced.shape[3]):
                            reachable_rect = transform_to_frame(
                                np.array([Symbolic_reduced[s_ind, other_u_ind, np.arange(n), t_ind],
                                          Symbolic_reduced[s_ind, other_u_ind, n + np.arange(n), t_ind]]),
                                global_transformation_list[s_ind][u_ind][other_u_ind],
                                overapproximate=False)
                            reachable_rect = transform_to_frames(
                                reachable_rect[0, :],
                                reachable_rect[1, :],
                                initial_rect[0, :],
                                initial_rect[1, :])
                            rtree_idx3d.insert(rect_global_cntr, (reachable_rect[0, 0], reachable_rect[0, 1],
                                                                  reachable_rect[0, 2],
                                                                  reachable_rect[1, 0], reachable_rect[1, 1],
                                                                  reachable_rect[1, 2]),
                                               obj=(s_ind, other_u_ind, t_ind))
                            rect_global_cntr += 1
                            if t_ind == 0:
                                initial_discovered_rect.append(reachable_rect)
                            elif t_ind == Symbolic_reduced.shape[3] - 1:
                                target_discovered_rect.append(reachable_rect)
                            else:
                                discovered_rect.append(reachable_rect)
                ####################################
                if curr_node is None:
                    if nearest_node is not None:
                        curr_node = nearest_node
                    else:
                        break
                else:
                    curr_node = curr_node.parent
                if not curr_node is None:
                    reachable_set = curr_node.reachable_set
                    hits = list(
                        rtree_idx3d.intersection(
                            (reachable_set[-1][0, 0], reachable_set[-1][0, 1], reachable_set[-1][0, 2],
                             reachable_set[-1][1, 0], reachable_set[-1][1, 1],
                             reachable_set[-1][1, 2]),
                            objects=True))
                    inter_num = len(hits)
                    if inter_num > 0:
                        hits = np.array([np.array([hit.bbox[:n], hit.bbox[n:]]) for hit in hits])
                        cur_solver.reset()
                        cur_solver = add_rects_to_solver(hits, var_dict, cur_solver)
                        uncovered_state = do_rects_list_contain_smt(reachable_set[-1], var_dict, cur_solver)
                        if uncovered_state is None:
                            if curr_node.object[0] == -1:  # root node
                                print("A path from every initial state to the target has been found!!!")
                                success = True
                                break
                            reachable_set_list.append(reachable_set)
                            abstract_state_control_list.append((curr_node.s_ind, curr_node.u_ind))
                            curr_node_list.append(curr_node)

            # if success:
            #    break
            progress_indicator = True

        if progress_indicator:
            print(time.time() - t_start, " ", rect_global_cntr - rect_curr_cntr,
                  " new controllable states have been found in this synthesis iteration\n")
            rect_curr_cntr = rect_global_cntr;
            print(rect_global_cntr, ' symbols are controllable to satisfy the reach-avoid specification\n')
            # trying to enlarge each of the rectangles in rect_curr
            print("itr: ", itr)
            print("checking_covered_tracking_ctr: ", checking_covered_tracking_ctr)
            print("tracking_start_ctr: ", tracking_start_ctr)
            print("new_start_ctr: ", new_start_ctr)
            print("tracking_rect_global_cntr: ", tracking_rect_global_cntr)
        else:
            print('No new controllable state has been found in this synthesis iteration\n', time.time() - t_start)
            itr += 1;

    print(['Controller synthesis for reach-avoid specification: ', time.time() - t_start, ' seconds'])
    # controllable_states = np.nonzero(Controller);
    if rect_global_cntr:
        print(rect_global_cntr, ' symbols are controllable to satisfy the reach-avoid specification\n')
    else:
        print('The reach-avoid specification cannot be satisfied from any initial state\n')

    # print("The number of states that were supposed to be explored but weren't is:", len(initial_sets_to_explore));

    # for c in cur_solver.assertions():
    #    print(c)

    # print("discovered rect: ", discovered_rect)
    plt.figure("Original coordinates")
    currentAxis = plt.gca()
    color = 'r'
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
    edge_color = 'k'
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

    color = 'y';
    print("obstacles_intersecting_rect: ", obstacles_intersecting_rect)
    for rect in obstacles_intersecting_rect:
        rect_patch = Rectangle(rect[0, [0, 1]], rect[1, 0] - rect[0, 0],
                               rect[1, 1] - rect[0, 1], linewidth=1,
                               edgecolor=edge_color, facecolor=color)
        # currentAxis.add_patch(rect_patch)

    plt.ylim([X_low[1], X_up[1]])
    plt.xlim([X_low[0], X_up[0]])

    plt.figure("Reduced coordinates")
    color = 'orange'
    currentAxis_1 = plt.gca()
    for s_ind in range(Symbolic_reduced.shape[0]):
        for u_ind in range(Symbolic_reduced.shape[1]):
            for idx in range(Symbolic_reduced.shape[3]):
                abstract_rect_low = Symbolic_reduced[s_ind, u_ind, np.arange(n), idx];
                abstract_rect_up = Symbolic_reduced[s_ind, u_ind, n + np.arange(n), idx]
                rect_patch = Rectangle(abstract_rect_low[[0, 1]], abstract_rect_up[0] - abstract_rect_low[0],
                                       abstract_rect_up[1] - abstract_rect_low[1], linewidth=1,
                                       edgecolor=edge_color, facecolor=color)
                currentAxis_1.add_patch(rect_patch)
    plt.ylim([-1, 1])
    plt.xlim([-1.5, 1.5])

    plt.show()
