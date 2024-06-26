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
    translation_vector = state
    rot_angle = state[2]
    poly_out: pc.Polytope = poly.rotation(i=0, j=1, theta=rot_angle)
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


# translates and rotates a rectangle to a new coordinate system with the new origin: state.
# Then it under-approximates it with an axis-aligned rectangle.
'''
def transform_to_frame(rect, state):
    low_red = np.array(
        [(rect[0, 0]) * math.cos(state[2]) -
         (rect[0, 1]) * math.sin(state[2]) + state[0],
         (rect[0, 0]) * math.sin(state[2]) +
         (rect[0, 1]) * math.cos(state[2]) + state[1],
         rect[0, 2] + state[2]]);
    up_red = np.array(
        [(rect[1, 0]) * math.cos(state[2]) -
         (rect[1, 1]) * math.sin(state[2]) + state[0],
         (rect[1, 0]) * math.sin(state[2]) +
         (rect[1, 1]) * math.cos(state[2]) + state[1],
         rect[1, 2] + state[2]]);
    rotated_rect = np.array([low_red, up_red]);
    center = np.average(rotated_rect, axis=0);
    width = np.array(rect[1, :] - rect[0, :]);

    low_red_axis_aligned = np.array(
        [center[0] + width[0] * math.cos(state[2]), center[1] + width[0] * math.sin(state[2]), low_red[2]]);
    up_red_axis_aligned = np.array(
        [center[0] + width[0] * math.cos(state[2]), center[1] + width[0] * math.sin(state[2]), low_red[2]]);
    return np.array([low_red, up_red])
'''


def transform_to_frames(low_red, up_red, source_full_low, source_full_up):
    # poly_1 = pc.box2poly(np.column_stack((low_red, up_red)));
    # poly_1 = transform_poly_to_abstract(poly_1, source_full_low);
    # poly_2 = pc.box2poly(np.column_stack((low_red, up_red)));
    # poly_2 = transform_poly_to_abstract(poly_2, source_full_up);
    box_1 = transform_to_frame(np.array([low_red, up_red]), source_full_low, overapproximate=True)
    box_2 = transform_to_frame(np.array([low_red, up_red]), source_full_up, overapproximate=True)
    box_3 = transform_to_frame(np.array([low_red, up_red]), np.array([source_full_low[0], source_full_low[1],
                                                                      source_full_up[2]]), overapproximate=True)
    box_4 = transform_to_frame(np.array([low_red, up_red]), np.array([source_full_up[0], source_full_up[1],
                                                                      source_full_low[2]]), overapproximate=True)
    result = get_convex_union([box_1, box_2, box_3, box_4])
    # result = get_convex_union([np.column_stack(poly_1.bounding_box).T, np.column_stack(poly_2.bounding_box).T]);
    '''
    low_red_1 = np.array(
        [(low_red[0]) * math.cos(source_full_low[2]) -
         (low_red[1]) * math.sin(source_full_low[2]) + source_full_low[0],
         (low_red[0]) * math.sin(source_full_low[2]) +
         (low_red[1]) * math.cos(source_full_low[2]) + source_full_low[1],
         low_red[2] + source_full_low[2]]);
    up_red_1 = np.array(
        [(up_red[0]) * math.cos(source_full_low[2]) -
         (up_red[1]) * math.sin(source_full_low[2]) + source_full_low[0],
         (up_red[0]) * math.sin(source_full_low[2]) +
         (up_red[1]) * math.cos(source_full_low[2]) + source_full_low[1],
         up_red[2] + source_full_low[2]]);
    low_red_2 = np.array(
        [(source_full_up[0]) * math.cos(source_full_up[2]) -
         (source_full_up[1]) * math.sin(source_full_up[2]) + source_full_up[0],
         (source_full_up[0]) * math.sin(source_full_up[2]) +
         (source_full_up[1]) * math.cos(source_full_up[2]) + source_full_up[1],
         low_red[2] + source_full_up[2]]);
    up_red_2 = np.array(
        [(source_full_up[0]) * math.cos(source_full_up[2]) -
         (source_full_up[1]) * math.sin(source_full_up[2]) + source_full_up[0],
         (source_full_up[0]) * math.sin(source_full_up[2]) +
         (source_full_up[1]) * math.cos(source_full_up[2]) + source_full_up[1],
         up_red[2] + source_full_up[2]]);
    width = np.array(up_red - low_red);
    # TODO: fix the overapproximation below, June 16th, 2022
    result_low = np.minimum(np.minimum(low_red_1, up_red_1), np.minimum(low_red_2, up_red_2)) - width;  # , up_red_1),
    result_up = np.maximum(np.maximum(low_red_1, up_red_1), np.maximum(low_red_2, up_red_2)) + width;
    '''
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
        '''
        m = cur_solver.model()
        uncovered_state_frac = []
        for var in var_dict:
            uncovered_state_frac.append(m[var].as_fraction())
        uncovered_state = []
        for frac in uncovered_state_frac:
            uncovered_state.append(float(frac.numerator) / float(frac.denominator))
        '''
        uncovered_state = np.average(rect1, axis=0)
        cur_solver.pop()
        return np.array(uncovered_state)
    cur_solver.pop()
    # print("rect1: ", rect1, " is already contained by previous rectangles");
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
'''
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
'''


def get_convex_union(list_array: List[np.array]) -> np.array:
    assert len(list_array) > 0, "list array length should be larger than zero"
    result: np.array = np.copy(list_array[0])
    for i in range(1, len(list_array)):
        result[0, :] = np.minimum(result[0, :], list_array[i][0, :])
        result[1, :] = np.maximum(result[1, :], list_array[i][1, :])
    return result


'''
def build_unified_abstraction(Symbolic_reduced, state_dimensions):
    #   input: reachable sets of the abstract (symmetry-reduced) system
    #   output:
    #   a) a list indexed by the abstract states such that for each such state,
    #   it stores a hyperrectangle bounding the set of transformed (last) reachable sets transformed using symmetries to have the same center
    #   b) a list indexed by the abstract states such that for each such state,
    #   it stores a list indexed by the abstract controls representing the set of transformations that map the reachable sets to match the same center of the first one.
    unified_reachable_sets = [];
    transformation_list = [];
    n = state_dimensions.shape[1];
    for s_ind in range(Symbolic_reduced.shape[0]):
        unified_reachable_sets.append(
            np.vstack(
                (Symbolic_reduced[s_ind, 0, np.arange(n), -1], Symbolic_reduced[s_ind, 0, n + np.arange(n), -1])));
        unified_center = (Symbolic_reduced[s_ind, 0, np.arange(n), -1] + Symbolic_reduced[
            s_ind, 0, n + np.arange(n), -1]) / 2;
        transformation_list.append([np.zeros(n, )]);
        for u_ind in range(1, Symbolic_reduced.shape[1]):
            specific_center = (Symbolic_reduced[s_ind, u_ind, np.arange(n), -1] + Symbolic_reduced[
                s_ind, u_ind, n + np.arange(n), -1]) / 2;
            transformation_vec = find_frame(specific_center, specific_center, unified_center, unified_center);
            # TODO: this should be replaced by transform_to_frame
            transformed_rect = transform_to_frame(np.array([Symbolic_reduced[s_ind, u_ind, np.arange(n), -1],
                                                            Symbolic_reduced[s_ind, u_ind, n + np.arange(n), -1]]),
                                                  transformation_vec[0, 0, :]);

            
            # print("transformation_vec: ", transformation_vec)
            # print("specific center: ", specific_center)
            # print("transformed_rect: ", transformed_rect)
            # print("transformed_rect_center: ", np.average(transformed_rect, axis=0))
            # print("unified_center: ", unified_center)
            # print("transforming_center: ", transform_to_frame(np.array([specific_center, specific_center]),
            #                                                   transformation_vec[0, 0, :]))
            # 
            # if np.any(np.average(transformed_rect, axis=0) != unified_center):
            #     print("ALERT!!!!!", "transformed_rect center ", np.average(transformed_rect, axis=0),
            #           " is different from unified center ", unified_center)
            
            # assert np.all(np.average(transformed_rect, axis=0) == unified_center), \
            #    "Transformed does not align with uniform!"
            unified_reachable_sets[-1] = get_convex_union([unified_reachable_sets[-1], transformed_rect]);
            transformation_list[-1].append(transformation_vec[0, 0, :]);
    # print("transformation_list: ", transformation_list)
    return unified_reachable_sets, transformation_list
'''


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


def reach_avoid_synthesis_sets(Symbolic_reduced, sym_x, sym_u, state_dimensions, Target_low, Target_up,
                               Obstacle_low, Obstacle_up, X_low, X_up, U_low, U_up):
    # defining the z3 solver that we'll use to check if a rectangle is in a set of rectangles
    n = state_dimensions.shape[1]
    cur_solver = Solver()
    var_dict = []
    for dim in range(n):
        var_dict.append(Real("x" + str(dim)))
    init_radius = n * [0.01]  # unified_reachable_sets[-1][1, :] - unified_reachable_sets[-1][0, :];
    init_radius[0] = 0.3
    init_radius[1] = 0.3
    # TODO: design an optimization procedure or make init_radius an array with increasing radii.
    itr = 0
    fail_itr = 0
    succ_itr = 0
    num_trials = 10
    # num_success_trials = 20
    target_aiming_prob = 0.8
    tracking_start_prob = 0.0
    success_continue_prob = 0.5
    num_nearest_controls = int(Symbolic_reduced.shape[1] / 2)

    t_start = time.time()

    # print("global_transformation_list: ", global_transformation_list)
    # print("global_unified_reachable_sets: ", global_unified_reachable_sets)
    color_initial = 'y'
    color_reach = 'b'
    color_target = 'm'
    p = index.Property()
    p.dimension = 3
    p.dat_extension = 'data'
    p.idx_extension = 'index'
    rtree_idx3d = index.Index('3d_index', properties=p)
    discovered_rect = []  # a list version of rtree_idx3d
    initial_discovered_rect = []
    target_discovered_rect = []
    obstacles_intersecting_rect = []

    tracking_rtree_idx3d = index.Index('3d_index_tracking', properties=p)
    tracking_rect_global_cntr_ids = 0
    tracking_rect_global_cntr = 0
    deleted_rects_cntr = 0
    # tracking_rects = [];
    # tracking_abstract_state_control = []  # this tracks the corresponding control and the
    # abstract discrete state of tracking_rects.
    # tracking_rrt_nodes = [] # this should replace the previous two variables.

    # rtree_idx3d_control = index.Index('3d_index_control', properties=p);

    abstract_rect_global_cntr = 0
    abstract_rtree_idx3d = index.Index('3d_index_abstract',
                                       properties=p)  # contains the reachsets of the abstract system.
    obstacles_rtree_idx3d = index.Index('3d_index_obstacles',
                                        properties=p)
    good_targets_rtree_idx3d = index.Index('3d_index_good_targets',
                                           properties=p)
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

    global_transformation_list, global_unified_reachable_sets = \
        build_global_unified_abstraction(Symbolic_reduced, state_dimensions)
    transformed_symbolic_rects = []
    initial_transformed_symbolic_rects = []
    target_transformed_symbolic_rects = []
    initial_set = np.array([[1, 1, 1.5], [1.2, 1.2, 1.6]])
    rect = initial_set  # np.array([[1, 1, 1.5],[1.2, 1.2, 1.6]])
    target_t_ind = 1
    for s_ind in range(Symbolic_reduced.shape[0]):
        for u_ind in range(Symbolic_reduced.shape[1]):
            for t_ind in range(target_t_ind + 1):  # Symbolic_reduced.shape[3]):
                reachable_rect = transform_to_frame(
                    # np.array([Symbolic_reduced[s_ind, u_ind, np.arange(n), t_ind],
                    #          Symbolic_reduced[s_ind, u_ind, n + np.arange(n), t_ind]]),
                    np.array([Symbolic_reduced[s_ind, u_ind, np.arange(n), t_ind],
                              Symbolic_reduced[s_ind, u_ind, n + np.arange(n), t_ind]]),
                    global_transformation_list[s_ind][0][target_t_ind][u_ind],
                    overapproximate=False)
                reachable_rect = transform_to_frames(
                    reachable_rect[0, :],
                    reachable_rect[1, :],
                    rect[0, :],
                    rect[1, :])
                if t_ind == 0:
                    initial_transformed_symbolic_rects.append(reachable_rect)
                elif t_ind == target_t_ind:  # Symbolic_reduced.shape[3] - 1:
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
    plt.close()

    # unified_reachable_sets, unifying_transformation_list = build_unified_abstraction(Symbolic_reduced,
    #                                                                                 state_dimensions);

    # TODO in the case of the existence of non-symmetric coordinates, the following line may need to be changed to be
    #  the max over all the radii of the unified reachable sets over all cells in the grid over the
    #  non-symmetric coordinates

    # plt.figure("Reduced coordinates with transformed reachable sets")
    # color = 'orange'
    # currentAxis_4 = plt.gca()
    # # temp = np.array([[1, 1, 1], [2, 2, 2]]);
    # temp = np.array([[0, 0, 0], [1, 1, -1]]);
    # for s_ind in range(Symbolic_reduced.shape[0]):
    #     for u_ind in range(Symbolic_reduced.shape[1]):
    #         for t_ind in range(Symbolic_reduced.shape[3]):
    #             rect = transform_to_frame(np.array([Symbolic_reduced[s_ind, u_ind, np.arange(n), t_ind],
    #                                                 Symbolic_reduced[s_ind, u_ind, n + np.arange(n), t_ind]]),
    #                                       unifying_transformation_list[s_ind][u_ind],
    #                                       overapproximate=True);
    #             rect = transform_to_frames(rect[0, :],
    #                                        rect[1, :],
    #                                        temp[1, :],
    #                                        temp[1, :]);
    #             rect_patch = Rectangle(rect[0, [0, 1]], rect[1, 0] - rect[0, 0],
    #                                    rect[1, 1] - rect[0, 1], linewidth=1,
    #                                    edgecolor='b', facecolor=color)
    #             currentAxis_4.add_patch(rect_patch)
    #             non_transformed_rect = np.array([Symbolic_reduced[s_ind, u_ind, np.arange(n), -1],
    #                                              Symbolic_reduced[s_ind, u_ind, n + np.arange(n), -1]]);
    #             rect_patch = Rectangle(non_transformed_rect[0, [0, 1]],
    #                                    non_transformed_rect[1, 0] - non_transformed_rect[0, 0],
    #                                    non_transformed_rect[1, 1] - non_transformed_rect[0, 1], linewidth=1,
    #                                    edgecolor='b', facecolor='g')
    #             currentAxis_4.add_patch(rect_patch)
    # plt.ylim([-10, 10])
    # plt.xlim([-10, 10])
    #
    # plt.show()
    # TODO: stop wasting time and implement this. Pick one last reachable set. Get the transformed reachtubes.
    #  Iterate over the reachtube and check if the transformed selected target to align (centerwise) with the
    #  current reachable set is included in all the intersecting rectangle, requires calls to Z3 and rtree.

    init_radius = np.array(init_radius)
    initial_set = np.array([-init_radius, init_radius])
    bloated_initial_set = initial_set
    # np.array([[initial_set[0,0] * math.sqrt(2), initial_set[0,1] * math.sqrt(2), initial_set[0,2]],
    #                                [initial_set[1,0] * math.sqrt(2), initial_set[1,1] * math.sqrt(2), initial_set[0,2]]])
    rect = initial_set

    ########### plot abstraction starting from a set of initial states ###############

    transformed_symbolic_rects = []
    initial_transformed_symbolic_rects = []
    target_transformed_symbolic_rects = []
    for s_ind in range(Symbolic_reduced.shape[0]):
        for u_ind in range(Symbolic_reduced.shape[1]):
            for t_ind in range(Symbolic_reduced.shape[3]):  # the -1 is there because obviously the
                reachable_rect = transform_to_frames(
                    Symbolic_reduced[s_ind, u_ind, np.arange(n), t_ind],
                    Symbolic_reduced[s_ind, u_ind, n + np.arange(n), t_ind],
                    rect[0, :],
                    rect[1, :])
                if t_ind == 0:
                    initial_transformed_symbolic_rects.append(reachable_rect)
                elif t_ind == Symbolic_reduced.shape[3] - 1:
                    target_transformed_symbolic_rects.append(reachable_rect)
                else:
                    transformed_symbolic_rects.append(reachable_rect)

    plt.figure("Abstract reachable sets")
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

    plt.ylim([-5, 5])
    plt.xlim([-5, 5])
    plt.show()
    plt.close()
    currentAxis.clear()

    #########################################
    # for s_ind in range(Symbolic_reduced.shape[0]):
    #    for u_ind in range(Symbolic_reduced.shape[1]):
    #        for target_t_ind in range(Symbolic_reduced.shape[3] - 3,
    #                                  Symbolic_reduced.shape[3] - 2):  # the -1 is there because obviously the
    # last reachable set is covered.
    '''
    for s_ind in range(Symbolic_reduced.shape[0]):
        for u_ind in range(Symbolic_reduced.shape[1]):
            for target_t_ind in range(Symbolic_reduced.shape[3] - 5,
                                      Symbolic_reduced.shape[3]-4):
                all_rects = []
                transformed_symbolic_rects = []
                initial_transformed_symbolic_rects = []
                target_transformed_symbolic_rects = []
                new_initial_sets = []
                could_have_been_new_initial_sets = []
                # target_reachtubes = []
                transformed_reachtubes = []
                # unified_center = np.zeros((n,))
                # unified_center[2] = math.pi / 2
                reachable_rect = np.array([Symbolic_reduced[s_ind, u_ind, np.arange(n), target_t_ind],
                                                      Symbolic_reduced[s_ind, u_ind, n + np.arange(n), target_t_ind]])
                reachable_rect = transform_to_frames(
                    reachable_rect[0, :],
                    reachable_rect[1, :],
                    initial_set[0, :],
                    initial_set[1, :])
                unified_center = np.average(reachable_rect, axis=0)
                target_rect = reachable_rect
                # np.average(np.array([Symbolic_reduced[s_ind, u_ind, np.arange(n), target_t_ind],
                #                                      Symbolic_reduced[s_ind, u_ind, n + np.arange(n), target_t_ind]]),
                #                            axis=0)
                for other_u_ind in range(Symbolic_reduced.shape[1]):
                    # target_reachtube = []
                    transformed_reachtube = []
                    for t_ind in range(target_t_ind + 1):
                        # reachable_rect = transform_to_frame(
                        #    np.array([Symbolic_reduced[s_ind, other_u_ind, np.arange(n), t_ind],
                        #              Symbolic_reduced[s_ind, other_u_ind, n + np.arange(n), t_ind]]),
                        #    global_transformation_list[s_ind][u_ind][target_t_ind][other_u_ind],
                        #    overapproximate=False)
                        reachable_rect = np.array([Symbolic_reduced[s_ind, other_u_ind, np.arange(n), t_ind],
                                                   Symbolic_reduced[s_ind, other_u_ind, n + np.arange(n), t_ind]])
                        reachable_rect = transform_to_frames(
                            reachable_rect[0, :],
                            reachable_rect[1, :],
                            initial_set[0, :],
                            initial_set[1, :])
                        transformed_reachtube.append(reachable_rect)
                        
                        # if t_ind == 0:
                        #    initial_transformed_symbolic_rects.append(reachable_rect)
                        # elif t_ind == target_t_ind:  # Symbolic_reduced.shape[3] - 1:
                        #    target_transformed_symbolic_rects.append(reachable_rect)
                        # else:
                        #    transformed_symbolic_rects.append(reachable_rect)
                        # transformed_reachtube.append(reachable_rect)
                        # all_rects.append(reachable_rect)
                    specific_center = np.average(transformed_reachtube[-1], axis=0)
                    transformation_vec = find_frame(specific_center, specific_center, unified_center,
                                                    unified_center)
                    new_target_rect = np.array([Symbolic_reduced[s_ind, u_ind, np.arange(n), target_t_ind],
                                                Symbolic_reduced[s_ind, u_ind, n + np.arange(n), target_t_ind]])
                    new_target_rect = transform_to_frames(
                        new_target_rect[0, :],
                        new_target_rect[1, :],
                        bloated_initial_set[0, :],
                        bloated_initial_set[1, :])
                    new_target_rect = transform_to_frame(
                        new_target_rect,
                        transformation_vec[0, 0, :], overapproximate=False)
                    # if other_u_ind == 0:
                    #    target_rect = new_target_rect
                    #else:
                    # target_rect[0, 2] = min(target_rect[0, 2], new_target_rect[0, 2]) # get_convex_union([target_rect,
                    # new_target_rect])
                    # target_rect[1, 2] = max(target_rect[1, 2], new_target_rect[1, 2])
                    for t_ind in range(target_t_ind + 1):
                        transformed_reachtube[t_ind] = transform_to_frame(
                            transformed_reachtube[t_ind],
                            transformation_vec[0, 0, :], overapproximate=False)
                        #    global_transformation_list[s_ind][u_ind][target_t_ind][other_u_ind],
                        #    overapproximate=False)
                        all_rects.append(transformed_reachtube[t_ind])
                        if t_ind == 0:
                            initial_transformed_symbolic_rects.append(transformed_reachtube[t_ind])
                        elif t_ind == target_t_ind:  # Symbolic_reduced.shape[3] - 1:
                            target_transformed_symbolic_rects.append(transformed_reachtube[t_ind])
                        else:
                            transformed_symbolic_rects.append(transformed_reachtube[t_ind])
                    transformed_reachtubes.append(transformed_reachtube)
                    # if other_u_ind == u_ind:

                # reachable_rect = transform_to_frame(
                # np.array([Symbolic_reduced[s_ind, other_u_ind, np.arange(n), t_ind],
                #           Symbolic_reduced[s_ind, other_u_ind, n + np.arange(n), t_ind]]),
                # global_unified_reachable_sets[s_ind, u_ind, target_t_ind],
                # global_transformation_list[s_ind][u_ind][target_t_ind][other_u_ind],
                # overapproximate=True)
                #target_rect = transform_to_frames(
                #    global_unified_reachable_sets[s_ind][u_ind][target_t_ind][0, :],
                #    global_unified_reachable_sets[s_ind][u_ind][target_t_ind][1, :],
                #    initial_set[0, :],
                #    initial_set[1, :])
                
                # target_reachtube.append(reachable_rect)
                # target_reachtubes.append(target_reachtube)
                cur_solver.reset()
                cur_solver = add_rects_to_solver(np.array(all_rects), var_dict, cur_solver)
                # (Symbolic_reduced[s_ind, u_ind, np.arange(n), target_t_ind]
                # + Symbolic_reduced[s_ind, u_ind, n + np.arange(n), target_t_ind]) / 2
                # np.average(global_unified_reachable_sets[s_ind][u_ind][target_t_ind], axis=0)
                interval_partition = np.linspace(0, 1, 9).tolist()
                interval_partition.pop()
                # interval_partition.pop(0)
                specific_center = np.average(target_rect, axis=0)  # target_reachtube[-1]
                for other_u_ind in range(Symbolic_reduced.shape[1]):
                    # target_reachtube = target_reachtubes.pop(0)
                    transformed_reachtube = transformed_reachtubes.pop(0)
                    for t_ind in range(target_t_ind):
                        for step in interval_partition:
                            # target_reachtube[t_ind]
                            # target_reachtube[t_ind + 1]
                            unified_center = (np.average(transformed_reachtube[t_ind], axis=0) * step +
                                              np.average(transformed_reachtube[t_ind + 1], axis=0)) * (1 - step)
                            # np.average(target_reachtube[t_ind], axis=0)
                            # (Symbolic_reduced[s_ind, u_ind, np.arange(n), t_ind]
                            # + Symbolic_reduced[s_ind, u_ind, n + np.arange(n), t_ind]) / 2
                            # np.average(global_unified_reachable_sets[s_ind][u_ind][t_ind], axis=0)
                            transformation_vec = find_frame(specific_center, specific_center, unified_center,
                                                            unified_center)
                            # transformed_rect = transform_to_frame(global_unified_reachable_sets[s_ind][u_ind][target_t_ind],
                            #                                       transformation_vec[0, 0, :], overapproximate=True)
                            # target_reachtube[-1]
                            transformed_rect = transform_to_frame(target_rect, transformation_vec[0, 0, :],
                                                                  overapproximate=False)
                            # np.array([Symbolic_reduced[s_ind, u_ind, np.arange(n), target_t_ind],
                            #          Symbolic_reduced[s_ind, u_ind, n + np.arange(n), target_t_ind]]),
                            # print("unified_center: ", unified_center)
                            # print("specific_center: ", specific_center)
                            # print("transformed_rect_center: ", np.average(transformed_rect, axis=0))
                            uncovered_state = do_rects_list_contain_smt(transformed_rect, var_dict,
                                                                        cur_solver)
                            large_initial_set_found = uncovered_state is None
                            if large_initial_set_found:
                                print(transformed_rect, " is a new initial set that can reach ",
                                      global_unified_reachable_sets[s_ind][u_ind][target_t_ind], "using u_ind ", u_ind,
                                      "at t_ind ", t_ind + step)
                                new_initial_sets.append(transformed_rect)
                            else:
                                could_have_been_new_initial_sets.append(transformed_rect)
                                # print("Contracting!!!", transformed_rect, ", the transformed version of ",
                                # global_unified_reachable_sets[0][0][-1], " to the frame ", unified_center, "using the
                                # transformation vector", transformation_vec[0, 0, :], " at t_ind ", t_ind, "is inside the
                                # set of rects ", all_rects) return
                plt.figure("Example transformed coordinates" + str(s_ind) + "_" + str(u_ind) + "_" + str(target_t_ind))
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

                for rect in new_initial_sets:
                    rect_patch = Rectangle(rect[0, [0, 1]], rect[1, 0] - rect[0, 0],
                                           rect[1, 1] - rect[0, 1], linewidth=1,
                                           edgecolor='k', facecolor='r')
                    currentAxis.add_patch(rect_patch)

                for rect in could_have_been_new_initial_sets:
                    rect_patch = Rectangle(rect[0, [0, 1]], rect[1, 0] - rect[0, 0],
                                           rect[1, 1] - rect[0, 1], linewidth=1,
                                           edgecolor='k', facecolor='c')
                    # currentAxis.add_patch(rect_patch)

                plt.ylim([-50, 50])
                plt.xlim([-50, 50])
                plt.show()
                plt.close()
                currentAxis.clear()
    '''
    rect_curr_cntr = 0
    rect_global_cntr = 0
    good_targets_cntr = 0
    intersection_time = 0
    contain_time = 0
    insert_time = 0
    nearest_time = 0

    '''
    for rect_ind in range(Target_low.shape[0]):
        cur_solver.push();
        for dim in range(n):
            cur_solver.add(var_dict[dim] < Target_low[rect_ind, dim]);
            cur_solver.add(var_dict[dim] > Target_up[rect_ind, dim]);
    '''
    '''
    for rect_ind in range(Target_low.shape[0]):
        cur_solver.add(
            Or([var_dict[0] < Target_low[rect_ind, 0], var_dict[1] < Target_low[rect_ind, 1],
                var_dict[2] < Target_low[rect_ind, 2], var_dict[0] > Target_up[rect_ind, 0],
                var_dict[1] > Target_up[rect_ind, 1], var_dict[2] > Target_up[rect_ind, 2]]));
    '''
    '''
    # sym_x_reduced = np.add(np.dot(sym_x[0,:], state_dimensions[0,:]), ~state_dimensions[0,:]);
    sym_x_reduced = sym_x;
    if np.all(max(sym_u) - min(sym_u) == 0):
        sym_u = sym_u[0, 0];
        ui_values = np.arange(sym_u);
        U_discrete = gray(ui_values, U_low.shape[0], sym_u);
        U_discrete = np.add(np.matlib.repmat(np.reshape(U_low,(U_low.shape[0], 1)), 1, int(np.power(sym_u, U_low.shape[0]))),
                            np.multiply((U_discrete - 1), np.matlib.repmat(np.reshape((U_up - U_low) / (sym_u - 1),(U_low.shape[0],1)),
                                                                       1, np.power(sym_u, U_low.shape[0]).astype(int))));
    else:
        print('Non-homogeneous input discretization cannot be handled yet')
    '''

    # matrix_dim_reduced = [int(np.prod(sym_x_reduced)), int(U_discrete.shape[1]), 2 * n];
    # symbol_step = (X_up - X_low) / sym_x;

    # targets = [];
    for target_idx in range(Target_low.shape[0]):
        # targets.append([Target_low[target_idx, :], Target_up[target_idx, :]]);
        rtree_idx3d.insert(rect_global_cntr, (Target_low[target_idx, 0], Target_low[target_idx, 1],
                                              Target_low[target_idx, 2], Target_up[target_idx, 0],
                                              Target_up[target_idx, 1], Target_up[target_idx, 2]),
                           obj=(-1, -1, 1))
        rect_global_cntr += 1
        good_targets_rtree_idx3d.insert(good_targets_cntr, (Target_low[target_idx, 0], Target_low[target_idx, 1],
                                                            Target_low[target_idx, 2], Target_up[target_idx, 0],
                                                            Target_up[target_idx, 1], Target_up[target_idx, 2]),
                                        obj=(-1, -1, 1))
        good_targets_cntr += 1

    obstacle_cntr = 0
    for obstacle_idx in range(Obstacle_low.shape[0]):
        obstacles_rtree_idx3d.insert(obstacle_cntr, (Obstacle_low[obstacle_idx, 0], Obstacle_low[obstacle_idx, 1],
                                                     Obstacle_low[obstacle_idx, 2], Obstacle_up[obstacle_idx, 0],
                                                     Obstacle_up[obstacle_idx, 1], Obstacle_up[obstacle_idx, 2]),
                                     obj=(-1, -1, 1))
        obstacle_cntr += 1

    # targets = np.array(targets);
    print('\n%s\tStart of the control synthesis\n', time.time() - t_start)

    # initial_sets_to_explore = [];  # contains tuples of initial sets, s_ind, u_ind, and target sets because of which
    # they were partitioned.

    '''
    sampling_radius = np.array(n * [5]);
    sampling_rectangle = np.array([Target_low[0] - sampling_radius, Target_up[0] + sampling_radius]);
    sampling_rectangle_low = np.maximum(sampling_rectangle[0, :], X_low);
    sampling_rectangle_low = np.minimum(sampling_rectangle_low, X_up);
    sampling_rectangle_up = np.minimum(sampling_rectangle[1, :], X_up);
    sampling_rectangle_up = np.maximum(sampling_rectangle_up, X_low);
    sampling_rectangle = np.array([sampling_rectangle_low, sampling_rectangle_up]);
    '''
    sampling_rectangle = np.array([X_low, X_up])

    checking_covered_tracking_ctr = 0
    tracking_start_ctr = 0
    new_start_ctr = 0

    while succ_itr < num_trials:  # or len(initial_sets_to_explore):
        # targets_temp = [];
        progress_indicator = False
        useless = True
        tracking_start = False
        while useless:  # and itr < num_trials:  # or len(initial_sets_to_explore)):
            # print("sampling_rectangle[1,:].tolist(): ", sampling_rectangle, sampling_rectangle[1,:].tolist())
            # if len(initial_sets_to_explore) == 0:

            sampled_state = sampling_rectangle[0, :] + np.array(
                [random.random() * ub for ub in
                 sampling_rectangle[1, :].tolist()])  # sample_random_state(X_low, X_up);
            # itr = 0;
            # sampled_state = initial_sets_to_explore.pop(random.randint(0, len(initial_sets_to_explore) - 1));

            # if len(tracking_rects) == 0:
            hit = list(rtree_idx3d.nearest((sampled_state[0], sampled_state[1], sampled_state[2],
                                            sampled_state[0], sampled_state[1], sampled_state[2]), 1,
                                           objects=True))
            if len(hit) == 0:
                useless = False
                break

            nearest_rect = np.array(
                [hit[0].bbox[:n], hit[0].bbox[n:]])  # TODO: change when considering multiple neighbors
            # hits = list(rtree_idx3d.intersection((nearest_rect[0, 0], nearest_rect[0, 1], nearest_rect[0, 2],
            #                                      nearest_rect[1, 0], nearest_rect[1, 1], nearest_rect[1, 2]),
            #                                     objects=True));

            # print("Nearest rectangle before enlarging: ", nearest_rect);
            # print("Number of intersecting rectangles: ", len(hits));

            # for hit in hits:
            #    nearest_rect[0, :] = np.minimum(nearest_rect[0, :], hit.bbox[:n])
            #    nearest_rect[1, :] = np.maximum(nearest_rect[1, :], hit.bbox[n:])
            # print("Nearest rectangle after enlarging: ", nearest_rect, " and before enlarging: ", );

            if tracking_rect_global_cntr and random.random() < tracking_start_prob:
                tracking_start = True
                break

            useless = does_rect_contain(np.array([sampled_state, sampled_state]), nearest_rect)
            if not useless:
                '''
                rect = nearest_rect;
                hits = list(rtree_idx3d.intersection((rect[0,0], rect[0,1], rect[0,2],
                                                rect[1, 0], rect[1, 1], rect[1, 2]),
                                               objects=True));
                union_rect = get_convex_union([np.array([hit.bbox[:n],hit.bbox[n:]]) for hit in hits]);
                rtree_idx3d.insert(rect_global_cntr, (union_rect[0, 0], union_rect[0, 1], union_rect[0, 2],
                                                      union_rect[1, 0], union_rect[1, 1], union_rect[1, 2]),
                                   obj=(s_ind, u_ind, 1));
                rect_global_cntr +=1;
                '''
                # check if it belongs to the obstacles
                '''
                for i in range(Obstacle_up.shape[0]):  # and np.any(rect_curr):
                    rect_obs = np.array([Obstacle_low[i, :], Obstacle_up[i, :]])
                    # TODO: replace this with R-tree intersection call, much faster when many obstacles exist.
                    if does_rect_contain(np.array([sampled_state, sampled_state]), rect_obs):
                        useless = True
                        # print("Sampled state ", sampled_state, " is not useless.")
                        break
                '''
                rect_curr = np.array([sampled_state - init_radius,
                                      sampled_state + init_radius])
                obstacle_hits = list(
                    obstacles_rtree_idx3d.intersection(
                        (rect_curr[0, 0], rect_curr[0, 1], rect_curr[0, 2], rect_curr[1, 0], rect_curr[1, 1],
                         rect_curr[1, 2]), objects=True))

                if len(obstacle_hits) > 0 or (np.any(rect_curr[0, :] < X_low) or np.any(
                        rect_curr[0, :] > X_up) or np.any(rect_curr[1, :] < X_low) or np.any(
                    rect_curr[1, :] > X_up)):
                    useless = True
            if useless:
                '''
                sampling_rectangle = np.array([sampling_rectangle_low - sampling_radius,
                                               sampling_rectangle_up + sampling_radius]);
                sampling_rectangle_low = np.maximum(sampling_rectangle[0, :], X_low);
                sampling_rectangle_low = np.minimum(sampling_rectangle_low, X_up);
                sampling_rectangle_up = np.minimum(sampling_rectangle[1, :], X_up);
                sampling_rectangle_up = np.maximum(sampling_rectangle_up, X_low);
                sampling_rectangle = np.array([sampling_rectangle_low, sampling_rectangle_up]);
                '''
                fail_itr += 1
                itr += 1

            '''
            else:
                # this exists a part of a previously investigated initial set from whcih t
                rect_curr, s_ind, u_ind, rect_target = initial_sets_to_explore.pop(0);
                rect_curr_center = np.average(rect_curr, axis=0);
                if rtree_idx3d.count((rect_curr_center[0], rect_curr_center[1], rect_curr_center[2],
                                      rect_curr_center[0], rect_curr_center[1], rect_curr_center[2])) == 0:
                    i = 0;
                    intersects_obstacle = False;
                    while i < Obstacle_up.shape[0]:  # and np.any(rect_curr):
                        rect_obs = np.array([Obstacle_low[i, :], Obstacle_up[i, :]]);
                        # Hussein: TODO: here we need to check if the reachable sets intersect the unsafe
                        #  sets as well.
                        if do_rects_inter(rect_curr, rect_obs):
                            print("rect_curr before: ", rect_curr)
                            for dim in range(n):
                                if rect_curr[1,dim] - rect_curr[0, dim] > \
                                        unified_reachable_sets[s_ind][1, dim] - unified_reachable_sets[s_ind][0, dim]:
                                    smaller_curr_low = copy.deepcopy(rect_curr[0, :]);
                                    smaller_curr_low[dim] = (rect_curr[0, dim] + rect_curr[1, dim]) / 2;
                                    initial_sets_to_explore.append((np.array(
                                        [[smaller_curr_low[0], smaller_curr_low[1], smaller_curr_low[2]],
                                         [rect_curr[1, 0], rect_curr[1, 1], rect_curr[1, 2]]]), s_ind, u_ind,
                                                                    rect_target));
                                    smaller_curr_up = copy.deepcopy(rect_curr[1, :]);
                                    smaller_curr_up[dim] = (rect_curr[0, dim] + rect_curr[1, dim]) / 2;
                                    initial_sets_to_explore.append(
                                        (np.array([[rect_curr[0, 0], rect_curr[0, 1], rect_curr[0, 2]],
                                                   [smaller_curr_up[0], smaller_curr_up[1],
                                                    smaller_curr_up[2]]]), s_ind, u_ind, rect_target));
                            intersects_obstacle = True;
                            print("rect_curr after: ", rect_curr)
                            print("The rectangle ", rect_curr, " intersects the obstacle ", rect_obs,
                                  ". Does it?")
                            obstacles_intersecting_rect.append(rect_curr);
                            break;
                        i += 1;
                    if not intersects_obstacle:
                        rtree_idx3d.insert(rect_global_cntr, (rect_curr[0, 0], rect_curr[0, 1], rect_curr[0, 2],
                                                              rect_curr[1, 0], rect_curr[1, 1], rect_curr[1, 2]),
                                           obj=(s_ind, u_ind, 1));
                        discovered_rect.append(rect_curr);
                        rect_global_cntr += 1;
                        progress_indicator = True;
                        useless = False;
                        break
                    itr += 1;
            '''
        if useless and not tracking_start:
            print("Couldn't find non-explored state in ", num_trials, " uniformly random samples")
            break
        # if not progress_indicator:
        # steer #
        nearest_rect_center = np.average(nearest_rect, axis=0)
        # print("nearest_rect_center is: ", nearest_rect_center)
        # path = []
        # path_resolution = 0.1;
        # TODO: make it half the distance from the center of any cell in X to the center
        # of the last reachable set.
        # path_vector = [];
        path_distance = np.linalg.norm(sampled_state - nearest_rect_center)
        # for dim in range(nearest_rect_center.shape[0]):
        #    path_vector.append((nearest_rect_center[dim] - sampled_state[dim]) / num_steps);
        # path_vector = (sampled_state - nearest_rect_center) / path_distance  # np.array(path_vector);
        # sampled_state = nearest_rect_center + num_steps * max_reachable_distance * path_vector;
        # print("path distance is: ", path_distance)
        num_steps = math.ceil(path_distance / max_reachable_distance)
        # for step_idx in range(1, num_steps):  # step_idx in range(math.floor(path_distance / path_resolution)):
        # print("sampled_state: ", sampled_state)
        # print("path_vector: ", step_idx * path_vector)
        # print("path_state: ", sampled_state + step_idx * path_vector)
        # path.append(nearest_rect_center + step_idx * max_reachable_distance * path_vector);

        path = [sampled_state]
        # July 14th, 2022, since we are considering initial sets with fixed size, we don't have
        # to start near the target

        # nearest_poly = pc.box2poly(nearest_rect.T);

        # Hussein: we might need to steer the system to there, can we check if there is a control input with a
        # reachable set that intersects the line from the center of the nearest rectangle to the sampled point?

        # valid_hit = False;
        # for path_state in path:
        while len(path):
            path_state = path.pop()
            # this for loop searches for a good initial point to start the RRT
            # in search for the target
            if rtree_idx3d.count((path_state[0], path_state[1], path_state[2],
                                  path_state[0] + 0.01, path_state[1] + 0.01, path_state[2] + 0.01)) > 0:
                continue
            # print("path state: ", path_state)
            # path_state = copy.deepcopy(sampled_state);
            # abstract_nearest_poly = transform_poly_to_abstract(nearest_poly, path_state);
            # abstract_nearest_rect = np.column_stack(abstract_nearest_poly.bounding_box).T;
            # hits = list(abstract_rtree_idx3d.nearest(
            #    (abstract_nearest_rect[0, 0], abstract_nearest_rect[0, 1], abstract_nearest_rect[0, 2],
            #     abstract_nearest_rect[1, 0] + 0.01, abstract_nearest_rect[1, 1] + 0.01, abstract_nearest_rect[1, 2]
            #     + 0.01), 1, objects=True));
            # u_ind = hits[0].id[1];
            # this is the control that corresponds to the reachable set that is nearest to
            # the nearest target set # [np.array([hit.bbox[:n], hit.bbox[n:]]) for hit in hits];
            # print("Number of symbolic control signals that can drive the system from state ", path_state,
            #      " to the box ", nearest_rect, " is ", len(hits))
            # for hit in hits:
            # rects_curr = find_frame(hit.bbox[:n], hit.bbox[n:],
            #                        nearest_rect[0, :], nearest_rect[1, :]);
            # TODO: change the following line to a for loop that iterates over all abstract states.
            # rects_curr = find_frame(unified_reachable_sets[0][0, :], unified_reachable_sets[0][1, :],
            #                        nearest_rect[0, :], nearest_rect[1, :]);

            '''
            tracking_rects.append(rect_curr);
            tracking_rrt_nodes.append(Node(reachable_set=rect_curr, s_ind=None))
            tracking_rtree_idx3d.insert(tracking_rect_global_cntr, (
                rects_curr[0, 0], rects_curr[0, 1], rects_curr[0, 2],
                rects_curr[1, 0], rects_curr[1, 1], rects_curr[1, 2]),
                                        obj=(s_ind, u_ind, 1));
            '''
            # tracking_rect_global_cntr += 1;
            # TODO: implement an RRT that grows the tracking tree then do a backward breadth-first-search
            #  to add the reachable initial sets
            sample_cntr = 0
            rrt_done = False
            while not rrt_done and sample_cntr < 2 * num_steps:  # this should depend on the step size
                # here we use a simple version of RRT to find a path from the sampled state towards rtree_idx3d.
                sample_cntr += 1
                included_idx = 0
                # if random.random() < 0.8:  # self.goal_sample_rate:
                sampled_state = sampling_rectangle[0, :] + np.array([random.random() * ub for ub in
                                                                     sampling_rectangle[1, :].tolist()])
                # else:
                #    # Change this to sample randomly from the targets / rects in rtree_id3d.
                #  Actually, I'm already doing that with the nearest rectangle search but the following helps
                #  in directing the search towards the original target. Actually, it should be directed larger
                # constellations with large sizes.
                #    sampled_state = np.average(np.array([Target_low[0, :].tolist(),
                #                                         Target_up[0, :].tolist()]), axis=0);

                hits_tracking = list(tracking_rtree_idx3d.nearest(
                    (nearest_rect_center[0], nearest_rect_center[1], nearest_rect_center[2],
                     nearest_rect_center[0] + 0.01, nearest_rect_center[1] + 0.01,
                     nearest_rect_center[2] + 0.01), 1, objects=True))
                if 0:  # tracking_start and len(hits_tracking):
                    # July 25th, 2022: get all the RRT nodes that intersect with the currently added rectangle
                    # and check if their last reachable sets is now contained in the discovered ones.
                    # One can also move this further down to check intersection with rotated
                    '''
                    tracking_hits = list(
                        tracking_rtree_idx3d.intersection(
                            (sampled_state[0], sampled_state[1], sampled_state[2],
                             sampled_state[0] + 0.01, sampled_state[1] + 0.01, sampled_state[2] + 0.01),
                            objects=True))
                    '''
                    # inter_num = len(tracking_hits)
                    # if inter_num == 0:
                    #    break
                    # hits = np.array([np.array([hit.bbox[:n], hit.bbox[n:]]) for hit in hits])
                    # July 26th, 2022, the following line is to reduce the number of checks,
                    # most of which might not be covered.
                    hits_tracking = [random.choice(hits_tracking)]
                    for tracking_hit in hits_tracking:
                        cur_solver.reset()
                        tracked_rect = np.array([tracking_hit.bbox[:n], tracking_hit.bbox[n:]])
                        discovered_hits = list(
                            rtree_idx3d.intersection(
                                (tracked_rect[0, 0], tracked_rect[0, 1], tracked_rect[0, 2],
                                 tracked_rect[1, 0], tracked_rect[1, 1], tracked_rect[1, 2]),
                                objects=True))
                        discovered_hits = np.array(
                            [np.array([hit.bbox[:n], hit.bbox[n:]]) for hit in discovered_hits])
                        cur_solver = add_rects_to_solver(discovered_hits, var_dict, cur_solver)
                        uncovered_state = do_rects_list_contain_smt(tracked_rect, var_dict, cur_solver)
                        if uncovered_state is None:  # then tracked_rect is covered now
                            reachable_set = tracking_hit.object.reachable_set  # reachable_set_list.append(
                            s_ind = tracking_hit.object.s_ind
                            u_ind = tracking_hit.object.u_ind
                            # abstract_state_control_list.append(
                            #    (tracking_hit.object.s_ind, tracking_hit.object.u_ind))
                            # curr_node_list.append(tracking_hit.object)
                            # curr_node.discovered = 1
                            nearest_node = tracking_hit.object
                            tracking_rtree_idx3d.delete(nearest_node.id_in_tracking_rtree,
                                                        (nearest_node.reachable_set[-1][0, 0],
                                                         nearest_node.reachable_set[-1][0, 1],
                                                         nearest_node.reachable_set[-1][0, 2],
                                                         nearest_node.reachable_set[-1][1, 0],
                                                         nearest_node.reachable_set[-1][1, 1],
                                                         nearest_node.reachable_set[-1][1, 2]))
                            tracking_rect_global_cntr -= 1
                            rrt_done = True
                            checking_covered_tracking_ctr += 1
                            break
                        else:
                            tracking_hit.object.uncovered_state = copy.deepcopy(uncovered_state)
                    break
                elif sample_cntr >= 2 and len(hits_tracking):  # or (sample_cntr == 1 and tracking_start))
                    nearest_tracking_rect = np.array([hits_tracking[0].bbox[:n], hits_tracking[0].bbox[n:]])
                    nearest_node = hits_tracking[0].object
                    nearest_tracking_rect_center = \
                        nearest_tracking_rect[0, :] + np.array([random.random() * ub for ub in nearest_tracking_rect[1,
                                                                                               :].tolist()])
                    # nearest_tracking_rect_center = copy.deepcopy(nearest_node.uncovered_state)
                    nearest_tracking_rect = np.array([np.maximum(nearest_tracking_rect_center - init_radius,
                                                                 nearest_tracking_rect[0, :]),
                                                      np.minimum(nearest_tracking_rect_center + init_radius,
                                                                 nearest_tracking_rect[1, :])])
                    tracking_start_ctr += 1
                else:
                    # TODO: check if path_state is already in a node.
                    nearest_tracking_rect_center = path_state
                    nearest_tracking_rect = rect_curr
                    if len(hits_tracking) and \
                            does_rect_contain(np.array([path_state.tolist(), path_state.tolist()]),
                                              np.array([hits_tracking[0].bbox[:n], hits_tracking[0].bbox[n:]])):
                        nearest_node = hits_tracking[0].object
                        tracking_start_ctr += 1
                    else:
                        nearest_node = None
                        new_start_ctr += 1
                # nearest_tracking_rect_center =
                # nearest_tracking_rect[0, :] + np.array([random.random() * ub for ub in
                #                                                                       nearest_tracking_rect[1,
                #                                                                       :].tolist()])

                # np.average(nearest_tracking_rect, axis=0)

                if random.random() < target_aiming_prob:  # self.goal_sample_rate:
                    hits = list(good_targets_rtree_idx3d.nearest(
                        (nearest_tracking_rect_center[0], nearest_tracking_rect_center[1],
                         nearest_tracking_rect_center[2],
                         nearest_tracking_rect_center[0] + 0.01, nearest_tracking_rect_center[1] + 0.01,
                         nearest_tracking_rect_center[2] + 0.01), 1, objects=True))
                    nearest_rect = np.array([hits[0].bbox[:n], hits[0].bbox[n:]])  # TODO: this should be changed if
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
                     + 0.01), num_nearest_controls, objects=True))
                # print("nearest abstract reachable set: ", hits[0].bbox)
                initial_set = nearest_tracking_rect  # np.array([nearest_tracking_rect_center - init_radius,
                #          nearest_tracking_rect_center + init_radius]);
                for hit in hits:
                    s_ind = hit.object[0]
                    u_ind = hit.object[1]
                    reachable_set = []
                    included_idx = 0
                    intersects_obstacle = False
                    for t_ind in range(Symbolic_reduced.shape[3]):  # the last reachable_rect
                        # we get from the unified one to make sure all transformed reachable sets end up in the target.
                        # reachable_set.append(transform_to_frames(Symbolic_reduced[s_ind, u_ind, np.arange(n), t_ind],
                        #                                         Symbolic_reduced[
                        #                                             s_ind, u_ind, n + np.arange(n), t_ind],
                        #                                         initial_set[0, :], initial_set[1, :]))
                        reachable_set.append(
                            transform_to_frames(global_unified_reachable_sets[s_ind][u_ind][t_ind][0, :],
                                                global_unified_reachable_sets[s_ind][u_ind][t_ind][1, :],
                                                initial_set[0, :], initial_set[1, :]))
                    # reachable_set.append(transform_to_frames(global_unified_reachable_sets[s_ind][u_ind][-1][0, :],
                    #                                         global_unified_reachable_sets[s_ind][u_ind][-1][1, :],
                    #                                         initial_set[0, :],
                    #                                         initial_set[1, :]))
                    # discovered_rect.append(global_unified_reachable_sets[s_ind][u_ind])
                    # check if the reachable set intersects the unsafe sets.
                    # if not, define rect_curr to be the initial set of the reachable set.
                    # below code is for checking
                    # rect_curr = reachable_set[0];  # or tracking_rects[-1];
                    for idx, reachable_rect in enumerate(reachable_set):
                        # i = 0
                        '''
                        while i < Obstacle_up.shape[0]:  # and np.any(rect_curr):
                            rect_obs = np.array([Obstacle_low[i, :], Obstacle_up[i, :]])
                            if do_rects_inter(reachable_rect,  # TODO: change it back.
                                              rect_obs):
                                obstacles_intersecting_rect.append(reachable_rect)
                                intersects_obstacle = True
                                break
                            i = i + 1
                        '''
                        obstacle_hits = list(
                            obstacles_rtree_idx3d.intersection(
                                (reachable_rect[0, 0], reachable_rect[0, 1], reachable_rect[0, 2], reachable_rect[1, 0],
                                 reachable_rect[1, 1],
                                 reachable_rect[1, 2]), objects=True))
                        if len(obstacle_hits) > 0 or (np.any(reachable_rect[0, :] < X_low) or np.any(
                                reachable_rect[0, :] > X_up) or np.any(reachable_rect[1, :] < X_low) or np.any(
                            reachable_rect[1, :] > X_up)):
                            intersects_obstacle = True
                            included_idx = idx - 1
                            break
                    if not intersects_obstacle:
                        included_idx = len(reachable_set) - 1
                    if included_idx > 0:  # not intersects_obstacle:
                        break
                    # else:
                    #    curr_low = reachable_set[-1][0, :];
                    #    curr_up = reachable_set[-1][1, :];
                # if included_idx == 0:
                #    Obstacle_low = np.concatenate((Obstacle_low, reachable_set[0][0, :]), axis=0)
                #    Obstacle_up = np.concatenate((Obstacle_up, reachable_set[0][1, :]), axis=0)
                if included_idx > 0:  # not intersects_obstacle: #
                    # print("Adding ", rect_curr, " to tracking_rects")
                    # TODO: set the parent of the node.
                    # if len(tracking_rrt_nodes) == 1:
                    #    tracking_rrt_nodes[-1].parent = None
                    # tracking_rects.append(rect_curr);
                    # tracking_abstract_state_control.append((s_ind, u_ind));
                    # discovered_rect.append(rect_curr);
                    # rect_curr = reachable_set[-1];
                    # if np.any(reachable_set[-1][0, :] > X_up) or np.any(reachable_set[-1][1, :] < X_low):
                    #    continue
                    # if np.any(rect_curr[1, :] > X_up):
                    #    if np.all(rect_curr[0, :] < X_up):
                    #        curr_up = np.minimum(rect_curr[1, :], X_up);
                    #    else:
                    #        continue
                    # else:
                    #    curr_up = rect_curr[1, :];
                    # rect_curr = np.array([curr_low, curr_up]);
                    # check if the last rect in tracking_rects is covered
                    # for idx in range(included_idx):
                    idx = 0
                    hits = list(
                        rtree_idx3d.intersection(
                            (reachable_set[included_idx - idx][0, 0], reachable_set[included_idx - idx][0, 1],
                             reachable_set[included_idx - idx][0, 2],
                             reachable_set[included_idx - idx][1, 0], reachable_set[included_idx - idx][1, 1],
                             reachable_set[included_idx - idx][1, 2]),
                            objects=True))
                    inter_num = len(hits)
                    if inter_num > 0:
                        # continue
                        # inter_num = rtree_idx3d.count((rect_low[0], rect_low[1], rect_low[2],
                        #                               rect_up[0], rect_up[1], rect_up[2]));
                        hits = np.array([np.array([hit.bbox[:n], hit.bbox[n:]]) for hit in hits])
                        if inter_num >= 30:
                            sampled_indices = np.linspace(0, len(hits), num=30, endpoint=False).astype(int)
                            hits = np.array(hits)[sampled_indices]
                            # hits = hits[:50]
                        cur_solver.reset()
                        cur_solver = add_rects_to_solver(hits, var_dict, cur_solver)
                        # not_useful = len(hits) > 0 and do_rects_list_contain(rect_curr[j, :, :],
                        #                                                     [hit.bbox for hit in hits]);

                        uncovered_state = do_rects_list_contain_smt(reachable_set[included_idx - idx], var_dict,
                                                                    cur_solver)
                        rrt_done = uncovered_state is None
                        if rrt_done:
                            included_idx = included_idx - idx
                            # break
                    else:
                        rrt_done = False
                        uncovered_state = np.average(reachable_set[included_idx], axis=0)
                    under_approximated_reachable_set = []
                    for t_ind in range(included_idx + 1):
                        under_approximated_reachable_set.append(
                            transform_to_frames(Symbolic_reduced[s_ind, u_ind, np.arange(n), t_ind],
                                                Symbolic_reduced[
                                                    s_ind, u_ind, n + np.arange(n), t_ind],
                                                initial_set[0, :], initial_set[1, :]))
                        # discovered_rect.append(under_approximated_reachable_set[-1])
                    if not rrt_done:  # only add to the RRT if the last reachable set is not in the target.
                        new_node = Node(reachable_set=under_approximated_reachable_set, s_ind=s_ind, u_ind=u_ind)
                        new_node.parent = nearest_node
                        new_node.id_in_tracking_rtree = tracking_rect_global_cntr_ids
                        new_node.uncovered_state = copy.deepcopy(uncovered_state)
                        reachable_rect = reachable_set[included_idx]
                        new_node.last_over_approximated_reachable_set = reachable_rect
                        # tracking_rrt_nodes.append()
                        # tracking_rrt_nodes[-1].parent = nearest_node
                        # for reachable_rect in reachable_set:
                        tracking_rtree_idx3d.insert(tracking_rect_global_cntr_ids, (
                            reachable_rect[0, 0], reachable_rect[0, 1], reachable_rect[0, 2],
                            reachable_rect[1, 0], reachable_rect[1, 1], reachable_rect[1, 2]),
                                                    obj=new_node)
                        tracking_rect_global_cntr += 1
                        tracking_rect_global_cntr_ids += 1
                    # rrt_done = do_rects_list_contain_smt(tracking_rects[-1], var_dict=var_dict,
                    # cur_solver=cur_solver);
            if rrt_done:
                '''
                for rect in tracking_rects:
                    rtree_idx3d.insert(rect_global_cntr, (rect[0, 0], rect[0, 1], rect[0, 2],
                                                          rect[1, 0], rect[1, 1], rect[1, 2]),
                                       obj=(s_ind, u_ind, 1))  # TODO: this should change to the correct u_ind,
                    discovered_rect.append(rect);
                    rect_global_cntr += 1;
                '''
                # while curr_node.discovered == 0:
                # reachable_rect_list = copy.deepcopy(reachable_set)
                # abstract_state_control_list = [(s_ind,u_ind)] * len(reachable_set)
                # if nearest_node is not None:
                reachable_set_list = [under_approximated_reachable_set[:included_idx + 1]]
                abstract_state_control_list = [(s_ind, u_ind)]
                curr_node_list = [None]
                while len(reachable_set_list) > 0:
                    reachable_set = reachable_set_list.pop()
                    s_ind, u_ind = abstract_state_control_list.pop()
                    curr_node = curr_node_list.pop()
                    initial_rect = reachable_set[0]
                    for rect_idx in range(len(reachable_set)):
                        rect = reachable_set[rect_idx]
                        rect_global_cntr += 1
                        if rect_idx == 0:
                            initial_discovered_rect.append(rect)
                        elif rect_idx == len(reachable_set) - 1:  # len(reachable_set) - 1:
                            target_discovered_rect.append(rect)
                            good_targets_rtree_idx3d.insert(good_targets_cntr, (rect[0, 0], rect[0, 1], rect[0, 2],
                                                                                rect[1, 0], rect[1, 1], rect[1, 2]),
                                                            obj=(s_ind, u_ind, 1))
                            covered_hits = rtree_idx3d.contains((rect[0, 0], rect[0, 1], rect[0, 2],
                                                                 rect[1, 0], rect[1, 1], rect[1, 2]),
                                                                objects=True)
                            for covered_hit in covered_hits:
                                rtree_idx3d.delete(covered_hit.id, covered_hit.bbox)
                                deleted_rects_cntr += 1
                            good_targets_cntr += 1
                        else:
                            discovered_rect.append(rect)
                        rtree_idx3d.insert(rect_global_cntr, (rect[0, 0], rect[0, 1], rect[0, 2],
                                                              rect[1, 0], rect[1, 1], rect[1, 2]),
                                           obj=(s_ind, u_ind, 1))

                    ####################################
                    for other_u_ind in range(len(global_transformation_list[s_ind][u_ind][len(reachable_set) - 1])):
                        start_t_ind = 0
                        for t_ind in range(len(reachable_set)):  # range(Symbolic_reduced.shape[3]):
                            reachable_rect = transform_to_frame(
                                np.array(
                                    [Symbolic_reduced[s_ind, other_u_ind, np.arange(n), len(reachable_set) - 1 - t_ind],
                                     Symbolic_reduced[
                                         s_ind, other_u_ind, n + np.arange(n), len(reachable_set) - 1 - t_ind]]),
                                global_transformation_list[s_ind][u_ind][len(reachable_set) - 1][other_u_ind],
                                overapproximate=True)
                            reachable_rect = transform_to_frames(
                                reachable_rect[0, :],
                                reachable_rect[1, :],
                                initial_rect[0, :],
                                initial_rect[1, :])
                            # discovered_rect.append(reachable_rect)
                            '''
                            i = 0
                            while i < Obstacle_up.shape[0]:  # and np.any(rect_curr):
                                rect_obs = np.array([Obstacle_low[i, :], Obstacle_up[i, :]])
                                if do_rects_inter(reachable_rect,  # TODO: change it back.
                                                  rect_obs):
                                    obstacles_intersecting_rect.append(reachable_rect)
                                    intersects_obstacle = True
                                    break
                                i = i + 1
                            '''
                            obstacle_hits = list(
                                obstacles_rtree_idx3d.intersection(
                                    (reachable_rect[0, 0], reachable_rect[0, 1], reachable_rect[0, 2],
                                     reachable_rect[1, 0], reachable_rect[1, 1],
                                     reachable_rect[1, 2]), objects=True))
                            if len(obstacle_hits) > 0 or (np.any(reachable_rect[0, :] < X_low) or np.any(
                                    reachable_rect[0, :] > X_up) or np.any(reachable_rect[1, :] < X_low) or np.any(
                                reachable_rect[1, :] > X_up)):
                                start_t_ind = t_ind + 1
                                break
                        if start_t_ind < len(reachable_set):  # not intersects_obstacle:
                            for t_ind in range(start_t_ind, len(reachable_set)):  # range(Symbolic_reduced.shape[3]):
                                reachable_rect = transform_to_frame(
                                    np.array([Symbolic_reduced[s_ind, other_u_ind, np.arange(n), t_ind],
                                              Symbolic_reduced[s_ind, other_u_ind, n + np.arange(n), t_ind]]),
                                    global_transformation_list[s_ind][u_ind][len(reachable_set) - 1][other_u_ind],
                                    overapproximate=False)
                                reachable_rect = transform_to_frames(
                                    reachable_rect[0, :],
                                    reachable_rect[1, :],
                                    initial_rect[0, :],
                                    initial_rect[1, :])
                                rect_global_cntr += 1
                                if t_ind == start_t_ind:  # 0
                                    initial_discovered_rect.append(reachable_rect)
                                elif t_ind == len(reachable_set) - 1:  # Symbolic_reduced.shape[3] - 1:
                                    target_discovered_rect.append(reachable_rect)
                                    covered_hits = rtree_idx3d.contains(
                                        (reachable_rect[0, 0], reachable_rect[0, 1], reachable_rect[0, 2],
                                         reachable_rect[1, 0], reachable_rect[1, 1], reachable_rect[1, 2]),
                                        objects=True)
                                    for covered_hit in covered_hits:
                                        rtree_idx3d.delete(covered_hit.id, covered_hit.bbox)
                                        deleted_rects_cntr += 1
                                else:
                                    discovered_rect.append(reachable_rect)
                                rtree_idx3d.insert(rect_global_cntr, (reachable_rect[0, 0], reachable_rect[0, 1],
                                                                      reachable_rect[0, 2],
                                                                      reachable_rect[1, 0], reachable_rect[1, 1],
                                                                      reachable_rect[1, 2]),
                                                   obj=(s_ind, other_u_ind, t_ind))
                    ####################################
                    if curr_node is None:
                        if tracking_start:
                            curr_node = nearest_node.parent
                        elif nearest_node is not None:
                            curr_node = nearest_node
                        else:
                            break
                    else:
                        curr_node = curr_node.parent
                    if not tracking_start and not curr_node is None:
                        reachable_set = curr_node.reachable_set
                        last_reachable_set = curr_node.last_over_approximated_reachable_set
                        hits = list(
                            rtree_idx3d.intersection(
                                (last_reachable_set[0, 0], last_reachable_set[0, 1], last_reachable_set[0, 2],
                                 last_reachable_set[1, 0], last_reachable_set[1, 1],
                                 last_reachable_set[1, 2]),
                                objects=True))
                        inter_num = len(hits)
                        if inter_num > 0:
                            hits = np.array([np.array([hit.bbox[:n], hit.bbox[n:]]) for hit in hits])
                            cur_solver.reset()
                            cur_solver = add_rects_to_solver(hits, var_dict, cur_solver)
                            uncovered_state = do_rects_list_contain_smt(last_reachable_set, var_dict, cur_solver)
                            if uncovered_state is None:
                                reachable_set_list.append(reachable_set)
                                abstract_state_control_list.append((curr_node.s_ind, curr_node.u_ind))
                                curr_node_list.append(curr_node)
                            '''
                            else:
                                while itr < num_trials:
                                    sampled_state = \
                                        last_reachable_set[0, :] + np.array(
                                            [random.random() * ub for ub in last_reachable_set[1,
                                                                            :].tolist()])
                                    rect_curr = np.array([sampled_state - init_radius,
                                                          sampled_state + init_radius])
                                    obstacle_hits = list(
                                        obstacles_rtree_idx3d.intersection(
                                            (rect_curr[0, 0], rect_curr[0, 1], rect_curr[0, 2], rect_curr[1, 0],
                                             rect_curr[1, 1],
                                             rect_curr[1, 2]), objects=True))

                                    if len(obstacle_hits) > 0 or (np.any(rect_curr[0, :] < X_low) or np.any(
                                            rect_curr[0, :] > X_up) or np.any(rect_curr[1, :] < X_low) or np.any(
                                        rect_curr[1, :] > X_up)):
                                        itr += 1
                                        fail_itr += 1
                                        if random.random() > success_continue_prob:
                                            continue
                                        else:
                                            break
                                    else:
                                        path.append(sampled_state)
                                        rrt_done = False
                                        break
                            '''
                        # else:
                        #    rrt_done = False
                        # uncovered_state = np.average(reachable_set[-1], axis=0)

                # curr_node = nearest_node  # tracking_rrt_nodes[-1];
                # reachable_set = curr_node.reachable_set
                # s_ind = curr_node.s_ind
                # u_ind = curr_node.u_ind
                progress_indicator = True
            else:
                print("Maximum number of ", 2 * num_steps,
                      " steps has been taken and the target has not been reached :(")

            # tracking_rects = []
            # tracking_abstract_state_control = []
            # tracking_rect_global_cntr = 0
            # os.remove("3d_index_tracking.data")
            # os.remove("3d_index_tracking.index")
            # tracking_rect_global_cntr = 0
            # TODO: reset the tracking_rtree or do not add existing rectangles to rtree_idx3d again.
            if rrt_done:
                break

        # hits = list(rtree_idx3d.intersection(
        #    (rect_curr_center[0], rect_curr_center[1], rect_curr_center[2],
        #     rect_curr_center[0] + 0.01, rect_curr_center[1] + 0.01,
        #     rect_curr_center[2] + 0.01), objects=True));

        # rects_curr = np.array([rects_curr]);
        # print("rects_curr: ", rects_curr)
        # rects_curr_extended = [];
        # the following list stores underapproximations of the rotated rectangles representing the initial sets
        # of the reachable sets in rects_curr_extended
        # potential_initial_sets = [];

        if 0:  # not np.any(np.isnan(rects_curr)):

            for rect_ind in range(rects_curr.shape[0]):
                for s_ind in range(len(unifying_transformation_list)):  # TODO: this might need to be removed as we
                    # TODO: we should be exploring a single s_ind and many u_ind.
                    for u_ind in range(len(unifying_transformation_list[s_ind])):
                        '''
                        rects_curr = np.concatenate((rects_curr, transform_to_frames(rects_curr[rect_ind,0,:], rects_curr[rect_ind,1,:],
                                                                                     unifying_transformation_list[u_ind],
                                                                                     unifying_transformation_list[u_ind])), 0);
                        '''
                        rects_curr_extended.append([]);
                        # TODO: add the s_ind , u_ind info to the boxes in rects_curr_extended.
                        # first transform the full reachable set according to the set of transformations defined by
                        # rect_curr[rect_ind,:,:]. We didn't have to do that when we weren't considering the full
                        # reachable set since we only needed the initial set which we already have from find_frame.

                        for t_ind in range(Symbolic_reduced.shape[3]):
                            rects_curr_extended[-1].append(
                                transform_to_frame(
                                    np.array([Symbolic_reduced[s_ind, u_ind, np.arange(n), t_ind],
                                              Symbolic_reduced[s_ind, u_ind, n + np.arange(n), t_ind]]),
                                    unifying_transformation_list[s_ind][u_ind],
                                    overapproximate=True));

                        print("The last reachable set ", rects_curr_extended[-1][-1],
                              " before the second transformation ");
                        # Second, transform the obtained full reachable set according to the transformation
                        # that sets the last set in the full reachable set to align its center with the center
                        # of the unified reachable set.
                        for t_ind in range(Symbolic_reduced.shape[3]):
                            rects_curr_extended[-1][t_ind] = transform_to_frames(
                                rects_curr_extended[-1][t_ind][0, :],
                                rects_curr_extended[-1][t_ind][1, :],
                                rects_curr[rect_ind, 0, :],
                                rects_curr[rect_ind, 1, :]);

                        if not does_rect_contain(rects_curr_extended[-1][-1], nearest_rect):
                            print("The last reachable set ", rects_curr_extended[-1][-1],
                                  " is not in the target ", nearest_rect, "!!!!");

                        # print("initial_set before transformation: ",
                        #      np.array([rects_curr[rect_ind, 0, :],
                        #                rects_curr[rect_ind, 1, :]]), " with state ",
                        #      unifying_transformation_list[s_ind][u_ind])
                        transformed_initial_set = transform_to_frame(
                            np.array([Symbolic_reduced[s_ind, u_ind, np.arange(n), 0],
                                      Symbolic_reduced[s_ind, u_ind, n + np.arange(n), 0]]),
                            unifying_transformation_list[s_ind][u_ind],
                            overapproximate=False);
                        transformed_initial_set = transform_to_frames(
                            transformed_initial_set[0, :],
                            transformed_initial_set[1, :],
                            rects_curr[rect_ind, 0, :],
                            rects_curr[rect_ind, 1, :]);  # TODO: change it to false
                        # print("transformed_initial_set: ", transformed_initial_set)
                        potential_initial_sets.append(transformed_initial_set);

                        # TODO: update the following line to take into account s_ind and thus the non-symmetric
                        #  coordinates.
            # rects_curr = np.array([rects_curr]);  # just to turn it to 3-dimensional array again.
            # print("rects_curr_extended: ", rects_curr_extended)
            #  for rect_ind in range(rects_curr_extended.shape[0]):
            print("potential_initial_sets: ", potential_initial_sets)
            for idx, reachable_set in enumerate(rects_curr_extended):
                intersects_obstacle = False;
                rect_curr = potential_initial_sets[idx];  # reachable_set[0];
                # rect_curr = rects_curr[rect_ind, :, :];
                if np.any(rect_curr[0, :] < X_low):
                    if np.all(rect_curr[1, :] > X_low):
                        curr_low = np.maximum(rect_curr[0, :], X_low);
                    else:
                        continue
                else:
                    curr_low = rect_curr[0, :];
                if np.any(rect_curr[0, :] > X_up) or np.any(rect_curr[1, :] < X_low):
                    continue
                if np.any(rect_curr[1, :] > X_up):
                    if np.all(rect_curr[0, :] < X_up):
                        curr_up = np.minimum(rect_curr[1, :], X_up);
                    else:
                        continue
                else:
                    curr_up = rect_curr[1, :];

                # rect_curr = np.array([[curr_low, curr_up]]);  # np.array([[curr_low, curr_up]]);
                rect_curr = np.array([[curr_low, curr_up]]);
                # print("rect_curr: ", rect_curr)
                rect_curr_center = np.average(rect_curr[0, :, :], axis=0);
                if does_rect_contain(np.array([path_state, path_state]), rect_curr[0, :, :]) or \
                        rtree_idx3d.count((rect_curr_center[0], rect_curr_center[1], rect_curr_center[2],
                                           rect_curr_center[0] + 0.01, rect_curr_center[1] + 0.01,
                                           rect_curr_center[2] + 0.01)) < 50:
                    # TODO: check if rects_curr projected to the
                    # non-cyclic coordinates contains the range in X that is specified by ind2sub(s_ind)
                    # TODO: check if the reachable set intersects the obstacle.
                    for reachable_rect in reachable_set:
                        i = 0
                        while i < Obstacle_up.shape[0]:  # and np.any(rect_curr):
                            rect_obs = np.array([Obstacle_low[i, :], Obstacle_up[i, :]]);
                            '''
                            if do_rects_inter(rect_curr, rect_obs):
                                intersects_obstacle = True;
                                print(rect_curr, " intersects the obstacle ", rect_obs)
                                break
                            '''
                            # rect_c = float('nan');
                            # for j in range(reachable_rect.shape[0]):
                            # Hussein: TODO: here we need to check if the reachable sets intersect the unsafe
                            #  sets as well. Replace get_rects_inter with partitioning the target box instead.
                            if do_rects_inter(reachable_rect,  # TODO: change it back.
                                              rect_obs):  # do_rects_inter(rect_curr[j, :, :], rect_obs):
                                # print("rect_curr[j, :, :] before: ", rect_curr[j, :, :])
                                # [rect_c_temp, rect_i] = get_rects_inter(rect_curr[j, :, :], rect_obs);
                                # TODO: change the following to the correct values
                                # TODO: maybe do the partitioning till finding the parts with reachable sets
                                # that do not intersect obstacles without adding them and then retreiving them
                                # at the outer while loop start.
                                # s_ind = 0;
                                # u_ind = 0;
                                '''
                                if np.any(np.isnan(rect_c)):
                                    rect_c = copy.deepcopy(rect_c_temp);
                                else:
                                    rect_c = np.concatenate((rect_c, rect_c_temp), 0);
                                '''
                                '''
                                for dim in range(n):
                                    if rect_curr[j, 1, dim] - rect_curr[j, 0, dim] > \
                                            unified_reachable_sets[s_ind][1, dim] - \
                                            unified_reachable_sets[s_ind][0, dim]:
                                        smaller_curr_low = copy.deepcopy(rect_curr[j, 0, :]);
                                        smaller_curr_low[dim] = (rect_curr[j, 0, dim] + rect_curr[j, 1, dim]) / 2;
                                        initial_sets_to_explore.append((np.array(
                                            [[smaller_curr_low[0], smaller_curr_low[1], smaller_curr_low[2]],
                                             [rect_curr[j, 1, 0], rect_curr[j, 1, 1], rect_curr[j, 1, 2]]]), s_ind,
                                                                        u_ind,
                                                                        nearest_rect));
                                        smaller_curr_up = copy.deepcopy(rect_curr[j, 1, :]);
                                        smaller_curr_up[dim] = (rect_curr[j, 0, dim] + rect_curr[j, 1, dim]) / 2;
                                        initial_sets_to_explore.append(
                                            (np.array([[rect_curr[j, 0, 0], rect_curr[j, 0, 1], rect_curr[j, 0, 2]],
                                                       [smaller_curr_up[0], smaller_curr_up[1],
                                                        smaller_curr_up[2]]]), s_ind, u_ind, nearest_rect));
                                '''

                                # This for loop partitions the target to smaller ones to avoid obstacles. TODO:
                                #  Maybe partitioning the source set might better? print("rect_curr[j, :,
                                #  :] after: ", rect_curr[j, :, :]) print("The rectangle ", rect_curr[j,:,:],
                                #  " intersects the obstacle ", rect_obs, ". Does it?")
                                #  obstacles_intersecting_rect.append(rect_curr[j, :, :]);
                                obstacles_intersecting_rect.append(reachable_rect);
                                intersects_obstacle = True;
                                '''
                                for dim in range(n):
                                    smaller_target_low = nearest_rect[0, :];
                                    smaller_target_low[dim] = (nearest_rect[0, dim] + nearest_rect[1, dim]) / 2;
                                    rtree_idx3d.insert(rect_global_cntr, (
                                        smaller_target_low[0], smaller_target_low[1], smaller_target_low[2],
                                        nearest_rect[1, 0], nearest_rect[1, 1], nearest_rect[1, 2]),
                                                       obj=(s_ind, u_ind, 1));
                                    rect_global_cntr += 1;
                                    smaller_target_up = nearest_rect[1, :];
                                    smaller_target_up[dim] = (nearest_rect[0, dim] + nearest_rect[1, dim]) / 2
                                    rtree_idx3d.insert(rect_global_cntr,
                                                       (nearest_rect[0, 0], nearest_rect[0, 1], nearest_rect[0, 2],
                                                        smaller_target_up[0], smaller_target_up[1],
                                                        smaller_target_up[2]),
                                                       obj=(s_ind, u_ind, 1));
                                    rect_global_cntr += 1;
                                '''
                                break
                            '''
                            else:
                                if np.any(np.isnan(rect_c)):
                                    rect_c = np.array([rect_curr[j, :, :]]);
                                else:
                                    rect_c = np.concatenate((rect_c, np.array([rect_curr[j, :, :]])), 0);

                            rect_curr = copy.deepcopy(rect_c);
                            '''
                            i = i + 1;
                        if intersects_obstacle:
                            print("The reachable rect ", reachable_rect, " intersects obstacle ", rect_obs, " :/")
                            break
                        # print("Sampled state intersects obstacle")
                        '''
                        sampling_rectangle = np.array([sampling_rectangle_low - sampling_radius,
                                                       sampling_rectangle_up + sampling_radius]);
                        sampling_rectangle_low = np.maximum(sampling_rectangle[0, :], X_low);
                        sampling_rectangle_low = np.minimum(sampling_rectangle_low, X_up);
                        sampling_rectangle_up = np.minimum(sampling_rectangle[1, :], X_up);
                        sampling_rectangle_up = np.maximum(sampling_rectangle_up, X_low);
                        sampling_rectangle = np.array([sampling_rectangle_low, sampling_rectangle_up]);
                        break
                        '''
                    #    continue
                    # obj_value = hit.object;
                    # s_ind = obj_value[0];
                    # u_ind = obj_value[1];
                    if not intersects_obstacle:
                        s_ind = 0;
                        u_ind = 0;
                        for rect_ind in range(rect_curr.shape[0]):  # this is a single iteration for now as we
                            # are not partitioning the initial set upon intersection with obstacle.
                            curr_low = rect_curr[rect_ind, 0, :];
                            curr_up = rect_curr[rect_ind, 1, :];
                            curr_average = np.average(rect_curr[rect_ind, :, :], axis=0);
                            if rtree_idx3d.count((curr_average[0], curr_average[1], curr_average[2],
                                                  curr_average[0] + 0.01, curr_average[1] + 0.01,
                                                  curr_average[2] + 0.01)) > 10:
                                continue
                            print("Adding ", np.array([curr_low, curr_up]), " to rtree")
                            rtree_idx3d.insert(rect_global_cntr, (curr_low[0], curr_low[1], curr_low[2],
                                                                  curr_up[0], curr_up[1], curr_up[2]),
                                               obj=(s_ind, u_ind, 1));
                            # for reachable_rect in reachable_set:
                            #    discovered_rect.append(reachable_rect);  #
                            discovered_rect.append(np.array([curr_low, curr_up]));
                            rect_global_cntr += 1;
                        progress_indicator = True;
                    '''
                    valid_hit = True;
                    sampling_rectangle = np.array(
                        [curr_low - sampling_radius,
                         curr_up + sampling_radius]);
                    sampling_rectangle_low = np.maximum(sampling_rectangle[0, :], X_low);
                    sampling_rectangle_low = np.minimum(sampling_rectangle_low, X_up);
                    sampling_rectangle_up = np.minimum(sampling_rectangle[1, :], X_up);
                    sampling_rectangle_up = np.maximum(sampling_rectangle_up, X_low);
                    sampling_rectangle = np.array([sampling_rectangle_low, sampling_rectangle_up]);
                    break;
                    '''
                else:
                    print("Box ", rect_curr, " does not contain the state ", path_state,
                          " and the center of the box is contained in previous existing boxes :(, like,",
                          list(rtree_idx3d.intersection(
                              (rect_curr_center[0], rect_curr_center[1], rect_curr_center[2],
                               rect_curr_center[0] + 0.01, rect_curr_center[1] + 0.01,
                               rect_curr_center[2] + 0.01), objects=True))[0].bbox);

            # if valid_hit:
            #    break;
        elif 0:
            print("Target is too small to get a meaningful initial set.")
            # create a bounding box around the sampled state using the nearest_rect[0, :], nearest_rect[1, :]
            # compute the reachable set by transforming the abstract reachable set corresponding to u_ind = hit.object[1]
            # the one above needs a new function that is similar to find_frame but instead transform a reachable set according to a set of frames
            # add the vertices of the reachable set (the last one) to the set of unexplored vertices
            # adjust the sampling procedure to sample from that set if it is not empty, otherwise sample from the state space.
            nearest_rect_radius = (nearest_rect[1, :] - nearest_rect[0, :]) / 2;
            curr_low = sampled_state - nearest_rect_radius;
            curr_up = sampled_state + nearest_rect_radius;
            if any(curr_up < curr_low):
                assert any(curr_low <= curr_up), "curr_up is less than curr_low!!!!!"
            curr_low = np.maximum(curr_low, X_low);
            curr_low = np.minimum(curr_low, X_up);
            curr_up = np.minimum(curr_up, X_up);
            curr_up = np.maximum(curr_up, X_low);
            # TODO: uncomment the following line and comment the one after if we are using a single control
            #  instead of all controls
            # new_target = transform_to_frames(np.array(hit.bbox[:n]), np.array(hit.bbox[n:]), curr_low, curr_up);
            new_target = transform_to_frames(unified_reachable_sets[0][0, :],
                                             unified_reachable_sets[0][1, :], curr_low, curr_up);
            vertices = rectangle_to_vertices(new_target);
            initial_sets_to_explore.extend(vertices);
            # TODO create two R-trees, one for growing it and one for storing control.
            # obj_value = hit.object;
            # s_ind = obj_value[0];
            # u_ind = obj_value[1];
            s_ind = 0;
            u_ind = 0;
            rtree_idx3d.insert(rect_global_cntr, (curr_low[0], curr_low[1], curr_low[2],
                                                  curr_up[0], curr_up[1], curr_up[2]),
                               obj=(s_ind, u_ind, 0));
            discovered_rect.append(np.array([curr_low, curr_up]));
            rect_global_cntr += 1;
            # print("The following states are to be explored in the future: ", vertices);
            # TODO: keep track the list of targets that were not covered yet.
        elif 0:
            hits = list(rtree_idx3d.nearest((nearest_rect[0, 0], nearest_rect[0, 1], nearest_rect[0, 2],
                                             nearest_rect[1, 0], nearest_rect[1, 1], nearest_rect[1, 2]), 5,
                                            objects=True));
            for hit in hits:
                enlarged_rect = get_convex_union([np.array([hit.bbox[0:n], hit.bbox[n:]]), nearest_rect]);
                rtree_idx3d.insert(rect_global_cntr, (enlarged_rect[0, 0], enlarged_rect[0, 1], enlarged_rect[0, 2],
                                                      enlarged_rect[1, 0], enlarged_rect[1, 1],
                                                      enlarged_rect[1, 2]),
                                   obj=(s_ind, u_ind, 1));
                rect_global_cntr += 1;

        # if valid_hit:
        #    break
        '''
        if valid_hit:
            itr = 0;
            progress_indicator = True;
        '''

        # enlarged_rect = get_convex_union([np.array([hit.bbox[0:n], hit.bbox[n:]]) for hit in hits]);
        '''
        for s_ind in range(Symbolic_reduced.shape[0]):
            for u_ind in range(Symbolic_reduced.shape[1]):
                # target_ind = random.randint(0, targets.shape[0]);
                # for target_ind in range(targets.shape[0]):
                # Target_low = targets[target_ind, 0, :];
                # Target_up = targets[target_ind, 1, :];

                # print("frames that put ", Symbolic_reduced[s_ind, u_ind, np.arange(n)], Symbolic_reduced[s_ind, u_ind,
                #                                                                                       n + np.arange(
                #                                                                                           n)], " in ", Target_low, Target_up, " are:")
                rects_curr = find_frame(Symbolic_reduced[s_ind, u_ind, np.arange(n)], Symbolic_reduced[s_ind, u_ind,
                                                                                                       n + np.arange(
                                                                                                           n)],
                                        Target_low, Target_up);
                # print(rects_curr);

                if not np.any(np.isnan(rects_curr)):
                    for rect_ind in range(rects_curr.shape[0]):
                        rect_curr = rects_curr[rect_ind, :, :];
                        curr_low = np.maximum(rect_curr[0, :], X_low);
                        curr_low = np.minimum(curr_low, X_up);
                        curr_up = np.minimum(rect_curr[1, :], X_up);
                        curr_up = np.maximum(curr_up, X_low);
                        rect_curr = np.array([curr_low, curr_up]); # np.array([[curr_low, curr_up]]);
                        i = 0;
                        # print("rect_curr: ", rect_curr)
                        while i < Obstacle_up.shape[0]: # and np.any(rect_curr):
                            rect_obs = np.array([Obstacle_low[i, :], Obstacle_up[i, :]]);
                            if do_rects_inter(rect_curr, rect_obs):
                                useless = True;
                                break
                            '''
        '''
        rect_c = [];
        for j in range(rect_curr.shape[0]):
            # Hussein: TODO: here we need to check if the reachable sets intersect the unsafe
            #  sets as well.
            if do_rects_inter(rect_curr[j, :, :], rect_obs):
                [rect_c_temp, rect_i] = get_rects_inter(rect_curr[j, :, :], rect_obs);
                if np.any(rect_c_temp):
                    if len(rect_c) == 0:
                        rect_c = rect_c_temp;
                    else:
                        rect_c = np.concatenate((rect_c, rect_c_temp), 0);
            else:
                if len(rect_c) == 0:
                    rect_c = np.array([rect_curr[j, :, :]]);
                else:
                    rect_c = np.concatenate((rect_c, np.array([rect_curr[j, :, :]])), 0);
        rect_curr = rect_c;
        '''
        '''
        i = i + 1;
        # print("done obstacles intersection: ", rect_curr)
        if useless:
            break
        # if np.any(rect_curr):
        for j in range(rect_curr.shape[0]):
            if not check_rect_empty(rect_curr[j, :, :], 1):
                rect_low = rect_curr[j, 0, :];
                rect_up = rect_curr[j, 1, :];
                # print("checking intersection with existing rectangles")
                # print("rect_global_cntr:", rect_global_cntr)
                time_before = time.time();

                # hits = list(rtree_idx3d.intersection((rect_low[0], rect_low[1], rect_low[2],
                #                                      rect_up[0], rect_up[1], rect_up[2]),
                #                                     objects=True));
                inter_num = rtree_idx3d.count((rect_low[0], rect_low[1], rect_low[2],
                                               rect_up[0], rect_up[1], rect_up[2]));
                # hits = np.array([[hit.bbox[:n], hit.bbox[n:]] for hit in hits]);
                intersection_time += time.time() - time_before;
                # cur_solver.reset();
                # cur_solver = add_rects_to_solver(hits, var_dict, cur_solver);
                # print("rectangle being searched for:", rect_curr[j, :, :])
                # print("hits: ", [hit.bbox for hit in hits])
                # if len(hits) >= 20:
                #    sampled_indices = np.linspace(0, len(hits), num=20, endpoint=False).astype(int);
                #    hits = np.array(hits)[sampled_indices];

                # print("hits len", len(hits))
                time_before = time.time();
                # not_useful = len(hits) > 0 and do_rects_list_contain(rect_curr[j, :, :],
                #                                                     [hit.bbox for hit in hits]);
                not_useful = inter_num > 0; # do_rects_list_contain_smt(rect_curr[j, :, :], var_dict, cur_solver);
                # print("Is rectangle ", rect_curr[j, :, :], " contained in the previous rectangles?", not_useful);
                contain_time += time.time() - time_before;
                if not not_useful:
                    # TODO: this should be changed to support any dimension instead of just 3
                    time_before = time.time();
                    rtree_idx3d.insert(rect_global_cntr, (
                        rect_low[0], rect_low[1], rect_low[2], rect_up[0], rect_up[1], rect_up[2]),
                                       obj=u_ind);

                    insert_time += time.time() - time_before;
                    # adding to the Z3 solver the complement of the rectangle.
                    # print("rectangle: ", rect_curr[j, :, :], " is being added to the solver");
                    # cur_solver.push();
                    # for dim in range(n):
                    '''
        '''
                    cur_solver.add(Or([var_dict[0] < rect_low[0], var_dict[1] < rect_low[1],
                                       var_dict[2] < rect_low[2], var_dict[0] > rect_up[0],
                                       var_dict[1] > rect_up[1], var_dict[2] > rect_up[2]]));
                    '''
        '''
                    # cur_solver.add(var_dict[dim] > rect_up[dim]);

                    rect_global_cntr = rect_global_cntr + 1;
                    # print("Added a new rectangle to Rtree: ", rect_curr[j, :, :])
                    progress_indicator = True;
                    if len(targets_temp) == 0:
                        targets_temp = np.array([rect_curr[j, :, :]]);
                    else:
                        targets_temp = np.concatenate((targets_temp, np.reshape(rect_curr[j, :, :],
                                                                                (1, rect_curr[j, :,
                                                                                    :].shape[0],
                                                                                 rect_curr[j, :,
                                                                                 :].shape[1]))), 0);
        '''
        itr += 1
        if progress_indicator:
            succ_itr += 1
            print(time.time() - t_start, " ", rect_global_cntr - rect_curr_cntr,
                  " new controllable states have been found in this synthesis iteration\n")
            rect_curr_cntr = rect_global_cntr;
            print(rect_global_cntr, ' symbols are controllable to satisfy the reach-avoid specification\n')
            # trying to enlarge each of the rectangles in rect_curr
            print("success iterations: ", succ_itr)
            print("deleted_rects: ", deleted_rects_cntr)
            # print("checking_covered_tracking_ctr: ", checking_covered_tracking_ctr)
            print("tracking_start_ctr: ", tracking_start_ctr)
            print("new_start_ctr: ", new_start_ctr)
            print("tracking_rect_global_cntr: ", tracking_rect_global_cntr)
            '''
            new_targets_temp = [];
            for target_idx in range(targets_temp.shape[0]):
                rect_curr = targets_temp[target_idx, :, :];
                time_before = time.time();
                hits = list(rtree_idx3d.nearest((rect_curr[0, 0], rect_curr[0, 1], rect_curr[0, 2],
                                                 rect_curr[1, 0], rect_curr[1, 1], rect_curr[1, 2]), 5,
                                                objects=True));
                nearest_time += time.time() - time_before;
                enlarged_rect = get_convex_union([np.array([hit.bbox[0:n], hit.bbox[n:]]) for hit in hits]);
                time_before = time.time();
                hits = list(rtree_idx3d.nearest((enlarged_rect[0, 0], enlarged_rect[0, 1], enlarged_rect[0, 2],
                                                 enlarged_rect[1, 0], enlarged_rect[1, 1], enlarged_rect[1, 2]), 5,
                                                objects=True));
                nearest_time += time.time() - time_before;
                if len(hits) >= 20:
                    sampled_indices = np.linspace(0, len(hits), num=20, endpoint=False).astype(int);
                    hits = np.array(hits)[sampled_indices];
                if not do_rects_list_contain(rect_curr, [hit.bbox for hit in hits]):
                    new_targets_temp.append(enlarged_rect);
                else:
                    new_targets_temp.append(targets_temp);
            '''
            ####
            # targets = targets_temp;  # np.array(new_targets_temp);
            # print("targets: ", targets)
            # print("intersection_time, contain_time, insert_time, nearest_time)", intersection_time, contain_time,
            #      insert_time, nearest_time)
        else:
            fail_itr += 1
            print('No new controllable state has been found in this synthesis iteration\n', time.time() - t_start)
            print("fail iterations: ", fail_itr)
            # break;
        print("# of RRT iterations so far: ", itr)

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

    '''
    plt.figure("Reduced coordinates with combined reachable sets")
    color = 'orange'
    currentAxis_2 = plt.gca()
    for rect in unified_reachable_sets:
        rect_patch = Rectangle(rect[0, [0, 1]], rect[1, 0] - rect[0, 0],
                               rect[1, 1] - rect[0, 1], linewidth=1,
                               edgecolor=color, facecolor=color)
        currentAxis_2.add_patch(rect_patch)
    # TODO plot transformation points as well.
    plt.ylim([-2, 2])
    plt.xlim([-2.5, 2.5])
    '''

    '''
    plt.figure("Reduced coordinates with transformed reachable sets")
    color = 'orange'
    currentAxis_3 = plt.gca()
    for s_ind in range(Symbolic_reduced.shape[0]):
        # print("unifying_transformation_list[s_ind]: ", unifying_transformation_list[s_ind])
        for u_ind in range(Symbolic_reduced.shape[1]):
            rect = transform_to_frame(np.array([Symbolic_reduced[s_ind, u_ind, np.arange(n), 0],
                                                Symbolic_reduced[s_ind, u_ind, n + np.arange(n), 0]]),
                                      unifying_transformation_list[s_ind][u_ind],
                                      overapproximate=False);  # TODO: change it to false
            rect_patch = Rectangle(rect[0, [0, 1]], rect[1, 0] - rect[0, 0],
                                   rect[1, 1] - rect[0, 1], linewidth=1,
                                   edgecolor=color, facecolor=color)
            currentAxis_3.add_patch(rect_patch)
            # unified_reachable_sets[-1] = get_convex_union([unified_reachable_sets[-1], transformed_rect]);
    plt.ylim([-2, 2])
    plt.xlim([-2.5, 2.5])
    '''

    plt.show()
