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
        ang += 2 * math.pi;
    while ang > 2 * math.pi:
        ang -= 2 * math.pi;

    low_red = np.array(
        [(rect[0, 0] - state[0]) * math.cos(ang) -
         (rect[0, 1] - state[1]) * math.sin(ang),
         (rect[0, 0] - state[0]) * math.sin(ang) +
         (rect[0, 1] - state[1]) * math.cos(ang),
         rect[0, 2] + state[2]]);
    up_red = np.array(
        [(rect[1, 0] - state[0]) * math.cos(ang) -
         (rect[1, 1] - state[1]) * math.sin(ang),
         (rect[1, 0] - state[0]) * math.sin(ang) +
         (rect[1, 1] - state[1]) * math.cos(ang),
         rect[1, 2] + state[2]]);

    if 0 <= ang <= math.pi / 2:
        x_bb_up = up_red[0] + (rect[1, 1] - rect[0, 1]) * math.sin(ang);
        y_bb_up = up_red[1];
        x_bb_low = low_red[0] - (rect[1, 1] - rect[0, 1]) * math.sin(ang);
        y_bb_low = low_red[1];
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
    ang = state[2];  # psi = 0 is North, psi = pi/2 is east

    while ang < 0:
        ang += 2 * math.pi;
    while ang > 2 * math.pi:
        ang -= 2 * math.pi;

    low_red = np.array(
        [(rect[0, 0]) * math.cos(ang) -
         (rect[0, 1]) * math.sin(ang) + state[0],
         (rect[0, 0]) * math.sin(ang) +
         (rect[0, 1]) * math.cos(ang) + state[1],
         rect[0, 2] + state[2]]);
    up_red = np.array(
        [(rect[1, 0]) * math.cos(ang) -
         (rect[1, 1]) * math.sin(ang) + state[0],
         (rect[1, 0]) * math.sin(ang) +
         (rect[1, 1]) * math.cos(ang) + state[1],
         rect[1, 2] + state[2]]);

    if 0 <= ang <= math.pi / 2:
        x_bb_up = up_red[0] + (rect[1, 1] - rect[0, 1]) * math.sin(ang);
        y_bb_up = up_red[1];
        x_bb_low = low_red[0] - (rect[1, 1] - rect[0, 1]) * math.sin(ang);
        y_bb_low = low_red[1];
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
    w = bb[1, 0] - bb[0, 0];
    h = bb[1, 1] - bb[0, 1];

    if overapproximate:
        return bb;

    bb_center = np.average(bb, axis=0);

    new_w, new_h = rotatedRectWithMaxArea(w, h, state[2]);

    low_new_rect = np.array([bb_center[0] - new_w / 2.0,
                             bb_center[1] - new_h / 2.0,
                             bb[0, 2]]);
    up_new_rect = np.array([bb_center[0] + new_w / 2.0,
                            bb_center[1] + new_h / 2.0,
                            bb[1, 2]]);

    # print("low_new_rect: ", low_new_rect)
    # print("up_new_rect: ", up_new_rect)
    result = np.array([low_new_rect, up_new_rect]);

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
    box_1 = transform_to_frame(np.array([low_red, up_red]), source_full_low, overapproximate=True);
    box_2 = transform_to_frame(np.array([low_red, up_red]), source_full_up, overapproximate=True);
    result = get_convex_union([box_1, box_2]);
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

    angle_step = 2 * (up_red[2] - low_red[2]);
    rect_curr = [];
    if angle_step > 0:
        # print("int(np.floor(target_full_up[2] - target_full_low[2]) / angle_step): ",
        #      int(np.floor(target_full_up[2] - target_full_low[2]) / angle_step));
        itr_num = int(np.ceil(target_full_up[2] - target_full_low[2]) / angle_step);
    else:
        itr_num = 1;
    for idx in range(itr_num):
        low_angle = target_full_low[2] + idx * angle_step;
        high_angle = min(target_full_low[2] + (idx + 1) * angle_step, target_full_up[2]);
        theta_low_sys = low_angle - low_red[2];
        theta_up_sys = high_angle - up_red[2];

        theta_low = theta_low_sys;
        while theta_low < -math.pi:
            theta_low += 2 * math.pi;
        while theta_low > math.pi:
            theta_low -= 2 * math.pi;
        theta_up = theta_up_sys;
        while theta_up < -math.pi:
            theta_up += 2 * math.pi;
        while theta_up > math.pi:
            theta_up -= 2 * math.pi;

        if 0 <= theta_low <= math.pi / 2:
            x_target_up_1 = target_full_up[0] - (up_red[1] - low_red[1]) * math.cos(math.pi / 2 - theta_low);
            y_target_up_1 = target_full_up[1];
        elif math.pi / 2 <= theta_low <= math.pi:
            x_target_up_1 = target_full_up[0] - (up_red[0] - low_red[0]) * math.sin(theta_low - math.pi / 2) - (
                    up_red[1] - low_red[1]) * math.cos(theta_low - math.pi / 2);
            y_target_up_1 = target_full_up[1] - (up_red[1] - low_red[1]) * math.sin(theta_low - math.pi / 2);
        elif 0 > theta_low >= - math.pi / 2:
            x_target_up_1 = target_full_up[0];
            y_target_up_1 = target_full_up[1] - (up_red[0] - low_red[0]) * math.sin(-1 * theta_low);
        else:
            x_target_up_1 = target_full_up[0];
            y_target_up_1 = target_full_up[1] - (up_red[1] - low_red[1]) * math.sin(-1 * theta_low - math.pi / 2);
            x_target_up_1 = x_target_up_1 - (up_red[0] - low_red[0]) * math.sin(-1 * theta_low - math.pi / 2);
            y_target_up_1 = y_target_up_1 - (up_red[0] - low_red[0]) * math.cos(-1 * theta_low - math.pi / 2);

        if 0 <= theta_low <= math.pi / 2:
            x_target_low_1 = target_full_low[0] + (up_red[1] - low_red[1]) * math.cos(math.pi / 2 - theta_low);
            y_target_low_1 = target_full_low[1];

        elif math.pi / 2 <= theta_low <= math.pi:
            x_target_low_1 = target_full_low[0] + (up_red[0] - low_red[0]) * math.cos(math.pi - theta_low) + (
                    up_red[1] - low_red[1]) * math.cos(theta_low - math.pi / 2);
            y_target_low_1 = target_full_low[1] + (up_red[1] - low_red[1]) * math.sin(theta_low - math.pi / 2);

        elif 0 > theta_low >= -math.pi / 2:
            x_target_low_1 = target_full_low[0];
            y_target_low_1 = target_full_low[1] + (up_red[0] - low_red[0]) * math.cos(math.pi / 2 + theta_low);

        else:
            x_target_low_1 = target_full_low[0] + (up_red[0] - low_red[0]) * math.cos(math.pi + theta_low);
            y_target_low_1 = target_full_low[1] + (up_red[0] - low_red[0]) * math.sin(math.pi + theta_low) + \
                             (up_red[1] - low_red[1]) * math.cos(math.pi + theta_low);

        curr_low_1 = np.array(
            [x_target_low_1 - (low_red[0]) * math.cos(theta_low_sys) + (low_red[1]) * math.sin(theta_low_sys),
             y_target_low_1 - (low_red[0]) * math.sin(theta_low_sys) - (low_red[1]) * math.cos(theta_low_sys),
             theta_low_sys]);
        curr_up_1 = np.array(
            [x_target_up_1 - (up_red[0]) * math.cos(theta_low_sys) + (up_red[1]) * math.sin(theta_low_sys),
             y_target_up_1 - (up_red[0]) * math.sin(theta_low_sys) - (up_red[1]) * math.cos(theta_low_sys),
             theta_low_sys]);

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
            x_target_up_2 = target_full_up[0] - (up_red[1] - low_red[1]) * math.cos(math.pi / 2 - theta_up);
            y_target_up_2 = target_full_up[1];
        elif math.pi / 2 <= theta_up <= math.pi:
            x_target_up_2 = target_full_up[0] - (up_red[0] - low_red[0]) * math.sin(theta_up - math.pi / 2) - (
                    up_red[1] - low_red[1]) * math.cos(theta_up - math.pi / 2);
            y_target_up_2 = target_full_up[1] - (up_red[1] - low_red[1]) * math.sin(theta_up - math.pi / 2);
        elif 0 > theta_up >= - math.pi / 2:
            x_target_up_2 = target_full_up[0];
            y_target_up_2 = target_full_up[1] - (up_red[0] - low_red[0]) * math.sin(-1 * theta_up);
        else:
            x_target_up_2 = target_full_up[0];
            y_target_up_2 = target_full_up[1] - (up_red[1] - low_red[1]) * math.sin(-1 * theta_up - math.pi / 2);
            x_target_up_2 = x_target_up_2 - (up_red[0] - low_red[0]) * math.sin(-1 * theta_up - math.pi / 2);
            y_target_up_2 = y_target_up_2 - (up_red[0] - low_red[0]) * math.cos(-1 * theta_up - math.pi / 2);

        if 0 <= theta_up <= math.pi / 2:
            x_target_low_2 = target_full_low[0] + (up_red[1] - low_red[1]) * math.cos(math.pi / 2 - theta_up);
            y_target_low_2 = target_full_low[1];

        elif math.pi / 2 <= theta_up <= math.pi:
            x_target_low_2 = target_full_low[0] + (up_red[0] - low_red[0]) * math.cos(math.pi - theta_up) + (
                    up_red[1] - low_red[1]) * math.cos(
                theta_up - math.pi / 2);
            y_target_low_2 = target_full_low[1] + (up_red[1] - low_red[1]) * math.sin(theta_up - math.pi / 2);

        elif 0 > theta_up >= - math.pi / 2:
            x_target_low_2 = target_full_low[0];
            y_target_low_2 = target_full_low[1] + (up_red[0] - low_red[0]) * math.cos(math.pi / 2 + theta_up);

        else:
            x_target_low_2 = target_full_low[0] + (up_red[0] - low_red[0]) * math.cos(math.pi + theta_up);
            y_target_low_2 = target_full_low[1] + (up_red[0] - low_red[0]) * math.sin(math.pi + theta_up) + (
                    up_red[1] - low_red[1]) * math.cos(
                math.pi + theta_up);

        curr_low_2 = np.array(
            [x_target_low_2 - (low_red[0]) * math.cos(theta_up_sys) + (low_red[1]) * math.sin(theta_up_sys),
             y_target_low_2 - (low_red[0]) * math.sin(theta_up_sys) - (low_red[1]) * math.cos(theta_up_sys),
             theta_up_sys]);
        curr_up_2 = np.array(
            [x_target_up_2 - (up_red[0]) * math.cos(theta_up_sys) + (up_red[1]) * math.sin(theta_up_sys),
             y_target_up_2 - (up_red[0]) * math.sin(theta_up_sys) - (up_red[1]) * math.cos(theta_up_sys),
             theta_up_sys]);

        if np.all(curr_low_1 <= curr_up_1) and np.all(curr_low_2 <= curr_up_2) \
                and do_rects_inter(np.array([curr_low_1[:2], curr_up_1[:2]]),
                                   np.array([curr_low_2[:2], curr_up_2[:2]])):
            curr_low_temp = np.maximum(curr_low_1, curr_low_2);
            curr_up_temp = np.minimum(curr_up_1, curr_up_2);
            curr_low = curr_low_temp;
            curr_low[2] = curr_low_1[2];  # np.minimum(curr_low_temp[2], curr_up_temp[2]);
            curr_up = curr_up_temp;
            curr_up[2] = curr_up_2[2];  # np.maximum(curr_low_temp[2], curr_up_temp[2]);
            # print("curr_low_1: ", curr_low_1, " curr_low_2: ", curr_low_2, " curr_up_1: ", curr_up_1, "curr_up_2: ",
            #      curr_up_2)
            # if np.all(curr_low <= curr_up):
            rect_curr.append([curr_low.tolist(), curr_up.tolist()]);
        # else:
        #    print("find_frame resulted in a rotated rectangle ", np.array([curr_low_1, curr_up_1]),
        #          " that is non-intersecting the rotated rectangle  ", np.array([curr_low_2, curr_up_2]))

    if len(rect_curr) == 0:
        return float('nan');

    rect_curr = np.array(rect_curr);

    # Note: the following block of code was removed since find_frame is now used also for single states instead of boxes.
    # if check_rect_empty(rect_curr[0, :, :], 1):
    #    print('Result is an empty rectangle!!', rect_curr)
    #    return float('nan');
    # rect_curr = np.concatenate((rect_curr, [curr_low, curr_up]), 0);
    return rect_curr;


def gray(xi, n, k):
    # From: Guan, Dah - Jyh(1998). "Generalized Gray Codes with Applications".
    # Proc.Natl.Sci.Counc.Repub.Of China(A) 22: 841???848.
    # http: // nr.stpi.org.tw / ejournal / ProceedingA / v22n6 / 841 - 848.pdf.
    x = np.zeros((n, int(pow(k, n))));  # The matrix with all combinations
    a = np.zeros((n + 1, 1));  # The current combination following(k, n) - Gray code
    b = np.ones((n + 1, 1));  # +1 or -1
    c = k * np.ones((n + 1, 1));  # The maximum for each digit
    j = 0;
    while a[n] == 0:
        # Write current combination in the output
        x[:, j] = xi[np.reshape(a[0:n], (n,)).astype(int)];
        j = j + 1;

        # Compute the next combination
        i = 0;
        l = a[0] + b[0];
        while (l >= c[i]) or (l < 0):
            b[i] = -b[i];
            i = i + 1;
            l = a[i] + b[i];
        a[i] = int(l);
    return x;


def check_rect_empty(rect, allow_zero_dim):
    if allow_zero_dim and np.all(rect[0, :] <= rect[1, :]) and np.any(rect[0, :] < rect[1, :]):
        return False;
    elif (not allow_zero_dim) and np.all(rect[0, :] < rect[1, :]):
        return False;
    else:
        return True;


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
            return False;
    return True;


def does_rect_contain(rect1, rect2):  # does rect2 contains rect1
    for i in range(rect1.shape[1]):
        if rect1[0, i] + 0.01 < rect2[0, i] or rect1[1, i] - 0.01 > rect2[1, i]:
            print(rect2, " does not contain ", rect1, " since ", rect1[0, i], "<", rect2[0, i], " or ", rect1[1, i],
                  ">", rect2[1, i])
            return False
    return True


def add_rects_to_solver(rects, var_dict, cur_solver):
    print("Adding the following rectangles to solver: ", rects.shape)
    for rect_ind in range(rects.shape[0]):
        rect = rects[rect_ind, :, :];
        print(rect)
        c = [];
        for dim in range(rect.shape[1]):
            c.append(var_dict[dim] < rect[0, dim]);
            c.append(var_dict[dim] > rect[1, dim]);
        cur_solver.add(Or(c));
    return cur_solver;


def do_rects_list_contain_smt(rect1, var_dict, cur_solver):
    # adding the rectangle to the z3 solver
    print("The rectangles in the solver do not contain ", rect1)
    cur_solver.push();
    for dim in range(rect1.shape[1]):
        cur_solver.add(var_dict[dim] >= rect1[0, dim]);
        cur_solver.add(var_dict[dim] <= rect1[1, dim]);
    res = cur_solver.check();
    cur_solver.pop();
    if res == sat:
        return False;
    # print("rect1: ", rect1, " is already contained by previous rectangles");
    return True;


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
                    next_key[reset_dim] = 0;  # quantized_key_range[0, reset_dim]
                return next_key
        raise ValueError("curr_key should not exceed the bounds of the bounding box.")

    n = rect1.shape[1];
    list_rects = [np.array([rect[:n], rect[n:]]) for rect in list_rects]
    rect2 = get_convex_union(list_rects);
    if not does_rect_contain(rect1, rect2):
        return False
    if check_rect_empty(rect1, 1) or len(list_rects) == 0:
        print(rect1)
        print("Do not pass empty rectangles to cover function")
        return 0;
    else:
        partition_list = [];
        for dim in range(n):
            dim_partition_list = [rect1[0, dim], rect1[1, dim]];
            for rect_item in list_rects:
                if rect1[0, dim] <= rect_item[0, dim] <= rect1[1, dim]:
                    dim_partition_list.append(rect_item[0, dim]);
                if rect1[0, dim] <= rect_item[1, dim] <= rect1[1, dim]:
                    dim_partition_list.append(rect_item[1, dim]);
            dim_partition_list = np.sort(np.array(dim_partition_list));
            partition_list.append(dim_partition_list);

        curr_key = np.zeros((n,));
        quantized_key_range = np.array([len(dim_partition_list) - 2 for dim_partition_list in partition_list])
        while True:
            cell_covered = False;
            for rect_item in list_rects:
                if does_rect_contain(np.array([[partition_list[dim][int(curr_key[dim])] for dim in range(n)],
                                               [partition_list[dim][int(curr_key[dim]) + 1] for dim in range(n)]]),
                                     np.array([rect_item[0, :], rect_item[1, :]])):
                    cell_covered = True;
                    break;
            if not cell_covered:
                # print("ending do_rects_list_contain with cell not covered")
                return False;
            if np.all(curr_key == quantized_key_range):
                break
            curr_key = next_quantized_key(curr_key, quantized_key_range).astype(int);
    # print("ending do_rects_list_contain")
    return True;


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


def build_unified_abstraction(Symbolic_reduced, state_dimensions, s_ind, u_ind):
    transformation_list = [];
    n = state_dimensions.shape[1];
    unified_reachable_set = np.vstack(
        (Symbolic_reduced[s_ind, u_ind, np.arange(n), -1], Symbolic_reduced[s_ind, u_ind, n + np.arange(n), -1]));
    unified_center = (Symbolic_reduced[s_ind, u_ind, np.arange(n), -1] + Symbolic_reduced[
        s_ind, u_ind, n + np.arange(n), -1]) / 2;
    transformation_list.append([np.zeros(n, )]);
    for other_u_ind in range(1, Symbolic_reduced.shape[1]):
        specific_center = (Symbolic_reduced[s_ind, other_u_ind, np.arange(n), -1] + Symbolic_reduced[
            s_ind, other_u_ind, n + np.arange(n), -1]) / 2;
        transformation_vec = find_frame(specific_center, specific_center, unified_center, unified_center);
        # TODO: this should be replaced by transform_to_frame
        transformed_rect = transform_to_frame(np.array([Symbolic_reduced[s_ind, other_u_ind, np.arange(n), -1],
                                                        Symbolic_reduced[s_ind, other_u_ind, n + np.arange(n), -1]]),
                                              transformation_vec[0, 0, :]);
        unified_reachable_set = get_convex_union([unified_reachable_set, transformed_rect]);
        transformation_list[-1].append(transformation_vec[0, 0, :]);
    # print("transformation_list: ", transformation_list)
    return unified_reachable_set, transformation_list


def reach_avoid_synthesis_sets(Symbolic_reduced, sym_x, sym_u, state_dimensions, Target_low, Target_up,
                               Obstacle_low, Obstacle_up, X_low, X_up, U_low, U_up):
    t_start = time.time()
    n = state_dimensions.shape[1];
    p = index.Property();
    p.dimension = 3;
    p.dat_extension = 'data';
    p.idx_extension = 'index';
    rtree_idx3d = index.Index('3d_index', properties=p);
    discovered_rect = [];  # a list version of rtree_idx3d
    obstacles_intersecting_rect = [];

    # tracking_rtree_idx3d = index.Index('3d_index_tracking', properties=p);
    tracking_rect_global_cntr = 0;
    tracking_rects = [];
    tracking_abstract_state_control = []; # this tracks the corresponding control and the
    # abstract discrete state of tracking_rects.

    # rtree_idx3d_control = index.Index('3d_index_control', properties=p);

    abstract_rect_global_cntr = 0;
    abstract_rtree_idx3d = index.Index('3d_index_abstract',
                                       properties=p);  # contains the reachsets of the abstract system.
    max_reachable_distance = 0;
    for s_ind in range(Symbolic_reduced.shape[0]):
        for u_ind in range(Symbolic_reduced.shape[1]):
            abstract_rect_low = Symbolic_reduced[s_ind, u_ind, np.arange(n), -1];
            abstract_rect_up = Symbolic_reduced[s_ind, u_ind, n + np.arange(n), -1];
            abstract_rtree_idx3d.insert(abstract_rect_global_cntr, (abstract_rect_low[0], abstract_rect_low[1],
                                                                    abstract_rect_low[2], abstract_rect_up[0],
                                                                    abstract_rect_up[1], abstract_rect_up[2]),
                                        obj=(s_ind, u_ind));
            max_reachable_distance = max(max_reachable_distance,
                                         np.linalg.norm(np.average(np.array([abstract_rect_low.tolist(),
                                                                             abstract_rect_up.tolist()]))));
            abstract_rect_global_cntr = abstract_rect_global_cntr + 1;

    # unified_reachable_sets, unifying_transformation_list = build_unified_abstraction(Symbolic_reduced,
    #                                                                                 state_dimensions);

    # TODO in the case of the existence of non-symmetric coordinates, the following line may need to be changed to be
    #  the max over all the radii of the unified reachable sets over all cells in the grid over the
    #  non-symmetric coordinates
    init_radius = n * [0.1]; # unified_reachable_sets[-1][1, :] - unified_reachable_sets[-1][0, :];

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

    rect_curr_cntr = 0;
    rect_global_cntr = 0;
    intersection_time = 0;
    contain_time = 0;
    insert_time = 0;
    nearest_time = 0;

    # defining the z3 solver that we'll use to check if a rectangle is in a set of rectangles
    cur_solver = Solver();
    var_dict = [];
    for dim in range(n):
        var_dict.append(Real("x" + str(dim)));

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
                           obj=(-1, -1, 1));

    # targets = np.array(targets);
    print('\n%s\tStart of the control synthesis\n', time.time() - t_start);

    itr = 0;
    num_trials = 100;

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
    sampling_rectangle = np.array([X_low, X_up]);

    while itr < num_trials:  # or len(initial_sets_to_explore):
        # targets_temp = [];
        progress_indicator = False;
        useless = True;
        while useless and itr < num_trials:  # or len(initial_sets_to_explore)):
            # print("sampling_rectangle[1,:].tolist(): ", sampling_rectangle, sampling_rectangle[1,:].tolist())
            # if len(initial_sets_to_explore) == 0:
            sampled_state = sampling_rectangle[0, :] + np.array(
                [random.random() * ub for ub in
                 sampling_rectangle[1, :].tolist()]);  # sample_random_state(X_low, X_up);

            # itr = 0;
            # sampled_state = initial_sets_to_explore.pop(random.randint(0, len(initial_sets_to_explore) - 1));

            # if len(tracking_rects) == 0:
            hit = list(rtree_idx3d.nearest((sampled_state[0], sampled_state[1], sampled_state[2],
                                            sampled_state[0], sampled_state[1], sampled_state[2]), 1,
                                           objects=True));
            if len(hit) == 0:
                useless = False;
                break

            nearest_rect = np.array(
                [hit[0].bbox[:n], hit[0].bbox[n:]]);  # TODO: change when considering multiple neighbors
            # hits = list(rtree_idx3d.intersection((nearest_rect[0, 0], nearest_rect[0, 1], nearest_rect[0, 2],
            #                                      nearest_rect[1, 0], nearest_rect[1, 1], nearest_rect[1, 2]),
            #                                     objects=True));

            # print("Nearest rectangle before enlarging: ", nearest_rect);
            # print("Number of intersecting rectangles: ", len(hits));

            # for hit in hits:
            #    nearest_rect[0, :] = np.minimum(nearest_rect[0, :], hit.bbox[:n])
            #    nearest_rect[1, :] = np.maximum(nearest_rect[1, :], hit.bbox[n:])
            # print("Nearest rectangle after enlarging: ", nearest_rect, " and before enlarging: ", );
            useless = does_rect_contain(np.array([sampled_state, sampled_state]), nearest_rect);
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
                for i in range(Obstacle_up.shape[0]):  # and np.any(rect_curr):
                    rect_obs = np.array([Obstacle_low[i, :], Obstacle_up[i, :]]);
                    # TODO: replace this with R-tree intersection call, much faster when many obstacles exist.
                    if does_rect_contain(np.array([sampled_state, sampled_state]), rect_obs):
                        useless = True;
                        # print("Sampled state ", sampled_state, " is not useless.")
                        break
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
                itr += 1;
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
        if useless:
            print("Couldn't find non-explored state in ", num_trials, " uniformly random samples")
            break
        # if not progress_indicator:
        # steer #
        nearest_rect_center = np.average(nearest_rect, axis=0);
        # print("nearest_rect_center is: ", nearest_rect_center)
        path = [];
        path_resolution = 0.1;
        # TODO: make it half the distance from the center of any cell in X to the center
        # of the last reachable set.
        # path_vector = [];
        num_steps = 50;
        path_distance = np.linalg.norm(sampled_state - nearest_rect_center);
        # for dim in range(nearest_rect_center.shape[0]):
        #    path_vector.append((nearest_rect_center[dim] - sampled_state[dim]) / num_steps);
        path_vector = (sampled_state - nearest_rect_center) / path_distance;  # np.array(path_vector);
        # sampled_state = nearest_rect_center + num_steps * max_reachable_distance * path_vector;
        # print("path distance is: ", path_distance)
        for step_idx in range(num_steps):  # step_idx in range(math.floor(path_distance / path_resolution)):
            # print("sampled_state: ", sampled_state)
            # print("path_vector: ", step_idx * path_vector)
            # print("path_state: ", sampled_state + step_idx * path_vector)
            path.append(nearest_rect_center + step_idx * path_vector);

        # nearest_poly = pc.box2poly(nearest_rect.T);

        # Hussein: we might need to steer the system to there, can we check if there is a control input with a
        # reachable set that intersects the line from the center of the nearest rectangle to the sampled point?

        # valid_hit = False;

        rrt_done = False;
        for path_state in path:
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
            rect_curr = np.array([path_state - init_radius, path_state + init_radius]);
            # tracking_rects.append(rect_curr);
            # tracking_rtree_idx3d.insert(tracking_rect_global_cntr, (
            #    rects_curr[0,0], rects_curr[0,1], rects_curr[0,2],
            #    rects_curr[1, 0], rects_curr[1, 1], rects_curr[1, 2]),
            #                   obj=(s_ind, u_ind, 1));
            # tracking_rect_global_cntr += 1;
            # TODO: implement an RRT that grows the tracking tree then do a backward breadth-first-search
            #  to add the reachable initial sets
            sample_cntr = 0;
            while not rrt_done and sample_cntr < 2 * num_steps:  # this should depend on the step size
                # here we use a simple version of RRT to find a path from the sampled state towards rtree_idx3d.
                sample_cntr += 1;
                sampled_state = sampling_rectangle[0, :] + np.array([random.random() * ub for ub in
                                                                     sampling_rectangle[1, :].tolist()]);
                hits = list(rtree_idx3d.nearest(
                    (sampled_state[0], sampled_state[1], sampled_state[2],
                     sampled_state[0] + 0.01, sampled_state[1] + 0.01,
                     sampled_state[2] + 0.01), 1, objects=True));
                nearest_rect = np.array([hits[0].bbox[:n], hits[0].bbox[n:]]);  # TODO: this should be changed if
                # number of nearest rectangles in the previous command is larger than 1.
                # print("nearest_rect: ", nearest_rect)
                # nearest_poly = pc.box2poly(nearest_rect.T);
                # if rtree_idx3d.count((sampled_state[0], sampled_state[1], sampled_state[2],
                #                      sampled_state[0] + 0.01,  sampled_state[1] + 0.01, sampled_state[2] + 0.01)) > 0:
                #    continue
                # print("path state: ", path_state)
                # abstract_nearest_poly = transform_poly_to_abstract(nearest_poly, np.average(tracking_rects[-1], axis=0));
                abstract_nearest_rect = transform_rect_to_abstract(nearest_rect, np.average(rect_curr, axis=0));
                # np.column_stack(
                # abstract_nearest_poly.bounding_box).T;
                # print("abstract_nearest_rect: ", abstract_nearest_rect)
                hits = list(abstract_rtree_idx3d.nearest(
                    (abstract_nearest_rect[0, 0], abstract_nearest_rect[0, 1], abstract_nearest_rect[0, 2],
                     abstract_nearest_rect[1, 0] + 0.01, abstract_nearest_rect[1, 1] + 0.01,
                     abstract_nearest_rect[1, 2]
                     + 0.01), 1, objects=True));
                # print("nearest abstract reachable set: ", hits[0].bbox)
                s_ind = hits[0].object[0];
                u_ind = hits[0].object[1];
                reachable_set = [];
                for t_ind in range(Symbolic_reduced.shape[3]):
                    reachable_set.append(transform_to_frames(Symbolic_reduced[s_ind, u_ind, np.arange(n), t_ind],
                                                             Symbolic_reduced[
                                                                 s_ind, u_ind, n + np.arange(n), t_ind],
                                                             rect_curr[0, :], rect_curr[1, :]));
                    # discovered_rect.append(reachable_set[-1]);
                # check if the reachable set intersects the unsafe sets.
                # if not, define rect_curr to be the initial set of the reachable set.
                # below code is for checking
                # rect_curr = reachable_set[0];  # or tracking_rects[-1];
                intersects_obstacle = False;
                for reachable_rect in reachable_set:
                    i = 0
                    while i < Obstacle_up.shape[0]:  # and np.any(rect_curr):
                        rect_obs = np.array([Obstacle_low[i, :], Obstacle_up[i, :]]);
                        if do_rects_inter(reachable_rect,  # TODO: change it back.
                                          rect_obs):
                            obstacles_intersecting_rect.append(reachable_rect);
                            intersects_obstacle = True;
                            break
                        i = i + 1;
                    if intersects_obstacle:
                        # print("The reachable rect ", reachable_rect, " intersects obstacle ", rect_obs, " :/")
                        break
                if not intersects_obstacle:
                    # print("Adding ", rect_curr, " to tracking_rects")
                    tracking_rect_global_cntr += 1;
                    tracking_rects.append(rect_curr);
                    tracking_abstract_state_control.append((s_ind, u_ind));
                    # discovered_rect.append(rect_curr);
                    # rect_curr = reachable_set[-1];
                    if np.any(reachable_set[-1][0, :] < X_low) or np.any(reachable_set[-1][0, :] > X_up) \
                            or np.any(reachable_set[-1][1, :] < X_low) or np.any(reachable_set[-1][1, :] > X_up):
                        # if np.all(rect_curr[1, :] > X_low):
                        #    curr_low = np.maximum(rect_curr[0, :], X_low);
                        # else:
                        continue
                    else:
                        curr_low = reachable_set[-1][0, :];
                        curr_up = reachable_set[-1][1, :];
                    # if np.any(reachable_set[-1][0, :] > X_up) or np.any(reachable_set[-1][1, :] < X_low):
                    #    continue
                    # if np.any(rect_curr[1, :] > X_up):
                    #    if np.all(rect_curr[0, :] < X_up):
                    #        curr_up = np.minimum(rect_curr[1, :], X_up);
                    #    else:
                    #        continue
                    # else:
                    #    curr_up = rect_curr[1, :];
                    rect_curr = np.array([curr_low, curr_up]);
                    # check if the last rect in tracking_rects is covered
                    hits = list(
                        rtree_idx3d.intersection(
                            (reachable_set[-1][0, 0], reachable_set[-1][0, 1], reachable_set[-1][0, 2],
                             reachable_set[-1][1, 0], reachable_set[-1][1, 1],
                             reachable_set[-1][1, 2]),
                            objects=True));
                    inter_num = len(hits);
                    if inter_num == 0:
                        continue
                    # inter_num = rtree_idx3d.count((rect_low[0], rect_low[1], rect_low[2],
                    #                               rect_up[0], rect_up[1], rect_up[2]));
                    hits = np.array([np.array([hit.bbox[:n], hit.bbox[n:]]) for hit in hits]);
                    cur_solver.reset();
                    # if inter_num >= 20:
                    #    sampled_indices = np.linspace(0, len(hits), num=20, endpoint=False).astype(int);
                    #    hits = np.array(hits)[sampled_indices];
                    cur_solver = add_rects_to_solver(hits, var_dict, cur_solver);
                    # not_useful = len(hits) > 0 and do_rects_list_contain(rect_curr[j, :, :],
                    #                                                     [hit.bbox for hit in hits]);
                    rrt_done = do_rects_list_contain_smt(tracking_rects[-1], var_dict, cur_solver);
                    # rrt_done = do_rects_list_contain_smt(tracking_rects[-1], var_dict=var_dict, cur_solver=cur_solver);
            if rrt_done:
                for rect in tracking_rects:
                    rtree_idx3d.insert(rect_global_cntr, (rect[0, 0], rect[0, 1], rect[0, 2],
                                                          rect[1, 0], rect[1, 1], rect[1, 2]),
                                       obj=(s_ind, u_ind, 1))  # TODO: this should change to the correct u_ind,
                    discovered_rect.append(rect);
                    rect_global_cntr += 1;
                progress_indicator = True;
            else:
                print("Maximum number of ", 2 * num_steps, " steps has been taken and the target has not been reached :(")

            tracking_rects = [];
            tracking_rect_global_cntr = 0;
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

        if progress_indicator:
            print('%s\t%d new controllable states have been found in this synthesis iteration\n',
                  time.time() - t_start, rect_global_cntr - rect_curr_cntr)
            rect_curr_cntr = rect_global_cntr;
            # trying to enlarge each of the rectangles in rect_curr

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
            print("intersection_time, contain_time, insert_time, nearest_time)", intersection_time, contain_time,
                  insert_time, nearest_time)
        else:
            print('%s\tNo new controllable state has been found in this synthesis iteration\n', time.time() - t_start)
            itr += 1;
            # break;

    print(['Controller synthesis for reach-avoid specification: ', time.time() - t_start, ' seconds'])
    # controllable_states = np.nonzero(Controller);
    if rect_global_cntr:
        print('%d symbols are controllable to satisfy the reach-avoid specification\n', rect_global_cntr)
    else:
        print('The reach-avoid specification cannot be satisfied from any initial state\n')

    # print("The number of states that were supposed to be explored but weren't is:", len(initial_sets_to_explore));

    # for c in cur_solver.assertions():
    #    print(c)

    # print("discovered rect: ", discovered_rect)
    plt.figure("Original coordinates")
    currentAxis = plt.gca()
    color = 'r';
    for i in range(Obstacle_up.shape[0]):  # and np.any(rect_curr):
        rect = np.array([Obstacle_low[i, :], Obstacle_up[i, :]]);
        rect_patch = Rectangle(rect[0, [0, 1]], rect[1, 0] - rect[0, 0],
                               rect[1, 1] - rect[0, 1], linewidth=1,
                               edgecolor=color, facecolor=color)
        currentAxis.add_patch(rect_patch)

    color = 'g';
    for target_idx in range(Target_low.shape[0]):
        rect = np.array([Target_low[target_idx, :], Target_up[target_idx, :]]);
        rect_patch = Rectangle(rect[0, [0, 1]], rect[1, 0] - rect[0, 0],
                               rect[1, 1] - rect[0, 1], linewidth=1,
                               edgecolor=color, facecolor=color)
        currentAxis.add_patch(rect_patch)

    color = 'b';
    edge_color = 'k';
    for rect in discovered_rect:
        rect_patch = Rectangle(rect[0, [0, 1]], rect[1, 0] - rect[0, 0],
                               rect[1, 1] - rect[0, 1], linewidth=1,
                               edgecolor=edge_color, facecolor=color)
        currentAxis.add_patch(rect_patch)

    color = 'y';
    print("obstacles_intersecting_rect: ", obstacles_intersecting_rect)
    for rect in obstacles_intersecting_rect:
        rect_patch = Rectangle(rect[0, [0, 1]], rect[1, 0] - rect[0, 0],
                               rect[1, 1] - rect[0, 1], linewidth=1,
                               edgecolor=edge_color, facecolor=color)
        currentAxis.add_patch(rect_patch)

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
