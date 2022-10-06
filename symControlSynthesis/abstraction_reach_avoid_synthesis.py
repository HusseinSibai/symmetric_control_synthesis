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


def transform_poly_to_abstract(poly: pc.polytope, state: np.array):
    # this function takes a polytope in the state space and transforms it to the abstract coordinates.
    # this should be provided by the user as it depends on the symmetries
    # Hussein: unlike the work in SceneChecker, we here rotate then translate, non-ideal, I prefer translation
    # then rotation but this requires changing find_frame which would take time.
    translation_vector = np.array([-1 * state[0], -1 * state[1], -1 * state[2]])
    rot_angle = -1 * state[2]
    poly_out: pc.Polytope = poly.translation(translation_vector)
    return poly_out.rotation(i=0, j=1, theta=rot_angle)


def fix_angle(angle):
    while angle < -1 * math.pi:
        angle += 2 * math.pi
    while angle > math.pi:
        angle -= 2 * math.pi
    return angle


def fix_rect_angles(rect: np.array):
    rect[0, 2] = fix_angle(rect[0, 2])
    rect[1, 2] = fix_angle(rect[1, 2])
    if rect[0, 2] > rect[1, 2]:
        temp = rect[0, 2]
        rect[0, 2] = rect[1, 2]
        rect[1, 2] = temp
    return rect


# https://stackoverflow.com/questions/11406189/determine-if-angle-lies-between-2-other-angles
def is_within_range(angle, a, b):
    a -= angle
    b -= angle
    a = fix_angle(a)
    b = fix_angle(b)
    if a * b >= 0:
        return False
    return abs(a - b) <= math.pi


def does_interval_contain(a_s, b_s, a_l, b_l):
    if is_within_range(a_s, a_l, b_l) and is_within_range(b_s, a_l, b_l) \
            and is_within_range(a_s, a_l, b_s) and is_within_range(b_s, a_s, b_l):
        return True
    return False


def transform_rect_to_abstract(rect: np.array, state: np.array, overapproximate=False):
    ang = -1 * state[2]  # psi = 0 is North, psi = pi/2 is east

    while ang < 0:
        ang += 2 * math.pi
    while ang > 2 * math.pi:
        ang -= 2 * math.pi

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


def transform_rect_to_abstract_frames(concrete_rect, frames_rect, over_approximate=False):
    box_1 = transform_rect_to_abstract(concrete_rect, frames_rect[0, :], overapproximate=over_approximate)
    box_2 = transform_rect_to_abstract(concrete_rect, frames_rect[1, :], overapproximate=over_approximate)
    box_3 = transform_rect_to_abstract(concrete_rect, np.array([frames_rect[0, 0], frames_rect[0, 1],
                                                                frames_rect[1, 2]]), overapproximate=over_approximate)
    box_4 = transform_rect_to_abstract(concrete_rect, np.array([frames_rect[1, 0], frames_rect[1, 1],
                                                                frames_rect[0, 2]]), overapproximate=over_approximate)
    if over_approximate:
        result = get_convex_union([box_1, box_2, box_3, box_4])
    else:
        result = get_intersection([box_1, box_2, box_3, box_4])
    return result  # np.array([result_low, result_up]);


def transform_poly_to_abstract_frames(concrete_poly, frames_rect):
    poly_1 = transform_poly_to_abstract(concrete_poly, frames_rect[0, :])
    poly_2 = transform_poly_to_abstract(concrete_poly, frames_rect[1, :])
    poly_3 = transform_poly_to_abstract(concrete_poly, np.array([frames_rect[0, 0], frames_rect[0, 1],
                                                                 frames_rect[1, 2]]))
    poly_4 = transform_poly_to_abstract(concrete_poly, np.array([frames_rect[1, 0], frames_rect[1, 1],
                                                                 frames_rect[0, 2]]))

    result = pc.union(poly_1, poly_2)
    result = pc.union(result, poly_3)
    result = pc.union(result, poly_4)
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
    for i in range(rect1.shape[1]):
        if rect1[0, i] > rect2[1, i] + 0.01 or rect1[1, i] + 0.01 < rect2[0, i]:
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
        print("The interval ", rect2[0, 2], rect2[1, 2], " contains ", rect1[0, 2], rect1[1, 2], "? ",
              does_interval_contain(rect1[0, 2], rect1[1, 2], rect2[0, 2], rect2[1, 2]))
        return does_interval_contain(rect1[0, 2], rect1[1, 2], rect2[0, 2], rect2[1, 2])
    return False


def get_convex_union(list_array: List[np.array]) -> np.array:
    assert len(list_array) > 0, "list array length should be larger than zero"
    result: np.array = np.copy(list_array[0])
    for i in range(1, len(list_array)):
        result[0, :] = np.minimum(result[0, :], list_array[i][0, :])
        result[1, :] = np.maximum(result[1, :], list_array[i][1, :])
    return result


def get_intersection(list_array: List[np.array]) -> np.array:
    assert len(list_array) > 0, "list array length should be larger than zero"
    result: np.array = np.copy(list_array[0])
    for i in range(len(list_array)):
        if do_rects_inter(list_array[i], result):
            result[0, :] = np.maximum(result[0, :], list_array[i][0, :])
            result[1, :] = np.minimum(result[1, :], list_array[i][1, :])
        else:
            return None
    return result


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

    obstacles = []
    targets = []
    for obstacle_idx in range(Obstacle_low.shape[0]):
        obstacle_rect = np.array([Obstacle_low[obstacle_idx, :], Obstacle_up[obstacle_idx, :]])
        obstacle_poly = pc.box2poly(obstacle_rect.T)
        obstacles.append(obstacle_poly)

    for target_idx in range(Target_low.shape[0]):
        target_rect = np.array([Target_low[target_idx, :], Target_up[target_idx, :]])
        target_rect = fix_rect_angles(target_rect)
        # target_poly = pc.box2poly(target_rect.T)
        targets.append(target_rect)

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
    initial_set = np.array([[6, 2, math.pi / 2], [6.1, 2.1, math.pi / 2 + 0.01]])
    traversal_stack = [copy.deepcopy(initial_set)]
    result_abstract_paths = []
    while len(traversal_stack) and fail_itr < M:
        initial_set = traversal_stack.pop()

        result, result_abstract_path = synthesize_helper(Symbolic_reduced, sym_x, sym_u, state_dimensions, targets,
                                                         obstacles, X_low, X_up, abstract_rtree_idx3d,
                                                         abstract_rect_global_cntr, abstract_paths,
                                                         abstract_path_last_set_parts,
                                                         initial_set, N)

        itr += 1
        if result != -1:
            succ_itr += 1
            print(time.time() - t_start, " ", abstract_rect_global_cntr,
                  " new controllable states have been found in this synthesis iteration\n")
            print(abstract_rect_global_cntr, ' symbols are controllable to satisfy the reach-avoid specification\n')
            print("success iterations: ", succ_itr)
            result_abstract_paths.append((initial_set, result_abstract_path))
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
    if len(traversal_stack) == 0:  # abstract_rect_global_cntr:
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
                rect = transform_to_frames(rect[0, :], rect[1, :], initial_set[0, :], initial_set[1, :])
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


def synthesize_helper(Symbolic_reduced, sym_x, sym_u, state_dimensions, targets,
                      obstacles, X_low, X_up, abstract_rtree_idx3d, abstract_rect_global_cntr,
                      abstract_paths, abstract_path_last_set_parts, initial_set, max_path_length):
    n = state_dimensions.shape[1]
    num_nearest_controls = int(Symbolic_reduced.shape[1] / 2)
    debugging = False
    target_aiming_prob = 0.8
    symbol_step = (X_up - X_low) / sym_x
    print("new synthesize_helper call ")
    print("initial_set: ", initial_set)
    print("max_path_length: ", max_path_length)
    abstract_targets = []

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
            return -1, None  # Failure

        if debugging:
            print("hits_not_intersecting_obstacles: ", len(hits_not_intersecting_obstacles))

        for hit in hits_not_intersecting_obstacles:
            s_ind, u_ind = hit.object
            if len(abstract_path_last_set_parts[s_ind][u_ind]):
                success = True
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
                                                                 abstract_path_last_set_parts,
                                                                 new_curr_initset, max_path_length - 1)

                if curr_result_list:  # in range(len(abstract_paths))
                    # iterate over results and take the union of the paths.
                    new_path = copy.deepcopy(abstract_paths[hit.id])
                    # print("Length of previous path: ", len(new_path))
                    for rect in result_abstract_path:  # abstract_paths[result]:
                        # also this should be fixed, changed last_set to abstract_paths[hit.id][-1] temporarily
                        new_path.append(transform_to_frames(rect[0, :], rect[1, :], abstract_paths[hit.id][-1][0, :],
                                                            abstract_paths[hit.id][-1][1, :]))
                    if debugging:
                        print("new path: ", new_path)
                    abstract_paths.append(new_path)
                    result_rect = new_path[-1]
                    abstract_rtree_idx3d.insert(abstract_rect_global_cntr, (result_rect[0, 0], result_rect[0, 1],
                                                                            result_rect[0, 2], result_rect[1, 0],
                                                                            result_rect[1, 1], result_rect[1, 2]),
                                                obj=(s_ind, u_ind))
                    abstract_rect_global_cntr += 1
                else:
                    success = False
                if success:
                    # print("length of abstract_paths[hid.id]: ", len(new_path))
                    result_list.append((curr_initset, hit.id, new_path))
                    break
                    # return hit.id, new_path
        if np.all(curr_key == quantized_key_range[1, :]):
            break
        curr_key = next_quantized_key(curr_key, quantized_key_range)

    print("All sampled paths do not reach target")
    return -1, None
