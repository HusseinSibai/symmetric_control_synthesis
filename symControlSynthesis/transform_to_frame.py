import math
import numpy as np
import polytope as pc
from typing import List, Dict


def transform_poly_to_abstract(poly: pc.polytope, state: np.array):
    # this function takes a polytope in the state space and transforms it to the abstract coordinates.
    # this should be provided by the user as it depends on the symmetries
    # Hussein: unlike the work in SceneChecker, we here rotate then translate, non-ideal, I prefer translation
    # then rotation but this requires changing find_frame which would take time.
    translation_vector = state;
    rot_angle = state[2];
    poly_out: pc.Polytope = poly.rotation(i=0, j=1, theta=rot_angle);
    return poly_out.translation(translation_vector)


def get_convex_union(list_array: List[np.array]) -> np.array:
    assert len(list_array) > 0, "list array length should be larger than zero"
    result: np.array = np.copy(list_array[0])
    for i in range(1, len(list_array)):
        result[0, :] = np.minimum(result[0, :], list_array[i][0, :])
        result[1, :] = np.maximum(result[1, :], list_array[i][1, :])
    return result


def find_frame(low_red, up_red, target_full_low, target_full_up):
    # reimplement using Z3? does it return a set of states instead of just a counter-example?
    # print(low_red, up_red, target_full_low, target_full_up)
    if up_red[2] - low_red[2] > target_full_up[2] - target_full_low[2]:
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
        theta_up = theta_up_sys;

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

        curr_low_temp = np.maximum(curr_low_1, curr_low_2);
        curr_up_temp = np.minimum(curr_up_1, curr_up_2);
        curr_low = np.minimum(curr_low_temp, curr_up_temp);
        curr_up = np.maximum(curr_low_temp, curr_up_temp);
        rect_curr.append([curr_low.tolist(), curr_up.tolist()]);

    rect_curr = np.array(rect_curr);

    # Note: the following block of code was removed since find_frame is now used also for single states instead of boxes.
    # if check_rect_empty(rect_curr[0, :, :], 1):
    #    print('Result is an empty rectangle!!', rect_curr)
    #    return float('nan');
    # rect_curr = np.concatenate((rect_curr, [curr_low, curr_up]), 0);
    return rect_curr;


def transform_to_frame(rect: np.array, state: np.array, overapproximate=True):
    # print("rect: ", rect)
    # print("state: ", state)
    ang = state[2];

    while ang < 0:
        ang += 2 * math.pi;
    while ang > 2 * math.pi:
        ang -= 2 * math.pi;

    state[2] = ang;
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

    if 0 <= state[2] <= math.pi / 2:
        x_bb_up = up_red[0] + (rect[1, 1] - rect[0, 1]) * math.sin(state[2]);
        y_bb_up = up_red[1];
        x_bb_low = low_red[0] - (rect[1, 1] - rect[0, 1]) * math.sin(state[2]);
        y_bb_low = low_red[1];
    elif math.pi / 2 <= state[2] <= math.pi:
        x_bb_up = low_red[0];
        y_bb_up = low_red[1] + (rect[1, 0] - rect[0, 0]) * math.cos(state[2] - math.pi / 2);
        x_bb_low = up_red[0];
        y_bb_low = up_red[1] - (rect[1, 0] - rect[0, 0]) * math.cos(state[2] - math.pi / 2);
    elif math.pi <= state[2] <= 3 * math.pi / 2.0:
        x_bb_up = low_red[0] + (rect[1, 1] - rect[0, 1]) * math.cos(3 * math.pi / 2 - state[2]);
        y_bb_up = low_red[1];
        x_bb_low = up_red[0] - (rect[1, 1] - rect[0, 1]) * math.cos(3 * math.pi / 2 - state[2])
        y_bb_low = up_red[1];
    else:
        x_bb_up = up_red[0];
        y_bb_up = up_red[1] + (rect[1, 0] - rect[0, 0]) * math.cos(state[2] - 3 * math.pi / 2);
        x_bb_low = low_red[0];
        y_bb_low = low_red[1] - (rect[1, 0] - rect[0, 0]) * math.cos(state[2] - 3 * math.pi / 2);

    bb = np.array([[x_bb_low, y_bb_low, low_red[2]], [x_bb_up, y_bb_up, up_red[2]]])
    rect_range = rect[1, :] - rect[0, :];
    quadrant = math.floor(ang / (math.pi / 2));
    # print("ang: ", ang)
    # print("rect_range: ", rect_range)
    # print("quadrant: ", quadrant)
    if quadrant == 0 or quadrant == 2:
        sign_alpha = ang;
    else:
        sign_alpha = math.pi - ang;
    alpha = (sign_alpha % math.pi + math.pi) % math.pi;
    # print("alpha: ", alpha)

    # w = rect_range[0] * math.cos(alpha) + rect_range[1] * math.sin(alpha);
    # h = rect_range[0] * math.sin(alpha) + rect_range[1] * math.cos(alpha);
    w = bb[1,0] - bb[0,0];
    h = bb[1, 1] - bb[0, 1];

    # print("w: ", w)
    # print("h: ", h)

    bb_center = np.average(bb, axis=0); # np.average(rect, axis=0);
    '''
    bb_center = np.array([bb_center[0] * math.cos(state[2]) -
                          (bb_center[1]) * math.sin(state[2]) + state[0],
                          (bb_center[0]) * math.sin(state[2]) +
                          (bb_center[1]) * math.cos(state[2]) + state[1],
                          bb_center[2] + state[2]]);
    '''

    # print("bb_center: ", bb_center);

    '''
    upper_left_in_global_coordinate = np.array([bb_center[0] - (rect_range[0] / 2.0),
                                                bb_center[1] + (rect_range[1] / 2.0),
                                                0]);
    print("upper_left_in_global_coordinate: ", upper_left_in_global_coordinate)
    '''
    if overapproximate:
        return bb;
        '''
        low_new_rect = np.array(
            [upper_left_in_global_coordinate[0], upper_left_in_global_coordinate[1] - h,
             bb_center[2] - rect_range[2]/2.0]);
        up_new_rect = np.array(
            [upper_left_in_global_coordinate[0] + w, upper_left_in_global_coordinate[1],
             bb_center[2] + rect_range[2]/2.0]);
        return np.array([low_new_rect, up_new_rect]);
        '''

    if rect_range[0] < rect_range[1]:
        gamma = math.atan2(w, h)
    else:
        gamma = math.atan2(h, w);

    # print("gamma: ", gamma)

    delta = math.pi - alpha - gamma;

    # print("delta: ", delta);

    if rect_range[0] < rect_range[1]:
        length = rect_range[1];
    else:
        length = rect_range[0];

    # print("length: ", length);

    d = length * math.cos(alpha);
    a = d * math.sin(alpha) / math.sin(delta);

    # print("d: ", d)
    # print("a: ", a)

    y = a * math.cos(gamma);
    x = y * math.tan(gamma);

    # print("y: ", y)
    # print("x: ", x)

    '''
    low_new_rect = np.array([x + upper_left_in_global_coordinate[0],
                             upper_left_in_global_coordinate[1] - y - h + 2 * y,
                             bb_center[2] - rect_range[2] / 2.0]);
    up_new_rect = np.array([x + upper_left_in_global_coordinate[0] + w - 2 * x,
                            upper_left_in_global_coordinate[1] - y,
                            bb_center[2] + rect_range[2] / 2.0]);
    '''
    low_new_rect = np.array([bb[0,0] + x,
                             bb[0,1] + y,
                             bb[0,2]]);
    up_new_rect = np.array([low_new_rect[0] + w - 2 * x,
                            low_new_rect[1] + h - 2 * y,
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
    print("Input to transform_to_frame: ", low_red, up_red, source_full_low)
    box_1 = transform_to_frame(np.array([low_red, up_red]), source_full_low, overapproximate=False);
    print("Output of transform_to_frame: ", box_1)
    print("Input to transform_to_frame: ", low_red, up_red, source_full_up)
    box_2 = transform_to_frame(np.array([low_red, up_red]), source_full_up, overapproximate=False);
    print("Output of transform_to_frame: ", box_2)
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
    return result


'''
def transform_to_frames(low_red, up_red, source_full_low, source_full_up):
    poly_1 = pc.box2poly(np.column_stack((low_red, up_red)));
    poly_1 = transform_poly_to_abstract(poly_1, source_full_low);
    poly_2 = pc.box2poly(np.column_stack((low_red, up_red)));
    poly_2 = transform_poly_to_abstract(poly_2, source_full_up);
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
    return get_convex_union([np.column_stack(poly_1.bounding_box).T, np.column_stack(poly_2.bounding_box).T])
    # return np.array([result_low, result_up]);
'''

rect = np.array([[5.55352, 0.56819, 2.0944],
                 [7.3257, 0.88228, 2.1944]]);
state = [-0.10383, -0.44535, -1.];
print(transform_to_frame(rect, state, overapproximate=False));

low_red = np.array([-1, 0.45, math.pi / 6.0]);
up_red = np.array([-0.75, 0.55, math.pi / 4.0]);
source_full_low = np.array([-1, 2, math.pi / 2.5]);
source_full_up = np.array([-0.5, 3, math.pi / 2.0]);
target = transform_to_frames(low_red, up_red, source_full_low, source_full_up);
print(target)

new_source = find_frame(low_red, up_red, target[0, :], target[1, :])
print(new_source)

import matplotlib

matplotlib.use("macOSX")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, Polygon

plt.figure("Reduced coordinates")
color = 'orange'
currentAxis_1 = plt.gca()
rect = np.array([low_red, up_red]);
rect_patch = Rectangle(rect[0, [0, 1]], rect[1, 0] - rect[0, 0],
                       rect[1, 1] - rect[0, 1], linewidth=1,
                       edgecolor=color, facecolor=color)
currentAxis_1.add_patch(rect_patch)
rect = np.array([source_full_low, source_full_up]);
rect_patch = Rectangle(rect[0, [0, 1]], rect[1, 0] - rect[0, 0],
                       rect[1, 1] - rect[0, 1], linewidth=1,
                       edgecolor='r', facecolor='r')
currentAxis_1.add_patch(rect_patch)
rect = target;
rect_patch = Rectangle(rect[0, [0, 1]], rect[1, 0] - rect[0, 0],
                       rect[1, 1] - rect[0, 1], linewidth=1,
                       edgecolor='b', facecolor='b')
currentAxis_1.add_patch(rect_patch)

for rect_idx in range(new_source.shape[0]):
    rect = new_source[rect_idx, :, :];
    rect_patch = Rectangle(rect[0, [0, 1]], rect[1, 0] - rect[0, 0],
                           rect[1, 1] - rect[0, 1], linewidth=1,
                           edgecolor='g', facecolor='g')
    currentAxis_1.add_patch(rect_patch)
# TODO plot transformation points as well.
plt.ylim([-4, 4])
plt.xlim([-5, 5])
plt.show()
