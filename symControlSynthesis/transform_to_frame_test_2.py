import math
import numpy as np
import polytope as pc

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

def transform_poly_to_abstract(poly: pc.polytope, state: np.array):
    # this function takes a polytope in the state space and transforms it to the abstract coordinates.
    # this should be provided by the user as it depends on the symmetries
    # Hussein: unlike the work in SceneChecker, we here rotate then translate, non-ideal, I prefer translation
    # then rotation but this requires changing find_frame which would take time.
    translation_vector = np.array([-1 * state[0], -1 * state[1], -1 * state[2]])
    rot_angle = -1 * state[2]
    poly_out: pc.Polytope = poly.translation(translation_vector)
    return poly_out.rotation(i=0, j=1, theta=rot_angle)


rect = np.array([[5.55352, 0.56819, 2.0944],
                 [7.3257, 0.88228, 2.1944]]);
state = [0, 0, 1.];
print(rect)
result = transform_to_frame(rect, state, overapproximate=False)
print(result)
poly = transform_poly_to_abstract(pc.box2poly(result.T), state)
box = np.column_stack(poly.bounding_box).T
print(box)
