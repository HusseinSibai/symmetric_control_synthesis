import numpy as np
import math


# A function that rotates a rectangle by angle alpha (the xy-coordinate system by angle -alpha) and translates
# the third coordinate accordingly.
def get_crop_coordinates(rect: np.array, ang: float, overapproximate=True):
    if ang < 0:
        ang += 2 * math.pi;
    elif ang > 2 * math.pi:
        ang -= 2 * math.pi;
    rect_range = rect[1, :] - rect[0, :];
    quadrant = math.floor(ang / (math.pi / 2));
    if quadrant == 0 or quadrant == 2:
        sign_alpha = ang;
    else:
        ang: math.pi - ang;
    alpha = (sign_alpha % math.pi + math.pi) % math.pi;

    w = rect_range[0] * math.cos(alpha) + rect_range[1] * math.sin(alpha);
    h = rect_range[0] * math.sin(alpha) + rect_range[1] * math.cos(alpha);
    bb_center = np.average(rect, axis=0);
    upper_left_in_global_coordinate = np.array([bb_center[0] - (rect_range[0] / 2.0),
                                                bb_center[1] + (rect_range[1] / 2.0),
                                                0]);
    if overapproximate:
        low_new_rect = np.array(
            [upper_left_in_global_coordinate[0], upper_left_in_global_coordinate[1] - h,
             rect[0, 2] + alpha]);
        up_new_rect = np.array(
            [upper_left_in_global_coordinate[0] + w, upper_left_in_global_coordinate[1],
             rect[1, 2] + alpha]);
        return np.array([low_new_rect, up_new_rect]);

    if rect_range[0] < rect_range[1]:
        gamma = math.atan2(w, h)
    else:
        gamma = math.atan2(h, w);

    delta = math.pi - alpha - gamma;

    if rect_range[0] < rect_range[1]:
        length = rect_range[1];
    else:
        length = rect_range[0];
    d = length * math.cos(alpha);
    a = d * math.sin(alpha) / math.sin(delta);

    y = a * math.cos(gamma);
    x = y * math.tan(gamma);

    low_new_rect = np.array([x + upper_left_in_global_coordinate[0], upper_left_in_global_coordinate[1] - y - h + 2 * y,
                             rect[0, 2] + alpha]);
    up_new_rect = np.array([x + upper_left_in_global_coordinate[0] + w - 2 * x, upper_left_in_global_coordinate[1] - y,
                            rect[1, 2] + alpha]);
    result = np.array([low_new_rect, up_new_rect]);

    return result


rect = np.array([[-1, -0.5, -0.8], [1, 2, 3]]);
alpha = math.pi / 3.0;
new_rect = get_crop_coordinates(rect, alpha, overapproximate=False);
print(new_rect)
