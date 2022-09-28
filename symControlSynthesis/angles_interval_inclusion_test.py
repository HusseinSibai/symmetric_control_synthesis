import math
import numpy as np


def fix_angle(angle):
    while angle < -1 * math.pi:
        angle += 2 * math.pi
    while angle > math.pi:
        angle -= 2 * math.pi
    return angle


def is_within_range(angle, a, b):
    a -= angle
    b -= angle
    a = fix_angle(a)
    b = fix_angle(b)
    if a * b >= 0:
        print(angle, a, b, False)
        return False
    print(angle, a, b, abs(a - b))
    return abs(a - b) <= math.pi


def does_interval_contain(a_s, b_s, a_l, b_l):
    if is_within_range(a_s, a_l, b_l) and is_within_range(b_s, a_l, b_l) \
            and is_within_range(a_s, a_l, b_s) and is_within_range(b_s, a_s, b_l):
        return True
    return False


print(does_interval_contain(-0.2, 0.2, -0.3, 0.3))