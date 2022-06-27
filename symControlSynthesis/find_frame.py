import math

def find_frame(low_red, up_red, target_full_low, target_full_up):
    if up_red[2] - low_red[2] > target_full_up[2] - target_full_low[2]:
        return float('nan')
    rect_curr = [];
    theta_low_sys = target_full_low[2] - low_red[2];
    theta_up_sys = target_full_up[2] - up_red[2];

    theta_low = theta_low_sys;
    theta_up = theta_up_sys;

    if theta_low >= 0 and theta_low <= math.pi / 2:
        x_target_up_1 = target_full_up[1] - (up_red[1] - low_red[1]) * math.cos(math.pi / 2 - theta_low);
        y_target_up_1 = target_full_up[1];
    elif theta_low >= math.pi / 2 and theta_low <= math.pi:
        x_target_up_1 = target_full_up[0] - (up_red[0] - low_red[0]) * math.sin(theta_low - math.pi / 2) - (
                    up_red[1] - low_red[1]) * math.cos(theta_low - math.pi / 2);
        y_target_up_1 = target_full_up[1] - (up_red[1] - low_red[1]) * math.sin(theta_low - math.pi / 2);
    elif theta_low < 0 and theta_low >= - math.pi / 2:
        x_target_up_1 = target_full_up[0];
        y_target_up_1 = target_full_up[1] - (up_red[0] - low_red[0]) * math.sin(-1 * theta_low);
    else:
        x_target_up_1 = target_full_up[0];
        y_target_up_1 = target_full_up[1] - (up_red[1] - low_red[1]) * math.sin(-1 * theta_low - math.pi / 2);
        x_target_up_1 = x_target_up_1 - (up_red[0] - low_red[0]) * math.sin(-1 * theta_low - math.pi / 2);
        y_target_up_1 = y_target_up_1 - (up_red[0] - low_red[0]) * math.cos(-1 * theta_low - math.pi / 2);

    if theta_low >= 0 and theta_low <= math.pi / 2:
        x_target_low_1 = target_full_low[0] + (up_red[1] - low_red[1]) * math.cos(math.pi / 2 - theta_low);
        y_target_low_1 = target_full_low[1];

    elif theta_low >= math.pi / 2 and theta_low <= math.pi:
        x_target_low_1 = target_full_low[0] + (up_red[0] - low_red[0]) * math.cos(math.pi - theta_low) + (
                    up_red[1] - low_red[1]) * math.cos(theta_low -math.pi / 2);
        y_target_low_1 = target_full_low[1] + (up_red[1] - low_red[1]) * math.sin(theta_low -math.pi / 2);

    elif theta_low < 0 and theta_low >= -math.pi / 2:
        x_target_low_1 = target_full_low[0];
        y_target_low_1 = target_full_low[1] + (up_red[0] - low_red[0]) * math.cos(math.pi / 2 + theta_low);
    
    else:
        x_target_low_1 = target_full_low[0] + (up_red[0] - low_red[0]) * math.cos(math.pi + theta_low);
        y_target_low_1 = target_full_low[1] + (up_red[0] - low_red[0]) * math.sin(math.pi + theta_low) + \
                         (up_red[1] - low_red[1]) * math.cos(math.pi + theta_low);

    curr_low_1 = [x_target_low_1 - (low_red(1)) * math.cos(theta_low_sys) + (low_red(2)) * math.sin(theta_low_sys),
                  y_target_low_1 - (low_red(1)) * math.sin(theta_low_sys) - (low_red(2)) * math.cos(theta_low_sys),
                  theta_low_sys];
    curr_up_1 = [x_target_up_1 - (up_red(1)) * math.cos(theta_low_sys) + (up_red(2)) * math.sin(theta_low_sys),
                 y_target_up_1 - (up_red(1)) * math.sin(theta_low_sys) - (up_red(2)) * math.cos(theta_low_sys), theta_low_sys];

    #####################################

    if theta_up >= 0 and theta_up <= math.pi / 2:
        x_target_up_2 = target_full_up(1) - (up_red(2) - low_red(2)) * math.cos(math.pi / 2 - theta_up);
        y_target_up_2 = target_full_up(2);
    elif theta_up >= math.pi / 2 and theta_up <= math.pi:
        x_target_up_2 = target_full_up(1) - (up_red(1) - low_red(1)) * math.sin(theta_up - math.pi / 2) - (
                    up_red(2) - low_red(2)) * math.cos(theta_up - math.pi / 2);
        y_target_up_2 = target_full_up(2) - (up_red(2) - low_red(2)) * math.sin(theta_up - math.pi / 2);
    elif theta_up < 0 and theta_up >= - math.pi / 2:
        x_target_up_2 = target_full_up(1);
        y_target_up_2 = target_full_up(2) - (up_red(1) - low_red(1)) * math.sin(-1 * theta_up);
    else:
        x_target_up_2 = target_full_up(1);
        y_target_up_2 = target_full_up(2) - (up_red(2) - low_red(2)) * math.sin(-1 * theta_up - math.pi / 2);
        x_target_up_2 = x_target_up_2 - (up_red(1) - low_red(1)) * math.sin(-1 * theta_up - math.pi / 2);
        y_target_up_2 = y_target_up_2 - (up_red(1) - low_red(1)) * math.cos(-1 * theta_up - math.pi / 2);

    if theta_up >= 0 and theta_up <= math.pi / 2:
        x_target_low_2 = target_full_low(1) + (up_red(2) - low_red(2)) * math.cos(math.pi / 2 - theta_up);
        y_target_low_2 = target_full_low(2);

    elif theta_up >= math.pi / 2 and theta_up <= math.pi:
        x_target_low_2 = target_full_low(1) + (up_red(1) - low_red(1)) * math.cos(math.pi - theta_up) + (up_red(2) - low_red(2)) * math.cos(
            theta_up - math.pi / 2);
        y_target_low_2 = target_full_low(2) + (up_red(2) - low_red(2)) * math.sin(theta_up - math.pi / 2);
        
    elif theta_up < 0 and theta_up >= - math.pi / 2:
        x_target_low_2 = target_full_low(1);
        y_target_low_2 = target_full_low(2) + (up_red(1) - low_red(1)) * math.cos(math.pi / 2 + theta_up);

    else:
        x_target_low_2 = target_full_low(1) + (up_red(1) - low_red(1)) * math.cos(math.pi + theta_up);
        y_target_low_2 = target_full_low(2) + (up_red(1) - low_red(1)) * math.sin(math.pi + theta_up) + (up_red(2) - low_red(2)) * math.cos(
            math.pi + theta_up);

    curr_low_2 = [x_target_low_2 - (low_red(1)) * math.cos(theta_up_sys) + (low_red(2)) * math.sin(theta_up_sys),
                  y_target_low_2 - (low_red(1)) * math.sin(theta_up_sys) - (low_red(2)) * math.cos(theta_up_sys), theta_up_sys];
    curr_up_2 = [x_target_up_2 - (up_red(1)) * math.cos(theta_up_sys) + (up_red(2)) * math.sin(theta_up_sys),
                 y_target_up_2 - (up_red(1)) * math.sin(theta_up_sys) - (up_red(2)) * math.cos(theta_up_sys), theta_up_sys];

    curr_low_temp = max(curr_low_1, curr_low_2);
    curr_up_temp = min(curr_up_1, curr_up_2);
    curr_low = min(curr_low_temp, curr_up_temp);
    curr_up = max(curr_low_temp, curr_up_temp);

    if check_rect_empty([curr_low, curr_up]):
        print('Result is an empty rectangle!!')
        curr_low
        curr_up
        return [];

    rect_curr = np.concatenate(rect_curr, [curr_low, curr_up], 2);
    return rect_curr;