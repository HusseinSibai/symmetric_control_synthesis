import matlab.engine
import numpy as np
import time
import math
from plot import plot

import matplotlib

matplotlib.use("macOSX")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, Polygon

from Reach_avoid_synthesis_sets import reach_avoid_synthesis_sets

## Initialization of the problem

# Reachability method(3 = CTMM, 4 = SDMM with sampling, 5=SDMM with IA)
OA_method = 3.1;  # Hussein: this is 3.1 instead of 3 since we are computing the full reachable set instead of the last one.

# State space definition N / E for camera range: 10 * 6.5 m
# orientation range[-pi, pi]; psi = 0 is pointing North
X_up = np.array([10, 6.5, math.pi]);
X_low = np.array([0, 0, -math.pi]);
n_x = len(X_up);

# Input space surge velocity: [-0.18, 0.18] m / s
# sway velocity: [-0.1, 0.1] m / s
# yaw rate: [-0.1, 0.1] rad / s
# U_up = [0.18; 0.1; 0.1];
# U_low = [-0.18; -0.1; -0.1];
U_up = np.array([0.18, 0.05, 0.1]);
U_low = np.array([-0.18, -0.05, -0.1]);
n_u = len(U_up);

# Disturbance space: current velocity
W_up = np.array([0.01, 0.01, 0.01]);  # assume low currents around the docks
W_low = -W_up;

# Target interval for the reachability problem
# go North towards a West - East docking band(N in [6, 10], E in [0, 6.5])
# with the ship facing East (+ / -30° around 90° = >[pi / 3, 2 * pi / 3])
# Target_up = [10; 6.5; 2 * pi / 3];
# Target_low = [6; 0; pi / 3];
# Target_up = np.array([[10, 6.5, 2 * math.pi / 3]]);
# Target_low = np.array([[7, 0, math.pi / 3]]);
Target_up = np.array([[10, 2, 2 * math.pi / 3], [10, 4, 2 * math.pi / 3], [10, 6.5, 2 * math.pi / 3]]);
Target_low = np.array([[7, 0, math.pi / 3], [7, 2, math.pi / 3], [7, 4, math.pi / 3]]);

# Obstacles
# Obstacle_up = [[5; 2; pi] [3; 6; pi]];
# Obstacle_low = [[3; 0; -pi] [0; 5; -pi]];
# Hussein: change the angles bounds from -pi, pi to - 100, 100, respectively
# Obstacle_up = [[[2.5], [5.5]], [[3],[6.5]], [[math.pi], [math.pi]]];
# Obstacle_low = [[[2], [5]], [[0],[3.5]], [[-math.pi], [-math.pi]]];
Obstacle_up = np.array([[2.5, 3, 100], [5.5, 6.5, 100]]);
Obstacle_low = np.array([[2, 0, -100], [5, 3.5, -100]]);

sym_x = 30 * np.ones((1, n_x));
sym_x[0, 2] = 10;

sym_u = 3 * np.ones((1, n_u));

time_step = np.linspace(0, 1, 5).reshape((1,5));

state_dimensions = np.zeros((1, n_x));

## Update specifications with error bounds

# Shrink state space / safety
X_up = X_up;  # - error_6D(1:n_x);
X_low = X_low;  # + error_6D(1:n_x);

# Shrink target set
Target_up = Target_up;  # - error_6D(1:n_x);
Target_low = Target_low;  # + error_6D(1:n_x);

# Moved this from being the first step in Control synthesis to here
# Convert target interval to the largest set of symbols it fully contains
# symbol_step = (X_up - X_low) / sym_x;

# Abstraction creation
t_abstraction = time.time();
eng = matlab.engine.start_matlab();
eng.addpath(r'/Users/hsibai/Downloads/IFAC20_ship_matlab');
eng.addpath(
    r'/Users/hsibai/Downloads/IFAC20_ship_matlab/TIRA:/Users/hsibai/Downloads/IFAC20_ship_matlab/TIRA/Input_files'
    r':/Users/hsibai/Downloads/IFAC20_ship_matlab/TIRA/OA_methods:/Users/hsibai/Downloads/IFAC20_ship_matlab/TIRA'
    r'/SDMM_hybrid:/Users/hsibai/Downloads/IFAC20_ship_matlab/TIRA/Utilities:');

X_low_matlab = matlab.double(X_low.tolist());
X_up_matlab = matlab.double(X_up.tolist());
sym_x_matlab = matlab.double(sym_x.tolist());
U_low_matlab = matlab.double(U_low.tolist());
U_up_matlab = matlab.double(U_up.tolist());
sym_u_matlab = matlab.double(sym_u.tolist());
W_low_matlab = matlab.double(W_low.tolist());
W_up_matlab = matlab.double(W_up.tolist());
state_dimensions_matlab = matlab.double(state_dimensions.tolist());
time_step_matlab = matlab.double(time_step.tolist());  # matlab.double(time_step);
OA_method_matlab = OA_method;
Symbolic_reduced, U_discrete, unsafe_trans = eng.Centralized_abstraction(X_low_matlab, X_up_matlab, sym_x_matlab,
                                                                         U_low_matlab, U_up_matlab,
                                                                         sym_u_matlab, W_low_matlab, W_up_matlab,
                                                                         time_step_matlab, state_dimensions_matlab,
                                                                         OA_method, nargout=3);

Symbolic_reduced = np.array(Symbolic_reduced);
U_discrete = np.array(U_discrete);
print("Symbolic_reduced: ", Symbolic_reduced)
print("U_discrete: ", U_discrete)
print("unsafe_trans: ", unsafe_trans)
# X_low = np.transpose(X_low);
# X_up = np.transpose(X_up);
# for i in range(Obstacle_up.shape[0]):
#    Obstacle_up[i, :] = np.minimum(Obstacle_up[i, :], X_up);  # + error_6D[1:n_x]
#    Obstacle_low[i, :] = np.maximum(Obstacle_low[i, :], X_low);  # - error_6D[1:n_x]

'''
Symbolic_reduced_list = [];
Symbolic_reduced = np.array(Symbolic_reduced);
for s_ind in range(Symbolic_reduced.shape[0]):
    for u_ind in range(Symbolic_reduced.shape[1]):
        Symbolic_reduced_list.append([Symbolic_reduced[s_ind, u_ind, np.arange(n_x)], Symbolic_reduced[s_ind, u_ind,
                                                                                                           n_x + np.arange(
                                                                                                               n_x)]]);
Symbolic_reduced_list = np.array(Symbolic_reduced_list)
print(Symbolic_reduced_list)
'''
plt.figure("Reduced coordinates")
color = 'orange'
edge_color = 'k'
currentAxis_1 = plt.gca()
n = state_dimensions.shape[1];
for s_ind in range(Symbolic_reduced.shape[0]):
    for u_ind in range(Symbolic_reduced.shape[1]):
        for idx in range(Symbolic_reduced.shape[3]):
            abstract_rect_low = Symbolic_reduced[s_ind, u_ind, np.arange(n), idx];
            abstract_rect_up = Symbolic_reduced[s_ind, u_ind, n + np.arange(n), idx];
            # abstract_rect_low = Symbolic_reduced[s_ind, u_ind, np.arange(n)];
            # abstract_rect_up = Symbolic_reduced[s_ind, u_ind, n + np.arange(n)];
            rect_patch = Rectangle(abstract_rect_low[[0, 1]], abstract_rect_up[0] - abstract_rect_low[0],
                                   abstract_rect_up[1] - abstract_rect_low[1], linewidth=1,
                                   edgecolor=edge_color, facecolor=color)
            currentAxis_1.add_patch(rect_patch)
plt.ylim([-0.5, 0.5])
plt.xlim([-1, 1])
plt.show()
# plot(Symbolic_reduced_list);
# print(np.array(mat_out).shape)
# print("Symbolic_reduced,U_discrete,unsafe_trans")
# print(np.array(Symbolic_reduced).shape,np.array(U_discrete).shape,np.array(unsafe_trans).shape)
# print(Symbolic_reduced,U_discrete,unsafe_trans)
# print("matout", mat_out)
t_abstraction = time.time() - t_abstraction;

U_discrete = np.array(U_discrete);

# Synthesize a controller

Controller = reach_avoid_synthesis_sets(Symbolic_reduced, sym_x, sym_u, state_dimensions, Target_low,
                                         Target_up, Obstacle_low, Obstacle_up, X_low, X_up, U_low, U_up);
