import matlab.engine
import numpy as np
import time
import math
import os
import sys
import pwd
import platform
import matplotlib

if platform.system() == 'Darwin':
    matplotlib.use("macOSX")

import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, Polygon

from new_abstraction_based_synthesis import abstract_synthesis

argv_split = sys.argv[1].split()
test_to_run = int(argv_split[0])
first_val = int(argv_split[1])
second_val = int(argv_split[2])
third_val = int(argv_split[3])

bypass = True
parallel = True

#determine if we are running sequentially or without clustering
if len(argv_split) > 4:
    if int(argv_split[-1]) == -1 or int(argv_split[-2]) == -1:
        parallel = False
    if int(argv_split[-1]) == -2 or int(argv_split[-2]) == -2:
        bypass = False

#avoid block if parallel non-clustered
if bypass:

    #set the wait condition to nothing so we can still run unclustered
    from shared_memory_dict import SharedMemoryDict
    wait_cond = SharedMemoryDict(name='lock', size=128)
    wait_cond[-1] = 1

if __name__ == '__main__':

    username = pwd.getpwuid(os.getuid())[0]
    OA_method = 3.1  

    X_up = np.array([10, 6.5, 2 * math.pi])
    X_low = np.array([0, 0, 0])
    n_x = len(X_up)

    U_up = np.array([0.18, 0.05, 0.1])
    U_low = np.array([-0.18, -0.05, -0.1])
    n_u = len(U_up)

    W_up = np.array([0.01, 0.01, 0.01])  
    W_low = -W_up

    Target_up = np.array([[10, 6.5, 2 * math.pi / 3]])
    Target_low = np.array([[7, 0, math.pi / 3]]) 

    Obstacle_up = np.array([[2.5, 3, 100], [5.5, 9.5, 100], [0, 9.5, 100], [13, 0, 100],
                            [13, 9.5, 100], [13, 9.5, 100]])
    Obstacle_low = np.array([[2, -3, -100], [5, 3.5, -100], [-3, -3, -100], [-3, -3, -100],
                            [-3, 6.5, -100], [10, -3, -100]])

    #check to see if we are running a special case
    if first_val == second_val:
        sym_x = first_val * np.ones((1, n_x))
        sym_x[0, 2] = second_val
        sym_u = third_val * np.ones((1, n_u))

    else:
        sym_x = np.ones((1, n_x))
        sym_x[0, 0] = first_val
        sym_x[0, 1] = second_val
        sym_x[0, 2] = third_val
        sym_u = 9 * np.ones((1, n_u))


    time_step = np.linspace(0, 3, 3).reshape((1, 3))

    reachability_abstraction_level = 2
    if reachability_abstraction_level == 2:
        state_dimensions = np.zeros((1, n_x))

    X_up = X_up + 3 
    X_low = X_low - 3 
    X_low[2] = 0
    X_up[2] = 2 * math.pi

    # Shrink target set
    Target_up = Target_up 
    Target_low = Target_low 

    symbol_step = (X_up - X_low) / sym_x

    # Abstraction creation
    t_abstraction = time.time()
    eng = matlab.engine.start_matlab()

    #import additional matlab files
    if platform.system() == 'Darwin':

        eng.addpath(r'/Users/' + username + r'/Downloads/IFAC20_ship_matlab')
        eng.addpath(
            r'/Users/' + username + r'/Downloads/IFAC20_ship_matlab/TIRA:/Users/' + username + r'/Downloads/IFAC20_ship_matlab/TIRA/Input_files'
            r':/Users/' + username + r'/Downloads/IFAC20_ship_matlab/TIRA/OA_methods:/Users/' + username + r'/Downloads/IFAC20_ship_matlab/TIRA'
            r'/SDMM_hybrid:/Users/' + username + r'/Downloads/IFAC20_ship_matlab/TIRA/Utilities:')

        print("Looking for IFAC20_ship_matlab in directory:")
        print(r'/Users/' + username + r'/Downloads')
    
    else:

        eng.addpath(r'/home/' + username + r'/Downloads/IFAC20_ship_matlab')
        eng.addpath(
            r'/home/' + username + r'/Downloads/IFAC20_ship_matlab/TIRA:/home/' + username + r'/Downloads/IFAC20_ship_matlab/TIRA/Input_files'
            r':/home/' + username + r'/Downloads/IFAC20_ship_matlab/TIRA/OA_methods:/home/' + username + r'/Downloads/IFAC20_ship_matlab/TIRA'
            r'/SDMM_hybrid:/home/' + username + r'/Downloads/IFAC20_ship_matlab/TIRA/Utilities:')

        print("Looking for IFAC20_ship_matlab in directory:")
        print(r'/home/' + username + r'/Downloads')

    X_low_matlab = matlab.double(X_low.tolist())
    X_up_matlab = matlab.double(X_up.tolist())
    sym_x_matlab = matlab.double(sym_x.tolist())
    U_low_matlab = matlab.double(U_low.tolist())
    U_up_matlab = matlab.double(U_up.tolist())
    sym_u_matlab = matlab.double(sym_u.tolist())
    W_low_matlab = matlab.double(W_low.tolist())
    W_up_matlab = matlab.double(W_up.tolist())
    state_dimensions_matlab = matlab.double(state_dimensions.tolist())
    time_step_matlab = matlab.double(time_step.tolist())
    OA_method_matlab = OA_method
    Symbolic_reduced, U_discrete, unsafe_trans = eng.Centralized_abstraction(X_low_matlab, X_up_matlab, sym_x_matlab,
                                                                            U_low_matlab, U_up_matlab,
                                                                            sym_u_matlab, W_low_matlab, W_up_matlab,
                                                                            time_step_matlab, state_dimensions_matlab,
                                                                            OA_method, nargout=3)

    Symbolic_reduced = np.array(Symbolic_reduced)

    U_discrete = np.array(U_discrete)
    print("Symbolic_reduced: ", Symbolic_reduced)
    print("U_discrete: ", U_discrete)
    print("U_discrete shape: ", U_discrete.shape)
    print("unsafe_trans: ", unsafe_trans)

    plt.figure("Reduced coordinates")
    color = 'orange'
    edge_color = 'k'
    currentAxis_1 = plt.gca()
    n = state_dimensions.shape[1]
    for s_ind in range(Symbolic_reduced.shape[0]):
        for u_ind in range(Symbolic_reduced.shape[1]):
            for idx in range(Symbolic_reduced.shape[3]):
                abstract_rect_low = Symbolic_reduced[s_ind, u_ind, np.arange(n), idx]
                abstract_rect_up = Symbolic_reduced[s_ind, u_ind, n + np.arange(n), idx]
                rect_patch = Rectangle(abstract_rect_low[[0, 1]], abstract_rect_up[0] - abstract_rect_low[0],
                                    abstract_rect_up[1] - abstract_rect_low[1], linewidth=1,
                                    edgecolor=edge_color, facecolor=color)
                currentAxis_1.add_patch(rect_patch)
    plt.ylim([-0.35, 0.35])
    plt.xlim([-0.75, 0.75])
    plt.savefig("Abstract reachable sets")

    t_abstraction = time.time() - t_abstraction

    U_discrete = np.array(U_discrete)

    N = 30
    M = 5
    abstract_synthesis(U_discrete, time_step, W_low, W_up,
                    Symbolic_reduced, sym_x, sym_u, state_dimensions,
                    Target_low, Target_up, Obstacle_low, Obstacle_up, X_low, X_up, eng, int(test_to_run) + 1, parallel)