import numpy as np
from Reach_avoid_synthesis_sets import transform_to_frames, find_frame, build_unified_abstraction


def unifying_reachable_set_test(Symbolic_abstraction, ):

nearest_rect = [[7.,    0.,    1.0472], [10.,  6.5, 2.0944]];



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
                    transform_to_frames(Symbolic_reduced[s_ind, u_ind, np.arange(n), t_ind],
                                        Symbolic_reduced[s_ind, u_ind, n + np.arange(n), t_ind],
                                        rects_curr[rect_ind, 0, :],
                                        rects_curr[rect_ind, 1, :]));