#Headers

#multiprocessing stuff
import platform

#cpu-only
if platform.system() == 'Darwin':

    from multiprocess import Process, Queue, shared_memory, Manager
    import multiprocess
    import concurrent.futures
    import signal

else:
    
    from multiprocessing import Process, Queue, shared_memory, Manager
    import multiprocessing as multiprocess
    import concurrent.futures
    import signal

#lock
from shared_memory_dict import SharedMemoryDict
wait_cond = SharedMemoryDict(name='lock', size=128)

# Use 'multiprocessing.cpu_count()' to determine the number of available CPU cores.
cpu_count = multiprocess.cpu_count()

#make a future pool
future_pool = [None] * (cpu_count * 10)

#time we spend spinning
time_spinning = 0

#Classes
class ThreadedAbstractState:

    def __init__(self, state_id, quantized_abstract_target, u_idx,
                 abstract_obstacles, concrete_state_indices_in, obstructed_u_idx_set, manager):
        self.id = state_id # tuple of centers
        self.quantized_abstract_target = quantized_abstract_target
        self.u_idx = u_idx
        self.abstract_obstacles = abstract_obstacles
        self.concrete_state_indices = manager.list(concrete_state_indices_in)
        self.obstructed_u_idx_set = obstructed_u_idx_set

class AbstractState:

    def __init__(self, state_id, quantized_abstract_target, u_idx,
                 abstract_obstacles, concrete_state_indices, obstructed_u_idx_set):
        self.id = state_id # tuple of centers
        self.quantized_abstract_target = quantized_abstract_target
        self.u_idx = u_idx
        self.abstract_obstacles = abstract_obstacles
        self.concrete_state_indices = concrete_state_indices
        self.obstructed_u_idx_set = obstructed_u_idx_set


class RelCoordState:
    def __init__(self, concrete_state_idx, abstract_targets,
         abstract_obstacles):
        self.idx = concrete_state_idx
        self.abstract_targets = abstract_targets
        self.abstract_obstacles = abstract_obstacles

#functions
def create_symmetry_abstract_states_threaded(lock_one, lock_two, symbols_to_explore, symbol_step, targets, obstacles, sym_x, X_low, X_up,
                                    reachability_rtree_idx3d, abstract_reachable_sets, symmetry_transformed_targets_and_obstacles, concrete_to_abstract,
                                    abstract_to_concrete, symmetry_abstract_states, u_idx_to_abstract_states_indices, nearest_target_of_concrete, valid_hit_idx_of_concrete,
                                    next_abstract_state_id, threshold_num_results, Q, thread_index, manager, stolen_work, steal_send_lock, steal_receive_lock, stealQueue, sendQueue,
                                    obstacles_rects, obstacle_indices, targets_rects, target_indices, concrete_edges, neighbor_map):

    #each new execution requires new opening of the rtree files
    p = index.Property()
    p.dimension = 3
    p.dat_extension = 'data'
    p.idx_extension = 'index'
    reachability_rtree_idx3d = index.Index('3d_index_abstract',
                                           properties=p)

    get_concrete_transition_calls = 0

    #grab indices for work
    first_index = int(len(symbols_to_explore)/cpu_count) * thread_index
    second_index = (int(len(symbols_to_explore)/cpu_count) * (thread_index+1)) - 1 if thread_index+1 != cpu_count else int(len(symbols_to_explore))

    #keep track of how much work we should be processing
    work_processed = 0
    
    #check if we are stealing work or assigned work
    if (stolen_work):

        #grab work from other tasks
        waiting_at_lock = time.time()
        with steal_send_lock:

            #if we have waited 50 seconds at the lock, die
            if time.time() - waiting_at_lock > 50:
                exit(0)

            stealQueue.put(1)

            #if we spin for 50 seconds, give up
            start_spin_timer = time.time()
            while(sendQueue.empty()):
                time.sleep(1) #trade off of cpu usage vs time jump
                if (time.time() - start_spin_timer > 50):
                    exit(0)
                pass

            #grab our new indices
            assignment = sendQueue.get()
            first_index = assignment[0]
            second_index = assignment[1]


    #keep track of position
    current_index = first_index
    
    #split task
    for s in symbols_to_explore[first_index : ]:

        if work_processed == (second_index + 1) - first_index:
            break

        current_index += 1

        #see if we should allow someone to steal work
        if second_index - current_index > 10:
            with steal_receive_lock:
                if not stealQueue.empty():

                    #take message
                    stealQueue.get()

                    #determine partition
                    remaining_work = second_index - current_index 
                    held = int(remaining_work/2)
                    stolen = remaining_work - held

                    #update our work and the new task's work
                    stolen_second_index = second_index
                    second_index = current_index + held

                    #adjust first index based on rounding
                    stolen_first_index = current_index + stolen
                    if (remaining_work % 2) == 0:
                        stolen_first_index += 1

                    #print("thread: ", thread_index, "giving work: ", stolen_first_index, "-",stolen_second_index, " | keeping ", current_index, "-", second_index)

                    sendQueue.put((stolen_first_index, stolen_second_index))

        s_subscript = np.array(np.unravel_index(s, tuple((sym_x[0, :]).astype(int))))
        s_rect: np.array = np.row_stack((s_subscript * symbol_step + X_low,
                                         s_subscript * symbol_step + symbol_step + X_low))
        s_rect[0, :] = np.maximum(X_low, s_rect[0, :])
        s_rect[1, :] = np.minimum(X_up, s_rect[1, :])

        # transforming the targets and obstacles to a new coordinate system relative to the states in s.
        abstract_targets_polys = []
        abstract_targets_rects = []
        abstract_targets_polys_over_approx = []
        abstract_targets_rects_over_approx = []
        abstract_pos_targets_polys = []

        for target_idx, target_poly in enumerate(targets):
            abstract_target_poly = transform_poly_to_abstract_frames(target_poly, s_rect, over_approximate=False)
            abstract_pos_target_poly = transform_poly_to_abstract_frames(target_poly, s_rect, over_approximate=False, project_to_pos=True)
            abstract_target_poly_over_approx = transform_poly_to_abstract_frames(
                target_poly, s_rect, over_approximate=True)  # project_to_pos=True
            if not pc.is_empty(abstract_target_poly):
                rc, x1 = pc.cheby_ball(abstract_target_poly)
                abstract_target_rect = np.array([x1 - rc, x1 + rc])
            elif not pc.is_empty(abstract_pos_target_poly):
                # pdb.set_trace()
                raise "abstract target is empty for a concrete state"

            else:
                # pdb.set_trace()
                print("empty abstract_target_poly: ", abstract_target_poly)
                raise "empty abstract_target_poly error, grid must be refined, it's too far to see the position of " \
                      "the target similarly even within the same grid cell! "
            abstract_target_rect_over_approx = np.column_stack(pc.bounding_box(abstract_target_poly_over_approx)).T
            abstract_targets_rects.append(abstract_target_rect)
            abstract_targets_polys.append(abstract_target_poly)
            abstract_targets_rects_over_approx.append(abstract_target_rect_over_approx)
            abstract_targets_polys_over_approx.append(abstract_target_poly_over_approx)
            abstract_pos_targets_polys.append(abstract_pos_target_poly)

        if len(abstract_targets_polys) == 0:
            # pdb.set_trace()
            raise "Abstract target is empty"

        abstract_obstacles = pc.Region(list_poly=[])
        for obstacle_poly in obstacles:
            abstract_obstacle = transform_poly_to_abstract_frames(obstacle_poly, s_rect,
                                                                  over_approximate=True)  # project_to_pos=True
            abstract_obstacles = get_poly_union(abstract_obstacles, abstract_obstacle)

        with lock_one:
            symmetry_transformed_targets_and_obstacles[s] = RelCoordState(s, abstract_targets_polys,
                                                                        abstract_obstacles)
                

        min_dist = np.inf
        for curr_target_idx, curr_target_poly in enumerate(abstract_targets_polys):

            # CHANGE: compute the closest point of the target polytope to the origin
            curr_nearest_point, curr_dist = nearest_point_to_the_origin(curr_target_poly)
            if min_dist > curr_dist:
                nearest_point = curr_nearest_point
                min_dist = curr_dist

        with lock_one:
            nearest_target_of_concrete[s] = nearest_point
        
        curr_num_results = 3
        is_obstructed_u_idx = {}
        added_to_existing_state = False
        while curr_num_results < threshold_num_results:

            if strategy_3 or strategy_6 or curr_num_results == len(abstract_reachable_sets):
                hits = list(range(len(abstract_reachable_sets)))
            else:
                rtree_hits = list(reachability_rtree_idx3d.nearest(
                    (nearest_point[0], nearest_point[1], nearest_point[2],
                    nearest_point[0]+0.001, nearest_point[1]+0.001, nearest_point[2]+0.001),
                    num_results=curr_num_results, objects=True))
                hits = [hit.object for hit in rtree_hits]
                #bbox_hits = [hit.bbox for hit in rtree_hits]
        
            if len(hits):
                for idx, hit_object in enumerate(hits):
                    if not hit_object in is_obstructed_u_idx:
                        next_concrete_state_indices, _ = get_concrete_transition(s, hit_object, concrete_edges, neighbor_map,
                                                                sym_x, symbol_step, abstract_reachable_sets,
                                                                obstacles_rects, obstacle_indices, targets_rects,
                                                                target_indices, X_low, X_up, benchmark)
                        get_concrete_transition_calls += 1

                        is_obstructed_u_idx[hit_object] = (next_concrete_state_indices == [-2])
                        
                    if not is_obstructed_u_idx[hit_object]:
                        with lock_one:
                            if not hit_object in u_idx_to_abstract_states_indices:
                                rect = get_bounding_box(abstract_reachable_sets[hit_object][-1])
                                new_abstract_state = ThreadedAbstractState(next_abstract_state_id['next_abstract_state_id'],
                                                        np.average(rect, axis=0),
                                                        hit_object,
                                                        copy.deepcopy(symmetry_transformed_targets_and_obstacles[s].abstract_obstacles),
                                                        [s],
                                                        set([k for k, v in is_obstructed_u_idx.items() if v == True]), 
                                                        manager)
                                symmetry_abstract_states.append(new_abstract_state)
                                concrete_to_abstract[s] = next_abstract_state_id['next_abstract_state_id']
                                u_idx_to_abstract_states_indices[hit_object] = manager.list([next_abstract_state_id['next_abstract_state_id']])
                                abstract_to_concrete[next_abstract_state_id['next_abstract_state_id']] = manager.list([s])
                                next_abstract_state_id['next_abstract_state_id'] += 1
                                added_to_existing_state = True
                                valid_hit_idx_of_concrete[s] = idx
                                break
                            else:
                                if len(u_idx_to_abstract_states_indices[hit_object]):
                                    add_concrete_state_to_symmetry_abstract_state(s, u_idx_to_abstract_states_indices[hit_object][0],
                                        symmetry_transformed_targets_and_obstacles[s].abstract_obstacles, symmetry_abstract_states,
                                        concrete_to_abstract, abstract_to_concrete, is_obstructed_u_idx)
                                    added_to_existing_state = True
                                    valid_hit_idx_of_concrete[s] = idx
                                    break
                                else:
                                    raise "No abstract states for u_idx when one was expected"
                if added_to_existing_state:
                    break
            else:
                raise "No hits but rtree's nearest should always return a result"
            if added_to_existing_state:
                break
            else:
                if curr_num_results == threshold_num_results - 1:
                    break
                else:
                    curr_num_results = min(5 * curr_num_results, threshold_num_results - 1)
        if not added_to_existing_state:

            with lock_one:
                add_concrete_state_to_symmetry_abstract_state(s, 0, pc.Region(list_poly=[]),
                    symmetry_abstract_states, concrete_to_abstract, abstract_to_concrete, {})

                valid_hit_idx_of_concrete[s] = len(abstract_reachable_sets)

        work_processed += 1
        
    Q.put([symmetry_transformed_targets_and_obstacles, work_processed, concrete_edges, get_concrete_transition_calls])
    exit(0)

def create_symmetry_abstract_states(symbols_to_explore, symbol_step, targets, targets_rects, target_indices, obstacles,  obstacles_rects, obstacle_indices,
                                    sym_x, X_low, X_up, reachability_rtree_idx3d, abstract_reachable_sets):
    t_start = time.time()
    print('\n%s\tStart of the symmetry abstraction \n', time.time() - t_start)

    #make manaing object
    manager = Manager()

    #make all managed dictionaries
    symmetry_transformed_targets_and_obstacles = {}
    concrete_to_abstract = manager.dict()
    abstract_to_concrete = manager.dict()
    symmetry_abstract_states = manager.list()
    u_idx_to_abstract_states_indices = manager.dict()
    nearest_target_of_concrete = manager.dict()
    valid_hit_idx_of_concrete = manager.dict()
    next_abstract_state_id = manager.dict()

    #we now pickel the edges dict
    concrete_edges = {}
    neighbor_map = {}

    obstacle_state = ThreadedAbstractState(0, None, None, [], [], set(), manager)
    symmetry_abstract_states.append(obstacle_state)
    abstract_to_concrete[0] = manager.list()
    get_concrete_transition_calls = 0

    next_abstract_state_id['next_abstract_state_id'] = 1

    if strategy_5 or strategy_2:
        threshold_num_results = 376
    elif strategy_1 or strategy_4:
        threshold_num_results = len(abstract_reachable_sets) + 1
    else:
        threshold_num_results = 4

    #process locks (incase I need them)
    lock_one = multiprocess.Lock()
    lock_two = multiprocess.Lock()
    steal_receive_lock = multiprocess.Lock()
    steal_send_lock = multiprocess.Lock()

    #close file
    reachability_rtree_idx3d.close()

    #queue for communication
    Q = Queue()
    sendQueue = Queue()
    stealQueue = Queue()

    #spawn up threadpool and submit tasks
    max_assignment = len(symbols_to_explore)
    process_count = cpu_count

    #only assign as many threads as we have work for
    if max_assignment < cpu_count:
        process_count = max_assignment

    #create our pool
    for i in range(process_count):
        future_pool[i] = Process(target=create_symmetry_abstract_states_threaded, args=(lock_one, lock_two, list(symbols_to_explore), symbol_step, targets, obstacles, sym_x, X_low, X_up,
                                    reachability_rtree_idx3d, abstract_reachable_sets, symmetry_transformed_targets_and_obstacles, concrete_to_abstract,
                                    abstract_to_concrete, symmetry_abstract_states, u_idx_to_abstract_states_indices, nearest_target_of_concrete, valid_hit_idx_of_concrete,
                                    next_abstract_state_id, threshold_num_results, Q, i, manager, False, steal_send_lock, steal_receive_lock, stealQueue, sendQueue,
                                    obstacles_rects, obstacle_indices, targets_rects, target_indices, concrete_edges, neighbor_map))
    #start them
    for i in range(process_count):
        future_pool[i].start()
    
    #get results from each process
    counter_threads = 0
    current_returns = 0
    current_thread_index_counter = process_count

    while current_returns != len(symbols_to_explore):
        print("Awaiting Processes: " + str(int((current_returns/len(symbols_to_explore))*100)) + "%", end="\r")  

        if (int((current_returns/len(symbols_to_explore))*100) == 100):
            print(current_returns)

        result = Q.get()
        current_returns += result[1]
        counter_threads += 1
        symmetry_transformed_targets_and_obstacles.update(result[0])
        concrete_edges.update(result[2])
        get_concrete_transition_calls += result[3]

        
        #spawn new thread again
        future_pool[current_thread_index_counter] = Process(target=create_symmetry_abstract_states_threaded, args=(lock_one, lock_two, list(symbols_to_explore), symbol_step, targets, obstacles, sym_x, X_low, X_up,
                                    reachability_rtree_idx3d, abstract_reachable_sets, symmetry_transformed_targets_and_obstacles, concrete_to_abstract,
                                    abstract_to_concrete, symmetry_abstract_states, u_idx_to_abstract_states_indices, nearest_target_of_concrete, valid_hit_idx_of_concrete,
                                    next_abstract_state_id, threshold_num_results, Q, current_thread_index_counter, manager, True, steal_send_lock, steal_receive_lock, stealQueue, sendQueue,
                                    obstacles_rects, obstacle_indices, targets_rects, target_indices, concrete_edges, neighbor_map))

        future_pool[current_thread_index_counter].start()

        current_thread_index_counter += 1

        
    #kill any waiting theif processes
    for i in future_pool:
        if i != None:
            try:
                os.kill(i.pid, signal.SIGTERM)
            except OSError:
                pass

    print("I counted: ", current_returns, " states returned out of ", len(symbols_to_explore), " symbols to explore")
    print(['Done creation of symmetry abstract states in: ', time.time() - t_start, ' seconds'])
    print("concrete_to_abstract: ", len(concrete_to_abstract))
    print("abstract_to_concrete: ", len(abstract_to_concrete))
    print("concrete states deemed 'obstacle': ", len(symmetry_abstract_states[0].concrete_state_indices))
    print("symmetry abstract states found: ", len(symmetry_abstract_states))

    #convert the ThreadedAbstractStates to AbstractStates
    symmetry_abstract_states_single = dict()
    for idx in range(len(symmetry_abstract_states)):

        #make new key
        symmetry_abstract_states_single[idx] =  AbstractState(symmetry_abstract_states[idx].id,
                                                symmetry_abstract_states[idx].quantized_abstract_target,
                                                symmetry_abstract_states[idx].u_idx,
                                                symmetry_abstract_states[idx].abstract_obstacles[:],
                                                symmetry_abstract_states[idx].concrete_state_indices[:],
                                                set(list(symmetry_abstract_states[idx].obstructed_u_idx_set)[:])) # copy to list to force shallow copy
                                                
    #overwrite
    symmetry_abstract_states = symmetry_abstract_states_single

    #grab all values from abstract_to_concrete
    abstract_to_concrete_single = dict()
    for key, value in abstract_to_concrete.items():
        hold_array = []
        for i in abstract_to_concrete[key]:
            hold_array.append(i)
        abstract_to_concrete_single[key] = copy.deepcopy(hold_array)

    #overwrite
    abstract_to_concrete = copy.deepcopy(abstract_to_concrete_single)

    return symmetry_transformed_targets_and_obstacles, concrete_to_abstract, abstract_to_concrete, symmetry_abstract_states, nearest_target_of_concrete, valid_hit_idx_of_concrete, concrete_edges, neighbor_map, get_concrete_transition_calls

