#########################################################################################################
'''

Simple launching utility which launches all 6 tests and allows each one to handle it's abstraction
before starting each synthesis so that we do not have any context switch slowdown as all should
be single threaded by the end

Does not handle context switching in the event that the system interrupts or that the system executing
this code has fewer than 6 physical cores for each process to sufficiently run on

'''
#########################################################################################################

import random
from datetime import datetime
import sys
import os
from multiprocessing import shared_memory
import subprocess
from subprocess import Popen, PIPE
from shared_memory_dict import SharedMemoryDict
import time

try: 
    sys.argv[1]

except:
    print("You must pass desired tests indices: ")
    print("possible configurations: [[25,25,9], [30,30,9], [40,40,9], [50,50,9]]")
    print("example (run only 25, 25, 9 and 30, 30 ,9): python3 executable.py '0 1' ")
    exit(0)

#lock
wait_cond = SharedMemoryDict(name='lock', size=128)
wait_cond[-1] = 0

#sequential bool
sequential = False
sequential_addon = " -2"

#possible configurations
possible_targets = [[25,25,9], [30,30,9], [40,40,9], [50,50,9]]
target_list = []

#get desired tests
argv_split = sys.argv[1].split()

#grab targets and add them to the list
for target in argv_split:
    if int(target) != -1:
        target_list.append(possible_targets[int(target)])
    else:
        sequential = True
        sequential_addon = " -1 -2"
        print("Sequential Launch")

#target tests
targets = ["1", "2", "3", "4", "5", "6"]

last_pid = 0

for configurations in target_list:
    for target in targets:

        target = int(target) - 1

        #set file name
        file_names = str(configurations[0]) + "-" + str(configurations[1]) + "-" + str(configurations[2])

        #current folder to work with
        current_folder = ("./" + file_names + "-" + str(target + 1))

        #see if the dirs we want are present
        if not os.path.exists(current_folder):
            os.mkdir(current_folder)

        #navigate to new dir
        wd = os.getcwd()
        os.chdir(current_folder)

        #spawn test in dir
        f = open("output.txt", "w")
        p = subprocess.Popen(["python3", wd + "/main.py", str(target) + " " + str(configurations[0]) + " " + str(configurations[1]) + " " + str(configurations[2]) + sequential_addon], stdout=f)
        last_pid = p.pid

        #head back to repeat
        os.chdir(wd)

        #set flag and wait
        wait_cond[p.pid] = -1

        if not sequential:
            while wait_cond[p.pid] == -1:
                time.sleep(100)

    #wait for last process to be finished with parallel execution
    if not sequential:
        while wait_cond[int(last_pid)] == -1:
            time.sleep(100)

#allow all processes to proceed
wait_cond[-1] = 1

#wait for all processes to finish
while(sum(wait_cond.values()) < (len(targets) * len(target_list)) + 1):
    time.sleep(1000)

#exit
exit(0)