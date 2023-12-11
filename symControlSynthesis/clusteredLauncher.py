#########################################################################################################
'''

Simple launching utility which launches all 6 tests and allows each one to handle it's abstraction
before starting each synthesis so that we do not have any context switch slowdown as all should
be single threaded by the end

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

#name of folders to make and work from
file_names = sys.argv[1]

#lock
wait_cond = SharedMemoryDict(name='lock', size=128)
wait_cond[-1] = 0

#get target tests
targets = sys.argv[2:]

for i in targets:

    #current folder to work with
    current_folder = ("./" + file_names + "-" + str(i))

    #see if the dirs we want are present
    if not os.path.exists(current_folder):
        os.mkdir(current_folder)

    #navigate to new dir
    wd = os.getcwd()
    os.chdir(current_folder)

    #spawn test in dir
    f = open("output.txt", "w")
    p = subprocess.Popen(["python3", wd + "/main.py", str(i)], stdout=f)

    #head back to repeat
    os.chdir(wd)

    #set flag and wait
    wait_cond[i] = -1
    while wait_cond[i] == -1:
        pass

#wait for last process to be finished with parallel execution
while wait_cond[5] == -1:
    pass

#allow all processes to proceed
wait_cond[-1] = 1

#wait for all processes to finish
while(sum(wait_cond.values()) < len(targets) + 1):
    pass

#exit
exit(0)