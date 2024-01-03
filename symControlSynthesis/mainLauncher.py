import random
from datetime import datetime
import sys
import os
from multiprocessing import shared_memory
import subprocess
from subprocess import Popen, PIPE
from shared_memory_dict import SharedMemoryDict
import time

possible_targets = [[25,25,9], [30,30,9], [40,40,9], [50,50,9]]

#determine if clustered or not 
print("++++++++++++++++++++++++++++++++++++++++++++++++++++++++++")
print("Would you like to run these tests clustered? (Y/N)\n")
print("This means that any selected tests will run strategies")
print("1, 2, 3, 4, 5, 6 before finishing.\n")
print("To run specfic strategies only, enter 'N'")
print("++++++++++++++++++++++++++++++++++++++++++++++++++++++++++\n")

tests = input()

#launch the clustering launcher
if tests.lower() == "y":

    #get desired tests to run from user
    print("++++++++++++++++++++++++++++++++++++++++++++++++++++++++++")
    print("\nTo begin running tests, please select the numbers of the ")
    print("tests you would like to run separated by spaces\n")
    print("Available Tests:")
    print("1. [25, 25, 9]")
    print("2. [30, 30, 9]")
    print("3. [40, 40, 9]")
    print("4. [50, 50, 9]")
    print("++++++++++++++++++++++++++++++++++++++++++++++++++++++++++\n")

    tests = input()
    tests = tests.split()

    #possible configurations
    target_list = []
    target_strategies = []
    test_list = ""
    parallel = False

    #grab targets and add them to the list
    for test in tests:
        try: 
            target_list.append(possible_targets[int(test) - 1])
            test_list += str(int(test) - 1) + " "
        except:
            print("Bad selection entered... Exiting.")
            exit(-1)

    print("++++++++++++++++++++++++++++++++++++++++++++++++++++++++++")
    print("Run the abstractions with multicore enabled? (Y/N)")
    print("++++++++++++++++++++++++++++++++++++++++++++++++++++++++++\n")
    tests = input()

    if tests.lower() == "y":
        test_list += ("-1")

    print("Running tests......")
    print("Terminal will exit when tests finish....")

    wd = os.getcwd()
    p = subprocess.Popen(["python3", wd + "/clusteredLauncher.py", test_list])
    p.wait()
    exit(0)

#get desired tests to run from user
print("++++++++++++++++++++++++++++++++++++++++++++++++++++++++++")
print("To begin running tests, please select the number ")
print("of the test you would like to run\n")
print("Available Tests:")
print("1. [25, 25, 9]")
print("2. [30, 30, 9]")
print("3. [40, 40, 9]")
print("4. [50, 50, 9]")
print("++++++++++++++++++++++++++++++++++++++++++++++++++++++++++\n")

while(True):
    test_to_run = input()
    try: 
        if int(test_to_run) < 1 or int(test_to_run) > 4:
            print("Please enter a number from the selection above:")
        else:
            break
    except:
        pass

#else, determine which tests we should run
print("++++++++++++++++++++++++++++++++++++++++++++++++++++++++++")
print("Which strategy would you like to run?")
print("Please enter the number of the strategy\n")
print("1. polls - all")
print("2. polls - 400")
print("3. polls + no closest")
print("4. polls -full + neighbors")
print("5. polls -400 + neighbors")
print("6. polls + no closest + neighbors")
print("++++++++++++++++++++++++++++++++++++++++++++++++++++++++++\n")

while(True):
    strategy_to_run = input()
    try: 
        if int(strategy_to_run) < 1 or int(strategy_to_run) > 6:
            print("Please enter a number from the selection above:")
        else:
            break
    except:
        pass

configurations = possible_targets[int(test_to_run) -1]

print("++++++++++++++++++++++++++++++++++++++++++++++++++++++++++")
print("Run the abstraction with multicore enabled? (Y/N)")
print("++++++++++++++++++++++++++++++++++++++++++++++++++++++++++\n")
tests = input()

if tests.lower() == "y":
    parallel = True

print("++++++++++++++++++++++++++++++++++++++++++++++++++++++++++")
print("Enter a folder name to store the results in:")
print("++++++++++++++++++++++++++++++++++++++++++++++++++++++++++\n")
tests = input()

#start the test
#set file name
file_names = tests

#current folder to work with
current_folder = "./" + file_names

#see if the dirs we want are present
if not os.path.exists(current_folder):
    os.mkdir(current_folder)

#navigate to new dir
wd = os.getcwd()
os.chdir(current_folder)

#build command line 
if (parallel):
    command_line = str(int(strategy_to_run)-1) + " " + str(configurations[0]) + " " + str(configurations[1]) + " " + str(configurations[2])
else:
    command_line = str(int(strategy_to_run)-1) + " " + str(configurations[0]) + " " + str(configurations[1]) + " " + str(configurations[2]) + " -1"

print("Running tests......")
print("Terminal will exit when tests finish....")

#spawn test in dir
f = open("output.txt", "w")
p = subprocess.Popen(["python3", wd + "/main.py", command_line], stdout=f)
p.wait()
exit(0)
