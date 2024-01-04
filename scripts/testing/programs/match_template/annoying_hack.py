import os

# A simple, if not annoying way to make the utils available to all the test scripts without using a proper package
# check a few steps up the relative path
max_tries = 5
n_try = 0
cistem_programs_path = os.path.abspath(os.path.join(__file__, '..'))

from  sys import path as sys_path

while n_try < max_tries:
    if os.path.basename(cistem_programs_path) == 'cistem_programs':
        break
    else:
        n_try += 1
        cistem_programs_path = os.path.abspath(os.path.join(cistem_programs_path, '..'))

if cistem_programs_path not in sys_path:
    sys_path.append(cistem_programs_path)
