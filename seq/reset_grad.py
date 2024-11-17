
import os
import sys
#*****************************************************************************
# Get the directory of the current script
main_directory = os.path.dirname(os.path.realpath(__file__))
parent_directory = os.path.dirname(main_directory)
parent_directory = os.path.dirname(parent_directory)

# Define the subdirectories you want to add to sys.path
subdirs = ['MaSeq', 'marcos_client']

# Add the subdirectories to sys.path
for subdir in subdirs:
    full_path = os.path.join(parent_directory, subdir)
    sys.path.append(full_path)
#******************************************************************************


import experiment as ex
import numpy as np

def halt():
    expt = ex.Experiment(lo_freq=5, halt_and_reset=True, grad_max_update_rate = 0.2, init_gpa=True)
    expt.add_flodict( {"tx0": (np.array([50, 130]), np.array([0.005, 0]))})
    rxd, msgs = expt.run()
    expt.close_server(only_if_sim=True)
    expt._s.close() 
if __name__ == '__main__':
    halt()