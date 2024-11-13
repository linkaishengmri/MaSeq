
import sys
sys.path.append('/home/lks/marcos-pypulseq/ocra-pulseq')
sys.path.append('/home/lks/marcos-pypulseq/marcos_client')


import experiment as ex
import numpy as np
import hw_params as hw
import dict_utils

def reset_grad():
    expt = ex.Experiment(lo_freq=5, halt_and_reset=True, grad_max_update_rate = 0.2, init_gpa=True)
    expt.add_flodict({ 
                        'grad_vx':[np.array([100, 100000]), np.array([.01,0])],
                        'grad_vy':[np.array([100, 100000]), np.array([.01,0])],
                        'grad_vz':[np.array([100, 100000]), np.array([.01,0])],
                        'grad_vz2':[np.array([100, 100000]), np.array([.01,0])],
                        'rx0_en':[np.array([100, 100000]), np.array([1,0])],
                        })
    rxd, msgs = expt.run()
    expt.close_server(only_if_sim=True)
    expt._s.close() # close socket on client
    # print(rxd, msgs)
    # dict_utils.save_dict(rxd, 'reset')
reset_grad()