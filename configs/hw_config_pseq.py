import numpy as np
Central_freq = 10.36e6 #Hz
gradFactor = [0.4113, 0.9094,1.0000] # (X, Y, Z) in T/m/o.u.
max_rf = 50 # uT
max_grad = 120  # mT/m
max_slew_rate = 50e-3  # mT/m/ms
grad_raster_time = 30e-6  # s
grad_rise_time = 400e-6 # s, time for gradient ramps
grad_steps = 5 # steps to gradient ramps
gammaB = 42.576e6 # Hz/T, Gyromagnetic ratio
blkTime = 15 # us, blanking time of Barthel's RFPA
gradDelay = 9 # Gradient amplifier delay (us)
oversamplingFactor = 6 # Rx oversampling
maxRdPoints = 2**18 # Maximum number of points to be acquired by the red pitaya
maxOrders = 2**14 # Maximum number of orders to be processed by the red pitaya
deadTime = 400 # us, RF coil dead time
# b1Efficiency = np.pi/(0.3*70) # rads / (a.u. * us)
larmorFreq = 3.066 # MHz
cic_delay_points = 3 # to account for signal delay from red pitaya due to cic filter
addRdPoints = 10 # to account for wrong first points after decimation
adcFactor = 13.788 # mV/adcUnit

RFgatePreTime = 1000e-6 # s Preparation time of RF gate.
bash_path = "gnome-terminal" #for genome linux
rp_ip_address = "10.42.0.92"
# antenna_dict = {"RF01": np.pi/(0.3*70), "RF02": np.pi/(0.3*70)}
# fov = [20.0, 20.0, 20.0]
# dfov = [0.0, 0.0, 0.0]

rp_version = "rp-122"
lnaGain = 50 # dB
# temperature = 293 # k
shimming_factor = 1e-5


