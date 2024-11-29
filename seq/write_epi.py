
import importlib
import os
import math
import numpy as np
import matplotlib.pyplot as plt

import pypulseq as pp

# from utils import animate, simulate_2d, sort_data_implicit, ifft_2d, combine_coils, plot_nd



plot = True
write_seq = True

# ======
# SETUP
# ======
import numpy as np
import matplotlib.pyplot as plt
import pypulseq as pp

# Set system limits
sys = pp.Opts(max_grad=28, grad_unit='mT/m', 
              max_slew=150, slew_unit='T/m/s', 
              rf_ringdown_time=20e-6, rf_dead_time=100e-6, 
              adc_dead_time=10e-6)

# Basic parameters
seq = pp.Sequence(sys)  # Create a new sequence object
fov = 256e-3
Nx = 32
Ny = Nx  # Same resolution in both directions
alpha = 90  # Flip angle in degrees
sliceThickness = 3e-3  # Slice thickness in meters
TR = 130e-3  # Repetition time (TR)
TE = 61e-3  # Echo time (TE)
nSeg = int(Ny/2)  # Number of segments equals number of phase encoding lines

# More in-depth parameters
rfSpoilingInc = 117  # RF spoiling increment
rfDuration = 3e-3  # RF pulse duration
roDuration = 640e-6  # Readout duration

# Create slice-selective pulse and gradients
rf, gz, gzReph = pp.make_sinc_pulse(
    flip_angle=alpha*np.pi/180, 
    duration=rfDuration,
    slice_thickness=sliceThickness, 
    apodization=0.42,
    time_bw_product=4, 
    system=sys,
    return_gz=True,
)

# Define other gradients and ADC events
deltak = 1 / fov  # k-space step in inverse meters
gxp = pp.make_trapezoid(channel='x', flat_area=Nx*deltak, flat_time=roDuration, system=sys)
gxm = pp.scale_grad(gxp, -1)
adc = pp.make_adc(Nx, duration=gxp.flat_time, delay=gxp.rise_time, system=sys)
gxPre = pp.make_trapezoid('x', area=-gxp.area/2, system=sys)

# Segmentation details
if nSeg % 2 == 0:
    print('Warning: For even number of segments, additional steps are required to avoid artifacts.')

phaseAreas = ((np.arange(int(Ny/nSeg)) - Ny/2) * deltak)

# Calculate blip gradient
gyBlip = pp.make_trapezoid('y', area=int(Ny/nSeg)*deltak, delay=gxp.rise_time+gxp.flat_time, system=sys)
if pp.calc_duration(gyBlip) - pp.calc_duration(gxp) < gyBlip.fall_time:
    gyBlip.delay += pp.calc_duration(gxp) - pp.calc_duration(gyBlip) + gyBlip.fall_time
gyBlip_parts = pp.split_gradient_at(gyBlip, pp.calc_duration(gxp), sys)
gyBlip_parts[1].delay = 0  # Reset delay for second part of the gradient

# Adjust gradient and ADC timing for echoes
gxp0 = gxp
adc0 = adc
gyBlip_part_tmp = gyBlip_parts[0]
if pp.calc_duration(gyBlip_parts[1]) > gxp.rise_time:
    gxp.delay = pp.calc_duration(gyBlip_parts[1]) - gxp.rise_time
    gxm.delay = pp.calc_duration(gyBlip_parts[1]) - gxm.rise_time
    adc.delay += gxp.delay
    gyBlip_part_tmp.delay += gxp.delay

gyBlip_down_up = pp.add_gradients(grads=[gyBlip_parts[1], gyBlip_part_tmp], system=sys)
gyBlip_up = gyBlip_parts[0]
gyBlip_down = gyBlip_parts[1]

# Gradient spoiling
spSign = -1 if np.size(np.array(TE)) % 2 == 0 else 1
gxSpoil = pp.make_trapezoid('x', area=2*Nx*deltak*spSign, system=sys)
gzSpoil = pp.make_trapezoid('z', area=4/sliceThickness, system=sys)

# Calculate timing delays
delayTE = TE - np.ceil((gz.fall_time + gz.flat_time/2 + (np.floor(nSeg/2)+0.5)*pp.calc_duration(gxp0) + np.floor((nSeg-1)/2)*gxp.delay)/seq.grad_raster_time) * seq.grad_raster_time
assert np.all(delayTE >= pp.calc_duration(gxPre, gzReph))
delayTR = np.round((TR - pp.calc_duration(gz) - delayTE - nSeg*pp.calc_duration(gxp0) - np.floor((nSeg-1)/2)*gxp.delay)/seq.grad_raster_time) * seq.grad_raster_time
assert np.all(delayTR >= pp.calc_duration(gxSpoil, gzSpoil))

# Initialize RF spoiling counters
rf_phase = 0
rf_inc = 0

# Define sequence blocks
for i in range(len(phaseAreas)):  # Loop over phase encodes
    rf.phaseOffset = rf_phase/180*np.pi
    adc.phaseOffset = rf_phase/180*np.pi
    adc0.phaseOffset = rf_phase/180*np.pi
    rf_inc = (rf_inc + rfSpoilingInc) % 360.0
    rf_phase = (rf_phase + rf_inc) % 360.0

    seq.add_block(rf, gz)
    gyPre = pp.make_trapezoid('y', area=phaseAreas[i], duration=pp.calc_duration(gxPre), system=sys)
    fff=(pp.align(left=[pp.make_delay(delayTE), gyPre, gzReph], right=[gxPre]))
    
    seq.add_block(fff[0],fff[1],fff[2],fff[3])
    for s in range(nSeg):  # Loop over segments
        if s == 0:
            seq.add_block(gxp0, adc0, gyBlip_up)
        else:
            gx = gxm if s % 2 == 1 else gxp
            if s != nSeg-1:
                seq.add_block(gx, adc, gyBlip_down_up)
            else:
                seq.add_block(gx, adc, gyBlip_down)

    gyPost = pp.make_trapezoid(channel='y', area=-phaseAreas[i] - gyBlip.area*(nSeg-1), duration=pp.calc_duration(gxPre), system=sys)
    delayEve = pp.make_delay(delayTR)
    seq.add_block(delayEve, gxSpoil, gyPost, gzSpoil)

# Check whether the timing of the sequence is correct
ok, error_report = seq.check_timing()


# Check timing
if plot:
    ok, error_report = seq.check_timing()
    
    if ok:
        print("Timing check passed successfully")
    else:
        print("Timing check failed! Error listing follows:")
        print((error_report))
    seq.plot()
    # Export definitions
    seq.set_definition("FOV", [fov, fov, sliceThickness])
    seq.set_definition("Name", "epi")

    seq.write("epi.seq")  # Write to pulseq file

    
    k_traj_adc, k_traj, t_excitation, t_refocusing, t_adc = seq.calculate_kspace()

    plt.figure(10)
    plt.plot(k_traj[0],k_traj[1],linewidth=1)
    plt.plot(k_traj_adc[0],k_traj_adc[1],'.', markersize=1.4)
    plt.axis("equal")
    plt.title("k-space trajectory (kx/ky)")

    plt.figure(11)
    plt.plot(t_adc, k_traj_adc.T, linewidth=1)
    plt.xlabel("Time of acqusition (s)")
    plt.ylabel("Phase")
    plt.show()
   

