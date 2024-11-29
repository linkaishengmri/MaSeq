
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
sys = pp.Opts(
    max_grad=38, grad_unit='mT/m', 
    max_slew=50, slew_unit='mT/m/ms', 
    rf_ringdown_time=20e-6,
    rf_dead_time=100e-6, 
    adc_dead_time=10e-6,
    adc_raster_time=1/(122.88e6),
    rf_raster_time=10e-6,
    grad_raster_time=10e-6,
    block_duration_raster=1e-6,
    B0 = .3 ##########################
)

# Basic parameters
seq = pp.Sequence(sys)  # Create a new sequence object

fov = 200e-3
Nx, Ny, Nslices  = 128, 128, 1
sliceThickness = 3e-3  # Slice thickness in meters
sliceGap = 1e-3
b_factor = 900  # Diffusion weighting factor in s/mm^2

b_factor = 1e-10 if b_factor < 1e-10 else b_factor
# Calculate slice positions
slice_positions = (sliceThickness + sliceGap) * (np.arange(Nslices) - (Nslices - 1) // 2)
slice_idx = np.concatenate((np.arange(Nslices)[::2],np.arange(Nslices)[1::2]))
slice_positions = slice_positions[slice_idx] # Reorder slices for an interleaved acquisition (optional)


TE = 140e-3  # Echo time (TE)
TR = 500e-3


# Number of segments equals number of phase encoding lines
nSeg = Ny 

dwell_set = 5.078125e-6 # adc dw time
dwell = np.round(np.array(dwell_set) * 122.88e6) / 122.88e6
roDuration = dwell * Nx # Duration of flat area of readout gradient (sec)
print(f'dwell time: {dwell}, readout time: {roDuration}')



tRFex = 3e-3  # Excitation pulse duration in seconds
tRFref = 3e-3  # Refocusing pulse duration in seconds
sat_ppm = -3.45
sat_freq = sat_ppm * 1e-6 * sys.B0 * sys.gamma
rf_fs = pp.make_gauss_pulse(
    flip_angle=np.deg2rad(110),
    system=sys,
    duration=50e-3,
    bandwidth=np.abs(sat_freq),
    freq_offset=sat_freq,
    use='saturation'
)

rf_fs.phase_offset = -2 * np.pi * rf_fs.freq_offset * pp.calc_rf_center(rf_fs)[0] # compensate for the frequency-offset induced phase    
gz_fs = pp.make_trapezoid(
    channel='z',
    system=sys, 
    delay=pp.calc_duration(rf_fs), 
    area=1 / 1e-4
) # spoil up to 0.1mm

# 90-degree slice selection pulse and gradient
rf, gz, gz_reph = pp.make_sinc_pulse(
    flip_angle=np.pi / 2,
    system=sys,
    duration=tRFex,
    slice_thickness=sliceThickness,
    phase_offset=np.pi / 2,
    apodization=0.5,
    time_bw_product=4,
    return_gz=True,
)
# 180-degree slice refocusing pulse and gradients
rf180, gz180, _ = pp.make_sinc_pulse(
    flip_angle=np.pi,
    system=sys,
    duration=tRFref,
    slice_thickness=sliceThickness,
    apodization=0.5,
    time_bw_product=4,
    use='refocusing',
    return_gz=True,
)

_, gzr_t, gzr_a = pp.make_extended_trapezoid_area(
    channel='z', 
    grad_start=gz180.amplitude,
    grad_end=0,
    area=-gz_reph.area + 0.5 * gz180.amplitude * gz180.fall_time, 
    system=sys
)

gz180n_times = np.concatenate(
    (np.array([gz180.delay, gz180.delay + gz180.rise_time]), 
    np.array(gz180.delay + gz180.rise_time + gz180.flat_time + gzr_t)))
gz180n_amp = np.concatenate((np.array([0, gz180.amplitude]), np.array(gzr_a)))
# Build the new trapezoid gradient
gz180n = pp.make_extended_trapezoid(
    channel='z',
    system=sys,
    times=gz180n_times,
    amplitudes=gz180n_amp
)



# Define other gradients and ADC events
deltak = 1 / fov  # k-space step in inverse meters
gxp = pp.make_trapezoid(channel='x', rise_time=100e-6, flat_area=Nx*deltak, flat_time=roDuration, system=sys)
gxm = pp.scale_grad(gxp, -1)
adc = pp.make_adc(Nx, duration=gxp.flat_time, delay=gxp.rise_time, system=sys)
gxPre = pp.make_trapezoid('x', area=-gxp.area/2, system=sys)


duration_to_center = (Ny / 2.0) * pp.calc_duration(gxp)
rf_center_incl_delay = rf.delay + pp.calc_rf_center(rf)[0]
rf180_center_incl_delay = rf180.delay + pp.calc_rf_center(rf180)[0]

delay_te1 = np.ceil((TE / 2 - pp.calc_duration(rf, gz) + rf_center_incl_delay - rf180_center_incl_delay) / sys.grad_raster_time) * sys.grad_raster_time
delay_te2_tmp = np.ceil((TE / 2 - pp.calc_duration(rf180, gz180n) + rf180_center_incl_delay - duration_to_center) / sys.grad_raster_time) * sys.grad_raster_time
assert delay_te1 >= 0
delay_te2 = delay_te2_tmp - pp.calc_duration(gxPre)
assert delay_te2 >= 0


def b_fact_calc(g, delta, DELTA):
        """
        Calculate the b-factor based on the gradient amplitude (g), 
        pulse duration (delta), and the interval between the gradient pulses (DELTA).
        
        Parameters:
        g : float
            Gradient amplitude in Hz/m.
        delta : float
            Duration of the gradient pulse (s).
        DELTA : float
            The interval between gradient pulses (s).
        
        Returns:
        b : float
            The b-factor (s/mm^2).
        """
        # Constants from the paper (for rectangular gradients)
        sigma = 1
        kappa_minus_lambda = 1/3 - 1/2  # For trapezoidal gradients, adjust later
        
        # Calculate b-factor
        b = (2 * np.pi * g * delta * sigma) ** 2 * (DELTA + 2 * kappa_minus_lambda * delta)
        
        return b

# Diffusion weighting calculation
small_delta = delay_te2 - np.ceil(sys.max_grad / sys.max_slew / sys.grad_raster_time) * sys.grad_raster_time
big_delta = delay_te1 + pp.calc_duration(rf180, gz180n)
g = np.sqrt(b_factor * 1e6 / b_fact_calc(1, small_delta, big_delta))
gr = np.ceil(g / sys.max_slew / sys.grad_raster_time) * sys.grad_raster_time
g_diff = pp.make_trapezoid('z', amplitude=g, rise_time=gr, flat_time=small_delta - gr, system=sys)
assert pp.calc_duration(g_diff) <= delay_te1
assert pp.calc_duration(g_diff) <= delay_te2

delay_tr = TR - (big_delta
            + pp.calc_duration(g_diff) 
            + pp.calc_duration(rf_fs,gz_fs) 
            + pp.calc_duration(rf) 
            + pp.calc_duration(gxPre)
            + duration_to_center * 2)
assert delay_tr >= 1e-6
# Segmentation details
phaseAreas = ((np.arange(int(Ny/nSeg)) - Ny/2) * deltak)

# Calculate blip gradient
gyBlip = pp.make_trapezoid('y', area=int(Ny/nSeg)*deltak, delay=gxp.rise_time+gxp.flat_time,
                            system=sys,rise_time=90e-6,flat_time=20e-6)
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
# Define sequence blocks
for Cs in range(Nslices):
    seq.add_block(rf_fs,gz_fs) 
    rf.freq_offset=gz.amplitude*slice_positions[Cs]
    rf.phase_offset=np.pi/2-2*np.pi*rf.freq_offset*pp.calc_rf_center(rf)[0] # compensate for the slice-offset induced phase
    rf180.freq_offset=gz180.amplitude*slice_positions[Cs]
    rf180.phase_offset=-2*np.pi*rf180.freq_offset*pp.calc_rf_center(rf180)[0] # compensate for the slice-offset induced phase
    seq.add_block(rf,gz)
    seq.add_block(pp.make_delay(delay_te1),g_diff)
    seq.add_block(rf180,gz180n)
    seq.add_block(pp.make_delay(delay_te2),g_diff)
 
    for i in range(len(phaseAreas)):  # Loop over phase encodes
        
        # seq.add_block(rf, gz)
        gyPre = pp.make_trapezoid('y', area=phaseAreas[i], duration=pp.calc_duration(gxPre), system=sys)
        seq.add_block(gyPre,gxPre)
        for s in range(nSeg):  # Loop over segments
            if s == 0:
                seq.add_block(gxp0, adc0, gyBlip_up)
            else:
                gx = gxm if s % 2 == 1 else gxp
                if s != nSeg-1:
                    seq.add_block(gx, adc, gyBlip_down_up)
                else:
                    seq.add_block(gx, adc, gyBlip_down)
        seq.add_block(delay_tr)
        # gyPost = pp.make_trapezoid(channel='y', area=-phaseAreas[i] - gyBlip.area*(nSeg-1), duration=pp.calc_duration(gxPre), system=sys)
        # seq.add_block(gxSpoil, gyPost, gzSpoil)

# Check whether the timing of the sequence is correct
ok, error_report = seq.check_timing()


# Check timing
if plot:
    ok, error_report = seq.check_timing()
    
    if ok:
        print("Timing check passed successfully")
        print(seq.test_report())
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
   

