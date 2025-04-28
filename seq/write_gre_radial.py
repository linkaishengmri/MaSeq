import math
import numpy as np
import matplotlib.pyplot as plt
from IPython.display import HTML

import pypulseq as pp
from utils import animate, simulate_2d, recon_nufft_2d, ifft_1d, plot_nd
plot = True
write_seq = True
animate_sequence = False
 
seq_filename = "gre_radial.seq"

# ======
# SETUP
# ======
fov = 256e-3
Nx = 128  # Define FOV and resolution
alpha = 30  # Flip angle
slice_thickness = 5e-3  # Slice thickness
TE = 5.5e-3  # Echo time
TR = 9e-3  # Repetition time
# Nr = 128 # Number of radial spokes

# Ex 1.2. Number of radial spokes for Nyquist sampling on the k-space edge
# Solved for arc length between two spokes = 1/fov
Nr = (math.ceil(Nx/2 * np.pi)-1)

# Ex 1.1 Increasing dummy scans to 50 is close enough to get to steady-state (see plots below)
N_dummy = 0  # Number of dummy scans
delta = np.pi / Nr  # Angular increment

# Ex 1.3: Golden angle increment
# delta = np.pi * (3 - np.sqrt(5))  # Angular increment

ro_os = 2 # Readout oversampling
rf_spoiling_inc = 117  # RF spoiling increment

# Set system limits
system = pp.Opts(
    max_grad=28,
    grad_unit="mT/m",
    max_slew=150,
    slew_unit="T/m/s",
    rf_ringdown_time=20e-6,
    rf_dead_time=100e-6,
    adc_dead_time=10e-6,
)

seq = pp.Sequence(system)  # Create a new sequence object

# ======
# CREATE EVENTS
# ======
# Create alpha-degree slice selection pulse and gradient
rf, gz, _ = pp.make_sinc_pulse(
    apodization=0.5,
    duration=1e-3,
    flip_angle=alpha * np.pi / 180,
    slice_thickness=slice_thickness,
    system=system,
    time_bw_product=4,
    return_gz=True,
)
gz_reph = pp.make_trapezoid(channel="z", area=-gz.area / 2, duration=2e-3, system=system)

# Define other gradients and ADC events
deltak = 1 / fov
gx = pp.make_trapezoid(channel="x", flat_area=Nx * deltak, flat_time=6.4e-3 / 5, system=system)
adc = pp.make_adc(num_samples=Nx*ro_os, duration=gx.flat_time, delay=gx.rise_time, system=system)
gx_pre = pp.make_trapezoid(channel="x", area=-gx.area / 2, duration=2e-3, system=system)

# Gradient spoiling
gx_spoil = pp.make_trapezoid(channel="x", area=0.5 * Nx * deltak, system=system)
gz_spoil = pp.make_trapezoid(channel="z", area=4 / slice_thickness, system=system)

# Calculate timing
delay_TE = (
    np.ceil(
        (
            TE
            - (pp.calc_duration(gz, rf) - pp.calc_rf_center(rf)[0] - rf.delay)
            - pp.calc_duration(gx_pre, gz_reph)
            - pp.calc_duration(gx) / 2
            - pp.eps
        )
        / seq.grad_raster_time
    )
    * seq.grad_raster_time
)

delay_TR = (
    np.ceil(
        (
            TR
            - pp.calc_duration(gz, rf)
            - pp.calc_duration(gx_pre, gz_reph)
            - pp.calc_duration(gx)
            - delay_TE
        )
        / seq.grad_raster_time
    )
    * seq.grad_raster_time
)
assert delay_TR >= pp.calc_duration(gx_spoil, gz_spoil)

rf_phase = 0
rf_inc = 0

# ======
# CONSTRUCT SEQUENCE
# ======
for i in range(-N_dummy, Nr):
    # Set RF/ADC phase for RF spoiling, and increment RF phase
    rf.phase_offset = rf_phase / 180 * np.pi
    adc.phase_offset = rf_phase / 180 * np.pi

    rf_inc = (rf_inc + rf_spoiling_inc) % 360.0
    rf_phase = (rf_phase + rf_inc) % 360.0

    # Slice-selective excitation pulse
    seq.add_block(rf, gz)

    # Slice rephaser and readout pre-phaser
    phi = delta * (i)
    seq.add_block(*pp.rotate(gx_pre, angle=phi, axis="z"), gz_reph)

    # Wait so readout is centered on TE
    seq.add_block(pp.make_delay(delay_TE))

    # Readout gradient, rotated by `phi`
    if i >= 0:
        # Real scan, readout gradient + ADC object
        seq.add_block(*pp.rotate(gx, angle=phi, axis="z"), adc)
    else:
        # Dummy scan, do not add ADC object
        seq.add_block(*pp.rotate(gx, angle=phi, axis="z"))

    # GX/GZ spoiler gradient, and wait for TR
    seq.add_block(*pp.rotate(gx_spoil, angle=phi, axis="z"), gz_spoil, pp.make_delay(delay_TR))

k_traj_adc, k_traj, t_excitation, t_refocusing, t_adc = seq.calculate_kspace()
deltak_edge = np.linalg.norm(k_traj_adc[:,adc.num_samples-1] - k_traj_adc[:,2*adc.num_samples-1])
if deltak_edge >= deltak*1.001: # Allow for small error
    print(f'Not Nyquist sampled! {deltak / deltak_edge * 100:.1f}% ')
else:
    print(f'Nyquist sampled! {deltak / deltak_edge * 100:.1f}% ')
# Timing check
ok, error_report = seq.check_timing()
if ok:
    print("Timing check passed successfully")
else:
    print("Timing check failed! Error listing follows:")
    print(error_report)

# ======
# VISUALIZATION
# ======
if plot:
    seq.plot()
    # Plot k-space trajectory
    k_traj_adc, k_traj, t_excitation, t_refocusing, t_adc = seq.calculate_kspace()

    plt.figure(9)
    plt.plot(k_traj[0],k_traj[1])
    plt.plot(k_traj_adc[0],k_traj_adc[1],'.')
    

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
    
    plt.figure(12)
    t = np.linspace(0, 1, k_traj_adc.shape[1])  # 归一化时间
    plt.scatter(k_traj_adc[0], k_traj_adc[1], c=t, cmap='viridis', s=2)  # 用颜色表示时间
    plt.axis("equal")
    plt.colorbar(label='Normalized Time')  # 添加颜色条
    plt.title("k-space trajectory (kx/ky) with Gradient")
    plt.show()

# Print test report
print(seq.test_report())

# =========
# WRITE .SEQ
# =========
seq.set_definition(key="FOV", value=[fov, fov, slice_thickness])

if write_seq:
    seq.set_definition(key="Name", value="gre_rad")
    seq.write(seq_filename)