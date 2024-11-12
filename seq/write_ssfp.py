# Created by Linkaisheng on Oct 14, 2024


# import importlib
# if importlib.util.find_spec('pypulseq') is None:
#     !pip install -q mrzerocore git+https://github.com/imr-framework/pypulseq.git

# import math
# import numpy as np
# import matplotlib.pyplot as plt

# import pypulseq as pp

# !wget -nc https://raw.githubusercontent.com/pulseq/MR-Physics-with-Pulseq/main/utils/utils.py
# from utils import animate, simulate_2d, sort_data_implicit, ifft_2d, combine_coils, plot_nd

# from IPython.display import HTML


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

# Define FOV and resolution
fov = [150e-3, 150e-3, 5e-3]
Nx = 256
Ny = 2
Nz = 1
alpha = 60  # flip angle
TR = 300e-3  # Repetition time
TE = 10e-3  # Echo time
dummy_scans = 0 #100 # Number of dummy repetitions

ro_os = 1 # Readout oversampling

# Here are the modifications that are needed to fulfil the different excercises
# For the RF-Spoiling, see below in the sequence block

# FISP
A = -0.5
C = 0.5
rf_inc = 0

# bSSFP
#A = -0.5
#C = -0.5
#rf_inc = 180
# To make the banding artefact visible, change C to, e.g., C = -0.4
# To shift the banding artefact, change rf_inc.

# PSIF
#A = 0.5
#C = -0.5
#rf_inc = 0

# k= -2 sequence
#A = 1.5
#C = -1.5
#rf_inc = 0

dwell_set = 25e-6 # adc dw time
dwell = np.round(np.array(dwell_set) * 122.88e6) / 122.88e6
readout_duration = dwell * Nx # Duration of flat area of readout gradient (sec)
print(f'dwell time: {dwell}, readout time: {readout_duration}')
# readout_duration = 3.2e-3 # Readout duration (sec)
pe_duration = 2e-3 # Duration of phase encoding gradients (sec)

# Create system object
system = pp.Opts(
    max_grad=28,
    grad_unit="mT/m",
    max_slew=150,
    slew_unit="T/m/s",
    rf_ringdown_time=20e-6,
    rf_dead_time=100e-6,
    adc_dead_time=10e-6,
    adc_raster_time=1/(122.88e6)
)

# Create a new sequence object
seq = pp.Sequence(system=system)

# ======
# CREATE EVENTS
# ======
rf, gz, _ = pp.make_sinc_pulse(
    flip_angle=alpha * math.pi / 180,
    duration=3e-3,
    slice_thickness=fov[2],
    apodization=0.42,
    time_bw_product=4,
    system=system,
    return_gz=True
)

# Define other gradients and ADC events
delta_kx = 1 / fov[0]
delta_ky = 1 / fov[1]
delta_kz = 1 / fov[2]

gx = pp.make_trapezoid(channel="x", flat_area=Nx * delta_kx, flat_time=readout_duration, system=system)
adc = pp.make_adc(num_samples=Nx * ro_os, duration=gx.flat_time, delay=gx.rise_time, system=system)
gx_pre = pp.make_trapezoid(channel="x", area=-gx.area / 2, duration=pe_duration, system=system)

gx_spoil = pp.make_trapezoid(channel="x", area=2 * Nx * delta_kx, system=system)
gz_spoil = pp.make_trapezoid(channel="z", area=4 / fov[2], system=system)

# Phase encoding
phase_areas_y = (np.arange(Ny) - Ny // 2) * delta_ky
phase_areas_z = (np.arange(Nz) - Nz // 2) * delta_kz

# Phase encoding table with YZ order (outer loop = Z, inner loop = Y)
phase_encode_table = [(y,z) for z in range(len(phase_areas_z)) for y in range(len(phase_areas_y))]

# Calculate timing
delay_TE = (
    np.ceil(
        (
            TE
            - (pp.calc_duration(gz, rf) - pp.calc_rf_center(rf)[0] - rf.delay)
            - pp.calc_duration(gx_pre)
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
            - pp.calc_duration(rf, gz)
            - pp.calc_duration(gx_pre)
            - pp.calc_duration(gx)
            - delay_TE
        )
        / seq.grad_raster_time
    )
    * seq.grad_raster_time
)

# Exercises: Possible that you need to comment out these
assert delay_TE >= 0
assert delay_TR >= pp.calc_duration(gx_spoil, gz_spoil)

N_pe = len(phase_encode_table)

# ======
# CONSTRUCT SEQUENCE
# ======

# Initialize label values
last_lin = 0
last_slc = 0

# Initialize RF phase cycling
rf_phase = 0


# Loop over phase encodes and define sequence blocks
for i in range(-dummy_scans, N_pe):

    # rf spoiling
    # rf_phase = i*(i-1)*117/2
    # no rf spoiling
    rf_phase = (rf_phase + rf_inc) % 360.0

    rf.phase_offset = rf_phase / 180 * np.pi
    adc.phase_offset = rf_phase / 180 * np.pi

    # RF excitation and slice/slab selection gradient
    seq.add_block(rf, gz)

    # Wait for TE
    seq.add_block(pp.make_delay(delay_TE))

    # Phase encoding gradients, combined with slice selection rephaser
    pe_index_y, pe_index_z = phase_encode_table[max(i, 0)]
    #
    gx_pre = pp.make_trapezoid(channel="x", area=A * gx.area, duration=pe_duration, system=system)
    gy_pre = pp.make_trapezoid(channel="y", area=phase_areas_y[pe_index_y], duration=pe_duration, system=system)
    gz_pre = pp.make_trapezoid(channel="z", area=phase_areas_z[pe_index_z] - gz.area / 2, duration=pe_duration, system=system)
    seq.add_block(gx_pre, gy_pre, gz_pre)

    # Readout, do not enable ADC/labels for dummy acquisitions
    if i < 0:
        seq.add_block(gx)
    else:
        # Readout with LIN (Y) and SLC (Z) labels (increment relative to previous label value)
        seq.add_block(gx, adc)#, pp.make_label('LIN', 'INC', pe_index_y - last_lin), pp.make_label('SLC', 'INC', pe_index_z - last_slc))

    # Balance phase encoding and slice selection gradients
    gy_post = pp.make_trapezoid(channel="y", area=-phase_areas_y[pe_index_y], duration=pe_duration, system=system) #jl
    gz_post = pp.make_trapezoid(channel="z", area=-phase_areas_z[pe_index_z] - gz.area / 2, duration=pe_duration, system=system) #jl
    gx_post = pp.make_trapezoid(channel="x", area=C * gx.area, duration=pe_duration, system=system)
    seq.add_block(gx_post, gy_post, gz_post)

    # wait for TR
    seq.add_block(pp.make_delay(delay_TR))

    # Update previous labels
    last_lin = pe_index_y
    last_slc = pe_index_z


# Check whether the timing of the sequence is correct
ok, error_report = seq.check_timing()
if ok:
    print("Timing check passed successfully")
else:
    print("Timing check failed. Error listing follows:")
    [print(e) for e in error_report]

# ======
# VISUALIZATION
# ======
if plot:
    #seq.plot()
    nn=6
    seq.plot()#time_range=np.array([50, 53])*TR, time_disp="ms")


# Very optional slow step, but useful for testing during development e.g. for the real TE, TR or for staying within
# slew-rate limits
print(seq.test_report())

# =========
# WRITE .SEQ
# =========
seq.set_definition(key="FOV", value=fov)

if write_seq:
    # Prepare the sequence output for the scanner
    seq.set_definition(key="Name", value="gre_3d")

    # Define the path to the test_files directory
    test_files_dir = os.path.join(os.path.dirname(__file__), '../test_files')

    # Define the file name for the output
    output_file_path = os.path.join(test_files_dir, 'ssfp.seq')

    seq.write(output_file_path)  # Save to disk
