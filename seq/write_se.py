# Created by Linkaisheng on Sep 30, 2024

import math
import warnings

import numpy as np

import pypulseq as pp


def main(plot: bool, write_seq: bool, seq_filename: str = "tse_pypulseq.seq"):
    # ======
    # SETUP
    # ======
    dG = 250e-6

    # Set system limits
    system = pp.Opts(
        max_grad=28,
        grad_unit="mT/m",
        max_slew=100,
        slew_unit="T/m/s",
        rf_ringdown_time=100e-6,
        rf_dead_time=100e-6,
        adc_dead_time=10e-6,
        rf_raster_time=10e-6, 
        adc_raster_time=1/(122.88e6)
    )
    seq = pp.Sequence(system)

    nsa = 1  # Number of averages
    n_slices = 1  # Number of slices
    Nx = 128
    Ny = 128
    fov = 220e-3  # mm
    slice_thickness = 5e-3  # s
    slice_gap = 15e-3  # s
    rf_flip = 90  # degrees
    rf_offset = 0
    print('User inputs setup')

    TE = 16e-3  # s
    TR = 3  # s
    tau = TE / 2  # s
    readout_time = 6.4e-3
    pre_time = 1e-3  # s


    flip90 = round(rf_flip * np.pi / 180, 3)
    flip180 = 180 * np.pi / 180
    rf90, gz90, _ = pp.make_sinc_pulse(
        flip_angle=flip90,
        system=system,
        duration=4e-3,
        slice_thickness=slice_thickness,
        apodization=0.5,
        time_bw_product=4,
        return_gz=True,
    )
    rf180, gz180, _ = pp.make_sinc_pulse(
        flip_angle=flip180,
        system=system,
        duration=2.5e-3,
        slice_thickness=slice_thickness,
        apodization=0.5,
        time_bw_product=4,
        phase_offset=90 * np.pi / 180,
        return_gz=True,
    )

    delta_k = 1 / fov
    k_width = Nx * delta_k
    gx =  pp.make_trapezoid(channel='x', system=system, flat_area=k_width, flat_time=readout_time)
    adc = pp.make_adc(num_samples=Nx, duration=gx.flat_time, delay=gx.rise_time)


    phase_areas = (np.arange(Ny) - (Ny / 2)) * delta_k
    gz_reph = pp.make_trapezoid(channel='z', system=system, area=-gz90.area / 2, duration=2.5e-3)
    gx_pre = pp.make_trapezoid(channel='x', system=system, flat_area=k_width / 2, flat_time=readout_time / 2)
    gy_pre = pp.make_trapezoid(channel='y', system=system, area=phase_areas[-1], duration=2e-3)
    gz_spoil = pp.make_trapezoid(channel='z', system=system, area=gz90.area * 1, duration=pre_time)



    delay1 = tau - pp.calc_duration(rf90) / 2 - pp.calc_duration(gx_pre)
    delay1 -= pp.calc_duration(gz_spoil) - pp.calc_duration(rf180) / 2
    delay1 = pp.make_delay(delay1)
    delay2 = tau - pp.calc_duration(rf180) / 2 - pp.calc_duration(gz_spoil)
    delay2 -= pp.calc_duration(gx) / 2
    delay2 = pp.make_delay(delay2)
    delay_TR = TR - pp.calc_duration(rf90) / 2 - pp.calc_duration(gx) / 2 - TE
    delay_TR -= pp.calc_duration(gy_pre)
    delay_TR = pp.make_delay(delay_TR)
    print(f'delay_1: {delay1}')
    print(f'delay_2: {delay1}')
    print(f'delay_TR: {delay_TR}')

    # Prepare RF offsets. This is required for multi-slice acquisition
    delta_z = n_slices * slice_gap
    z = np.linspace((-delta_z / 2), (delta_z / 2), n_slices) + rf_offset

    for _k in range(nsa):  # Averages
        for j in range(n_slices):  # Slices
            # Apply RF offsets
            freq_offset = gz90.amplitude * z[j]
            rf90.freq_offset = freq_offset

            freq_offset = gz180.amplitude * z[j]
            rf180.freq_offset = freq_offset

            for i in range(Ny):  # Phase encodes
                seq.add_block(rf90, gz90)
                gy_pre = pp.make_trapezoid(channel='y', system=system, area=phase_areas[-i - 1], duration=2e-3)
                seq.add_block(gx_pre, gy_pre, gz_reph)
                seq.add_block(delay1)
                seq.add_block(gz_spoil)
                seq.add_block(rf180, gz180)
                seq.add_block(gz_spoil)
                seq.add_block(delay2)
                seq.add_block(gx, adc)
                gy_pre = pp.make_trapezoid(channel='y', system=system, area=-phase_areas[-j - 1], duration=2e-3)
                seq.add_block(gy_pre, gz_spoil)
                seq.add_block(delay_TR)

    (
        ok,
        error_report,
    ) = seq.check_timing()  # Check whether the timing of the sequence is correct
    if ok:
        print("Timing check passed successfully")
    else:
        print("Timing check failed. Error listing follows:")
        [print(e) for e in error_report]
    
    # ======
    # VISUALIZATION
    # ======
    if plot:
        seq.plot()
        print(seq.test_report())

    # =========
    # WRITE .SEQ
    # =========
    seq.set_definition(key="FOV", value=[fov, fov, slice_thickness])
    if write_seq:
        import os

        # Define the path to the test_files directory
        test_files_dir = os.path.join(os.path.dirname(__file__), '../test_files')

        # Define the file name for the output
        output_file_path = os.path.join(test_files_dir, seq_filename)

        seq.write(output_file_path)  # Save to disk


if __name__ == "__main__":
    main(plot=True, write_seq=True, seq_filename = "tse.seq")
