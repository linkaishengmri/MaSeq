# Created by Linkaisheng on Sep 30, 2024

import math
import warnings
import matplotlib.pyplot as plt
import numpy as np

import pypulseq as pp


def main(plot: bool, write_seq: bool, seq_filename: str = "flash_pypulseq.seq"):
 
    # Define FOV and resolution
    fov = 256e-3
    slice_thickness = 5e-3
    Nx, Ny = 64,64
    
    dummy = 2

    # Define sequence parameters
    TE = 8e-3
    TR = 22e-3
    alpha = 30  # Flip angle in degrees


    dwell_set = 25e-6 # adc dw time
    dwell = np.round(np.array(dwell_set) * 122.88e6) / 122.88e6
    readout_duration = dwell * Nx # Duration of flat area of readout gradient (sec)
    print(f'dwell time: {dwell}, readout time: {readout_duration}')
    pe_duration = 2e-3 # Duration of phase encoding gradients (sec)

    # Set system limits
    sys = pp.Opts(
        max_grad=10,  # Maximum gradient amplitude [mT/m]
        grad_unit="mT/m",
        max_slew=100,  # Maximum slew rate [T/m/s]
        slew_unit="T/m/s",
        rf_ringdown_time=20e-6,
        rf_dead_time=100e-6,
        adc_dead_time=10e-6,
        adc_raster_time=1/(122.88e6),
        rf_raster_time=10e-6
    )

    # Create a new sequence object
    seq = pp.Sequence(sys)

    # Create slice-selective alpha-pulse and corresponding gradients
    rf, gz, gz_reph = pp.make_sinc_pulse(
        flip_angle=alpha * math.pi / 180,
        duration=4e-3,
        slice_thickness=slice_thickness,
        apodization=0.5,
        time_bw_product=4,
        system=sys,
        return_gz=True,
    )

    # Define other gradients and ADC events
    deltak = 1 / fov
    gx = pp.make_trapezoid(
        channel="x", flat_area=Nx * deltak, flat_time=readout_duration, system=sys
    )
    adc = pp.make_adc(num_samples=Nx, duration=gx.flat_time, delay=gx.rise_time)
    gx_pre = pp.make_trapezoid(channel="x", area=-gx.area / 2, duration=pe_duration, system=sys)
    phase_areas = ((np.arange(Ny) - Ny // 2) * deltak)

    def round_to_raster_time(value, raster_time):
        """Round a given time value to the nearest raster time."""
        return np.round(value / raster_time) * raster_time
    
    # Calculate timing
    delay_te = round_to_raster_time(
        TE - pp.calc_duration(gx_pre) - pp.calc_duration(gz) / 2 - pp.calc_duration(gx) / 2,
        sys.grad_raster_time,
    )
    delay_tr = round_to_raster_time(
        TR - pp.calc_duration(gx_pre) - pp.calc_duration(gz) - pp.calc_duration(gx) - delay_te,
        sys.grad_raster_time,
    )

    spoil_area = 10 * gx.area
    gx_post = pp.make_trapezoid(channel="x", area=spoil_area, system=sys)
    gy_post = pp.make_trapezoid(channel="y", area=-np.max(phase_areas), duration=2e-3, system=sys)
    gz_post = pp.make_trapezoid(channel="z", area=spoil_area, system=sys)

    delay_tr -= pp.calc_duration(gx_post, gy_post, gz_post)

    # Loop over phase encodes and define sequence blocks
    for i in range(-dummy, Ny):
        # Vary RF phase quasi-randomly
        rand_phase = (117 * (i ** 2 + i + 2) % 360) * np.pi / 180
        rf, gz, _ = pp.make_sinc_pulse(
            flip_angle=alpha * math.pi / 180,
            duration=4e-3,
            slice_thickness=5e-3,
            apodization=0.5,
            time_bw_product=4,
            system=sys,
            phase_offset=rand_phase,
            return_gz=True,
        )
        seq.add_block(rf, gz)
        
        if i >= 0:  # Negative or zero index -- dummy scans
            gy_pre = pp.make_trapezoid(channel="y", area=phase_areas[i], duration=2e-3, system=sys)
        else:
            gy_pre = pp.make_trapezoid(channel="y", area=0, duration=2e-3, system=sys)
        
        seq.add_block(gx_pre, gy_pre, gz_reph)
        seq.add_block(pp.make_delay(delay_te))
        
        # Make receiver phase follow transmitter phase
        adc = pp.make_adc(
            num_samples=Nx,
            duration=gx.flat_time,
            delay=gx.rise_time,
            phase_offset=rand_phase,
        )
        
        if i >= 0:  # Negative index -- dummy scans
            seq.add_block(gx, adc)
        else:
            seq.add_block(gx)
        
        gy_post = pp.make_trapezoid(channel="y", area=-gy_pre.area, duration=2e-3, system=sys)
        seq.add_block(gx_post, gy_post, gz_post)
        seq.add_block(pp.make_delay(delay_tr))

    # Check timing
    ok, error_report = seq.check_timing()
    if ok:
        print("Timing check passed successfully")
    else:
        print("Timing check failed! Error listing follows:")
        # print("\n".join(error_report))

    # Export definitions
    seq.set_definition("FOV", [fov, fov, slice_thickness])
    seq.set_definition("Name", "DEMO_gre5")

    #seq.write("DEMO_gre5.seq")  # Write to pulseq file

    

    # Optional k-space trajectory calculation
    # ktraj_adc, t_adc, ktraj, t_ktraj = seq.calculate_kspace()
    k_traj_adc, k_traj, t_excitation, t_refocusing, t_adc = seq.calculate_kspace()

    plt.figure()
    plt.plot(k_traj[0],k_traj[1])
    plt.plot(k_traj_adc[0],k_traj_adc[1],'.')
    plt.axis("equal")
    plt.title("k-space trajectory (kx/ky)")
    
    # plt.figure()
    # plt.plot(t_excitation, k_traj.T)
    # plt.plot(t_adc, k_traj[0], ".")
    # plt.title("k-space vector components as functions of time")
    # plt.legend(["kx", "ky", "kz"])


    # plt.plot(ktraj[0, :], ktraj[1, :], "b")
    # plt.plot(ktraj_adc[0, :], ktraj_adc[1, :], "r.")


    plt.show()

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
    main(plot=True, write_seq=False, seq_filename = "flash_pypulseq.seq")
