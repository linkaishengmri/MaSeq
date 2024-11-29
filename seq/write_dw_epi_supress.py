# Created by Linkaisheng on Nov 25, 2024
import numpy as np

xx=[(x/122.88) for x in range(10000)]


import math
import warnings
import matplotlib.pyplot as plt


import pypulseq as pp


def main(plot: bool, write_seq: bool, seq_filename: str = "dw_epi_pypulseq.seq"):
    import numpy as np
    sys = pp.Opts(
        max_grad=38,  # Maximum gradient amplitude [mT/m]
        grad_unit="mT/m",
        max_slew=180,  # Maximum slew rate [T/m/s]
        slew_unit="T/m/s",
        rf_ringdown_time=10e-6,
        rf_dead_time=100e-6,
        adc_dead_time=10e-6,
        adc_raster_time=1e-7,
        rf_raster_time=10e-6,
        grad_raster_time=10e-6,
        block_duration_raster=1e-6,
        B0=10.36 #hw.B0 #######################[TODO]
    )
    # Define FOV and resolution
    fov = 224e-3
    thickness = 2e-3
    Nx, Ny, Nslices  = 112, 112, 1
    b_factor = 900  # Diffusion weighting factor in s/mm^2
 

    # Define sequence parameters
    TE = 80e-3
    pe_enable = 1  # Phase encoding enable flag
    ro_os = 1  # Readout oversampling factor
    part_fourier_factor = 0.5  # Partial Fourier sampling factor


    dwell_set = 6.25e-6 # adc dw time
    dwell = np.round(np.array(dwell_set) * 122.88e6) / 122.88e6
    readout_time = dwell * Nx # Duration of flat area of readout gradient (sec)
    print(f'dwell time: {dwell}, readout time: {readout_time}')
    
    tRFex = 3e-3  # Excitation pulse duration in seconds
    tRFref = 3e-3  # Refocusing pulse duration in seconds
    sat_ppm = -3.45
    sat_freq = sat_ppm * 1e-6 * sys.B0 * sys.gamma
    rf_fs = pp.make_gauss_pulse(
        flip_angle=np.deg2rad(110),
        system=sys,
        duration=8e-3,
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
        slice_thickness=thickness,
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
        slice_thickness=thickness,
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

    # Define digital output trigger
    # trig = pp.make_digital_output_pulse(channel='osc0', duration=100e-6)



    # Gradients and ADC events
    deltak = 1 / fov
    k_width = Nx * deltak
    blip_dur = np.ceil(2 * np.sqrt(deltak / sys.max_slew) / 10e-6 / 2) * 10e-6 * 2
    print(f'blip duration: {blip_dur}')
    gy = pp.make_trapezoid(channel='y', system=sys, area=-deltak, duration=blip_dur, rise_time=2e-5)
    # Readout gradient
    # % readout gradient is a truncated trapezoid with dead times at the beginnig
    # % and at the end each equal to a half of blip_dur
    # % the area between the blips should be defined by kWidth
    # % we do a two-step calculation: we first increase the area assuming maximum
    # % slewrate and then scale down the amlitude to fix the area 

    extra_area = blip_dur / 2 * blip_dur / 2 * sys.max_slew
    gx = pp.make_trapezoid('x', system=sys, area=k_width + extra_area, duration=readout_time + blip_dur, rise_time=1.2e-4)
    actual_area = gx.area - gx.amplitude / gx.rise_time * blip_dur / 2 * blip_dur / 2 / 2 - gx.amplitude / gx.fall_time * blip_dur / 2 * blip_dur / 2 / 2
    gx.amplitude = gx.amplitude / actual_area * k_width
    gx.area = gx.amplitude * (gx.flat_time + gx.rise_time / 2 + gx.fall_time / 2)
    gx.flat_area = gx.amplitude * gx.flat_time

    # ADC settings (ramp sampling)
    adc_dwell_nyquist = deltak / gx.amplitude / ro_os
    adc_dwell = np.floor(adc_dwell_nyquist * 1e7) * 1e-7
    adc_samples = np.floor(readout_time / adc_dwell / 4) * 4
    adc = pp.make_adc(adc_samples, dwell=adc_dwell, delay=blip_dur / 2)

    # Realign ADC with respect to gradient
    time_to_center = adc.dwell * ((adc_samples - 1) / 2 + 0.5)
    adc.delay = np.round((gx.rise_time + gx.flat_time / 2 - time_to_center) * 1e6) * 1e-6

    # Split the blip into two halves and combine them
    gy_parts = pp.split_gradient_at(gy, blip_dur / 2, system=sys)
    gy_blipup, gy_blipdown, _ = pp.align(right=gy_parts[0], left= gy_parts[1], center=gx)
    
    gy_blipup.delay = sys.grad_raster_time * np.round(gy_blipup.delay / sys.grad_raster_time)
    
    gy_blipdownup = pp.add_gradients(grads=[gy_blipdown, gy_blipup], system=sys)

    # Enable or disable phase encoding
    gy_blipup.waveform = gy_blipup.waveform * pe_enable
    gy_blipdown.waveform = gy_blipdown.waveform * pe_enable
    gy_blipdownup.waveform = gy_blipdownup.waveform * pe_enable

    # Pre-phase gradients
    Ny_pre = round(part_fourier_factor * Ny / 2 - 1)
    Ny_post = round(Ny / 2 + 1)
    Ny_meas = Ny_pre + Ny_post

    gx_pre = pp.make_trapezoid('x', system=sys, area=-gx.area / 2)
    gy_pre = pp.make_trapezoid('y', system=sys, area=Ny_pre * deltak)
    gx_pre, gy_pre = pp.align(right= gx_pre, left= gy_pre)

    gy_pre = pp.make_trapezoid('y', system=sys, area=gy_pre.area, duration=pp.calc_duration(gx_pre, gy_pre))
    gy_pre.amplitude = gy_pre.amplitude * pe_enable

    # Calculate delays
    duration_to_center = (Ny_pre + 0.5) * pp.calc_duration(gx)
    rf_center_incl_delay = rf.delay + pp.calc_rf_center(rf)[0]
    rf180_center_incl_delay = rf180.delay + pp.calc_rf_center(rf180)[0]
    delay_te1 = np.ceil((TE / 2 - pp.calc_duration(rf, gz) + rf_center_incl_delay - rf180_center_incl_delay) / sys.grad_raster_time) * sys.grad_raster_time
    delay_te2_tmp = np.ceil((TE / 2 - pp.calc_duration(rf180, gz180n) + rf180_center_incl_delay - duration_to_center) / sys.grad_raster_time) * sys.grad_raster_time
    assert delay_te1 >= 0
    delay_te2 = delay_te2_tmp - pp.calc_duration(gx_pre, gy_pre)
    gx_pre.delay = 0
    gy_pre.delay = 0
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

    seq = pp.Sequence(system=sys)
    # Add sequence blocks for each slice
    # Define sequence blocks
    for s in range(0, Nslices):
        seq.add_block(rf_fs,gz_fs) 
        rf.freq_offset=gz.amplitude*thickness*(s-1-(Nslices-1)/2) 
        rf.phase_offset=np.pi/2-2*np.pi*rf.freq_offset*pp.calc_rf_center(rf)[0] # compensate for the slice-offset induced phase
        rf180.freq_offset=gz180.amplitude*thickness*(s-1-(Nslices-1)/2) 
        rf180.phase_offset=-2*np.pi*rf180.freq_offset*pp.calc_rf_center(rf180)[0] # compensate for the slice-offset induced phase
        seq.add_block(rf,gz)
        seq.add_block(pp.make_delay(delay_te1),g_diff)
        seq.add_block(rf180,gz180n)
        seq.add_block(pp.make_delay(delay_te2),g_diff)
        seq.add_block(gx_pre,gy_pre)
        for i in range(Ny_meas):
            if i==0:
                seq.add_block(gx,gy_blipup,adc) # Read the first line of k-space with a single half-blip at the end
            elif i==Ny_meas-1:
                seq.add_block(gx,gy_blipdown,adc) # Read the last line of k-space with a single half-blip at the beginning
            else:
                seq.add_block(gx,gy_blipdownup,adc) # Read an intermediate line of k-space with a half-blip at the beginning and a half-blip at the end
            gx.amplitude = -gx.amplitude # Reverse polarity of read gradient
         
    


    # Check timing
    ok, error_report = seq.check_timing()
    if ok:
        print("Timing check passed successfully")
    else:
        print("Timing check failed! Error listing follows:")
        print((error_report))

    # Export definitions
    seq.set_definition("FOV", [fov, fov, thickness])
    seq.set_definition("Name", "dw_epi")

    seq.write("dw_epi.seq")  # Write to pulseq file

    
    k_traj_adc, k_traj, t_excitation, t_refocusing, t_adc = seq.calculate_kspace()

    plt.figure(3)
    plt.plot(k_traj[0],k_traj[1],linewidth=1)
    plt.plot(k_traj_adc[0],k_traj_adc[1],'.', markersize=1.4)
    plt.axis("equal")
    plt.title("k-space trajectory (kx/ky)")

    plt.figure(4)
    plt.plot(t_adc, k_traj_adc.T, linewidth=1)
    plt.xlabel("Time of acqusition (s)")
    plt.ylabel("Phase")
   



    # plt.show()

    # ======
    # VISUALIZATION
    # ======
    if plot:
        seq.plot()
        print(seq.test_report())

    # =========
    # WRITE .SEQ
    # =========
    seq.set_definition(key="FOV", value=[fov, fov, thickness])
    if write_seq:
        import os

        # Define the path to the test_files directory
        test_files_dir = os.path.join(os.path.dirname(__file__), '../test_files')

        # Define the file name for the output
        output_file_path = os.path.join(test_files_dir, seq_filename)

        seq.write(output_file_path)  # Save to disk


if __name__ == "__main__":
    main(plot=True, write_seq=False, seq_filename = "dw_epi_pypulseq.seq")
