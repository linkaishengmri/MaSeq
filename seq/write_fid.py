# Created by Linkaisheng on Oct 12, 2024


import math
import warnings

import numpy as np

import pypulseq as pp


def main(plot: bool, write_seq: bool, seq_filename: str = "fid.seq"):
    
    # Set system limits
    system = pp.Opts(
        max_grad=80,
        grad_unit="mT/m",
        max_slew=100,
        slew_unit="T/m/s",
        rf_ringdown_time=100e-6,
        rf_dead_time=100e-6,
        rf_raster_time=10e-6, 
        adc_raster_time=1/(122.88e6) 
    )

    seq = pp.Sequence(system)  # Create a new sequence object
    
    Nx = 20480 # samples
    t_ex = 3.5e-3 # RF duration
    dwell_set = 3.125e-6 # adc dw time
    dwell = np.round(np.array(dwell_set) * 122.88e6) / 122.88e6
    dead_time = 100e-6 # dead time (total dead time = ringdown time + this dead_time)
    sampling_time = dwell * Nx
    print(dwell, sampling_time)

    flip_ex = 90 * np.pi / 180
    rf_ex = pp.make_sinc_pulse(
        flip_angle=flip_ex,
        system=system,
        duration=t_ex,
        apodization=0.5,
        time_bw_product=4,
        phase_offset=0,
        return_gz=False
    )

    deadtime = pp.make_delay(dead_time)
    adc = pp.make_adc(
        num_samples=Nx, duration=sampling_time
    )



    seq.add_block(rf_ex)
    seq.add_block(deadtime)
    
    seq.add_block(adc)
                

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

    # =========
    # WRITE .SEQ
    # =========
    if write_seq:
        import os

        # Define the path to the test_files directory
        test_files_dir = os.path.join(os.path.dirname(__file__), '../test_files')

        # Define the file name for the output
        output_file_path = os.path.join(test_files_dir, seq_filename)

        seq.write(output_file_path)  # Save to disk


if __name__ == "__main__":
    main(plot=True, write_seq=True, seq_filename = "fid.seq")
