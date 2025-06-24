"""
Created on Sunday, April 27th 2025
@author: Kaisheng Lin, School of Electronics, Peking University, China
@Summary: GRE sequence with radial sampling strategy, implemented with PyPulseq and compatible with MaSeq.
"""

import os
import sys
import matplotlib.pyplot as plt
import warnings

#*****************************************************************************
# Get the directory of the current script
main_directory = os.path.dirname(os.path.realpath(__file__))
parent_directory = os.path.dirname(main_directory)
parent_directory = os.path.dirname(parent_directory)

# Define the subdirectories you want to add to sys.path
subdirs = ['MaSeq', 'marcos_client']

# Add the subdirectories to sys.path
for subdir in subdirs:
    full_path = os.path.join(parent_directory, subdir)
    sys.path.append(full_path)
#******************************************************************************
from seq.utils import sort_data_implicit, plot_nd, ifft_2d, combine_coils, recon_nufft_2d
import math
import pypulseq as pp
import numpy as np
import seq.mriBlankSeq as blankSeq   
import configs.units as units
import scipy.signal as sig
import experiment_multifreq as ex
import configs.hw_config_pseq as hw
from flocra_pulseq.interpreter_pseq import PseqInterpreter
from pypulseq.convert import convert
# 8246716884 /122.88 = 67,111,953 us
# 9465307104
class GRERadialDebugPSEQ(blankSeq.MRIBLANKSEQ):
    def __init__(self):
        super(GRERadialDebugPSEQ, self).__init__()
        # Input the parameters
        self.output = None
        self.expt = None
        self.nScans = None
        self.larmorFreq = None
        self.rfExFA = None
        self.rfSincExTime = None
        self.repetitionTime = None
        self.echoTime = None
        self.fovInPlane = None
        self.dfov = None
        self.nPoints = None
        self.axesOrientation = None
        self.dummyPulses = None
        self.riseTime = None
        self.bandwidth = None
        self.DephTime = None
        self.shimming = None
        self.thickness = None
        self.sliceGap = None
        self.RFSpoilPhase = None
        self.fsp_r = None
        self.fsp_s = None
        self.Nr = None
        self.gx_comp = None
        self.gy_comp = None
        self.gz_comp = None
        self.compReadGrad = None
        self.adcDelayTime = None
        self.RFexDelayTime = None
        self.gradAmpDelayTime = None
        self.usePreemphasisGrad = None
        
        self.addParameter(key='seqName', string='gre', val='gre')
        self.addParameter(key='nScans', string='Number of scans', val=1, field='IM')
        self.addParameter(key='larmorFreq', string='Larmor frequency (MHz)', val=10.53380, units=units.MHz, field='IM')
        self.addParameter(key='rfExFA', string='Excitation flip angle (deg)', val=90, field='RF')
        self.addParameter(key='rfSincExTime', string='RF sinc excitation time (ms)', val=3.0, units=units.ms, field='RF')
        self.addParameter(key='repetitionTime', string='Repetition time (ms)', val=50.0, units=units.ms, field='SEQ')
        self.addParameter(key='echoTime', string='Echo time (ms)', val=10.0, units=units.ms, field='SEQ')
        
        self.addParameter(key='fovInPlane', string='FOV (mm)', val=100, units=units.mm, field='IM')
        self.addParameter(key='thickness', string='Slice thickness (mm)', val=5, units=units.mm, field='IM')
        self.addParameter(key='sliceGap', string='Slice gap (mm)', val=1, units=units.mm, field='IM')
        self.addParameter(key='dfov', string='dFOV[x,y,z] (mm)', val=[0.0, 0.0, 0.0], units=units.mm, field='IM',
                          tip="Position of the gradient isocenter")
        self.addParameter(key='nPoints', string='nPoints[rd, -, sl]', val=[128, 1, 1], field='IM')
        self.addParameter(key='axesOrientation', string='Axes[rd,ph,sl]', val=[1,2,0], field='IM',
                          tip="0=x, 1=y, 2=z")
        self.addParameter(key='dummyPulses', string='Dummy pulses', val=0, field='SEQ')
        
        self.addParameter(key='bandwidth', string='Acquisition Bandwidth (kHz)', val=21.3333333333333333, units=units.kHz, field='IM',
                          tip="The bandwidth of the acquisition (kHz). This value affects resolution and SNR.")
        self.addParameter(key='DephTime', string='Dephasing time (ms)', val=2.0, units=units.ms, field='OTH')
        self.addParameter(key='riseTime', string='Grad. rising time (ms)', val=0.25, units=units.ms, field='OTH')
        self.addParameter(key='shimming', string='Shimming', val=[0.0010, 0.0015, 0.0003], field='SEQ')
        self.addParameter(key='RFSpoilPhase', string='RF Spoiling Phase', val=117, field='OTH',
                          tip='117 deg is recommended')
        self.addParameter(key='fsp_r', string='Readout Spoiling', val=0.5, field='OTH',
                          tip="Gradient spoiling for readout.")
        self.addParameter(key='fsp_s', string='Slice Spoiling', val=4, field='OTH',
                          tip="Gradient spoiling for slice.")
        self.addParameter(key='Nr', string='Number of radial readouts', val=2, field='OTH',)
        self.addParameter(key='gx_comp', string='gx_comp', val=0.50, field='OTH',
                          tip="Gradient compensation for readout.") 
        self.addParameter(key='gy_comp', string='gy_comp', val=0.50, field='OTH',
                          tip="Gradient compensation for readout.") 
        
        self.addParameter(key='gz_comp', string='gz_comp', val=0.50, field='OTH',
                          tip="Gradient compensation for slice.") 
        self.addParameter(key='compReadGrad', string='Read Grad. Compensation', val=0, field='OTH')
        self.addParameter(key='adcDelayTime', string='ADC delay time (us)', val=180, units=units.us, field='OTH')
        self.addParameter(key='RFexDelayTime', string='RF ex delay time (us)', val=180, units=units.us, field='OTH')
        self.addParameter(key='gradAmpDelayTime', string='Grad. amp delay time (us)', val=[0,0,0], units=units.us, field='OTH',
                          tip="Compensation delay time for gradient amplifier.")
        self.addParameter(key='usePreemphasisGrad', string='usePreemphasisGrad', val=1, field='OTH')
        
     

    def sequenceTime(self):
        return (self.mapVals['repetitionTime'] *1e-3 * 
                self.mapVals['nScans'] *
                (math.ceil(self.mapVals['nPoints'][0]/2 * np.pi)-1) / 60)

    def sequenceAtributes(self):
        super().sequenceAtributes()

        # Convert files to a list
        # self.files = self.files.strip('[]').split(',')
        # self.files = [s.strip() for s in self.files]

    def sequenceRun(self, plotSeq=0, demo=False, standalone=False):
        init_gpa = False
        self.demo = demo
        self.plotSeq = plotSeq
        self.standalone = standalone
        
        # Calculate slice positions
        slice_positions = self.dfov[2] + (self.thickness + self.sliceGap) * (np.arange(self.nPoints[2]) - (self.nPoints[2] - 1) // 2)

        # slice idx
        slice_idx = np.concatenate((np.arange(self.nPoints[2])[::2],np.arange(self.nPoints[2])[1::2]))
        self.mapVals['sliceIdx'] = slice_idx

        # Reorder slices for an interleaved acquisition (optional)
        slice_positions = slice_positions[slice_idx]
        
        # redefine fov using slice thickness and gap
        self.fov = [self.fovInPlane, self.fovInPlane, np.max(slice_positions)-np.min(slice_positions)+self.thickness]       
        
        '''
        Step 1: Define the interpreter for FloSeq/PSInterpreter.
        The interpreter is responsible for converting the high-level pulse sequence description into low-level
        instructions for the scanner hardware. You will typically update the interpreter during scanner calibration.
        '''

        max_grad_Hz = convert(from_value=hw.max_grad, from_unit='mT/m', gamma=hw.gammaB, to_unit='Hz/m')
        rfExTime_us = int(np.round(self.rfSincExTime * 1e6))
        assert rfExTime_us in hw.max_sinc_rf_arr, f"RF excitation time '{rfExTime_us}' s is not found in the hw_config_pseq file; please search it in search_p90_pseq."
        max_rf_Hz = hw.max_sinc_rf_arr[rfExTime_us] * 1e-6 * hw.gammaB
        
        self.flo_interpreter = PseqInterpreter(
            tx_warmup=hw.blkTime,  # Transmit chain warm-up time (us)
            rf_center=hw.larmorFreq * 1e6 ,  # Larmor frequency (Hz)
            rf_amp_max=max_rf_Hz,  # Maximum RF amplitude (Hz)
            grad_max=max_grad_Hz,  # Maximum gradient amplitude (Hz/m)
            grad_t=10,  # Gradient raster time (us)
            orientation=self.axesOrientation, # gradient orientation
            grad_eff=hw.gradFactor, # gradient coefficient of efficiency
            use_multi_freq = True,
            add_rx_points = 0,
            tx_t= 1229/122.88, # us
            use_grad_preemphasis=True if self.usePreemphasisGrad == 1 else False,
            grad_preemphasis_coeff={
                        'zz':( (np.array([1.8061, 1.391, 0.2535, -0.0282]) * 1e-2, 
                            np.array([1567, 17510, 167180, 608533] ))),
                        'xx':( (np.array([-0.3031, 0.0782, 0.0227, 0.0]) * 1e-2,
                            np.array([2537, 87749, 986942, 0.1] ))),
                        'yy':( (np.array([1.7434, 2.0108, 0.4076, -0.1527]) *1e-2,
                            np.array([2151, 24193, 321545, 989703] ))),
                 },
            use_fir_decimation = (self.bandwidth < 30.007326007326007e3), # 30kHz
        )
        
        '''
        Step 2: Define system properties using PyPulseq (pp.Opts).
        These properties define the hardware capabilities of the MRI scanner, such as maximum gradient strengths,
        slew rates, and dead times. They are typically set based on the hardware configuration file (`hw_config`).
        '''
        self.system = pp.Opts(
            rf_dead_time=100 * 1e-6,  # Dead time between RF pulses (s)
            max_grad=38,  # Maximum gradient strength (mT/m)
            grad_unit='mT/m',  # Units of gradient strength
            max_slew=hw.max_slew_rate,  # Maximum gradient slew rate (mT/m/ms)
            slew_unit='mT/m/ms',  # Units of gradient slew rate
            grad_raster_time=hw.grad_raster_time, # hw.grad_raster_time,  # Gradient raster time (s)
            rise_time=hw.grad_rise_time,  # Gradient rise time (s)
            rf_raster_time=10e-6,
            block_duration_raster=1e-6,
            adc_raster_time=1/(122.88e6),
            adc_dead_time=0e-6,
            rf_ringdown_time=100e-6,
            

        )

        '''
        Step 3: Perform any calculations required for the sequence.
        In this step, students can implement the necessary calculations, such as timing calculations, RF amplitudes, and
        gradient strengths, before defining the sequence blocks.
        '''

        bw = self.bandwidth * 1e-6 # MHz
        bw_ov = self.bandwidth * 1e-6 # - hw.oversamplingFactor  # MHz
        sampling_period = 1 / bw_ov  # us, Dwell time

        '''
        Step 4: Define the experiment to get the true bandwidth
        In this step, student needs to get the real bandwidth used in the experiment. To get this bandwidth, an
        experiment must be defined and the sampling period should be obtained using get_rx_ts()[0]
        '''

        if not self.demo:
            expt = ex.ExperimentMultiFreq(
                lo_freq=hw.larmorFreq,  # Larmor frequency in MHz
                rx_t=sampling_period,  # Sampling time in us
                init_gpa=False,  # Whether to initialize GPA board (False for True)
                gpa_fhdo_offset_time=(1 / 0.2 / 3.1),  # GPA offset time calculation
                auto_leds=True  # Automatic control of LEDs (False or True)
            )
            sampling_period = expt.get_rx_ts()[0]  # us
            bw = 1 / sampling_period # / hw.oversamplingFactor  # MHz
            print("Acquisition bandwidth fixed to: %0.3f kHz" % (bw * 1e3))
            expt.__del__()
        self.mapVals['bw_MHz'] = bw
        self.mapVals['sampling_period_us'] = sampling_period

        '''
        Step 5: Define sequence blocks.
        In this step, you will define the building blocks of the MRI sequence, including the RF pulses and gradient pulses.
        '''
        TE = self.echoTime
        TR = self.repetitionTime
        N_dummy = self.dummyPulses
        
        rf_spoiling_inc = self.RFSpoilPhase
          
        Nx, Ny, n_slices = self.nPoints
        Nr = self.Nr
        delta = np.pi / Nr
        sampling_time = sampling_period * 1e-6 * self.nPoints[0]
        readout_time = sampling_time + 2 * self.system.adc_dead_time
        
        rf, gz, _ = pp.make_sinc_pulse(
            flip_angle=self.rfExFA * np.pi / 180,
            duration=self.rfSincExTime,
            slice_thickness=self.thickness,
            apodization=0.5,
            time_bw_product=4,
            system=self.system,
            phase_offset= np.pi / 2,
            return_gz=True
        )

        gz_reph = pp.make_trapezoid(channel="z", area=-gz.area * self.gz_comp, duration=self.DephTime, system=self.system)
        # Define other gradients and ADC events
        deltak = 1 / self.fovInPlane
        gx = pp.make_trapezoid(channel="x", flat_area=Nx * deltak, flat_time=readout_time, system=self.system)
        adc = pp.make_adc(num_samples=Nx, duration=gx.flat_time, delay=gx.rise_time, system=self.system)
        gx_pre = pp.make_trapezoid(channel="x", area=-gx.area * self.gx_comp + self.compReadGrad, duration=self.DephTime, system=self.system)

        # Gradient spoiling
        gx_spoil = pp.make_trapezoid(channel="x", area=self.fsp_r * Nx * deltak, duration=self.DephTime, system=self.system)
        gz_spoil = pp.make_trapezoid(channel="z", area=self.fsp_s / self.thickness, duration=self.DephTime, system=self.system)

        
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
                / self.system.grad_raster_time
            )
            * self.system.grad_raster_time
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
                / self.system.grad_raster_time
            )
            * self.system.grad_raster_time
        )
        assert delay_TE >= 0, f"delay_TE '{delay_TE}' should be greater than 0."
        
        assert delay_TR >= pp.calc_duration(gx_spoil, gz_spoil), f"delay_TR '{delay_TR}' should be greater than spoil time {gx_spoil}, {gz_spoil}."
        

        def runBatches_pseq(waveforms, n_readouts, frequency=hw.larmorFreq, bandwidth=0.03):
            """
            Execute multiple batches of waveforms for MRI data acquisition, handle scanning, and store oversampled data.

            Parameters:
            -----------
            waveforms : dict
                A dictionary of waveform sequences, where each key corresponds to a batch identifier and
                the value is the waveform data generated using PyPulseq.
            n_readouts : dict
                A dictionary that specifies the number of readout points for each batch. Keys correspond to
                the batch identifiers, and values specify the number of readout points for each sequence.
            frequency : float, optional
                Larmor frequency in MHz for the MRI scan (default is the system's Larmor frequency, hw.larmorFreq).
            bandwidth : float, optional
                Bandwidth in Hz used to calculate the sampling time (1 / bandwidth gives the sampling period).

            Returns:
            --------
            bool
                Returns True if all batches were successfully executed, and False if an error occurred (e.g.,
                sequence waveforms are out of hardware bounds).

            Notes:
            ------
            - The method will initialize the Red Pitaya hardware if not in demo mode.
            - The method converts waveforms from PyPulseq format to Red Pitaya compatible format.
            - If plotSeq is True, the sequence will be plotted instead of being executed.
            - In demo mode, the acquisition simulates random data instead of using actual hardware.
            - Oversampled data is stored in the class attribute `self.mapVals['data_over']`.
            - Data points are acquired in batches, with error handling in case of data loss, and batches are repeated if necessary.
            """
            

            # Initialize a list to hold oversampled data
            data_over = []
            real_bandwidth = bandwidth * self.flo_interpreter._fir_decimation_rate 
            
            # Iterate through each batch of waveforms
            for seq_num in waveforms.keys():
                # Initialize the experiment if not in demo mode
                if not self.demo:
                    self.expt = ex.ExperimentMultiFreq(
                        lo_freq=frequency,  # Larmor frequency in MHz
                        rx_t=1 / real_bandwidth,  # Sampling time in us
                        init_gpa=False,  # Whether to initialize GPA board (False for now)
                        gpa_fhdo_offset_time=(1 / 0.2 / 3.1),  # GPA offset time calculation
                        auto_leds=True,  # Automatic control of LEDs
                        allow_user_init_cfg=True # Allow lo*_freq and lo*_rst to be modified
                    )
                print(f"Center frequecy set: {frequency} MHz")
                # Convert the PyPulseq waveform to the Red Pitaya compatible format
                self.pypulseq2mriblankseq_ms(waveforms=waveforms[seq_num], shimming=self.shimming)

                # Load the waveforms into Red Pitaya
                if not self.floDict2Exp_ms():
                    print("ERROR: Sequence waveforms out of hardware bounds")
                    return False
                else:
                    encoding_ok = True
                    # print("Sequence waveforms loaded successfully")
                if self.plotSeq and not self.demo:
                    self.expt.plot_sequence()

                # If not plotting the sequence, start scanning
                if not self.plotSeq:
                    for scan in range(self.nScans):
                        print(f"Scan {scan + 1}, batch {seq_num.split('_')[-1]}/{1} running...")
                        acquired_points = 0
                        expected_points = n_readouts[seq_num] * self.flo_interpreter._fir_decimation_rate * hw.oversamplingFactor  # Expected number of points

                        # Continue acquiring points until we reach the expected number
                        while acquired_points != expected_points:
                            if not self.demo:
                                rxd, msgs = self.expt.run()  # Run the experiment and collect data
                            else:
                                # In demo mode, generate random data as a placeholder
                                rxd = {'rx0': np.random.randn(expected_points + self.flo_interpreter.get_add_rx_points()) + 1j * np.random.randn(expected_points + + self.flo_interpreter.get_add_rx_points())}
                            
                            # Update acquired points
                            self.rxChName = 'rx0'
                            rx_raw_data = rxd[self.rxChName]
                            rxdata = self.flo_interpreter.rx_points_added_for_img(rx_raw_data, self.nPoints[0])
                            rxdata = np.reshape(rxdata, newshape=(-1))
                            acquired_points = np.size(rxdata)

                            # Check if acquired points coincide with expected points
                            if acquired_points != expected_points:
                                print("WARNING: data apoints lost!")
                                print("Repeating batch...")

                        # Concatenate acquired data into the oversampled data array
                        data_over = np.concatenate((data_over, rxdata), axis=0)
                        print(f"Acquired points = {acquired_points}, Expected points = {expected_points}")
                        print(f"Scan {scan + 1}, batch {seq_num[-1]}/{1} ready!")

                    # Decimate the oversampled data and store it
                    self.mapVals['data_over'] = data_over
                    self.mapVals['data_full'] = np.concatenate((self.mapVals['data_full'], self.mapVals['data_over']), axis=0)
                    
                elif self.plotSeq and self.standalone:
                    # Plot the sequence if requested and return immediately
                    self.sequencePlot(standalone=self.standalone)

                if not self.demo:
                    self.expt.__del__()

            return True
        
        
        # Initialize batches dictionary to store different parts of the sequence.
        batches = {}
        n_rd_points_dict = {}  # Dictionary to track readout points for each batch
        n_rd_points = 0
        batch_idx = 1 # In this sequence, batch_idx is equivalent to the index of slice coding index 

        
        '''
        Step 7: Define your createBatches method.
        In this step you will populate the batches adding the blocks previously defined in step 4, and accounting for
        number of acquired points to check if a new batch is required.
        '''
        def createBatches():
            """
            Create batches for the full pulse sequence.

            Instructions:
            - This function creates the complete pulse sequence by iterating through repetitions.
            - Each iteration adds new blocks to the sequence, including the RF pulse, ADC block, and repetition delay.
            - If a batch exceeds the maximum number of readout points, a new batch is started.

            Returns:
                waveforms (dict): Contains the waveforms for each batch.
                n_rd_points_dict (dict): Dictionary of readout points per batch.
            """
            
            n_rd_points = 0
            rf_phase = 0
            rf_inc = 0
            for s in range(n_slices):    
                # slice offset
                rf_freq_offset_slice = gz.amplitude * slice_positions[s]
                rf_phase_offset_slice = 0 - 2 * np.pi * rf.freq_offset * pp.calc_rf_center(rf)[0]
                adc_phase_offset_slice = rf_phase_offset_slice
                for i in range(-N_dummy, Nr):
                     

                    # Set RF/ADC freq/phase for RF spoiling, and increment RF phase
                    rf.freq_offset = rf_freq_offset_slice
                    rf.phase_offset = rf_phase / 180 * np.pi + rf_phase_offset_slice
                    adc.phase_offset = rf_phase / 180 * np.pi + adc_phase_offset_slice

                    rf_inc = (rf_inc + rf_spoiling_inc) % 360.0
                    rf_phase = (rf_phase + rf_inc) % 360.0

                    # Slice-selective excitation pulse
                    batches[batch_num].add_block(rf, gz)

                    # Slice rephaser and readout pre-phaser
                    phi = delta * (i)
                    batches[batch_num].add_block(*pp.rotate(gx_pre, angle=phi, axis="z"), gz_reph)

                    # Wait so readout is centered on TE
                    batches[batch_num].add_block(pp.make_delay(delay_TE))

                    # Readout gradient, rotated by `phi`
                    if i >= 0:
                        # Real scan, readout gradient + ADC object
                        batches[batch_num].add_block(*pp.rotate(gx, angle=phi, axis="z"), adc)
                        assert n_rd_points + self.nPoints[0] < hw.maxRdPoints
                        n_rd_points = n_rd_points + self.nPoints[0]
                    else:
                        # Dummy scan, do not add ADC object
                        batches[batch_num].add_block(*pp.rotate(gx, angle=phi, axis="z"))

                    # GX/GZ spoiler gradient, and wait for TR
                    batches[batch_num].add_block(*pp.rotate(gx_spoil, angle=phi, axis="z"), gz_spoil, pp.make_delay(delay_TR))
            
            (
                ok,
                error_report,
            ) = batches[batch_num].check_timing()  # Check whether the timing of the sequence is correct
            
            if plotSeq:
                if ok:
                    print("Timing check passed successfully")
                else:
                    print("Timing check failed. Error listing follows:")
                    [print(e) for e in error_report]
                
                k_traj_adc, k_traj, t_excitation, t_refocusing, t_adc = batches[batch_num].calculate_kspace()
                deltak_edge = np.linalg.norm(k_traj_adc[:,adc.num_samples-1] - k_traj_adc[:,2*adc.num_samples-1])
                if deltak_edge >= deltak*1.001: # Allow for small error
                    print(f'Not Nyquist sampled! {deltak / deltak_edge * 100:.1f}% ')
                else:
                    print(f'Nyquist sampled! {deltak / deltak_edge * 100:.1f}% ')
                
                print(batches[batch_num].test_report())
                batches[batch_num].plot()
                k_traj_adc, k_traj, t_excitation, t_refocusing, t_adc = batches[batch_num].calculate_kspace()

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

            batches[batch_num].set_definition(key="Name", value="gre_radial")
            batches[batch_num].set_definition(key="FOV", value=self.fov)
            batches[batch_num].write(f"gre_radial_{batch_num}.seq")
            self.waveforms[batch_num], param_dict = self.flo_interpreter.interpret(f"gre_radial_{batch_num}.seq")
            rx0_en = self.waveforms[batch_num]['rx0_en']
            self.waveforms[batch_num]['rx0_en'] = (rx0_en[0] + self.adcDelayTime*1e6, rx0_en[1])
            tx0 = self.waveforms[batch_num]['tx0']
            self.waveforms[batch_num]['tx0'] = (tx0[0] + self.RFexDelayTime*1e6, tx0[1])
            grad_vx = self.waveforms[batch_num]['grad_vx'][0][1:]
            self.waveforms[batch_num]['grad_vx'][0][1:] = (grad_vx + self.gradAmpDelayTime[0]*1e6)
            grad_vy = self.waveforms[batch_num]['grad_vy'][0][1:]
            self.waveforms[batch_num]['grad_vy'][0][1:] = (grad_vy + self.gradAmpDelayTime[1]*1e6)
            grad_vz = self.waveforms[batch_num]['grad_vz'][0][1:]
            self.waveforms[batch_num]['grad_vz'][0][1:] = (grad_vz + self.gradAmpDelayTime[2]*1e6)
            print(f"gre_radial_{batch_num}.seq ready!")
            print(f"{len(batches)} batches created with {n_rd_points} read points. Sequence ready!")

            # Update the number of acquired ponits in the last batch
            self.n_rd_points_dict[batch_num] = n_rd_points
            self.lastseq = batches[batch_num]

            return 

        '''
        Step 8: Run the batches
        This step will handle the different batches, run it and get the resulting data. This should not be modified.
        Oversampled data will be available in self.mapVals['data_over']
        '''
        self.waveforms = {}  # Dictionary to store generated waveforms
        self.n_rd_points_dict = {}
        # self.rf_slice_freq_offset = []
        self.mapVals['data_full'] = []
        
            
        batch_num = f"batch_{batch_idx}"  # Initial batch name
        print(f"Creating {batch_num}.seq...")
        batches[batch_num] = pp.Sequence(system=self.system)

        createBatches()

        batches_list = [{key: value} for key, value in self.waveforms.items()]
        n_rd_points_list = [{key: value} for key, value in self.n_rd_points_dict.items()]
        
        assert runBatches_pseq(batches_list[0],
                            n_rd_points_list[0],
                            frequency=(self.larmorFreq)*1e-6 ,  # MHz
                            bandwidth=bw_ov,  # MHz
                            )
            
        self.mapVals['n_readouts'] = list(self.n_rd_points_dict.values())
        self.mapVals['n_batches'] = 1
        self.mapVals['Nr'] = Nr
        return True

        
    def sequenceAnalysis(self, mode=None, lambda_tv=0.0):
        def getFHWM(s,f_vector,bw):
            target = np.max(s) / 2
            p0 = np.argmax(s)
            f0 = f_vector[p0]
            s1 = np.abs(s[0:p0]-target)
            f1 = f_vector[np.argmin(s1)]
            s2 = np.abs(s[p0::]-target)
            f2 = f_vector[np.argmin(s2)+p0]
            return f2-f1

        self.mode = mode
        # Signal and spectrum from 'fir' and decimation
        # signal = self.mapVals['data_full']

         # Get data
        data_full_pre = self.mapVals['data_full']
        nRD, nPH, nSL = self.nPoints
        nRadial = self.mapVals['Nr']

        nRD = nRD + 2 * hw.addRdPoints
        n_batches = self.mapVals['n_batches']

        # fir decimator
        if self.flo_interpreter._fir_decimation_rate > 1:
            data_waiting_for_fir = np.reshape(data_full_pre, newshape=(-1, self.flo_interpreter._fir_decimation_rate * nRD))
            data_full = self.flo_interpreter.fir_decimator(input_matrix=data_waiting_for_fir, decimation_rate=3)
        else:
            data_full = data_full_pre
        signal = np.reshape(data_full,newshape=(-1) )

        bw = self.mapVals['bw_MHz']*1e3 # kHz
        nPoints = self.mapVals['nPoints'][0] * nRadial
        # deadTime = self.mapVals['deadTime']*1e-3 # ms
        rfSincExTime = self.mapVals['rfSincExTime']*1e-3 # ms
        tVector = np.linspace(rfSincExTime/2 + 0 + 0.5/bw, rfSincExTime/2  + (nPoints-0.5)/bw, nPoints)
        signal_chunks = signal.reshape(-1, nRD)

        fft_chunks = []
        for chunk in signal_chunks:
            fft_result = np.fft.ifftshift(np.fft.ifftn(np.fft.ifftshift(chunk)))
            fft_chunks.append(fft_result)

        spectrum = np.concatenate(fft_chunks)
        fVector = np.arange(len(spectrum))

        # find the max arg of abs(signal), and judge if it is in the center of the signal
        print(f'signal max index: {np.abs(signal).argmax()}/{len(signal)}, val={np.max(np.abs(signal))}')

        # Add time signal to the layout
        result1 = {'widget': 'curve',
                   'xData': tVector,
                   'yData': [np.real(signal), np.imag(signal)],
                   'xLabel': 'Time (ms)',
                   'yLabel': 'Signal amplitude (a.u.)',
                   'title': 'Signal vs time',
                   'legend': ['real','imag'],
                   'row': 0,
                   'col': 0}
        # Add time signal to the layout
        result2 = {'widget': 'curve',
                   'xData': fVector,
                   'yData': [np.abs(spectrum)],
                   'xLabel': 'index',
                   'yLabel': 'Amplitude',
                   'title': 'spectrum amplitude',
                   'legend': ['abs'],
                   'row': 1,
                   'col': 0}

        # Add frequency spectrum to the layout
        result3 = {'widget': 'curve',
                   'xData': fVector,
                   'yData': [np.angle(spectrum)],
                   'xLabel': 'index',
                   'yLabel': 'Phase (rad)',
                   'title': 'spectrum angle',
                   'legend': ['angle'],
                   'row': 2,
                   'col': 0}
        diff_phase = np.diff(np.angle(spectrum))
        diff_phase_padded = np.insert(diff_phase, 0, 0)
        result4 = {'widget': 'curve',
                   'xData': fVector,
                   'yData': [diff_phase_padded],
                   'xLabel': 'index',
                   'yLabel': 'Diff Phase (rad)',
                   'title': 'spectrum angle diff',
                   'legend': ['diff_angle'],
                   'row': 3,
                   'col': 0}

        # create self.out to run in iterative mode
        self.output = [result1, result2, result3, result4]
        self.saveRawData()

        if self.mode == 'Standalone':
            self.plotResults()
        return self.output
 
    
if __name__ == '__main__':
    seq = GRERadialDebugPSEQ()
    seq.sequenceAtributes()
    seq.sequenceRun(plotSeq=True, demo=False, standalone=True)
    seq.sequenceAnalysis(mode='Standalone')




