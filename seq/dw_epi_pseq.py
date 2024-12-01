"""
Created on Thursday, Nov 28th 2024
@author: Kaisheng Lin, School of Electronics, Peking University, China
@Summary: DW-EPI sequence, implemented with PyPulseq and compatible with MaSeq.
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
from seq.utils import sort_data_implicit, plot_nd, ifft_2d, combine_coils
import pypulseq as pp
import numpy as np
import seq.mriBlankSeq as blankSeq   
import configs.units as units
import scipy.signal as sig
import experiment_multifreq as ex
import configs.hw_config_pseq as hw
from flocra_pulseq.interpreter_pseq import PseqInterpreter
from pypulseq.convert import convert

class DWEPIPSEQ(blankSeq.MRIBLANKSEQ):
    def __init__(self):
        super(DWEPIPSEQ, self).__init__()
        # Input the parameters
        self.output = None
        self.expt = None
        self.nScans = None
        self.larmorFreq = None
        self.rfExFA = None
        self.rfReFA = None
        self.rfSaFA = None
        self.rfSincExTime = None
        self.rfSincReTime = None
        self.rfSaTime = None
 
        self.fovInPlane = None
        self.dfov = None
        self.nPoints = None
        self.axesOrientation = None
        # self.riseTime = None
        self.dwellTime = None
        # self.DephTime = None
        self.echoTime = None
        self.repetitionTime = None
        self.shimming = None
        self.thickness = None
        self.sliceGap = None
        self.bFactor = None
        self.enaSat = None
        self.addParameter(key='seqName', string='dw_epi', val='dw_epi')
        self.addParameter(key='nScans', string='Number of scans', val=1, field='IM')
        self.addParameter(key='larmorFreq', string='Larmor frequency (MHz)', val=10.35206, units=units.MHz, field='IM')
        self.addParameter(key='rfExFA', string='Excitation flip angle (deg)', val=90, field='RF')
        self.addParameter(key='rfReFA', string='Refocusing flip angle (deg)', val=180, field='RF')
        self.addParameter(key='rfSaFA', string='Saturation flip angle (deg)', val=110, field='RF')
        
        self.addParameter(key='rfSincExTime', string='RF sinc excitation time (ms)', val=3.0, units=units.ms, field='RF')
        self.addParameter(key='rfSincReTime', string='RF sinc refocusing time (ms)', val=3.0, units=units.ms, field='RF')
        self.addParameter(key='rfSaTime', string='RF saturation time (ms)', val=50.0, units=units.ms, field='RF')
        self.addParameter(key='enaSat', string='Enable Saturation', val=0, field='RF')
        
        self.addParameter(key='fovInPlane', string='FOV[Rd,Ph] (mm)', val=[200, 200], units=units.mm, field='IM')
        self.addParameter(key='thickness', string='Slice thickness (mm)', val=5, units=units.mm, field='IM')
        self.addParameter(key='sliceGap', string='Slice gap (mm)', val=1, units=units.mm, field='IM')
        self.addParameter(key='dfov', string='dFOV[x,y,z] (mm)', val=[0.0, 0.0, 0.0], units=units.mm, field='IM',
                          tip="Position of the gradient isocenter")
        self.addParameter(key='nPoints', string='nPoints[rd, ph, sl]', val=[128, 128, 1], field='IM')
        self.addParameter(key='axesOrientation', string='Axes[rd,ph,sl]', val=[1,2,0], field='IM',
                          tip="0=x, 1=y, 2=z")
        self.addParameter(key='dwellTime', string='Dwell Time (us)', val=5.078125, units=units.us, field='IM',
                          tip="Dwell Time of the acquisition (us). This value affects resolution and SNR.")
        self.addParameter(key='echoTime', string='Echo Time (ms)', val=140.0, units=units.ms, field='SEQ')
        self.addParameter(key='repetitionTime', string='Repetition time (ms)', val=500.0, units=units.ms, field='SEQ')
        self.addParameter(key='bFactor', string='Diffusion weighting factor (s/mm^2)', val=0, field='SEQ')

        # self.addParameter(key='DephTime', string='Dephasing time (ms)', val=2.0, units=units.ms, field='OTH')
        # self.addParameter(key='riseTime', string='Grad. rising time (ms)', val=0.25, units=units.ms, field='OTH')
        self.addParameter(key='shimming', string='Shimming', val=[0.0, 0.0, 0.0], field='SEQ')
 

    def sequenceInfo(self):
        print("Pulseq Reader")
        print("Author: PhD. J.M. Algarín")
        print("Contact: josalggui@i3m.upv.es")
        print("mriLab @ i3M, CSIC, Spain")
        print("Run a list of .seq files\n")
        

    def sequenceTime(self):
        return (self.mapVals['repetitionTime'] *1e-3 * 
                self.mapVals['nScans'] *
                self.mapVals['nPoints'][1] / self.mapVals['etl'] / 60)

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
        slice_positions = (self.thickness + self.sliceGap) * (np.arange(self.nPoints[2]) - (self.nPoints[2] - 1) // 2)

        # slice idx
        slice_idx = np.concatenate((np.arange(self.nPoints[2])[::2],np.arange(self.nPoints[2])[1::2]))
        self.mapVals['sliceIdx'] = slice_idx

        # Reorder slices for an interleaved acquisition (optional)
        slice_positions = slice_positions[slice_idx]
        
        # redefine fov using slice thickness and gap
        self.fov = [self.fovInPlane[0], self.fovInPlane[1], np.max(slice_positions)-np.min(slice_positions)+self.thickness]       
        
        '''
        Step 1: Define the interpreter for FloSeq/PSInterpreter.
        The interpreter is responsible for converting the high-level pulse sequence description into low-level
        instructions for the scanner hardware. You will typically update the interpreter during scanner calibration.
        '''

        max_grad_Hz = convert(from_value=hw.max_grad, from_unit='mT/m', gamma=hw.gammaB, to_unit='Hz/m')
        max_rf_Hz = hw.max_rf * 1e-6 * hw.gammaB
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
            tx_t= 1229/122.88 # us
        )
        
        '''
        Step 2: Define system properties using PyPulseq (pp.Opts).
        These properties define the hardware capabilities of the MRI scanner, such as maximum gradient strengths,
        slew rates, and dead times. They are typically set based on the hardware configuration file (`hw_config`).
        '''
        self.system = pp.Opts(
            rf_dead_time=100e-6,  # Dead time between RF pulses (s)
            max_grad=38,  # Maximum gradient strength (mT/m)
            grad_unit='mT/m',  # Units of gradient strength
            max_slew=50, # may be too large for system !!!!!!!!!!!!!!
            slew_unit='mT/m/ms',  # Units of gradient slew rate
            grad_raster_time=hw.grad_raster_time, # hw.grad_raster_time,  # Gradient raster time (s)
            rise_time=100e-6,  # Gradient rise time (s)
            rf_raster_time=10e-6,
            block_duration_raster=1e-6,
            adc_raster_time=1/(122.88e6),
            adc_dead_time=0e-6,
            rf_ringdown_time=100e-6,
            B0 = self.larmorFreq / hw.gammaB,
        )

        '''
        Step 3: Perform any calculations required for the sequence.
        In this step, students can implement the necessary calculations, such as timing calculations, RF amplitudes, and
        gradient strengths, before defining the sequence blocks.
        '''
        bw = 1 / self.dwellTime * 1e-6 # MHz
        bw_ov = bw
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
        
         
        Nx, Ny, Nslices  = self.nPoints
        sliceThickness = self.thickness
        b_factor = self.bFactor  # Diffusion weighting factor in s/mm^2

        b_factor = 1e-10 if b_factor < 1e-10 else b_factor
        

        TE = self.echoTime  # Echo time (TE)
        TR = self.repetitionTime


        # Number of segments equals number of phase encoding lines
        nSeg = Ny 

        roDuration = sampling_period * 1e-6 * Nx # Duration of flat area of readout gradient (sec)

        tRFex = self.rfSincExTime  # Excitation pulse duration in seconds
        tRFref = self.rfSincReTime  # Refocusing pulse duration in seconds
        sat_ppm = -3.45
        sat_freq = sat_ppm * 1e-6 * self.system.B0 * self.system.gamma
        if self.enaSat == 0:
            rf_fs = pp.make_delay(1e-3)
        else:
            rf_fs = pp.make_gauss_pulse(
                flip_angle=np.deg2rad(self.rfSaFA),
                system=self.system,
                duration=self.rfSaTime,
                bandwidth=np.abs(sat_freq),
                freq_offset=sat_freq,
                use='saturation'
            )
            rf_fs.phase_offset = -2 * np.pi * rf_fs.freq_offset * pp.calc_rf_center(rf_fs)[0] # compensate for the frequency-offset induced phase    
        
        if self.enaSat==0:
            gz_fs = pp.make_delay(1e-10)
        else:
            gz_fs = pp.make_trapezoid(
                channel='z',
                system=self.system, 
                delay=pp.calc_duration(rf_fs), 
                area=1 / 1e-4
            ) # spoil up to 0.1mm

        # 90-degree slice selection pulse and gradient
        rf, gz, gz_reph = pp.make_sinc_pulse(
            flip_angle=np.pi / 2,
            system=self.system,
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
            system=self.system,
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
            system=self.system
        )

        gz180n_times = np.concatenate(
            (np.array([gz180.delay, gz180.delay + gz180.rise_time]), 
            np.array(gz180.delay + gz180.rise_time + gz180.flat_time + gzr_t)))
        gz180n_amp = np.concatenate((np.array([0, gz180.amplitude]), np.array(gzr_a)))
        # Build the new trapezoid gradient
        gz180n = pp.make_extended_trapezoid(
            channel='z',
            system=self.system,
            times=gz180n_times,
            amplitudes=gz180n_amp
        )

        
        # Define other gradients and ADC events
        deltakx = 1 / self.fov[0]  # k-space step in inverse meters
        deltaky = 1 / self.fov[1]
        gxp = pp.make_trapezoid(channel='x', rise_time=100e-6, flat_area=Nx*deltakx, flat_time=roDuration, system=self.system)
        gxm = pp.scale_grad(gxp, -1)
        adc = pp.make_adc(Nx, duration=gxp.flat_time, delay=gxp.rise_time, system=self.system)
        gxPre = pp.make_trapezoid('x', area=-gxp.area/2, system=self.system)


        duration_to_center = (Ny / 2.0) * pp.calc_duration(gxp)
        rf_center_incl_delay = rf.delay + pp.calc_rf_center(rf)[0]
        rf180_center_incl_delay = rf180.delay + pp.calc_rf_center(rf180)[0]

        delay_te1 = np.ceil((TE / 2 - pp.calc_duration(rf, gz) + rf_center_incl_delay - rf180_center_incl_delay) / self.system.grad_raster_time) * self.system.grad_raster_time
        delay_te2_tmp = np.ceil((TE / 2 - pp.calc_duration(rf180, gz180n) + rf180_center_incl_delay - duration_to_center) / self.system.grad_raster_time) * self.system.grad_raster_time
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
        small_delta = delay_te2 - np.ceil(self.system.max_grad / self.system.max_slew / self.system.grad_raster_time) * self.system.grad_raster_time
        big_delta = delay_te1 + pp.calc_duration(rf180, gz180n)
        g = np.sqrt(b_factor * 1e6 / b_fact_calc(1, small_delta, big_delta))
        gr = np.ceil(g / self.system.max_slew / self.system.grad_raster_time) * self.system.grad_raster_time
        g_diff = pp.make_trapezoid('z', amplitude=g, rise_time=gr, flat_time=small_delta - gr, system=self.system)
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
        phaseAreas = ((np.arange(int(Ny/nSeg)) - Ny/2) * deltaky)

        # Calculate blip gradient
        gyBlip = pp.make_trapezoid('y', area=int(Ny/nSeg)*deltaky, delay=gxp.rise_time+gxp.flat_time,
                                   system=self.system, rise_time=90e-6, flat_time=20e-6)
        if pp.calc_duration(gyBlip) - pp.calc_duration(gxp) < gyBlip.fall_time:
            gyBlip.delay += pp.calc_duration(gxp) - pp.calc_duration(gyBlip) + gyBlip.fall_time
        gyBlip_parts = pp.split_gradient_at(gyBlip, pp.calc_duration(gxp), self.system)
        gyBlip_parts[1].delay = 0  # Reset delay for second part of the gradient

        # Adjust gradient and ADC timing for echoes
        gxp0 = gxp
        adc0 = adc
        gyBlip_part_tmp = gyBlip_parts[0]
        if pp.calc_duration(gyBlip_parts[1]) > gxp.rise_time:
            print('warning: gyBlip time may be too large! ')
            gxp.delay = pp.calc_duration(gyBlip_parts[1]) - gxp.rise_time
            gxm.delay = pp.calc_duration(gyBlip_parts[1]) - gxm.rise_time
            adc.delay += gxp.delay
            gyBlip_part_tmp.delay += gxp.delay

        gyBlip_down_up = pp.add_gradients(grads=[gyBlip_parts[1], gyBlip_part_tmp], system=self.system)
        gyBlip_up = gyBlip_parts[0]
        gyBlip_down = gyBlip_parts[1]

        # Gradient spoiling
        spSign = -1 if np.size(np.array(TE)) % 2 == 0 else 1
        gxSpoil = pp.make_trapezoid('x', area=2*Nx*deltakx*spSign, system=self.system)
        gzSpoil = pp.make_trapezoid('z', area=4/sliceThickness, system=self.system)


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

            # Iterate through each batch of waveforms
            for seq_num in waveforms.keys():
                # Initialize the experiment if not in demo mode
                if not self.demo:
                    self.expt = ex.ExperimentMultiFreq(
                        lo_freq=frequency,  # Larmor frequency in MHz
                        rx_t=1 / bandwidth,  # Sampling time in us
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
                if self.plotSeq:
                    self.expt.plot_sequence()

                # If not plotting the sequence, start scanning
                if not self.plotSeq:
                    for scan in range(self.nScans):
                        print(f"Scan {scan + 1}, batch {seq_num.split('_')[-1]}/{1} running...")
                        acquired_points = 0
                        expected_points = n_readouts[seq_num] * hw.oversamplingFactor  # Expected number of points

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
                            rxdata = rx_raw_data #  self.flo_interpreter.rx_points_added_for_img(rx_raw_data, self.nPoints[0])
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
            # Define sequence blocks
            for Cs in range(Nslices):
                batches[batch_num].add_block(rf_fs,gz_fs) 
                rf.freq_offset=gz.amplitude*slice_positions[Cs]
                rf.phase_offset=np.pi/2-2*np.pi*rf.freq_offset*pp.calc_rf_center(rf)[0] # compensate for the slice-offset induced phase
                rf180.freq_offset=gz180.amplitude*slice_positions[Cs]
                rf180.phase_offset=-2*np.pi*rf180.freq_offset*pp.calc_rf_center(rf180)[0] # compensate for the slice-offset induced phase
                batches[batch_num].add_block(rf,gz)
                batches[batch_num].add_block(pp.make_delay(delay_te1),g_diff)
                batches[batch_num].add_block(rf180,gz180n)
                batches[batch_num].add_block(pp.make_delay(delay_te2),g_diff)
            
                for i in range(len(phaseAreas)):  # Loop over phase encodes
                    gyPre = pp.make_trapezoid('y', area=phaseAreas[i], duration=pp.calc_duration(gxPre), system=self.system)
                    batches[batch_num].add_block(gyPre,gxPre)
                    for s in range(nSeg):  # Loop over segments
                        if s == 0:
                            batches[batch_num].add_block(gxp0, adc0, gyBlip_up)
                        else:
                            gx = gxm if s % 2 == 1 else gxp
                            if s != nSeg-1:
                                batches[batch_num].add_block(gx, adc, gyBlip_down_up)
                            else:
                                batches[batch_num].add_block(gx, adc, gyBlip_down)
                        assert n_rd_points + self.nPoints[0] < hw.maxRdPoints
                        n_rd_points = n_rd_points + self.nPoints[0]
                    batches[batch_num].add_block(delay_tr)
 
        
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
                plt.show()

            batches[batch_num].set_definition(key="Name", value="dw_epi")
            batches[batch_num].set_definition(key="FOV", value=self.fov)
            batches[batch_num].write(batch_num + ".seq")
            self.waveforms[batch_num], param_dict = self.flo_interpreter.interpret(batch_num + ".seq")
            print(f"{batch_num}.seq ready!")
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
        return True

        
    def sequenceAnalysis(self, mode=None):
        self.mode = mode
        self.etl = 1
        #self.axesOrientation = [0,1,2] # for ssfp
        self.unlock_orientation = 0 # for ssfp
        resolution = self.fov / self.nPoints
        self.mapVals['resolution'] = resolution

        # Get data
        data_full = self.mapVals['data_full']
        nRD, nPH, nSL = self.nPoints
        nRD = nRD + 2 * hw.addRdPoints
        n_batches = self.mapVals['n_batches']

        # Reorganize data_full
        data_prov = np.zeros([self.nScans, nRD * nPH * nSL], dtype=complex)
        if n_batches > 1:
            n_rds = self.mapVals['n_readouts']
            data_full_a = data_full[0:sum(n_rds[0:-1]) * self.nScans]
            data_full_b = data_full[sum(n_rds[0:-1]) * self.nScans:]
            data_full_a = np.reshape(data_full_a, newshape=(n_batches - 1, self.nScans, -1, nRD))
            data_full_b = np.reshape(data_full_b, newshape=(1, self.nScans, -1, nRD))
            for scan in range(self.nScans):
                data_scan_a = np.reshape(data_full_a[:, scan, :, :], -1)
                data_scan_b = np.reshape(data_full_b[:, scan, :, :], -1)
                data_prov[scan, :] = np.concatenate((data_scan_a, data_scan_b), axis=0)
        else:
            data_full = np.reshape(data_full, (1, self.nScans, -1, nRD))
            for scan in range(self.nScans):
                data_prov[scan, :] = np.reshape(data_full[:, scan, :, :], -1)
        
        # [TODO]: Add Rx phase here
        # expiangle = self.flo_interpreter.get_rx_phase_dict()['rx0']
        # raw_data = np.reshape(data_prov, newshape=(1, self.nScans, -1, nRD))
        # for scan in range(self.nScans):
        #     for line in range(raw_data.shape[2]):
        #         raw_data[0, scan, line, :] = raw_data[0, scan, line, :] * expiangle[line]
        # data_full = np.reshape(raw_data, -1)

        data_full = np.reshape(data_prov, -1)
        
        # Average data
        data_full = np.reshape(data_full, newshape=(self.nScans, -1))
        data = np.average(data_full, axis=0)
        self.mapVals['data'] = data

        # slice_idx = self.mapVals['sliceIdx']
        n_ex = int(np.floor(self.nPoints[1] / self.etl))
        # data_arrange_slice = np.zeros(shape=(nSL, n_ex, self.etl, nRD), dtype=complex)
        data_shape = np.reshape(data, newshape=(n_ex, nSL, self.etl, nRD))
        
        kdata_input = np.reshape(data_shape, newshape=(1, -1, nRD))
        data_ind = sort_data_implicit(kdata=kdata_input, seq=self.lastseq, shape=(nSL, nPH, nRD))
        data_ind = np.reshape(data_ind, newshape=(1, nSL, nPH, nRD))

        self.mapVals['kSpace'] = data_ind
        
        # Get images
        image_ind = np.zeros_like(data_ind)
        im = ifft_2d(data_ind[0])
        image_ind[0] = im

        # for echo in range(1):
        #     image_ind[echo] = np.fft.ifftshift(np.fft.ifftn(np.fft.ifftshift(data_ind[echo])))
        self.mapVals['iSpace'] = image_ind
        
        # Prepare data to plot (plot central slice)
        axes_dict = {'x': 0, 'y': 1, 'z': 2}
        axes_keys = list(axes_dict.keys())
        axes_vals = list(axes_dict.values())
        axes_str = ['', '', '']
        n = 0
        for val in self.axesOrientation:
            index = axes_vals.index(val)
            axes_str[n] = axes_keys[index]
            n += 1

        # Normalize image
        k_space = np.zeros((nSL, nPH, nRD - 2 * hw.addRdPoints))
        image = np.zeros(( nSL, nPH, nRD - 2 * hw.addRdPoints))

        
        n = 0
        for slice in range(nSL):
            for echo in range(1):
                k_space[n, :, :] = np.abs(data_ind[echo, slice, :, :])
                image[n, :, :] = np.abs(image_ind[echo, slice, :, :])
                n += 1
        image = image / np.max(image) * 100
        # plt.plot(np.real(k_space[0,0,:]))
        # plt.show()
        imageOrientation_dicom = [1.0, 0.0, 0.0, 0.0, 1.0, 0.0]
        if not self.unlock_orientation:  # Image orientation
            pass
            if self.axesOrientation[2] == 2:  # Sagittal
                title = "Sagittal"
                if self.axesOrientation[0] == 0 and self.axesOrientation[1] == 1:  # OK
                    image = np.flip(image, axis=2)
                    image = np.flip(image, axis=1)
                    k_space = np.flip(k_space, axis=2)
                    k_space = np.flip(k_space, axis=1)
                    x_label = "(-Y) A | PHASE | P (+Y)"
                    y_label = "(-X) I | READOUT | S (+X)"
                    imageOrientation_dicom = [0.0, 1.0, 0.0, 0.0, 0.0, -1.0]
                else:
                    image = np.transpose(image, (0, 2, 1))
                    image = np.flip(image, axis=2)
                    image = np.flip(image, axis=1)
                    k_space = np.transpose(k_space, (0, 2, 1))
                    k_space = np.flip(k_space, axis=2)
                    k_space = np.flip(k_space, axis=1)
                    x_label = "(-Y) A | READOUT | P (+Y)"
                    y_label = "(-X) I | PHASE | S (+X)"
                    imageOrientation_dicom = [0.0, 1.0, 0.0, 0.0, 0.0, -1.0]
            elif self.axesOrientation[2] == 1:  # Coronal
                title = "Coronal"
                if self.axesOrientation[0] == 0 and self.axesOrientation[1] == 2:  # OK
                    image = np.flip(image, axis=2)
                    image = np.flip(image, axis=1)
                    image = np.flip(image, axis=0)
                    k_space = np.flip(k_space, axis=2)
                    k_space = np.flip(k_space, axis=1)
                    k_space = np.flip(k_space, axis=0)
                    x_label = "(+Z) R | PHASE | L (-Z)"
                    y_label = "(-X) I | READOUT | S (+X)"
                    imageOrientation_dicom = [1.0, 0.0, 0.0, 0.0, 0.0, -1.0]
                else:
                    image = np.transpose(image, (0, 2, 1))
                    image = np.flip(image, axis=2)
                    image = np.flip(image, axis=1)
                    image = np.flip(image, axis=0)
                    k_space = np.transpose(k_space, (0, 2, 1))
                    k_space = np.flip(k_space, axis=2)
                    k_space = np.flip(k_space, axis=1)
                    k_space = np.flip(k_space, axis=0)
                    x_label = "(+Z) R | READOUT | L (-Z)"
                    y_label = "(-X) I | PHASE | S (+X)"
                    imageOrientation_dicom = [1.0, 0.0, 0.0, 0.0, 0.0, -1.0]
            elif self.axesOrientation[2] == 0:  # Transversal
                title = "Transversal"
                if self.axesOrientation[0] == 1 and self.axesOrientation[1] == 2:
                    image = np.flip(image, axis=2)
                    image = np.flip(image, axis=1)
                    k_space = np.flip(k_space, axis=2)
                    k_space = np.flip(k_space, axis=1)
                    x_label = "(+Z) R | PHASE | L (-Z)"
                    y_label = "(+Y) P | READOUT | A (-Y)"
                    imageOrientation_dicom = [1.0, 0.0, 0.0, 0.0, 1.0, 0.0]
                else:  # OK
                    image = np.transpose(image, (0, 2, 1))
                    image = np.flip(image, axis=2)
                    image = np.flip(image, axis=1)
                    k_space = np.transpose(k_space, (0, 2, 1))
                    k_space = np.flip(k_space, axis=2)
                    k_space = np.flip(k_space, axis=1)
                    x_label = "(+Z) R | READOUT | L (-Z)"
                    y_label = "(+Y) P | PHASE | A (-Y)"
                    imageOrientation_dicom = [1.0, 0.0, 0.0, 0.0, 1.0, 0.0]
        else:
            x_label = "%s axis" % axes_str[1]
            y_label = "%s axis" % axes_str[0]
            title = "Image"

        result1 = {'widget': 'image',
                   'data': image,
                   'xLabel': x_label,
                   'yLabel': y_label,
                   'title': title,
                   'row': 0,
                   'col': 0}

        result2 = {'widget': 'image',
                   'data': np.log10(k_space+0.01),
                   'xLabel': x_label,
                   'yLabel': y_label,
                   'title': "k_space",
                   'row': 0,
                   'col': 1}
 

        # Dicom tags
        image_DICOM = np.transpose(image, (0, 2, 1))
        slices, rows, columns = image_DICOM.shape
        self.meta_data["Columns"] = columns
        self.meta_data["Rows"] = rows
        self.meta_data["NumberOfSlices"] = slices
        self.meta_data["NumberOfFrames"] = slices
        img_full_abs = np.abs(image_DICOM) * (2 ** 15 - 1) / np.amax(np.abs(image_DICOM))
        img_full_int = np.int16(np.abs(img_full_abs))
        img_full_int = np.reshape(img_full_int, newshape=(slices, rows, columns))
        arr = img_full_int
        self.meta_data["PixelData"] = arr.tobytes()
        self.meta_data["WindowWidth"] = 26373
        self.meta_data["WindowCenter"] = 13194
        self.meta_data["ImageOrientationPatient"] = imageOrientation_dicom
        resolution = self.mapVals['resolution'] * 1e3
        self.meta_data["PixelSpacing"] = [resolution[0], resolution[1]]
        self.meta_data["SliceThickness"] = resolution[2]
        # Sequence parameters
        self.meta_data["RepetitionTime"] = self.mapVals['repetitionTime']
        self.meta_data["EchoTime"] = self.mapVals['echoTime']
        self.meta_data["FlipAngle"] = [self.mapVals['rfExFA'], self.mapVals['rfReFA']]
        self.meta_data["NumberOfAverages"] = self.mapVals['nScans']
        # self.meta_data["EchoTrainLength"] = self.mapVals['etl']
        

        self.meta_data["ScanningSequence"] = 'DW-EPI'

        # create self.out to run in iterative mode
        self.output = [result1, result2]

        # save data once self.output is created
        self.saveRawData()

        # Plot result in standalone execution
        if self.mode == 'Standalone':
            self.plotResults()

        return self.output
 
    
if __name__ == '__main__':
    seq = DWEPIPSEQ()
    seq.sequenceAtributes()
    seq.sequenceRun(plotSeq=True, demo=True, standalone=True)
    seq.sequenceAnalysis(mode='Standalone')




