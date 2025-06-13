"""
Created on Tuesday, Nov 18th 2024
@author: Kaisheng Lin, School of Electronics, Peking University, China
@Summary: TSE sequence (RARE), implemented with PyPulseq and compatible with MaSeq.
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

class TSE3DPSEQ(blankSeq.MRIBLANKSEQ):
    def __init__(self):
        super(TSE3DPSEQ, self).__init__()
        # Input the parameters
        self.output = None
        self.expt = None
        self.nScans = None
        self.larmorFreq = None
        self.rfExFA = None
        self.rfReFA = None
        self.rfExTime = None
        self.rfReTime = None
        self.inversionTime = None
        self.preExTime = None
        self.repetitionTime = None
        self.echoSpacing = None
        self.fov = None
        self.dfov = None
        self.nPoints = None
        self.axesOrientation = None
        self.riseTime = None
        self.bandwidth = None
        self.DephTime = None
        self.shimming = None
        self.etl = None
        self.angle = None
        self.rotationsAxis = None
        self.dummyPulses = None
        self.sweepMode = None
        
        self.addParameter(key='seqName', string='tse', val='tse')
        self.addParameter(key='nScans', string='Number of scans', val=1, field='IM')
        self.addParameter(key='larmorFreq', string='Larmor frequency (MHz)', val=10.33307, units=units.MHz, field='IM')
        self.addParameter(key='rfExFA', string='Excitation flip angle (deg)', val=90, field='RF')
        self.addParameter(key='rfReFA', string='Refocusing flip angle (deg)', val=180, field='RF')
        self.addParameter(key='rfExTime', string='RF excitation time (us)', val=400.0, units=units.us, field='RF')
        self.addParameter(key='rfReTime', string='RF refocusing time (us)', val=800.0, units=units.us, field='RF')
        self.addParameter(key='repetitionTime', string='Repetition time (ms)', val=1000.0, units=units.ms, field='SEQ')
        self.addParameter(key='preExTime', string='Preexitation time (ms)', val=0.0, units=units.ms, field='SEQ')
        self.addParameter(key='inversionTime', string='Inversion time (ms)', val=0.0, units=units.ms, field='SEQ', tip="0 to ommit this pulse")
        
        self.addParameter(key='fov', string='FOV[x,y,z] (mm)', val=[100.0, 100.0, 10.0], units=units.mm, field='IM')
        self.addParameter(key='dfov', string='dFOV[x,y,z] (mm)', val=[0.0, 0.0, 0.0], units=units.mm, field='IM',
                          tip="Position of the gradient isocenter")
        self.addParameter(key='nPoints', string='nPoints[rd, ph, sl]', val=[128, 128, 10], field='IM')
        self.addParameter(key='axesOrientation', string='Axes[rd,ph,sl]', val=[1,2,0], field='IM',
                          tip="0=x, 1=y, 2=z")
        self.addParameter(key='angle', string='Angle (degree)', val=0.0, field='IM')
        self.addParameter(key='rotationAxis', string='Rotation axis', val=[0, 0, 1], field='IM')
        self.addParameter(key='bandwidth', string='Acquisition Bandwidth (kHz)', val=32., units=units.kHz, field='IM',
                          tip="The bandwidth of the acquisition (kHz). This value affects resolution and SNR.")
        self.addParameter(key='DephTime', string='Dephasing time (ms)', val=1.0, units=units.ms, field='OTH')
        self.addParameter(key='riseTime', string='Grad. rising time (ms)', val=0.10, units=units.ms, field='OTH')
        self.addParameter(key='shimming', string='Shimming', val=[0.0015, 0.0020, 0.0015], field='SEQ')
        self.addParameter(key='etl', string='Echo train length', val=8, field='SEQ')
        self.addParameter(key='echoSpacing', string='Echo Spacing (ms)', val=10.0, units=units.ms, field='SEQ')
        self.addParameter(key='dummyPulses', string='Dummy pulses', val=0, field='SEQ', tip="Use last dummy pulse to calibrate k = 0")
        self.addParameter(key='sweepMode', string='Sweep mode', val=1, field='SEQ', tip="0: sweep from -kmax to kmax. 1: sweep from 0 to kmax. 2: sweep from kmax to 0")
        
     

    def sequenceTime(self):
        return (self.mapVals['repetitionTime'] *1e-3 * 
                self.mapVals['nScans'] *
                self.mapVals['nPoints'][1] *
                self.mapVals['nPoints'][2]
                / self.mapVals['etl'] / 60)
        

    def sequenceAtributes(self):
        super().sequenceAtributes()
        # Conversion of variables to non-multiplied units
        self.angle = self.angle * np.pi / 180 # rads

        # Add rotation, dfov and fov to the history
        self.rotation = self.rotationAxis.tolist()
        self.rotation.append(self.angle)
        self.rotations.append(self.rotation)
        self.dfovs.append(self.dfov.tolist())
        self.fovs.append(self.fov.tolist())

    def sequenceRun(self, plotSeq=0, demo=False, standalone=False):
        init_gpa = False
        self.demo = demo
        self.plotSeq = plotSeq
        self.standalone = standalone
        
        max_grad_Hz = convert(from_value=hw.max_grad, from_unit='mT/m', gamma=hw.gammaB, to_unit='Hz/m')
        
        rfExTime_us = int(np.round(self.rfExTime * 1e6))
        rfReTime_us = int(np.round(self.rfReTime * 1e6))
        assert rfExTime_us in hw.max_cpmg_rf_arr, f"RF excitation time '{rfExTime_us}' s is not found in the hw_config_pseq file; please search it in search_p90_pseq."
        assert rfReTime_us in hw.max_cpmg_rf_p180_arr, f"RF refocusing time '{rfReTime_us}' s is not found in the hw_config_pseq file; please search it in search_p180_pseq."
        max_rf_Hz = hw.max_cpmg_rf_arr[rfExTime_us] * 1e-6 * hw.gammaB
       
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
            use_grad_preemphasis=False, 
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
            rf_dead_time=0 * 1e-6,  # Dead time between RF pulses (s)
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
        self.internalAxesOrientation = [0, 1, 2]  # Default axes orientation
        # Set the fov
        self.dfov = self.getFovDisplacement()
        self.dfov = self.dfov[self.internalAxesOrientation]
        self.fov = self.fov[self.internalAxesOrientation]

        # Check for used axes
        axes_enable = []
        for ii in range(3):
            if self.nPoints[ii] == 1:
                axes_enable.append(0)
            else:
                axes_enable.append(1)
        self.mapVals['axes_enable'] = axes_enable
        resolution = self.fov / self.nPoints
        self.mapVals['resolution'] = resolution
        
        
        # Matrix size
        n_rd = self.nPoints[0]
        n_ph = self.nPoints[1]
        n_sl = self.nPoints[2]

        # ETL if etl>n_ph
        if self.etl > n_ph:
            self.etl = n_ph

        # Miscellaneous
        n_rd_points_per_train = self.etl * n_rd

        # par_acq_lines in case par_acq_lines = 0
        par_acq_lines = int(int(self.nPoints[2])-self.nPoints[2]/2)
        self.mapVals['partialAcquisition'] = par_acq_lines

        bw = self.bandwidth * 1e-6 # MHz
        bw_ov = self.bandwidth * 1e-6 
        sampling_period = 1 / bw_ov  # us, Dwell time
        acqTime = n_rd * sampling_period * 1e-6
        rdGradTime = acqTime + 2 * self.system.adc_dead_time  # s, Readout gradient time
        phGradTime = self.echoSpacing/2-self.rfReTime/2-1*self.riseTime - acqTime/2

        # Max gradient amplitude
        rd_grad_amplitude = self.nPoints[0] / (hw.gammaB * self.fov[0] * acqTime) 
        ph_grad_amplitude = n_ph / (2 * hw.gammaB * self.fov[1] * (phGradTime + self.riseTime)) 
        sl_grad_amplitude = n_sl / (2 * hw.gammaB * self.fov[2] * (phGradTime + self.riseTime)) 
        self.mapVals['rd_grad_amplitude'] = rd_grad_amplitude
        self.mapVals['ph_grad_amplitude'] = ph_grad_amplitude
        self.mapVals['sl_grad_amplitude'] = sl_grad_amplitude

        # Readout dephasing amplitude
        rd_deph_amplitude = 0.5 * rd_grad_amplitude * (self.riseTime+self.DephTime) / (self.riseTime + self.DephTime)
        self.mapVals['rd_deph_amplitude'] = rd_deph_amplitude
        print("Max rd gradient amplitude: %0.1f mT/m" % (max(rd_grad_amplitude, rd_deph_amplitude) * 1e3))
        print("Max ph gradient amplitude: %0.1f mT/m" % (ph_grad_amplitude * 1e3))
        print("Max sl gradient amplitude: %0.1f mT/m" % (sl_grad_amplitude * 1e3))

        # Phase and slice gradient vector
        ph_gradients = np.linspace(-ph_grad_amplitude, ph_grad_amplitude, num=n_ph, endpoint=False)
        sl_gradients = np.linspace(-sl_grad_amplitude, sl_grad_amplitude, num=n_sl, endpoint=False)

        # Now fix the number of slices to partially acquired k-space
        n_sl = (int(self.nPoints[2] / 2) + par_acq_lines) * axes_enable[2]+(1-axes_enable[2])
        print("Number of acquired slices: %i" % n_sl)

        # Set phase vector to given sweep mode
        ind = self.getIndex(self.etl, n_ph, self.sweepMode)
        self.mapVals['sweepOrder'] = ind
        ph_gradients = ph_gradients[ind]
        self.mapVals['ph_gradients'] = ph_gradients.copy()
        self.mapVals['sl_gradients'] = sl_gradients.copy()

        # Normalize gradient list
        if ph_grad_amplitude != 0:
            ph_gradients /= ph_grad_amplitude
        if sl_grad_amplitude != 0:
            sl_gradients /= sl_grad_amplitude

        # Get the rotation matrix
        rot = self.getRotationMatrix()
        grad_amp = np.array([0.0, 0.0, 0.0])
        grad_amp[self.internalAxesOrientation[0]] = 1
        grad_amp = np.reshape(grad_amp, (3, 1))
        result = np.dot(rot, grad_amp)

        # Map the axis to "x", "y", and "z" according ot axesOrientation
        axes_map = {0: "x", 1: "y", 2: "z"}
        rd_channel = axes_map.get(self.internalAxesOrientation[0], "")
        ph_channel = axes_map.get(self.internalAxesOrientation[1], "")
        sl_channel = axes_map.get(self.internalAxesOrientation[2], "")
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
        sampling_time = sampling_period * n_rd * 1e-6  # s
            
        self.mapVals['bw_MHz'] = bw
        self.mapVals['sampling_period_us'] = sampling_period
        self.mapVals['sampling_time_s'] = sampling_time

        '''
        Step 5: Define sequence blocks.
        In this step, you will define the building blocks of the MRI sequence, including the RF pulses and gradient pulses.
        '''
        # First delay, sequence will start after 1 repetition time, this ensure gradient and ADC latency is not an issue.
        if self.inversionTime==0 and self.preExTime==0:
            delay = self.repetitionTime - self.rfExTime / 2 - self.system.rf_dead_time
        elif self.inversionTime>0 and self.preExTime==0:
            delay = self.repetitionTime - self.inversionTime - self.rfReTime / 2 - self.system.rf_dead_time
        elif self.inversionTime==0 and self.preExTime>0:
            delay = self.repetitionTime - self.preExTime - self.rfExTime / 2 - self.system.rf_dead_time
        else:
            delay = self.repetitionTime - self.preExTime - self.inversionTime - self.rfExTime / 2 - self.system.rf_dead_time
        delay_first = pp.make_delay(delay)

        # ADC to get noise
        delay = 100e-6
        block_adc_noise = pp.make_adc(
            num_samples=n_rd,
            dwell=sampling_period * 1e-6,
            delay=delay,
        )

        # Pre-excitation pulse
        if self.preExTime>0:
            flip_pre = self.rfExFA * np.pi / 180
            delay = 0
            block_rf_pre_excitation = pp.make_block_pulse(
                flip_angle=flip_pre,
                system=self.system,
                duration=self.rfExTime,
                phase_offset=0.0,
                delay=0,
            )
            if self.inversionTime==0:
                delay = self.preExTime
            else:
                delay = self.rfExTime / 2 - self.rfReTime / 2 + self.preExTime
            delay_pre_excitation = pp.make_delay(delay)

        # Inversion pulse
        if self.inversionTime>0:
            flip_inv = self.rfReFA * np.pi / 180
            block_rf_inversion = pp.make_block_pulse(
                flip_angle=flip_inv,
                system=self.system,
                duration=self.rfReTime,
                phase_offset=0,
                delay=0,
                use='inversion'
            )
            delay = self.rfReTime / 2 - self.rfExTime / 2 + self.inversionTime
            delay_inversion = pp.make_delay(delay)

        # Excitation pulse
        flip_ex = self.rfExFA * np.pi / 180
        block_rf_excitation = pp.make_block_pulse(
            flip_angle=flip_ex,
            system=self.system,
            duration=self.rfExTime,
            phase_offset=0.0,
            delay=0.0,
            use='excitation'
        )

       

        # Delay to re-focusing pulse
        delay_preph = pp.make_delay(self.echoSpacing / 2 + self.rfExTime / 2 - self.rfReTime / 2)

        # Refocusing pulse
        flip_re = self.rfReFA * np.pi / 180
        block_rf_refocusing = pp.make_block_pulse(
            flip_angle=flip_re,
            system=self.system,
            duration=self.rfReTime,
            phase_offset=np.pi / 2,
            delay=0,
            use='refocusing'
        )

        # Delay to next refocusing pulse
        delay_reph = pp.make_delay(self.system.rf_dead_time+self.echoSpacing/2+rdGradTime+self.riseTime)

        # Phase gradient de-phasing
        delay = self.system.rf_dead_time + self.rfReTime
        block_gr_ph_deph = pp.make_trapezoid(
            channel=ph_channel,
            system=self.system,
            amplitude=ph_grad_amplitude * hw.gammaB + float(ph_grad_amplitude==0),
            flat_time=phGradTime-2*self.riseTime,
            rise_time=self.riseTime,
            delay=delay,
        )

        # Slice gradient de-phasing
        delay = self.system.rf_dead_time + self.rfReTime
        block_gr_sl_deph = pp.make_trapezoid(
            channel=sl_channel,
            system=self.system,
            amplitude=sl_grad_amplitude * hw.gammaB + float(sl_grad_amplitude==0),
            flat_time=phGradTime-2*self.riseTime,
            delay=delay,
            rise_time=self.riseTime,
        )

        # Readout gradient
        delay = self.system.rf_dead_time + self.rfReTime / 2 + self.echoSpacing / 2 - rdGradTime / 2 - \
                self.riseTime
        assert delay >= 0, f"Delay readout gradient is negative: {delay} s. Please check the parameters."
        block_gr_rd_reph = pp.make_trapezoid(
            channel=rd_channel,
            system=self.system,
            amplitude=rd_grad_amplitude * hw.gammaB,
            flat_time=rdGradTime,
            rise_time=self.riseTime,
            delay=delay,
        )
        
         # De-phasing gradient
        delay = self.system.rf_dead_time + self.rfExTime
        block_gr_rd_preph = pp.make_trapezoid(
            channel=rd_channel,
            system=self.system,
            area=block_gr_rd_reph.area/2,
            duration=sampling_time/2,
            rise_time=self.riseTime,
            delay=delay,
        )
        # ADC to get the signal
        delay = self.system.rf_dead_time + self.rfReTime / 2 + self.echoSpacing / 2 - sampling_time / 2
        assert delay >= 0, f"Delay ADC is negative: {delay} s. Please check the parameters."
        block_adc_signal = pp.make_adc(
            num_samples=n_rd,
            dwell=sampling_period * 1e-6,
            delay=delay,
        )

        # Phase gradient re-phasing
        delay = 0 
        assert delay >= 0, f"Delay phase gradient re-phasing is negative: {delay} s. Please check the parameters."
        block_gr_ph_reph = pp.make_trapezoid(
            channel=ph_channel,
            system=self.system,
            amplitude=ph_grad_amplitude * hw.gammaB + float(ph_grad_amplitude==0),
            flat_time=phGradTime-2*self.riseTime,
            rise_time=self.riseTime,
            delay=delay,
        )

        # Slice gradient re-phasing
        delay = 0 
        assert delay >= 0, f"Delay slice gradient re-phasing is negative: {delay} s. Please check the parameters."
        block_gr_sl_reph = pp.make_trapezoid(
            channel=sl_channel,
            system=self.system,
            amplitude=sl_grad_amplitude * hw.gammaB + float(sl_grad_amplitude==0),
            flat_time=phGradTime-2*self.riseTime,
            rise_time=self.riseTime,
            delay=delay,
        )

        # Delay TR
        delay = self.repetitionTime + self.rfReTime / 2 - self.rfExTime / 2 - (self.etl + 0.5) * self.echoSpacing - \
            self.inversionTime - self.preExTime
        if self.inversionTime > 0 and self.preExTime == 0:
            delay -= self.rfExTime / 2
        assert delay >= 0, f"Delay TR is negative: {delay} s. Please check the parameters."
        delay_tr = pp.make_delay(delay)
        
        '''
        # Step 6: Define your initializeBatch according to your sequence.
        # In this step, you will create the initializeBatch method to create dummy pulses that will be initialized for
        # each new batch.
        '''
        def runBatches(waveforms, n_readouts, frequency=hw.larmorFreq, bandwidth=0.03):
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
            self.mapVals['n_readouts'] = list(n_readouts.values())
            self.mapVals['n_batches'] = len(n_readouts.values())

            # Initialize a list to hold oversampled data
            data_over = []
            self.mapVals['data_full'] = []
        
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
                        auto_leds=True  # Automatic control of LEDs
                    )
                print(f"Center frequecy set: {frequency} MHz")
                
                # Convert the PyPulseq waveform to the Red Pitaya compatible format
                self.pypulseq2mriblankseq(waveforms=waveforms[seq_num], shimming=self.shimming)

                # Load the waveforms into Red Pitaya
                if not self.floDict2Exp():
                    print("ERROR: Sequence waveforms out of hardware bounds")
                    return False
                else:
                    encoding_ok = True
                if self.plotSeq and not self.demo:
                    self.expt.plot_sequence()

                # If not plotting the sequence, start scanning
                if not self.plotSeq:
                    for scan in range(self.nScans):
                        print(f"Scan {scan + 1}, batch {seq_num.split('_')[-1]}/{len(n_readouts)} running...")
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
                                print(f"WARNING: data points lost! (acquired: {acquired_points}, expected: {expected_points}) Repeating batch...")
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
    
        def initialize_batch():
            """
            Initializes a batch of MRI sequence blocks using PyPulseq for a given experimental configuration.

            Returns:
            --------
            tuple
                - `batch` (pp.Sequence): A PyPulseq sequence object containing the configured sequence blocks.
                - `n_rd_points` (int): Total number of readout points in the batch.
                - `n_adc` (int): Total number of ADC acquisitions in the batch.

            Workflow:
            ---------
            1. **Create PyPulseq Sequence Object**:
                - Instantiates a new PyPulseq sequence object (`pp.Sequence`) and initializes counters for
                  readout points (`n_rd_points`) and ADC events (`n_adc`).

            2. **Set Gradients to Zero**:
                - Initializes slice and phase gradients (`gr_ph_deph`, `gr_sl_deph`, `gr_ph_reph`, `gr_sl_reph`) to zero
                  by scaling predefined gradient blocks with a factor of 0.

            3. **Add Initial Delay and Noise Measurement**:
                - Adds an initial delay block (`delay_first`) and a noise measurement ADC block (`block_adc_noise`)
                  to the sequence.

            4. **Generate Dummy Pulses**:
                - Creates a specified number of dummy pulses (`self.dummyPulses`) to prepare the system for data acquisition:
                    - **Pre-excitation Pulse**:
                        - If `self.preExTime > 0`, adds a pre-excitation pulse with a readout pre-phasing gradient.
                    - **Inversion Pulse**:
                        - If `self.inversionTime > 0`, adds an inversion pulse with a scaled readout pre-phasing gradient.
                    - **Excitation Pulse**:
                        - Adds an excitation pulse followed by a readout de-phasing gradient (`block_gr_rd_preph`).

                - For each dummy pulse:
                    - **Echo Train**:
                        - For the last dummy pulse, appends an echo train that includes:
                            - A refocusing pulse.
                            - Gradients for readout re-phasing, phase de-phasing, and slice de-phasing.
                            - ADC signal acquisition block (`block_adc_signal`).
                            - Gradients for phase and slice re-phasing.
                        - For other dummy pulses, excludes the ADC signal acquisition.

                    - **Repetition Time Delay**:
                        - Adds a delay (`delay_tr`) to separate repetitions.

            5. **Return Results**:
                - Returns the configured sequence (`batch`), total readout points (`n_rd_points`), and number of ADC events (`n_adc`).

            """
            # Instantiate pypulseq sequence object
            batch = pp.Sequence(self.system)
            n_rd_points = 0
            n_adc = 0

            # Set slice and phase gradients to 0
            gr_ph_deph = pp.scale_grad(block_gr_ph_deph, scale=0.0)
            gr_sl_deph = pp.scale_grad(block_gr_sl_deph, scale=0.0)
            gr_ph_reph = pp.scale_grad(block_gr_ph_reph, scale=0.0)
            gr_sl_reph = pp.scale_grad(block_gr_sl_reph, scale=0.0)

            # Add first delay and first noise measurement
            batch.add_block(delay_first)#, block_adc_noise)
            # n_rd_points += n_rd
            n_adc += 1

            # Create dummy pulses
            for dummy in range(self.dummyPulses):
                # Pre-excitation pulse
                if self.preExTime>0:
                    gr_rd_preex = pp.scale_grad(block_gr_rd_preph, scale=1.0)
                    batch.add_block(block_rf_pre_excitation,
                                            gr_rd_preex,
                                            delay_pre_excitation)

                # Inversion pulse
                if self.inversionTime>0:
                    gr_rd_inv = pp.scale_grad(block_gr_rd_preph, scale=-1.0)
                    batch.add_block(block_rf_inversion,
                                            gr_rd_inv,
                                            delay_inversion)

                # Add excitation pulse and readout de-phasing gradient
                batch.add_block(block_gr_rd_preph,
                                        block_rf_excitation,
                                        delay_preph)

                # Add echo train
                for echo in range(self.etl):
                    if dummy == self.dummyPulses-1:
                        batch.add_block(block_rf_refocusing,
                                                block_gr_rd_reph,
                                                gr_ph_deph,
                                                gr_sl_deph,
                                                block_adc_signal,
                                                delay_reph)
                        batch.add_block(gr_ph_reph,
                                                gr_sl_reph)
                        n_rd_points += n_rd
                        n_adc += 1
                    else:
                        batch.add_block(block_rf_refocusing,
                                                block_gr_rd_reph,
                                                gr_ph_deph,
                                                gr_sl_deph,
                                                delay_reph)
                        batch.add_block(gr_ph_reph,
                                                gr_sl_reph)

                # Add time delay to next repetition
                batch.add_block(delay_tr)

            return batch, n_rd_points, n_adc

        '''
        Step 7: Define your createBatches method.
        In this step you will populate the batches adding the blocks previously defined in step 4, and accounting for
        number of acquired points to check if a new batch is required.
        '''
        def createBatches():
            """
            Creates and processes multiple batches of MRI sequence blocks for slice and phase encoding sweeps.

            Returns:
            --------
            tuple
                - `waveforms` (dict): Dictionary of interpreted waveform data for each batch.
                - `n_rd_points_dict` (dict): Dictionary mapping each batch to its total number of readout points.
                - `n_adc` (int): Total number of ADC acquisitions across all batches.

            Workflow:
            ---------
            1. **Initialization**:
                - Initializes dictionaries to store batches (`batches`), waveforms (`waveforms`), and readout points (`n_rd_points_dict`).
                - Tracks the current readout point count (`n_rd_points`), ADC window count (`n_adc`), and batch index (`seq_idx`).

            2. **Slice and Phase Sweep**:
                - Iterates over slices (`n_sl`) and phases (`n_ph`) to build and organize sequence blocks:
                    - **Batch Management**:
                        - Creates a new batch if no batch exists or the current batch exceeds the hardware limit (`hw.maxRdPoints`).
                        - Writes the previous batch to disk, interprets it using `flo_interpreter`, and updates dictionaries.
                        - Initializes the next batch with `initializeBatch()`.

                    - **Pre-Excitation and Inversion Pulses**:
                        - Optionally adds a pre-excitation pulse (`block_rf_pre_excitation`) and an inversion pulse (`block_rf_inversion`), if respective times (`self.preExTime`, `self.inversionTime`) are greater than zero.

                    - **Excitation and Echo Train**:
                        - Adds an excitation pulse followed by an echo train for phase and slice gradients:
                            - Gradients are scaled based on the current slice (`sl_idx`) and phase (`ph_idx`) indices.
                            - Includes ADC acquisition blocks (`block_adc_signal`) and refocusing pulses for each echo.

                    - **Repetition Time Delay**:
                        - Adds a delay (`delay_tr`) between repetitions.

            3. **Final Batch Processing**:
                - Writes and interprets the last batch after completing all slices and phases.
                - Updates the total readout points for the final batch.

            Returns:
            --------
            - `waveforms`: Interpreted waveforms for each batch, generated using the `flo_interpreter`.
            - `n_rd_points_dict`: Maps batch names to the total readout points per batch.
            - `n_adc`: Total number of ADC acquisition windows across all batches.
            """
            
            batches = {}  # Dictionary to save batches PyPulseq sequences
            waveforms = {}  # Dictionary to store generated waveforms per each batch
            n_rd_points_dict = {}  # Dictionary to track readout points for each batch
            n_rd_points = 0  # To account for number of acquired rd points
            seq_idx = 0  # Sequence batch index
            n_adc = 0  # To account for number of adc windows
            batch_num = "batch_0"  # Initial batch name

            # Slice sweep
            for sl_idx in range(n_sl):
                ph_idx = 0
                # Phase sweep
                while ph_idx < n_ph:
                    # Check if a new batch is needed (either first batch or exceeding readout points limit)
                    if seq_idx == 0 or n_rd_points + n_rd_points_per_train > hw.maxRdPoints:
                        # If a previous batch exists, write and interpret it
                        if seq_idx > 0:
                            batches[batch_num].write(batch_num + ".seq")
                            waveforms[batch_num], param_dict = self.flo_interpreter.interpret(batch_num + ".seq")
                            print(f"{batch_num}.seq ready!")

                        # Update to the next batch
                        seq_idx += 1
                        n_rd_points_dict[batch_num] = n_rd_points  # Save readout points count
                        n_rd_points = 0
                        batch_num = f"batch_{seq_idx}"
                        batches[batch_num], n_rd_points, n_adc_0 = initialize_batch()  # Initialize new batch
                        n_adc += n_adc_0
                        print(f"Creating {batch_num}.seq...")

                    # Pre-excitation pulse
                    if self.preExTime > 0:
                        gr_rd_preex = pp.scale_grad(block_gr_rd_preph, scale=+1.0)
                        batches[batch_num].add_block(block_rf_pre_excitation,
                                                     gr_rd_preex,
                                                     delay_pre_excitation)

                    # Inversion pulse
                    if self.inversionTime > 0:
                        gr_rd_inv = pp.scale_grad(block_gr_rd_preph, scale=-1.0)
                        batches[batch_num].add_block(block_rf_inversion,
                                                     gr_rd_inv,
                                                     delay_inversion)

                    # Add excitation pulse and readout de-phasing gradient
                    batches[batch_num].add_block(block_gr_rd_preph,
                                            block_rf_excitation,
                                            delay_preph)

                    # Add echo train
                    for echo in range(self.etl):
                        # Fix the phase and slice amplitude
                        gr_ph_deph = pp.scale_grad(block_gr_ph_deph, ph_gradients[ph_idx])
                        gr_sl_deph = pp.scale_grad(block_gr_sl_deph, sl_gradients[sl_idx])
                        gr_ph_reph = pp.scale_grad(block_gr_ph_reph, - ph_gradients[ph_idx])
                        gr_sl_reph = pp.scale_grad(block_gr_sl_reph, - sl_gradients[sl_idx])

                        # Add blocks
                        batches[batch_num].add_block(block_rf_refocusing,
                                                block_gr_rd_reph,
                                                gr_ph_deph,
                                                gr_sl_deph,
                                                block_adc_signal,
                        )#delay_reph)
                        batches[batch_num].add_block(gr_ph_reph,
                                                gr_sl_reph,
                                                
                                                #pp.make_delay(phGradTime + 2 * self.riseTime)
)
                        n_rd_points += n_rd
                        n_adc += 1
                        ph_idx += 1

                    # Add time delay to next repetition
                    batches[batch_num].add_block(delay_tr)
            # After final repetition, save and interpret the last batch
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
                
                batches[batch_num].plot()
                #print(batches[batch_num].test_report())
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

            batches[batch_num].set_definition(key="Name", value="tse3d")
            batches[batch_num].set_definition(key="FOV", value=self.fov)
            batches[batch_num].write(batch_num + ".seq")
            waveforms[batch_num], param_dict = self.flo_interpreter.interpret(batch_num + ".seq")
            print(f"{batch_num}.seq ready!")
            print(f"{len(batches)} batches created with {n_rd_points} read points. Sequence ready!")

            # Update the number of acquired points in the last batch
            n_rd_points_dict.pop('batch_0')
            n_rd_points_dict[batch_num] = n_rd_points
            self.lastseq = batches[batch_num]
            return waveforms, n_rd_points_dict, n_adc
        '''
        Step 8: Run the batches
        This step will handle the different batches, run it and get the resulting data. This should not be modified.
        Oversampled data will be available in self.mapVals['data_over']
        '''
        waveforms, n_readouts, n_adc = createBatches()
        assert runBatches(waveforms=waveforms,
                            n_readouts=n_readouts,
                            frequency=self.larmorFreq*1e-6,  # MHz
                            bandwidth=bw,  # MHz
                            )
        self.mapVals['n_readouts'] = list(n_readouts)
        self.mapVals['n_batches'] = 1
        return True

        
    def sequenceAnalysis(self, mode=None):
        self.mode = mode
 
        #self.axesOrientation = [0,1,2] # for ssfp
        self.unlock_orientation = 0 # for ssfp
        resolution = self.fov / self.nPoints
        self.mapVals['resolution'] = resolution

        # Get data
        data_full_pre = self.mapVals['data_full']
        n_rd, n_ph, n_sl = self.nPoints
        # The code is incrementing the variable `nRD` by twice the value of `hw.addRdPoints`.
        n_rd = n_rd + 2 * hw.addRdPoints
        n_batches = self.mapVals['n_batches']
        ind = self.getParameter('sweepOrder')
        # fir decimator
        if self.flo_interpreter._fir_decimation_rate > 1:
            data_waiting_for_fir = np.reshape(data_full_pre, newshape=(-1, self.flo_interpreter._fir_decimation_rate * n_rd))
            data_full = self.flo_interpreter.fir_decimator(input_matrix=data_waiting_for_fir, decimation_rate=3)
        else:
            data_full = data_full_pre

        # Reorganize data_full
        data_prov = np.zeros([self.nScans, n_rd * n_ph * n_sl], dtype=complex)
        if n_batches > 1:
            n_rds = self.mapVals['n_readouts']
            data_full_a = data_full[0:sum(n_rds[0:-1]) * self.nScans]
            data_full_b = data_full[sum(n_rds[0:-1]) * self.nScans:]
            data_full_a = np.reshape(data_full_a, newshape=(n_batches - 1, self.nScans, -1, n_rd))
            data_full_b = np.reshape(data_full_b, newshape=(1, self.nScans, -1, n_rd))
            for scan in range(self.nScans):
                data_scan_a = np.reshape(data_full_a[:, scan, :, :], -1)
                data_scan_b = np.reshape(data_full_b[:, scan, :, :], -1)
                data_prov[scan, :] = np.concatenate((data_scan_a, data_scan_b), axis=0)
        else:
            data_full = np.reshape(data_full, (1, self.nScans, -1, n_rd))
            for scan in range(self.nScans):
                data_prov[scan, :] = np.reshape(data_full[:, scan, :, :], -1)
        
        


        # Average data
        data_prov = np.reshape(data_full, newshape=(self.nScans, n_rd * n_ph * n_sl))
        data_prov = np.average(data_prov, axis=0)
        # Reorganize the data according to sweep mode
        data_prov = np.reshape(data_prov, newshape=(n_sl, n_ph, n_rd))
        data_temp = np.zeros_like(data_prov)
        for ii in range(n_ph):
            data_temp[:, ind[ii], :] = data_prov[:, ii, :]
        data_prov = data_temp
        # Get central line
        # data_prov = data_prov[int(self.nPoints[2] / 2), int(n_ph / 2), :]
        # ind_krd_0 = np.argmax(np.abs(data_prov))
        # if ind_krd_0 < n_rd / 2 - hw.addRdPoints or ind_krd_0 > n_rd / 2 + hw.addRdPoints:
        #     ind_krd_0 = int(n_rd / 2)

        # Get individual images
        # data_full = np.reshape(data_full, newshape=(self.nScans, n_sl, n_ph, n_rd))
        # data_full = data_full[:, :, :, ind_krd_0 - int(self.nPoints[0] / 2):ind_krd_0 + int(self.nPoints[0] / 2)]
        # data_temp = np.zeros_like(data_full)
        # for ii in range(n_ph):
        #     data_temp[:, :, ind[ii], :] = data_full[:, :, ii, :]
        # data_full = data_temp
        # self.mapVals['data_full'] = data_full

        data_full = data_prov
        self.mapVals['data_full'] = data_full

        
        # Do zero padding
        # data_temp = np.zeros(shape=(self.nPoints[2], self.nPoints[1], self.nPoints[0]), dtype=complex)
        # data_temp[0:n_sl, :, :] = data
        # data = np.reshape(data_temp, newshape=(1, self.nPoints[0] * self.nPoints[1] * self.nPoints[2]))

        # No zero-padding, just reshape to 1D for phase correction
        data = np.reshape(data_full, newshape=(1, n_sl * n_ph * n_rd))

        # Fix the position of the sample according to dfov
        bw = self.getParameter('bw_MHz')
        time_vector = np.linspace(-self.nPoints[0] / bw / 2 + 0.5 / bw, self.nPoints[0] / bw / 2 - 0.5 / bw,
                                       self.nPoints[0]) * 1e-6  # s
        kMax = np.array(self.nPoints) / (2 * np.array(self.fov)) * np.array(self.mapVals['axes_enable'])
        kRD = time_vector * hw.gammaB * self.getParameter('rd_grad_amplitude')
        kPH = np.linspace(-kMax[1], kMax[1], num=self.nPoints[1], endpoint=False)
        kSL = np.linspace(-kMax[2], kMax[2], num=self.nPoints[2], endpoint=False)
        kPH, kSL, kRD = np.meshgrid(kPH, kSL, kRD)
        kRD = np.reshape(kRD, newshape=(1, self.nPoints[0] * self.nPoints[1] * self.nPoints[2]))
        kPH = np.reshape(kPH, newshape=(1, self.nPoints[0] * self.nPoints[1] * self.nPoints[2]))
        kSL = np.reshape(kSL, newshape=(1, self.nPoints[0] * self.nPoints[1] * self.nPoints[2]))
        dPhase = np.exp(-2 * np.pi * 1j * (self.dfov[0] * kRD + self.dfov[1] * kPH + self.dfov[2] * kSL))
        data = np.reshape(data * dPhase, newshape=(self.nPoints[2], self.nPoints[1], self.nPoints[0]))
        self.mapVals['kSpace3D'] = data
        img = np.fft.ifftshift(np.fft.ifftn(np.fft.ifftshift(data)))
        self.mapVals['image3D'] = img
        data = np.reshape(data, newshape=(1, self.nPoints[0] * self.nPoints[1] * self.nPoints[2]))

        # Create sampled data
        kRD = np.reshape(kRD, newshape=(self.nPoints[0] * self.nPoints[1] * self.nPoints[2], 1))
        kPH = np.reshape(kPH, newshape=(self.nPoints[0] * self.nPoints[1] * self.nPoints[2], 1))
        kSL = np.reshape(kSL, newshape=(self.nPoints[0] * self.nPoints[1] * self.nPoints[2], 1))
        data = np.reshape(data, newshape=(self.nPoints[0] * self.nPoints[1] * self.nPoints[2], 1))
        self.mapVals['kMax_1/m'] = kMax
        self.mapVals['sampled'] = np.concatenate((kRD, kPH, kSL, data), axis=1)
        self.mapVals['sampledCartesian'] = self.mapVals['sampled']  # To sweep
        data = np.reshape(data, newshape=(self.nPoints[2], self.nPoints[1], self.nPoints[0]))

        axes_enable = self.mapVals['axes_enable']

        # Get axes in strings
        axes = self.mapVals['axesOrientation']
        axesDict = {'x':0, 'y':1, 'z':2}
        axesKeys = list(axesDict.keys())
        axesVals = list(axesDict.values())
        axesStr = ['','','']
        n = 0
        for val in axes:
            index = axesVals.index(val)
            axesStr[n] = axesKeys[index]
            n += 1
        if (axes_enable[1] == 0 and axes_enable[2] == 0):
            bw = self.mapVals['bw']*1e-3 # kHz
            acqTime = self.mapVals['acqTime'] # ms
            tVector = np.linspace(-acqTime/2, acqTime/2, self.nPoints[0])
            sVector = self.mapVals['sampled'][:, 3]
            fVector = np.linspace(-bw/2, bw/2, self.nPoints[0])
            iVector = np.fft.ifftshift(np.fft.ifftn(np.fft.ifftshift(sVector)))

            # Plots to show into the GUI
            result1 = {}
            result1['widget'] = 'curve'
            result1['xData'] = tVector
            result1['yData'] = [np.abs(sVector), np.real(sVector), np.imag(sVector)]
            result1['xLabel'] = 'Time (ms)'
            result1['yLabel'] = 'Signal amplitude (mV)'
            result1['title'] = "Signal"
            result1['legend'] = ['Magnitude', 'Real', 'Imaginary']
            result1['row'] = 0
            result1['col'] = 0

            result2 = {}
            result2['widget'] = 'curve'
            result2['xData'] = fVector
            result2['yData'] = [np.abs(iVector)]
            result2['xLabel'] = 'Frequency (kHz)'
            result2['yLabel'] = "Amplitude (a.u.)"
            result2['title'] = "Spectrum"
            result2['legend'] = ['Spectrum magnitude']
            result2['row'] = 1
            result2['col'] = 0

            self.output = [result1, result2]
            
        else:
            # Plot image
            image = np.abs(self.mapVals['image3D'])
            image = image/np.max(np.reshape(image,-1))*100

            # Image plot
            # result_1, image = self.fix_image_orientation(np.abs(image), axes=self.axesOrientation)
            result_1 = {
                'widget': 'image',
                'data': np.abs(image),
                'xLabel': 'X',
                'yLabel': 'Y',
                'title': 'Reconstructed image',
                'row': 0,
                'col': 0
            }
             

            # k-space plot
            result2 = {'widget': 'image'}
            if hasattr(self, 'parFourierFraction') and self.parFourierFraction == 1:
                result2['data'] = np.log10(np.abs(self.mapVals['kSpace3D']))
            else:
                result2['data'] = np.abs(self.mapVals['kSpace3D'])
            result2['xLabel'] = "k%s"%axesStr[1]
            result2['yLabel'] = "k%s"%axesStr[0]
            result2['title'] = "k-Space"
            result2['row'] = 0
            result2['col'] = 1

            # DICOM TAGS
            if self.axesOrientation[2]==2:
                image_orientation_dicom = [0.0, 1.0, 0.0, 0.0, 0.0, -1.0]
            elif self.axesOrientation[2]==1:
                image_orientation_dicom = [1.0, 0.0, 0.0, 0.0, 1.0, 0.0]
            elif self.axesOrientation[2]==0:
                image_orientation_dicom = [1.0, 0.0, 0.0, 0.0, 0.0, -1.0]

            # Image
            image_dicom = np.transpose(image, (0, 2, 1))
            # If it is a 3d image
            if len(image_dicom.shape) > 2:
                # Obtener dimensiones
                slices, rows, columns = image_dicom.shape
                self.meta_data["Columns"] = columns
                self.meta_data["Rows"] = rows
                self.meta_data["NumberOfSlices"] = slices
                self.meta_data["NumberOfFrames"] = slices
            # if it is a 2d image
            else:
                # Obtener dimensiones
                rows, columns = image_dicom.shape
                slices = 1
                self.meta_data["Columns"] = columns
                self.meta_data["Rows"] = rows
                self.meta_data["NumberOfSlices"] = 1
                self.meta_data["NumberOfFrames"] = 1
            img_full_abs = np.abs(image_dicom) * (2 ** 15 - 1) / np.amax(np.abs(image_dicom))
            img_full_int = np.int16(np.abs(img_full_abs))
            img_full_int = np.reshape(img_full_int, (slices, rows, columns))
            arr = img_full_int
            self.meta_data["PixelData"] = arr.tobytes()
            self.meta_data["WindowWidth"] = 26373
            self.meta_data["WindowCenter"] = 13194
            self.meta_data["ImageOrientationPatient"] = image_orientation_dicom
            resolution = self.mapVals['resolution'] * 1e3
            self.meta_data["PixelSpacing"] = [resolution[0], resolution[1]]
            self.meta_data["SliceThickness"] = resolution[2]
            # Sequence parameters
            self.meta_data["RepetitionTime"] = self.mapVals['repetitionTime']
            self.meta_data["EchoTime"] = self.mapVals['echoSpacing']
            self.meta_data["EchoTrainLength"] = self.mapVals['etl']

            # Add results into the output attribute (result1 must be the image to save in dicom)
            self.output = [result_1,  result2]

        # Reset rotation angle and dfov to zero
        self.mapVals['angle'] = self.angle
        self.mapVals['dfov'] = np.array(self.mapVals['dfov'])
        self.mapVals['dfov'][self.axesOrientation] = self.dfov.reshape(-1)
        self.mapVals['dfov'] = list(self.mapVals['dfov'])

        # Save results
        self.saveRawData()
        # self.save_ismrmrd()

        self.mapVals['angle'] = 0.0
        self.mapVals['dfov'] = [0.0, 0.0, 0.0]
        try:
            self.sequence_list['RARE'].mapVals['angle'] = 0.0
            self.sequence_list['RARE'].mapVals['dfov'] = [0.0, 0.0, 0.0]
        except:
            pass
        hw.dfov = [0.0, 0.0, 0.0]

        if self.mode == 'Standalone':
            self.plotResults()

        return self.output
 
    
if __name__ == '__main__':
    seq = TSE3DPSEQ()
    seq.sequenceAtributes()
    seq.sequenceRun(plotSeq=True, demo=True, standalone=True)
    seq.sequenceAnalysis(mode='Standalone')




