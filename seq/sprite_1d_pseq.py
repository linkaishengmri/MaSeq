"""
Created on Tuesday, Apr 25th 2025
@author: Kaisheng Lin, School of Electronics, Peking University, China
@Summary: SPRITE sequence for ultra-short T2 measurement, implemented with PyPulseq and compatible with MaSeq.
"""

import os
import sys
import matplotlib.pyplot as plt


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
import pypulseq as pp
import numpy as np
import seq.mriBlankSeq as blankSeq   
import configs.units as units
import scipy.signal as sig
import experiment_multifreq as ex
import configs.hw_config_pseq as hw
from flocra_pulseq.interpreter_pseq import PseqInterpreter
from pypulseq.convert import convert
# from seq.utils import sort_data_implicit, plot_nd, ifft_2d, combine_coils

# max_cpmg_rf_arr = { #us:uT
#     20: 365,
#     30: 296,
#     50: 224,
#     100:169,
#     150:149,
#     200:138,
#     250:133,
#     300:130,
#     400:126,
#     500:126,
#     600:126,
#     700:126,
#     800:127,
#     900:129,
#     1000:130,
#     1500:136,
#     2000:133,
# }
# max_cpmg_rf_p180_arr = {#us:uT
#     40:185,
#     100:114,
#     200:86,
#     300:76,
#     400:70.5,
#     500:68.5,
#     600:67,
#     700:67,
#     800:67,
#     900:67,
#     1000:67,
# }
class SPRITER1dSEQ(blankSeq.MRIBLANKSEQ):
    def __init__(self):
        super(SPRITER1dSEQ, self).__init__()
        # Input the parameters
        self.nScans = None
        self.larmorFreq = None
        self.rfExFA = None
        self.repetitionTime = None
        self.rfExTime = None
        self.echoTime = None
        self.nPoints = None
        self.riseTime = None
        self.SpoilingTimeAfterRising = None
        self.fov = None
        self.bandwidth = None
        self.SamplingPoints = None
        self.shimming = None
        self.txChannel = None
        self.rxChannel = None    
        self.axesOrientation = None

        self.addParameter(key='seqName', string='CPMGInfo', val='TSE')
        self.addParameter(key='nScans', string='Number of scans', val=1, field='SEQ')
        self.addParameter(key='larmorFreq', string='Larmor frequency (MHz)', val=10.33364, units=units.MHz, field='RF')
        self.addParameter(key='rfExFA', string='Excitation flip angle (deg)', val=90, field='RF')
        self.addParameter(key='repetitionTime', string='Repetition time (ms)', val=10.0, units=units.ms, field='SEQ')
        self.addParameter(key='rfExTime', string='RF excitation time (us)', val=30, units=units.us, field='RF')
        self.addParameter(key='echoTime', string='Echo time (ms)', val=.30, units=units.ms, field='SEQ')
        self.addParameter(key='nPoints', string='Number of acquired points', val=1, field='IM')
        self.addParameter(key='riseTime', string='Grad. Rise time (ms)', val=.5, units=units.ms, field='OTH')
        self.addParameter(key='SpoilingTimeAfterRising', string='Grad. soiling time after grad. rising (ms)', val=8, units=units.ms, field='OTH')
        self.addParameter(key='fov', string='FOV (mm)', val=150, units=units.mm, field='IM')
        self.addParameter(key='bandwidth', string='Acquisition Bandwidth (kHz)', val=106.66666666666667, units=units.kHz, field='IM',
                                tip="The bandwidth of the acquisition (kHz). This value affects resolution and SNR.")
        self.addParameter(key='SamplingPoints', string='Sampling Points Number', val=128, field='IM')
        self.addParameter(key='shimming', string='shimming', val=[0.001, -0.006, 0.001], units=units.sh, field='OTH')
        self.addParameter(key='txChannel', string='Tx channel', val=0, field='RF')
        self.addParameter(key='rxChannel', string='Rx channel', val=0, field='RF')
        self.addParameter(key='axesOrientation', string='Axes[rd,ph,sl]', val=[1,2,0], field='IM',
                          tip="0=x, 1=y, 2=z")
    def sequenceInfo(self):
        pass
        
    def sequenceTime(self):
        return (self.mapVals['repetitionTime'] *1e-3 * self.mapVals['SamplingPoints'] * self.mapVals['nScans'] / 60)  # minutes

    def sequenceAtributes(self):
        super().sequenceAtributes()
        
    
    def sequenceRun(self, plotSeq=0, demo=False, standalone=False):
        init_gpa = False
        self.demo = demo
        self.plotSeq = plotSeq
        self.standalone = standalone

        max_grad_Hz = convert(from_value=hw.max_grad, from_unit='mT/m', gamma=hw.gammaB, to_unit='Hz/m')
        
        rfExTime_us = int(np.round(self.rfExTime * 1e6))
        assert rfExTime_us in hw.max_cpmg_rf_arr, f"RF excitation time '{rfExTime_us}' s is not found in the hw_config_pseq file; please search it in search_p90_pseq."
        
        max_rf_Hz = hw.max_cpmg_rf_arr[rfExTime_us] * 1e-6 * hw.gammaB
        self.flo_interpreter = PseqInterpreter(
            tx_warmup=10,  # Transmit chain warm-up time (us)
            rf_center=hw.larmorFreq * 1e6 ,  # Larmor frequency (Hz)
            rf_amp_max=max_rf_Hz,  # Maximum RF amplitude (Hz)
            grad_max=max_grad_Hz,  # Maximum gradient amplitude (Hz/m)
            grad_t=10,  # Gradient raster time (us)
            orientation=self.axesOrientation, # gradient orientation
            grad_eff=hw.gradFactor, # gradient coefficient of efficiency
            tx_ch = self.txChannel, # Transmit channel
            rx_ch = self.rxChannel, # Receive channel
            add_rx_points = 0, # Additional points for the receiver
            tx_t= 1229/122.88, # us
            grad_zero_end=False,
            use_multi_freq=False, # Use multi-frequency mode
            use_grad_preemphasis=False,
            grad_preemphasis_coeff={
                        'xx':( (np.array([0.383494796, 0.159428847, 0.06601789, 0.03040273]), 
                            np.array([384.543433, 4353.01123, 46948.52793, 485123.9174] ))),
                        'yy':( (np.array([0.383494796, 0.159428847, 0.06601789, 0.03040273]),
                            np.array([384.543433, 4353.01123, 46948.52793, 485123.9174] ))),
                        'zz':( (np.array([0.383494796, 0.159428847, 0.06601789, 0.03040273]),
                            np.array([384.543433, 4353.01123, 46948.52793, 485123.9174] ))),
                 },
            use_fir_decimation = (self.bandwidth < 30.007326007326007e3), # 30kHz
        )
        assert (self.txChannel == 0 or self.txChannel == 1)
        assert (self.rxChannel == 0 or self.rxChannel == 1)
        self.rxChName = 'rx0' if (self.rxChannel == 0) else 'rx1'
        self.mapVals['rxChName'] = 'rx0'

        self.system = pp.Opts(
            rf_dead_time=10 * 1e-6,  # Dead time between RF pulses (s)
            rf_ringdown_time= 10 * 1e-6,
            max_grad=40,  # Maximum gradient strength (mT/m)
            grad_unit='mT/m',  # Units of gradient strength
            max_slew=hw.max_slew_rate,  # Maximum gradient slew rate (mT/m/ms)
            slew_unit='mT/m/ms',  # Units of gradient slew rate
            grad_raster_time=hw.grad_raster_time,  # Gradient raster time (s)
            rise_time=self.riseTime,  # Gradient rise time (s)
            rf_raster_time=10e-6,
            block_duration_raster=1e-6,
            adc_raster_time=1/(122.88e6)
        )

        bw = self.bandwidth * 1e-6 # MHz
        bw_ov = bw
        sampling_period = 1 / bw_ov  # us, Dwell time

        if not self.demo:
            expt = ex.Experiment(
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
        self.mapVals['acqTime'] = self.nPoints / bw * 1e-3 # ms

        readout_duration = sampling_period * 1e-6 * self.nPoints

        readout_duration_rounded = np.ceil(sampling_period * self.nPoints / 10) * 10 * 1e-6
        if not self.demo:
            print(f'dwell time: {sampling_period} us, readout time: {readout_duration} s')
        
    
        delay_TE = np.round((self.echoTime - 0.5 * self.rfExTime - self.system.rf_ringdown_time) / self.system.grad_raster_time) * self.system.grad_raster_time
        delay_Spoiling_before_rising = np.round((self.repetitionTime - 0.5 * self.rfExTime - self.system.rf_dead_time 
             -self.riseTime - self.SpoilingTimeAfterRising - readout_duration_rounded) / self.system.grad_raster_time) * self.system.grad_raster_time
        assert delay_TE > 0, f"Error: delay_TE is non-positive: {delay_TE}"
        assert delay_Spoiling_before_rising > 0, f"Error: SpoilingTimeAfterRising is non-positive: {delay_Spoiling_before_rising}"

        rf_ex = pp.make_block_pulse(
            flip_angle=self.rfExFA * np.pi / 180,
            duration=self.rfExTime,
            system=self.system,
            phase_offset=0 * np.pi / 180,
            delay=0,
        )
        adc = pp.make_adc(num_samples=self.nPoints, duration=readout_duration) 
        
        delta_kx = 1 / self.fov
        phase_areas_x = (np.arange(self.SamplingPoints) - self.SamplingPoints // 2) * delta_kx
        phase_amp_x = phase_areas_x / self.echoTime
        last_phase_amp_x = 0
        acq_points = 0
        seq = pp.Sequence(system=self.system)

        def make_flat_grad(duration, amplitude_list):
            dur = duration # np.floor(duration* 1e6 / self.system.grad_raster_time) *self.system.grad_raster_time
            grad_flat_time = np.array([0, dur])
            grad_x = pp.make_extended_trapezoid(channel="x", times=grad_flat_time, amplitudes=np.array([amplitude_list[0], amplitude_list[0]]), system=self.system)
            grad_y = pp.make_extended_trapezoid(channel="y", times=grad_flat_time, amplitudes=np.array([amplitude_list[1], amplitude_list[1]]), system=self.system)
            grad_z = pp.make_extended_trapezoid(channel="z", times=grad_flat_time, amplitudes=np.array([amplitude_list[2], amplitude_list[2]]), system=self.system)
            return grad_x, grad_y, grad_z

        for scan in range(self.nScans):
            last_phase_amp_x = 0
            for ind in range(self.SamplingPoints):
                
                current_phase_amp_x = phase_amp_x[ind]
                rise_x = pp.make_extended_trapezoid(
                    channel="x", 
                    times=np.array([0, self.riseTime]),
                    amplitudes=np.array([last_phase_amp_x, current_phase_amp_x]),
                    system=self.system)
                rise_y = pp.make_extended_trapezoid(
                    channel="y", 
                    times=np.array([0, self.riseTime]),
                    amplitudes=np.array([0, 0]),
                    system=self.system)
                rise_z = pp.make_extended_trapezoid(
                    channel="z", 
                    times=np.array([0, self.riseTime]),
                    amplitudes=np.array([0, 0]),
                    system=self.system)
                grad_Hz = np.array([current_phase_amp_x, 0, 0]) 
                seq.add_block(rise_x, rise_y, rise_z)
                seq.add_block(pp.make_delay(self.SpoilingTimeAfterRising), *make_flat_grad(pp.calc_duration(self.SpoilingTimeAfterRising), grad_Hz))
                seq.add_block(rf_ex, *make_flat_grad(pp.calc_duration(rf_ex), grad_Hz))
                seq.add_block(pp.make_delay(delay_TE), *make_flat_grad(pp.calc_duration(delay_TE), grad_Hz))
                seq.add_block(adc, pp.make_delay(readout_duration_rounded), *make_flat_grad(pp.calc_duration(readout_duration_rounded), grad_Hz))
                acq_points += self.nPoints
                delay_spoiling = pp.make_delay(delay_Spoiling_before_rising)
                seq.add_block(delay_spoiling, *make_flat_grad(pp.calc_duration(delay_spoiling), grad_Hz))
                last_phase_amp_x = current_phase_amp_x
            fall_x = pp.make_extended_trapezoid(
                channel="x", 
                times=np.array([0, self.riseTime]),
                amplitudes=np.array([last_phase_amp_x, 0]),
                system=self.system
            )
            seq.add_block(fall_x)
            last_phase_amp_x = 0

        if plotSeq:
            # Check whether the timing of the sequence is correct
            ok, error_report = seq.check_timing()
            if ok:
                print("Timing check passed successfully")
            else:
                print("Timing check failed. Error listing follows:")
                [print(e) for e in error_report]   
            print(seq.test_report())
            seq.plot(show_blocks =False)
            k_traj_adc, k_traj, t_excitation, t_refocusing, t_adc = seq.calculate_kspace()

            # plt.figure(10)
            # plt.plot(k_traj[0],k_traj[1],linewidth=1)
            # plt.plot(k_traj_adc[0],k_traj_adc[1],'.', markersize=1.4)
            # plt.axis("equal")
            # plt.title("k-space trajectory (kx/ky)")

            # plt.figure(11)
            # plt.plot(t_adc, k_traj_adc.T, linewidth=1)
            # plt.xlabel("Time of acqusition (s)")
            # plt.ylabel("Phase")
            
            # plt.figure(12)
            # t = np.linspace(0, 1, k_traj_adc.shape[1])  # 归一化时间
            # plt.scatter(k_traj_adc[0], k_traj_adc[1], c=t, cmap='viridis', s=2)  # 用颜色表示时间
            # plt.axis("equal")
            # plt.colorbar(label='Normalized Time')  # 添加颜色条
            # plt.title("k-space trajectory (kx/ky) with Gradient")
            # plt.show()

        seq.set_definition(key="Name", value="SPRITE1D")
        seq.write("SPRITE1D.seq")
        self.waveforms, param_dict = self.flo_interpreter.interpret("SPRITE1D.seq")
         
        larmorFreq = self.mapVals['larmorFreq']
        if not self.demo:
            self.expt = ex.Experiment(
                lo_freq=(self.larmorFreq + 0) * 1e-6,  # Larmor frequency in MHz
                rx_t= sampling_period,
                init_gpa=False,  # Whether to initialize GPA board (False for now)
                gpa_fhdo_offset_time=(1 / 0.2 / 3.1),  # GPA offset time calculation
                auto_leds=True  # Automatic control of LEDs
            )
        print(f"Center frequecy set: {(self.larmorFreq + 0) * 1e-6} MHz")
        # Convert the PyPulseq waveform to the Red Pitaya compatible format
        self.pypulseq2mriblankseqV2(waveforms=self.waveforms, shimming=self.shimming)
        
        # Load the waveforms into Red Pitaya
        if not self.floDict2Exp_ms():
            print("ERROR: Sequence waveforms out of hardware bounds")
            return False
        else:
            encoding_ok = True
            # print("Sequence waveforms loaded successfully")

        if self.plotSeq and self.standalone:
            # Plot the sequence if requested and return immediately
            self.expt.plot_sequence()
            self.sequencePlot(standalone=self.standalone)
            
        data_over = []
         
        # If not plotting the sequence, start scanning
        if not self.plotSeq:
            # for scan in range(self.nScans):
            if True:
                print(f"Scan running...")
                acquired_points = 0
                expected_points = self.nPoints * self.SamplingPoints * self.nScans  # Expected number of points

                # Continue acquiring points until we reach the expected number
                while acquired_points != expected_points:
                    if not self.demo:
                        rxd, msgs = self.expt.run()  # Run the experiment and collect data
                    else:
                        # In demo mode, generate random data as a placeholder
                        rxd = {self.rxChName: np.random.randn(expected_points + self.flo_interpreter.get_add_rx_points()) + 1j * np.random.randn(expected_points + + self.flo_interpreter.get_add_rx_points())}
                    # Update acquired points
                    rx_raw_data = rxd[self.rxChName]
                    add_rx_points = self.flo_interpreter.get_add_rx_points()
                    before_delete = np.reshape(rx_raw_data, newshape=(self.nScans, -1))
                    rxdataremove = before_delete[:, add_rx_points:]
                    rxdata = np.reshape(rxdataremove, newshape=(-1))
                    acquired_points = np.size(rxdata)


                    # Check if acquired points coincide with expected points
                    if acquired_points != expected_points:
                        print("WARNING: data apoints lost!")
                        print("Repeating batch...")

                # Concatenate acquired data into the oversampled data array
                data_over = np.concatenate((data_over, rxdata), axis=0)
                print(f"Acquired points = {acquired_points}, Expected points = {expected_points}")
                print(f"Scan ready!")
                # plt.plot(data_over)
                # plt.show()
            # Decimate the oversampled data and store it
            self.mapVals['data_over'] = data_over

            # Average data
            # data = np.average(np.reshape(data_over, (self.nScans, -1)), axis=0)
            self.mapVals['data'] = data_over

        

        if not self.demo:
            self.expt.__del__()

        self.mapVals['n_readouts'] = self.nPoints
        return True

        


    def sequenceAnalysis(self, mode=None):
        
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


        # [TODO]: Add Rx phase here
        if True:
            rawdata = self.mapVals['data']
            chName = self.mapVals['rxChName']
            expiangle = self.flo_interpreter.get_rx_phase_dict()[chName]
            raw_data_reshape = np.reshape(rawdata, newshape=(-1, self.mapVals['SamplingPoints']))
            
            for line in range(raw_data_reshape.shape[0]):
                raw_data_reshape[line, :] = raw_data_reshape[line, :] * expiangle[line]
            signal = np.reshape(raw_data_reshape, -1)
        else: 
            signal = self.mapVals['data']


        # Average data
        signal = np.average(np.reshape(signal, (self.nScans, -1)), axis=0)

          
        # average filter
        bw = self.mapVals['bw_MHz']*1e3 # kHz
        nPoints = self.mapVals['nScans'] * self.mapVals['SamplingPoints'] 
        deadTime = 0 #self.mapVals['deadTime']*1e-3 # ms
        rfRectExTime = self.mapVals['rfExTime']*1e-3 # ms
        

        def create_tVector(bw, nPoint, echoSpacing, etl, RFinterval = False):
            point_interval = 1 / bw
            # window_duration = nPoint * point_interval
            start_times = np.arange(0, etl * echoSpacing, echoSpacing)
            if RFinterval:
                tVector = np.concatenate([start_time + np.arange(nPoint) * point_interval for start_time in start_times])
            else:
                tVector = np.reshape([np.arange(nPoint * etl) * point_interval], newshape=(-1))
            return tVector

        #tVector = np.linspace(rfRectExTime/2 + deadTime + 0.5/bw, rfRectExTime/2 + deadTime + (nPoints-0.5)/bw, nPoints)
        tVector = np.arange(self.mapVals['SamplingPoints'])
        tVecRes = np.reshape(tVector, newshape=(-1, self.mapVals['SamplingPoints']))
        
        # fir_coefficients = np.ones(self.mapVals['filterWindowSize']) / self.mapVals['filterWindowSize']
        # num_taps = len(fir_coefficients)
        # signal_waiting_for_filters = np.reshape(signal, newshape=(-1, self.mapVals['SamplingPoints']))
        # output_length = signal_waiting_for_filters.shape[1] - num_taps + 1
        # filtered = np.zeros((signal_waiting_for_filters.shape[0], output_length), dtype=complex)
        # filtered_time = np.zeros((signal_waiting_for_filters.shape[0], output_length))

        # for i in range(signal_waiting_for_filters.shape[0]):
        #     real_filtered = np.convolve(signal_waiting_for_filters[i].real, fir_coefficients, mode='valid')
        #     imag_filtered = np.convolve(signal_waiting_for_filters[i].imag, fir_coefficients, mode='valid')
        #     filtered[i] = real_filtered + 1j * imag_filtered
        #     filtered_time[i] = tVecRes[i, num_taps - 1:] 
        # filtered_signal = np.reshape(filtered, newshape=(-1))
        # filtered_time_vector = np.reshape(filtered_time, newshape=(-1))
        
        fVector = np.linspace(-bw/2, bw/2, nPoints)
        spectrum = np.abs(np.fft.ifftshift(np.fft.ifftn(np.fft.ifftshift(signal))))
        fitedLarmor=self.mapVals['larmorFreq'] - fVector[np.argmax(np.abs(spectrum))] * 1e-3  #MHz
        # hw.larmorFreq=fitedLarmor
        # print(f"self{self.larmorFreq}, map{self.mapVals['larmorFreq'] }, fv{fVector[np.argmax(np.abs(spectrum))]},fit larmor{fitedLarmor}")
        fwhm=getFHWM(spectrum, fVector, bw)
        dB0=fwhm*1e6/fitedLarmor


        # t_filtered = tVector[:filtered_signal.shape[0]]

        # for sequence in self.sequenceList.values():
        #     if 'larmorFreq' in sequence.mapVals:
        #         sequence.mapVals['larmorFreq'] = hw.larmorFreq
        self.mapVals['larmorFreq'] = fitedLarmor

        # Get the central frequency
        print('Larmor frequency: %1.5f MHz' % fitedLarmor)
        print('FHWM: %1.5f kHz' % fwhm)
        print('dB0/B0: %1.5f ppm' % dB0)

        self.mapVals['signalVStime'] = [tVector, signal]
        self.mapVals['spectrum'] = [fVector, spectrum]
        # self.mapVals['filtered_signalVStime'] = [filtered_time_vector, filtered_signal]
        
        # Add time signal to the layout
        result1 = {'widget': 'curve',
                   'xData': tVector,
                   'yData': [np.abs(signal), np.real(signal), np.imag(signal)],
                   'xLabel': 'Points',
                   'yLabel': 'Signal amplitude',
                   'title': 'Signal vs Points',
                   'legend': ['abs', 'real', 'imag'],
                   'row': 0,
                   'col': 0}

        result2 = {'widget': 'curve',
                   'xData': tVector,
                   'yData': [spectrum],
                   'xLabel': 'Points',
                   'yLabel': 'Spectrum amplitude (a.u.)',
                   'title': 'Spectrum',
                   'legend': [''],
                   'row': 1,
                   'col': 0}

        # create self.out to run in iterative mode
        self.output = [result1, result2]
        self.saveRawData()

        if self.mode == 'Standalone':
            self.plotResults()
        return self.output
    
if __name__ == '__main__':
    seq = SPRITER1dSEQ()
    seq.sequenceAtributes()
    seq.sequenceRun(plotSeq=False, demo=False, standalone=True)
    seq.sequenceAnalysis(mode='Standalone')


