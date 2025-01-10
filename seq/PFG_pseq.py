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
class PFGPSEQ(blankSeq.MRIBLANKSEQ):
    def __init__(self):
        super(PFGPSEQ, self).__init__()
        # Input the parameters
        self.output = None
        self.expt = None

        self.seqName = None 
        self.nScans = None  
        self.larmorFreq = None 
        self.rfExFA = None
        self.rfReFA = None
        # self.rfExAmp = None  
        # self.rfReAmp = None  
        self.rfExTime = None 
        self.rfReTime = None 
        self.echoSpacing = None  
        self.repetitionTime = None  
        self.nPoints = None 
        self.etl = None  
        self.bandwidth = None
        self.acqTime = None  
        self.shimming = None 
        self.Exphase = None 
        self.Refphase = None
        self.Rxphase = None
        self.RxTimeOffset = None
        self.txChannel = None
        self.rxChannel = None


        # PFG params initialization
        self.p90GradDeadTime = None
        self.p180GradDeadTime = None
        self.riseTime = None
        self.gradTime = None
        self.gradAmp = None
        self.FirstEchoSpacing = None
        

        self.addParameter(key='seqName', string='CPMGInfo', val='TSE')
        self.addParameter(key='nScans', string='Number of scans', val=1, field='SEQ')
        self.addParameter(key='larmorFreq', string='Larmor frequency (MHz)', val=10.35358, units=units.MHz, field='RF')
        
        # PFG params
        self.addParameter(key='p90GradDeadTime', string='Dead time between p90 and grad.', val=1, units=units.ms, field='OTH')
        self.addParameter(key='p180GradDeadTime', string='Dead time between p180 and grad.', val=1, units=units.ms, field='OTH')
        self.addParameter(key='riseTime', string='Grad. Rise time (ms)', val=0.5, units=units.ms, field='OTH')
        self.addParameter(key='gradTime', string='Grad. duration (ms)', val=4, units=units.ms, field='SEQ')
        self.addParameter(key='gradAmp', string='Grad. amplitude (mT/m)', val=[20.0, 20.0, 20.0], field='SEQ')
        self.addParameter(key='FirstEchoSpacing', string='First echo spacing (ms)', val=40, units=units.ms, field='SEQ')
        
        # CPMG params
        self.addParameter(key='rfExFA', string='Excitation flip angle (deg)', val=90, field='RF')
        self.addParameter(key='rfReFA', string='Refocusing flip angle (deg)', val=180, field='RF')
        self.addParameter(key='repetitionTime', string='Repetition time (ms)', val=3500., units=units.ms, field='SEQ')
        self.addParameter(key='rfExTime', string='RF excitation time (us)', val=20.0, units=units.us, field='RF')
        self.addParameter(key='rfReTime', string='RF refocusing time (us)', val=40.0, units=units.us, field='RF')
        self.addParameter(key='echoSpacing', string='Echo spacing (ms)', val=0.7, units=units.ms, field='SEQ')
        self.addParameter(key='nPoints', string='Number of acquired points', val=10, field='IM')
        self.addParameter(key='filterWindowSize', string='Filter Window Size', val=10, field='IM')
        self.addParameter(key='etl', string='Echo train length', val=1024, field='SEQ')
        self.addParameter(key='bandwidth', string='Acquisition Bandwidth (kHz)', val=426.666667, units=units.kHz, field='IM',
                          tip="The bandwidth of the acquisition (kHz). This value affects resolution and SNR.")
        self.addParameter(key='shimming', string='shimming', val=[0.0, 0.0, 0.0], units=units.sh, field='OTH')
        self.addParameter(key='Exphase', string='Ex Phase (deg)', val=[0, 180, 90, 270], tip='Excitation Phase Cycling', field='RF')
        self.addParameter(key='Refphase', string='Ref Phase (deg)', val=[90, 90, 180, 180], tip='Refocusing Phase Cycling', field='RF')
        self.addParameter(key='Rxphase', string='Rx Phase (deg)', val=[0, 180, 90, 270], tip='Rx Phase Cycling', field='RF')
        self.addParameter(key='RxTimeOffset', string='Rx Time Offset (ms)', val=0, units=units.ms, field='SEQ')
        self.addParameter(key='txChannel', string='Tx channel', val=0, field='RF')
        self.addParameter(key='rxChannel', string='Rx channel', val=0, field='RF')

    def sequenceInfo(self):
        pass
        
    def sequenceTime(self):
        return (self.mapVals['repetitionTime'] *1e-3 * self.mapVals['nScans'] / 60)  # minutes

    def sequenceAtributes(self):
        super().sequenceAtributes()

    def sequenceRun(self, plotSeq=0, demo=False, standalone=False):
        init_gpa = False
        self.demo = demo
        self.plotSeq = plotSeq
        self.standalone = standalone

        max_grad_Hz = convert(from_value=hw.max_grad, from_unit='mT/m', gamma=hw.gammaB, to_unit='Hz/m')
        grad_Hz = np.zeros(len(self.gradAmp))
        for i in range(len(self.gradAmp)):
            grad_Hz[i] = convert(from_value=self.gradAmp[i], from_unit='mT/m', gamma=hw.gammaB, to_unit='Hz/m')
        rfExTime_us = int(np.round(self.rfExTime * 1e6))
        rfReTime_us = int(np.round(self.rfReTime * 1e6))
        assert rfExTime_us in hw.max_cpmg_rf_arr, f"RF excitation time '{rfExTime_us}' s is not found in the hw_config_pseq file; please search it in search_p90_pseq."
        assert rfReTime_us in hw.max_cpmg_rf_p180_arr, f"RF refocusing time '{rfReTime_us}' s is not found in the hw_config_pseq file; please search it in search_p180_pseq."
        
        max_rf_Hz = hw.max_cpmg_rf_arr[rfExTime_us] * 1e-6 * hw.gammaB
        rf_ref_correction_coeff = 0.5 * hw.max_cpmg_rf_arr[rfExTime_us] / hw.max_cpmg_rf_p180_arr[rfReTime_us]
        self.flo_interpreter = PseqInterpreter(
            tx_warmup=10,  # Transmit chain warm-up time (us)
            rf_center=hw.larmorFreq * 1e6 ,  # Larmor frequency (Hz)
            rf_amp_max=max_rf_Hz,  # Maximum RF amplitude (Hz)
            grad_max=max_grad_Hz,  # Maximum gradient amplitude (Hz/m)
            grad_t=10,  # Gradient raster time (us)
            grad_eff=hw.gradFactor, # gradient coefficient of efficiency
            tx_ch = self.txChannel, # Transmit channel
            rx_ch = self.rxChannel, # Receive channel
            add_rx_points = 8, # Additional points for the receiver
            use_multi_freq=True, # Use multi-frequency mode
        )
        assert (self.txChannel == 0 or self.txChannel == 1)
        assert (self.rxChannel == 0 or self.rxChannel == 1)
        self.rxChName = 'rx0' if (self.rxChannel == 0) else 'rx1'
        self.mapVals['rxChName'] = 'rx0'

        self.system = pp.Opts(
            rf_dead_time=10 * 1e-6,  # Dead time between RF pulses (s)
            rf_ringdown_time= 10 * 1e-6,
            max_grad=30,  # Maximum gradient strength (mT/m)
            grad_unit='mT/m',  # Units of gradient strength
            max_slew=hw.max_slew_rate,  # Maximum gradient slew rate (mT/m/ms)
            slew_unit='mT/m/ms',  # Units of gradient slew rate
            grad_raster_time=hw.grad_raster_time,  # Gradient raster time (s)
            rise_time=self.riseTime,  # Gradient rise time (s)
            rf_raster_time=0.25e-6,
            block_duration_raster=0.25e-6,
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

        readout_duration_rounded = np.ceil(sampling_period * self.nPoints * 4) / 4 * 1e-6
        if not self.demo:
            print(f'dwell time: {sampling_period} us, readout time: {readout_duration} s')
        
        RealExphase = np.tile(self.Exphase, int(np.ceil(self.nScans / len(self.Exphase))))
        RealRefphase = np.tile(self.Refphase, int(np.ceil(self.nScans / len(self.Refphase))))
        RealRxphase = np.tile(self.Rxphase, int(np.ceil(self.nScans / len(self.Rxphase))))
        

        # PFG seq design:
        # make a trapezoid gradient with the given parameters using pypulseq
        
        grad_diffusion_x = pp.make_trapezoid(channel='x', area=self.gradTime*grad_Hz[0], duration=self.gradTime, system=self.system)
        grad_diffusion_y = pp.make_trapezoid(channel='y', area=self.gradTime*grad_Hz[1], duration=self.gradTime, system=self.system)
        grad_diffusion_z = pp.make_trapezoid(channel='z', area=self.gradTime*grad_Hz[2], duration=self.gradTime, system=self.system)
        grad_duration = grad_diffusion_x.rise_time + grad_diffusion_x.flat_time + grad_diffusion_x.fall_time

        p902grad_delay = np.round((self.p90GradDeadTime - 0.5 * self.rfExTime - self.system.rf_ringdown_time)
                         / self.system.block_duration_raster) * self.system.block_duration_raster   
        grad2p180_delay = np.round((0.5 * self.FirstEchoSpacing - 0.5 * (self.rfReTime + self.rfExTime) - self.system.rf_dead_time - self.system.rf_dead_time - p902grad_delay - grad_duration)
                         / self.system.block_duration_raster) * self.system.block_duration_raster   
        p1802grad_delay = np.round((self.p180GradDeadTime - 0.5 * self.rfReTime - self.system.rf_ringdown_time)
                         / self.system.block_duration_raster) * self.system.block_duration_raster   
        p1802echo_delay = np.round((0.5 * self.FirstEchoSpacing - 0.5 * (self.rfReTime) - self.system.rf_ringdown_time - p1802grad_delay - grad_duration -  np.round(0.5 * readout_duration_rounded * 1e6) / 1e6)
                         / self.system.block_duration_raster) * self.system.block_duration_raster   
        assert p902grad_delay > 0, f"Error: p902grad_delay is non-positive: {p902grad_delay}"
        assert grad2p180_delay > 0, f"Error: grad2p180_delay is non-positive: {grad2p180_delay}"
        assert p1802grad_delay > 0, f"Error: p1802grad_delay is non-positive: {p1802grad_delay}"
        assert p1802echo_delay > 0, f"Error: p1802echo_delay is non-positive: {p1802echo_delay}"

        rf_ex = pp.make_block_pulse(
            flip_angle=self.rfExFA * np.pi / 180,
            duration=self.rfExTime,
            system=self.system,
            phase_offset=RealExphase[0] * np.pi / 180,
            delay=0,
        )
        rf_ref = pp.make_block_pulse(
            flip_angle=self.rfReFA * np.pi / 180,
            duration=self.rfReTime,
            system=self.system,
            phase_offset=RealRefphase[0] * np.pi / 180,
            delay=0,
        )
        
        # correct p180:
        rf_ref.signal = rf_ref_correction_coeff * rf_ref.signal 

        adc = pp.make_adc(num_samples=self.nPoints, duration=readout_duration) 
        # delay_te1 = np.round((0.5 * (self.echoSpacing - self.rfExTime - self.rfReTime) - (self.system.rf_dead_time+self.system.rf_ringdown_time))
        #                       / self.system.block_duration_raster) * self.system.block_duration_raster   
        delay_te2 = np.round((0.5 * (self.echoSpacing - self.rfReTime - readout_duration_rounded) - self.system.rf_ringdown_time)
                              / self.system.block_duration_raster) * self.system.block_duration_raster
        delay_te3 = np.round((0.5 * (self.echoSpacing - self.rfReTime - readout_duration_rounded) - self.system.rf_dead_time) 
                              / self.system.block_duration_raster) * self.system.block_duration_raster
        delay_te2_with_offset = np.round((delay_te2 + self.RxTimeOffset) / self.system.block_duration_raster) * self.system.block_duration_raster
        delay_te3_with_offset = np.round((delay_te3 - self.RxTimeOffset) / self.system.block_duration_raster) * self.system.block_duration_raster
        
        recovery_time = self.repetitionTime - ( 0.5 * self.rfExTime + self.system.rf_dead_time 
                      + (self.etl-1) * self.echoSpacing +  self.FirstEchoSpacing 
                      + delay_te3_with_offset + np.round(0.5 * readout_duration_rounded * 1e6) / 1e6)
        # Assertions to check if times are greater than zero
        # assert delay_te1 > 0, f"Error: delay_te1 is non-positive: {delay_te1}"
        assert recovery_time > 0, f"Error: recovery_time is non-positive: {recovery_time}"
        
        if self.etl == 1: # for debuging
            delay_te2 = 10e-6 
            delay_te3 = 10e-6 
            delay_te2_with_offset = delay_te2
            delay_te3_with_offset = delay_te3
        else:
            assert delay_te2 > 0, f"Error: delay_te2 is non-positive: {delay_te2}"
            assert delay_te3 > 0, f"Error: delay_te3 is non-positive: {delay_te3}"
            assert delay_te2_with_offset > 0, f"Error: delay_te2_with_offset is non-positive: {delay_te2_with_offset}"
            assert delay_te3_with_offset > 0, f"Error: delay_te3_with_offset is non-positive: {delay_te3_with_offset}"
        
        acq_points = 0
        seq = pp.Sequence(system=self.system)
        for scan in range(self.nScans):
            # Phase Cycling
            rf_ex.phase_offset = RealExphase[scan] * np.pi / 180
            adc.phase_offset = RealRxphase[scan] * np.pi / 180
            rf_ref.phase_offset = RealRefphase[scan] * np.pi / 180

            # PFG seq
            seq.add_block(rf_ex)
            seq.add_block(pp.make_delay(p902grad_delay))
            seq.add_block(grad_diffusion_x, grad_diffusion_y, grad_diffusion_z)
            seq.add_block(pp.make_delay(grad2p180_delay))
            seq.add_block(rf_ref)
            seq.add_block(pp.make_delay(p1802grad_delay))
            seq.add_block(grad_diffusion_x, grad_diffusion_y, grad_diffusion_z)
            seq.add_block(pp.make_delay(p1802echo_delay))

            seq.add_block(adc, pp.make_delay(readout_duration_rounded))
            seq.add_block(pp.make_delay(delay_te3_with_offset))
            acq_points += self.nPoints
           
            # Echo train
            for echoIndex in range(self.etl-1):
                seq.add_block(rf_ref)
                seq.add_block(pp.make_delay(delay_te2_with_offset))
                seq.add_block(adc, pp.make_delay(readout_duration_rounded))
                seq.add_block(pp.make_delay(delay_te3_with_offset))
                acq_points += self.nPoints
            if not scan == self.nScans-1: 
                seq.add_block(pp.make_delay(recovery_time))

        if plotSeq:
            # Check whether the timing of the sequence is correct
            ok, error_report = seq.check_timing()
            if ok:
                print("Timing check passed successfully")
            else:
                print("Timing check failed. Error listing follows:")
                [print(e) for e in error_report]   

            seq.plot(show_blocks =False)

        seq.set_definition(key="Name", value="PFG")
        seq.write("PFG.seq")
        self.waveforms, param_dict = self.flo_interpreter.interpret("PFG.seq")
         
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
                expected_points = self.nPoints * self.etl * self.nScans  # Expected number of points

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
                    before_delete = np.reshape(rx_raw_data, newshape=(self.etl * self.nScans, -1))
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
            raw_data_reshape = np.reshape(rawdata, newshape=(-1, self.mapVals['nPoints']))
            
            for line in range(raw_data_reshape.shape[0]):
                raw_data_reshape[line, :] = raw_data_reshape[line, :] * expiangle[line]
            signal = np.reshape(raw_data_reshape, -1)
        else: 
            signal = self.mapVals['data']


        # Average data
        signal = np.average(np.reshape(signal, (self.nScans, -1)), axis=0)

          
        # average filter
        bw = self.mapVals['bw_MHz']*1e3 # kHz
        nPoints = self.mapVals['nScans'] * self.mapVals['nPoints'] * self.mapVals['etl']
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
        tVector = create_tVector(bw * 1e3, self.mapVals['nPoints'], self.mapVals['echoSpacing'], self.mapVals['etl'])
        tVecRes = np.reshape(tVector, newshape=(-1, self.mapVals['nPoints']))
        
        fir_coefficients = np.ones(self.mapVals['filterWindowSize']) / self.mapVals['filterWindowSize']
        num_taps = len(fir_coefficients)
        signal_waiting_for_filters = np.reshape(signal, newshape=(-1, self.mapVals['nPoints']))
        output_length = signal_waiting_for_filters.shape[1] - num_taps + 1
        filtered = np.zeros((signal_waiting_for_filters.shape[0], output_length), dtype=complex)
        filtered_time = np.zeros((signal_waiting_for_filters.shape[0], output_length))

        for i in range(signal_waiting_for_filters.shape[0]):
            real_filtered = np.convolve(signal_waiting_for_filters[i].real, fir_coefficients, mode='valid')
            imag_filtered = np.convolve(signal_waiting_for_filters[i].imag, fir_coefficients, mode='valid')
            filtered[i] = real_filtered + 1j * imag_filtered
            filtered_time[i] = tVecRes[i, num_taps - 1:] 
        filtered_signal = np.reshape(filtered, newshape=(-1))
        filtered_time_vector = np.reshape(filtered_time, newshape=(-1))
        
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

        # Add time signal to the layout
        result1 = {'widget': 'curve',
                   'xData': tVector,
                   'yData': [np.abs(signal), np.real(signal), np.imag(signal)],
                   'xLabel': 'Time (ms)',
                   'yLabel': 'Signal amplitude',
                   'title': 'Signal vs time',
                   'legend': ['abs', 'real', 'imag'],
                   'row': 0,
                   'col': 0}

        # # Add frequency spectrum to the layout
        # result2 = {'widget': 'curve',
        #            'xData': tVector,#fVector,
        #            'yData': [np.angle(signal)], #[spectrum],
        #            'xLabel': 'Time (ms)',
        #            'yLabel': 'Angle (rad)',
        #            'title': 'Angle',
        #            'legend': [''],
        #            'row': 1,
        #            'col': 0}
        
        result2 = {'widget': 'curve',
                   'xData': filtered_time_vector,
                   'yData': [np.abs(filtered_signal), np.real(filtered_signal), np.imag(filtered_signal)],
                   'xLabel': 'Time (ms)',
                   'yLabel': 'Filtered signal amplitude',
                   'title': 'Signal vs time',
                   'legend': ['abs', 'real', 'imag'],
                   'row': 1,
                   'col': 0}

        # create self.out to run in iterative mode
        self.output = [result1, result2]
        self.saveRawData()

        if self.mode == 'Standalone':
            self.plotResults()
        return self.output
    
if __name__ == '__main__':
    seq = PFGPSEQ()
    seq.sequenceAtributes()
    seq.sequenceRun(plotSeq=True, demo=False, standalone=True)
    seq.sequenceAnalysis(mode='Standalone')


