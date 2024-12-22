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
import experiment as ex
import configs.hw_config_pseq as hw
from flocra_pulseq.interpreter_pseq import PseqInterpreter
from pypulseq.convert import convert
from seq.utils import sort_data_implicit, plot_nd, ifft_2d, combine_coils

class CPMGPSEQ(blankSeq.MRIBLANKSEQ):
    def __init__(self):
        super(CPMGPSEQ, self).__init__()
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
        self.phase_mode = None  
        self.txChannel = None
        self.rxChannel = None
        self.addParameter(key='seqName', string='CPMGInfo', val='TSE')
        self.addParameter(key='nScans', string='Number of scans', val=1, field='SEQ')
        self.addParameter(key='larmorFreq', string='Larmor frequency (MHz)', val=10.35376, units=units.MHz, field='RF')
        self.addParameter(key='rfExFA', string='Excitation flip angle (deg)', val=90, field='RF')
        self.addParameter(key='rfReFA', string='Refocusing flip angle (deg)', val=180, field='RF')
        
        # self.addParameter(key='rfExAmp', string='RF excitation amplitude (a.u.)', val=0.3, field='RF')
        # self.addParameter(key='rfReAmp', string='RF refocusing amplitude (a.u.)', val=0.3, field='RF')
        self.addParameter(key='rfExTime', string='RF excitation time (us)', val=300.0, units=units.us, field='RF')
        self.addParameter(key='rfReTime', string='RF refocusing time (us)', val=300.0, units=units.us, field='RF')
        self.addParameter(key='echoSpacing', string='Echo spacing (ms)', val=5.0, units=units.ms, field='SEQ')
        self.addParameter(key='repetitionTime', string='Repetition time (ms)', val=1000., units=units.ms, field='SEQ')
        self.addParameter(key='nPoints', string='Number of acquired points', val=50, field='IM')
        self.addParameter(key='etl', string='Echo train length', val=50, field='SEQ')
        self.addParameter(key='bandwidth', string='Acquisition Bandwidth (kHz)', val=32, units=units.kHz, field='IM',
                          tip="The bandwidth of the acquisition (kHz9. This value affects resolution and SNR.")
        self.addParameter(key='shimming', string='shimming', val=[0.0, 0.0, 0.0], units=units.sh, field='OTH')
        self.addParameter(key='phase_mode', string='Phase mode', val='CP', tip='CP, CPMG, APCP, APCPMG', field='SEQ')
        self.addParameter(key='txChannel', string='Tx channel', val=0, field='RF')
        self.addParameter(key='rxChannel', string='Rx channel', val=0, field='RF')

    def sequenceInfo(self):
        pass
        
    def sequenceTime(self):
        return 0 # (self.mapVals['repetitionTime'] *1e-3 * self.mapVals['nScans'] / 60)  # minutes

    def sequenceAtributes(self):
        super().sequenceAtributes()

    def sequenceRun(self, plotSeq=0, demo=False, standalone=False):
        init_gpa = False
        self.demo = demo
        self.plotSeq = plotSeq
        self.standalone = standalone

        max_grad_Hz = convert(from_value=hw.max_grad, from_unit='mT/m', gamma=hw.gammaB, to_unit='Hz/m')
        max_rf_Hz = hw.max_rf * 1e-6 * hw.gammaB
        self.flo_interpreter = PseqInterpreter(
            tx_warmup=hw.blkTime,  # Transmit chain warm-up time (us)
            rf_center=hw.larmorFreq * 1e6 ,  # Larmor frequency (Hz)
            rf_amp_max=max_rf_Hz,  # Maximum RF amplitude (Hz)
            grad_max=max_grad_Hz,  # Maximum gradient amplitude (Hz/m)
            grad_t=10,  # Gradient raster time (us)
            grad_eff=hw.gradFactor, # gradient coefficient of efficiency
            tx_ch = self.txChannel,
            rx_ch = self.rxChannel,
            add_rx_points = 0,
        )
        assert (self.txChannel == 0 or self.txChannel == 1)
        assert (self.rxChannel == 0 or self.rxChannel == 1)
        self.rxChName = 'rx0' if (self.rxChannel == 0) else 'rx1'
        self.mapVals['rxChName'] = 'rx0'

        self.system = pp.Opts(
            rf_dead_time=hw.blkTime * 1e-6,  # Dead time between RF pulses (s)
            max_grad=30,  # Maximum gradient strength (mT/m)
            grad_unit='mT/m',  # Units of gradient strength
            max_slew=hw.max_slew_rate,  # Maximum gradient slew rate (mT/m/ms)
            slew_unit='mT/m/ms',  # Units of gradient slew rate
            grad_raster_time=hw.grad_raster_time,  # Gradient raster time (s)
            rise_time=hw.grad_rise_time,  # Gradient rise time (s)
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
        if not self.demo:
            print(f'dwell time: {sampling_period} us, readout time: {readout_duration} s')
        
        rf_ex = pp.make_block_pulse(
            flip_angle=self.rfExFA * np.pi / 180,
            duration=self.rfExTime,
            system=self.system,
            phase_offset=0.0,
            delay=0,
        )
        rf_ref = pp.make_block_pulse(
            flip_angle=self.rfReFA * np.pi / 180,
            duration=self.rfReTime,
            system=self.system,
            phase_offset=0.0,
            delay=0,
        )

        # Check that self.phase_mode is once of the good values
        phase_modes = ['CP', 'CPMG', 'APCP', 'APCPMG']
        if not self.phase_mode in phase_modes:
            print('ERROR: unexpected phase mode.')
            print('ERROR: Please select one of possible modes:')
            print('ERROR: CP\nCPMG\nAPCP\nAPCPMG')
            return False

        adc = pp.make_adc(num_samples=self.nPoints, duration=readout_duration) 
        delay_te1 =  np.round((0.5 * (self.echoSpacing - self.rfExTime - self.rfReTime) - self.system.rf_dead_time)
                              / self.system.block_duration_raster) * self.system.block_duration_raster   
        delay_te2 =  np.round(0.5 * (self.echoSpacing - self.rfReTime - readout_duration)
                              / self.system.block_duration_raster) * self.system.block_duration_raster
        delay_te3 = np.round((delay_te2 - self.system.rf_dead_time) 
                              / self.system.block_duration_raster) * self.system.block_duration_raster
        
                             
        recovery_time = self.repetitionTime - 0.5*(self.rfExTime+self.echoSpacing) - self.etl * self.echoSpacing
        # Assertions to check if times are greater than zero
        assert delay_te1 > 0, f"Error: delay_te1 is non-positive: {delay_te1}"
        assert delay_te2 > 0, f"Error: delay_te2 is non-positive: {delay_te2}"
        assert recovery_time > 0, f"Error: recovery_time is non-positive: {recovery_time}"
        
        acq_points = 0
        seq = pp.Sequence(system=self.system)
        #for scan in range(self.nScans):
        if True:
            # Excitation pulse
            seq.add_block(pp.make_delay(0.003))
            seq.add_block(rf_ex)
            seq.add_block(pp.make_delay(delay_te1))
            # Echo train
            for echoIndex in range(self.etl):
                
                phase = 0
                if self.phase_mode == 'CP':
                    phase = 0.0
                elif self.phase_mode == 'CPMG':
                    phase = np.pi/2
                elif self.phase_mode == 'APCP':
                    phase = ((-1)**(echoIndex)+1)*np.pi/2
                elif self.phase_mode == 'APCPMG':
                    phase = (-1)**echoIndex*np.pi/2
                rf_ref.phase_offset = phase
                adc.phase_offset = phase
                seq.add_block(rf_ref)
                seq.add_block(pp.make_delay(delay_te2))
                seq.add_block(adc)
                seq.add_block(pp.make_delay(delay_te3))
                
                acq_points += self.nPoints

            seq.add_block(pp.make_delay(recovery_time))

        if plotSeq:
            # Check whether the timing of the sequence is correct
            ok, error_report = seq.check_timing()
            if ok:
                print("Timing check passed successfully")
            else:
                print("Timing check failed. Error listing follows:")
                [print(e) for e in error_report]   

            seq.plot()

        seq.set_definition(key="Name", value="cpmg")
        seq.write("cpmg.seq")
        self.waveforms, param_dict = self.flo_interpreter.interpret("cpmg.seq")
         
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
        if not self.floDict2Exp():
            print("ERROR: Sequence waveforms out of hardware bounds")
            return False
        else:
            encoding_ok = True
            # print("Sequence waveforms loaded successfully")

        if self.plotSeq and self.standalone:
            # Plot the sequence if requested and return immediately
            self.sequencePlot(standalone=self.standalone)
            
        data_over = []
        # If not plotting the sequence, start scanning
        if not self.plotSeq:
            for scan in range(self.nScans):
                print(f"Scan {scan + 1}, running...")
                acquired_points = 0
                expected_points = self.nPoints * self.etl  # Expected number of points

                # Continue acquiring points until we reach the expected number
                while acquired_points != expected_points:
                    if not self.demo:
                        rxd, msgs = self.expt.run()  # Run the experiment and collect data
                    else:
                        # In demo mode, generate random data as a placeholder
                        rxd = {self.rxChName: np.random.randn(expected_points) + 1j * np.random.randn(expected_points)}
                    # Update acquired points
                    rx_raw_data = rxd[self.rxChName]
                    rxdata = rx_raw_data
                    acquired_points = np.size(rxdata)

                    # Check if acquired points coincide with expected points
                    if acquired_points != expected_points:
                        print("WARNING: data apoints lost!")
                        print("Repeating batch...")

                # Concatenate acquired data into the oversampled data array
                data_over = np.concatenate((data_over, rxdata), axis=0)
                print(f"Acquired points = {acquired_points}, Expected points = {expected_points}")
                print(f"Scan {scan + 1}, ready!")
                # plt.plot(data_over)
                # plt.show()
            # Decimate the oversampled data and store it
            self.mapVals['data_over'] = data_over

            # Average data
            data = np.average(np.reshape(data_over, (self.nScans, -1)), axis=0)
            self.mapVals['data'] = data

        

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
            raw_data_reshape = np.reshape(rawdata, newshape=(self.mapVals['etl'], self.mapVals['nPoints']))
            
            for line in range(raw_data_reshape.shape[0]):
                raw_data_reshape[line, :] = raw_data_reshape[line, :] * expiangle[line]
            signal = np.reshape(raw_data_reshape, -1)
        else: 
            signal = self.mapVals['data']
 
        bw = self.mapVals['bw_MHz']*1e3 # kHz
        nPoints = self.mapVals['nPoints'] * self.mapVals['etl']
        deadTime = 0 #self.mapVals['deadTime']*1e-3 # ms
        rfRectExTime = self.mapVals['rfExTime']*1e-3 # ms
        tVector = np.linspace(rfRectExTime/2 + deadTime + 0.5/bw, rfRectExTime/2 + deadTime + (nPoints-0.5)/bw, nPoints)
        fVector = np.linspace(-bw/2, bw/2, nPoints)
        spectrum = np.abs(np.fft.ifftshift(np.fft.ifftn(np.fft.ifftshift(signal))))
        fitedLarmor=self.mapVals['larmorFreq'] - fVector[np.argmax(np.abs(spectrum))] * 1e-3  #MHz
        hw.larmorFreq=fitedLarmor
        # print(f"self{self.larmorFreq}, map{self.mapVals['larmorFreq'] }, fv{fVector[np.argmax(np.abs(spectrum))]},fit larmor{fitedLarmor}")
        fwhm=getFHWM(spectrum, fVector, bw)
        dB0=fwhm*1e6/hw.larmorFreq

        # for sequence in self.sequenceList.values():
        #     if 'larmorFreq' in sequence.mapVals:
        #         sequence.mapVals['larmorFreq'] = hw.larmorFreq
        self.mapVals['larmorFreq'] = hw.larmorFreq

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
                   'yLabel': 'Signal amplitude (mV)',
                   'title': 'Signal vs time',
                   'legend': ['abs', 'real', 'imag'],
                   'row': 0,
                   'col': 0}

        # Add frequency spectrum to the layout
        result2 = {'widget': 'curve',
                   'xData': fVector,
                   'yData': [spectrum],
                   'xLabel': 'Frequency (kHz)',
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
    seq = CPMGPSEQ()
    seq.sequenceAtributes()
    seq.sequenceRun(plotSeq=False, demo=False, standalone=True)
    seq.sequenceAnalysis(mode='Standalone')


