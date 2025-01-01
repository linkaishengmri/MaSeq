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

class SearchP90PSEQ(blankSeq.MRIBLANKSEQ):
    def __init__(self):
        super(SearchP90PSEQ, self).__init__()
        # Input the parameters
        self.output = None
        self.expt = None

        self.seqName = None 
        self.seqName = None             
        self.nScans = None              
        self.larmorFreq = None          
        self.rfExFA = None              
        self.rfExAmpSearchRange = None  
        self.rfExTime = None            
        self.deadTime = None            
        self.repetitionTime = None      
        self.nPoints = None             
        self.bandwidth = None                           
        self.shimming = None            
        self.txChannel = None           
        self.rxChannel = None           


        self.addParameter(key='seqName', string='search_p90', val='fid')
        self.addParameter(key='nScans', string='Number of scans', val=1, field='SEQ')
        self.addParameter(key='larmorFreq', string='Larmor frequency (MHz)', val=10.35393, units=units.MHz, field='RF')
        self.addParameter(key='rfExFA', string='Excitation flip angle (deg)', val=90, field='RF')
        
        self.addParameter(key='rfExAmpSearchRange', 
                          string='RF Amp Searching Range (uT)', val=[355,1,375], field='RF',
                          tip="[begin, step, end]")
        self.addParameter(key='rfExTime', string='RF excitation time (us)', val=20.0, units=units.us, field='RF')
        
        
        self.addParameter(key='deadTime', string='Dead Time (us)', val=300.0, units=units.us, field='SEQ')
        self.addParameter(key='repetitionTime', string='Repetition time (ms)', val=1000., units=units.ms, field='SEQ')
        self.addParameter(key='nPoints', string='Number of acquired points', val=500, field='IM')
        self.addParameter(key='bandwidth', string='Acquisition Bandwidth (kHz)', val=32, units=units.kHz, field='IM',
                          tip="The bandwidth of the acquisition (kHz9. This value affects resolution and SNR.")
        self.addParameter(key='shimming', string='shimming', val=[0.0, 0.0, 0.0], units=units.sh, field='OTH')
        self.addParameter(key='txChannel', string='Tx channel', val=0, field='RF')
        self.addParameter(key='rxChannel', string='Rx channel', val=0, field='RF')

    def sequenceInfo(self):
        pass
        
    def sequenceTime(self):
        return (self.mapVals['repetitionTime'] *1e-3 * self.mapVals['nScans'] / 60)  # minutes

    def sequenceAtributes(self):
        super().sequenceAtributes()

    def sequenceRunAndAnalysis(self, plotSeq=0, demo=False, standalone=False):
        init_gpa = False
        self.demo = demo
        self.plotSeq = plotSeq
        self.standalone = standalone

        max_grad_Hz = convert(from_value=hw.max_grad, from_unit='mT/m', gamma=hw.gammaB, to_unit='Hz/m')
        
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
            phase_offset=0,
            delay=0,
        )
        
        
        adc = pp.make_adc(num_samples=self.nPoints, duration=readout_duration) 
        
                             
        recovery_time = self.repetitionTime - 0.5*self.rfExTime - self.deadTime - sampling_period * 1e-6
        assert recovery_time > 0, f"Error: recovery_time is non-positive: {recovery_time}"
        
        acq_points = 0
        seq = pp.Sequence(system=self.system)
        for scan in range(self.nScans):      
            # Excitation pulse
            seq.add_block(rf_ex)
            seq.add_block(pp.make_delay(self.deadTime))
            seq.add_block(adc)
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
        seq.write("searchingp90.seq")
        self.mapVals['data'] = np.array([])
        RFp90 = np.arange(self.rfExAmpSearchRange[0], self.rfExAmpSearchRange[2], self.rfExAmpSearchRange[1])
        for p90 in RFp90:
            max_rf_Hz = p90 * 1e-6 * hw.gammaB
            self.flo_interpreter = PseqInterpreter(
                tx_warmup=10,  # Transmit chain warm-up time (us)
                rf_center=hw.larmorFreq * 1e6 ,  # Larmor frequency (Hz)
                rf_amp_max=max_rf_Hz,  # Maximum RF amplitude (Hz)
                grad_max=max_grad_Hz,  # Maximum gradient amplitude (Hz/m)
                grad_t=10,  # Gradient raster time (us)
                grad_eff=hw.gradFactor, # gradient coefficient of efficiency
                tx_ch = self.txChannel,
                rx_ch = self.rxChannel,
                add_rx_points = 0,
                use_multi_freq=True,
            )
            self.waveforms, param_dict = self.flo_interpreter.interpret("searchingp90.seq")
         
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
                self.sequencePlot(standalone=self.standalone)
                
            data_over = []
            # If not plotting the sequence, start scanning
            if not self.plotSeq:
                # for scan in range(self.nScans):
                if True:
                    print(f"Scan running...")
                    acquired_points = 0
                    expected_points = self.nPoints * self.nScans  # Expected number of points

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
                    print(f"Scan ready!")
                    # plt.plot(data_over)
                    # plt.show()
                # Decimate the oversampled data and store it
                self.mapVals['data_over'] = data_over

                # plot each nscan:
                if False:
                    data_to_plot = np.reshape(data_over, (self.nScans, -1))
                    for ind in range(self.nScans):
                        plt.plot(np.abs(data_to_plot[ind]), label=f'max RF:{p90} uT, Scan {ind+1}')
                        plt.legend()
                    plt.show()
                # Average data
                data_for_analysis = np.average(np.reshape(data_over, (self.nScans, -1)), axis=0)

                self.mapVals['data'] = np.concatenate((self.mapVals['data'],data_for_analysis))

        

            if not self.demo:
                self.expt.__del__()
        # analysis: 
        data_ready = self.mapVals['data']
        data_ready = np.reshape(data_ready, newshape=(RFp90.shape[0], -1))
        row_max = np.max(np.abs(data_ready), axis=1)
        plt.plot(RFp90, row_max)
        plt.show()                
        # self.mapVals['n_readouts'] = self.nPoints
        
        return True
 
    
if __name__ == '__main__':
    seq = SearchP90PSEQ()
    seq.sequenceAtributes()
    seq.sequenceRunAndAnalysis(plotSeq=False, demo=False, standalone=True)


