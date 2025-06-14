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
import pandas as pd
import time
from datetime import datetime
# from seq.utils import sort_data_implicit, plot_nd, ifft_2d, combine_coils

from seq.SR_T1_pseq import SRT1PSEQ

class SRT1SPECPSEQ(SRT1PSEQ):
    
    def __init__(self):
        super(SRT1SPECPSEQ, self).__init__()
        self.inversionTimeRange = None
        self.cycleNum = None
        self.addParameter(key='inversionTimeRange', string='Inversion Range (ms)', val=[1,1500], units=units.ms, field='SEQ')
        self.addParameter(key='cycleNum', string='Cycle Number', val=32, field='SEQ')
    
    def sequenceTime(self):
        return (self.mapVals['repetitionTime'] *1e-3 * self.mapVals['cycleNum'] * self.mapVals['nScans'] / 60)  # minutes

    def sequenceAtributes(self):
        super().sequenceAtributes()

    def cycleDataAnalysis(self, mode=None):
        self.mode = mode
        self.mapVals['etl'] = 1
        # Signal and spectrum from 'fir' and decimation 
        # Phase correction
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
        tVector = create_tVector(bw * 1e3, self.mapVals['nPoints'], 1, self.mapVals['etl'])
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
        # filtered_time_vector = np.reshape(filtered_time, newshape=(-1))
        filtered_time_vector = np.array([self.mapVals['inversionTime']])

        
        

        filtered_signalVStime = [filtered_time_vector,filtered_signal]

        if self.mode == 'Standalone':
            self.plotResults()
        return filtered_signalVStime
    
    def sequenceRun(self, plotSeq=0, demo=False, standalone=False):
        
        def calculate_repeat_para(nType, m_Min, m_Max, Counts):
            """
            Calculate Repeat_Para based on nType using numpy.
            
            Parameters:
                nType (int): Type of calculation (1 for FFG TE^3, 2 for PFG G^2, 3 for SR-CPMG, 4 for T2-T2)
                m_Min (float): Minimum value
                m_Max (float): Maximum value
                Counts (int): Number of steps
            
            Returns:
                numpy.ndarray: Array of Repeat_Para values
            """
            indices = np.arange(Counts)  # Create indices from 0 to Counts - 1
            Repeat_Para = np.zeros(Counts)  # Initialize Repeat_Para array

            if nType == 1:  # FFG TE^3
                min3 = m_Min ** 3
                max3 = m_Max ** 3
                values = 10 ** (((np.log10(max3) - np.log10(min3)) / (Counts - 1)) * indices + np.log10(min3))
                Repeat_Para = np.ceil(values ** (1 / 3))  # Apply cube root and round up

            elif nType == 2:  # PFG G^2
                min2 = m_Min ** 2
                max2 = m_Max ** 2
                values = 10 ** (((np.log10(max2) - np.log10(min2)) / (Counts - 1)) * indices + np.log10(min2))
                Repeat_Para = np.floor(values ** (1 / 2))  # Apply square root and round down

            elif nType == 3:  # SR-CPMG
                values = 10 ** (((np.log10(m_Max) - np.log10(m_Min)) / (Counts - 1)) * indices + np.log10(m_Min))
                Repeat_Para = np.floor(values)  # Round down

            elif nType == 4:  # T2-T2
                values = 10 ** (((np.log10(m_Max) - np.log10(m_Min)) / (Counts - 1)) * indices + np.log10(m_Min))
                Repeat_Para = np.floor(values)  # Round down

            return Repeat_Para.astype(int)  # Return as integer array

        repeatPara = calculate_repeat_para(4, self.inversionTimeRange[0]*1e6, self.inversionTimeRange[1]*1e6, self.cycleNum)
        
        now = datetime.now().strftime("%Y%m%d_%H%M%S")
        out_data_mat = np.zeros((self.cycleNum, 2))
        filteredsignal = np.zeros((self.cycleNum, 2), dtype=complex)
        
        for rindex in range(len(repeatPara)):
            # set inversion time
            self.mapVals['inversionTime'] = repeatPara[rindex] * 1e-3  # Convert to ms
            self.inversionTime = repeatPara[rindex] * 1e-6 # Convert to second
            # Run sequence
            super().sequenceRun(plotSeq=plotSeq, demo=demo, standalone=standalone)
            filtered_signalVStime = self.cycleDataAnalysis(mode=self.mode)
            print(f'-----TIval above: {repeatPara[rindex] * 1e-3} ms------- max abs value: {np.abs(filtered_signalVStime[1]).max()}')
            

            # Save results
            
            out_data_mat[rindex, 0] = repeatPara[rindex] 
            out_data_mat[rindex, 1] = np.abs(filtered_signalVStime[1]).max()
            filteredsignal[rindex, 0] = filtered_signalVStime[0]
            filteredsignal[rindex, 1] = np.abs(filtered_signalVStime[1])
            
            time.sleep(0.01)
        
        self.full_raw_data = filteredsignal
        return True
    
    def sequenceAnalysis(self, mode=None):
        self.mode = mode
        assert self.full_raw_data is not None, "Please run sequenceRun first to get the raw data."
        
        filteredsignal = self.full_raw_data
        # data export
        signal_pack = np.column_stack((np.abs(filteredsignal[:, 0])*1e-3, np.abs(filteredsignal[:, 1])))
        # export to .dat
        now = datetime.now().strftime("%Y%m%d_%H%M%S")
        path = os.path.join('experiments/SRT1', now)
        os.makedirs(path, exist_ok=True)
        filename = os.path.join(path, f'T1rawdata.txt')
        np.savetxt(filename, signal_pack, delimiter=',', fmt='%d')

        def display_plot(FiltersignalVStime):
            from flintpy.flintpy import Flint, FlintSignal
            signal_decay = FiltersignalVStime[:, 1]
            time_axis = FiltersignalVStime[:, 0]
            print(signal_decay)
            print(time_axis)
            signal = FlintSignal.load_from_data(signal_decay, time_axis, None)
            flint = Flint(
                signal, kernel_shape=[1000, 1], kernel_name="T1SR", alpha=1e-1, t1range=[1e-6, 1e1], t2range=None
            )
            flint.solve_flint()

            # Generate and capture the Plotly figure from flint.plot()
            fig = flint.plot()
            print()
            fig.update_layout(
                xaxis=dict(
                    tickvals=[1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1e0, 1e1]
                )
            )
            # Display the updated plot
            fig.show()
            plt.show()
            # plot FilterSingal
            # plt.figure()
            # plt.plot(time_axis, signal_decay)
            # plt.title('Filtered Signal')
            # plt.xlabel('Time (s)')
            # plt.ylabel('Amplitude')

        display_plot(signal_pack)
        self.mapVals['full_raw_data'] = self.full_raw_data
        result1 = {'widget': 'curve',
                   'xData': signal_pack[:,0],
                   'yData': [np.abs(signal_pack[:,1]), np.real(signal_pack[:,1]), np.imag(signal_pack[:,1])],
                   'xLabel': 'Time (ms)',
                   'yLabel': 'Filtered signal amplitude',
                   'title': 'Signal vs time',
                   'legend': ['abs', 'real', 'imag'],
                   'row': 1,
                   'col': 0}
        self.output = [result1]
        self.saveRawData()

        if self.mode == 'Standalone':
            self.plotResults()
        return self.output

if __name__ == '__main__':
    seq = SRT1SPECPSEQ()
    init_params = {
        'seqName': 'SRT1',
        'nScans': 4,
        'larmorFreq': 10.33316,
        'rfExFA': 90,
        'rfExTime': 25.0,
        'repetitionTime': 3000,
        'nPoints': 10,
        'filterWindowSize': 10,
        'bandwidth': 426.666667,
        'shimming': [0.0, 0.0, 0.0],
        'Exphase': [0, 180, 90, 270],
        'Refphase': [90, 90, 180, 180],
        'Rxphase': [0, 180, 90, 270],
        'txChannel': 0,
        'rxChannel': 0,
        'saturationPulseqNum': 10,
        'saturationIntervalDecay': 0.29,
        'firstInterval': 100,
        'inversionTime': 100,
        'deadTime': 500,
        'inversionTimeRange': [1, 1500],
        'cycleNum':32,
    }
    seq.mapVals.update(init_params)
    seq.sequenceAtributes()
    seq.sequenceRun(plotSeq=False, demo=True, standalone=True)
    seq.sequenceAnalysis(mode='Standalone')


