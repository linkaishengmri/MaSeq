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

from seq.PFG_pseq import PFGPSEQ

class PFGSPECPSEQ(PFGPSEQ):
    
    def __init__(self):
        super(PFGSPECPSEQ, self).__init__()
        self.gradAmpRange = None
        self.cycleNum = None
        self.enaGradAxis = None
        self.addParameter(key='gradAmpRange', string='Grad. Amplitude Range (mT/m)', val=[3.6622, 36.6222], field='SEQ')
        self.addParameter(key='cycleNum', string='Cycle Number', val=16, field='SEQ')
        self.addParameter(key='enaGradAxis', string='Enable Gradient Axis', val=[1, 0, 0], field='SEQ')
    
    def sequenceTime(self):
        return (self.mapVals['repetitionTime'] *1e-3 * self.mapVals['cycleNum'] * self.mapVals['nScans'] / 60)  # minutes

    def sequenceAtributes(self):
        super().sequenceAtributes()

    def cycleDataAnalysis(self, mode=None):
        self.mode = mode
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
        
        
        

        filtered_signalVStime = [filtered_time_vector,filtered_signal]

        if self.mode == 'Standalone':
            self.plotResults()
        return filtered_signalVStime
    
    def sequenceRun(self, plotSeq=0, demo=False, standalone=False):
        
        def gradAmpConvert2mT_m(gradAmp):
            return gradAmp / 32767.0 * hw.max_grad

        def gradAmpConvertFrommT_m(gradAmp):
            return int(gradAmp / hw.max_grad * 32767)

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

        gradAmpDigitized = [gradAmpConvertFrommT_m(gradAmp) for gradAmp in self.gradAmpRange]
        repeatPara = calculate_repeat_para(2, gradAmpDigitized[0], gradAmpDigitized[1], self.cycleNum)
        # txt format:
        # SI(1024), C2(5), TAU(100), RD(2000000), NS(32), D2(100000), D11(5000), D12(5000), D30(1499999), and G18(10000).
        txtPramsList = np.array([
            [self.mapVals['etl'], 0],
            [5, 0],
            [self.mapVals['echoSpacing']*500, 0],
            [self.mapVals['repetitionTime']*1000, 0],
            [self.mapVals['nScans'], 0],
            [8000, 0],
            [12000, 0],
            [60000, 0],
            [3000, 0],
            [gradAmpDigitized[0], 0],
        ])
        
        now = datetime.now().strftime("%Y%m%d_%H%M%S")
        result = np.zeros((len(repeatPara), self.mapVals['etl']), dtype=complex)
        plt.ion() 
        fig, ax = plt.subplots()    
        for rindex in range(len(repeatPara)):
            # set gradAmp
            for i in range(3):
                self.mapVals['gradAmp'][i] = gradAmpConvert2mT_m(repeatPara[rindex]) * self.enaGradAxis[i]
                self.gradAmp[i] = self.mapVals['gradAmp'][i]

            # Run sequence
            super().sequenceRun(plotSeq=plotSeq, demo=demo, standalone=standalone)
            filtered_signalVStime = self.cycleDataAnalysis(mode=self.mode)
            print(f'-----gradAmp above: {repeatPara[rindex]}/32767, {gradAmpConvert2mT_m(repeatPara[rindex]):.4f} mT/m,  ------- max abs value: {np.abs(filtered_signalVStime[1]).max()}')
            result[rindex] = filtered_signalVStime[1]

            # plotting
            ax.plot(filtered_signalVStime[0], np.abs(filtered_signalVStime[1]), label=f'Cycle #{rindex + 1}')  
            ax.legend()
            plt.draw()
            plt.pause(0.1) 

            # Save results
            resultMat = np.column_stack((np.real(result[rindex]), np.imag(result[rindex])))
            # Write the result to txt format:
            txtPramsList[9][0] = repeatPara[rindex] 
            outputMat = np.concatenate((txtPramsList,resultMat), axis=0)

            # Write outputMat to a text file with tab-separated values
            path = os.path.join('experiments/PFG', now)
            os.makedirs(path, exist_ok=True)
            filename = os.path.join(path, f'G18-{repeatPara[rindex]}.txt')
            np.savetxt(filename, outputMat, delimiter='\t', fmt='%s')

            time.sleep(0.01)
        plt.ioff() 
        self.full_raw_data = result
        return True
    
    def sequenceAnalysis(self, mode=None):
        self.mode = mode
        assert self.full_raw_data is not None, "Please run sequenceRun first to get the raw data."
        self.mapVals['full_raw_data'] = self.full_raw_data
        flattened_result = self.full_raw_data.flatten()
        time_vec = np.arange(flattened_result.size) * self.mapVals['echoSpacing']
        result1 = {'widget': 'curve',
                   'xData': time_vec,
                   'yData': [np.abs(flattened_result), np.real(flattened_result), np.imag(flattened_result)],
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
    seq = PFGSPECPSEQ()
    init_params = {
        "seqName": "PFGEcho",
        "nScans": 4,
        "larmorFreq": 10.35622,
        "rfExFA": 90,
        "rfReFA": 180,
        "rfExTime": 25.0,
        "rfReTime": 50.0,
        "echoSpacing": 0.7,
        "repetitionTime": 4000,
        "nPoints": 10,
        "filterWindowSize": 10,
        "etl": 4096,
        "bandwidth": 426.666667,
        "shimming": [0.0, 0.0, 0.0],
        "Exphase": [0, 180, 90, 270],
        "Refphase": [90, 90, 180, 180],
        "Rxphase": [0, 180, 90, 270],
        "RxTimeOffset": 0,
        "txChannel": 0,
        "rxChannel": 0,
        "p90GradDeadTime": 1,
        "p180GradDeadTime": 1,
        "riseTime": 0.5,
        "gradTime": 20,
        "gradAmp": [20.0, 0.0, 0.0],
        "FirstEchoSpacing": 50,
        'gradAmpRange': [3.663, 36.623],
        'cycleNum': 8,
        'enaGradAxis': [1, 0, 0],
    }
    seq.mapVals.update(init_params)
    seq.sequenceAtributes()
    seq.sequenceRun(plotSeq=False, demo=True, standalone=True)
    seq.sequenceAnalysis(mode='Standalone')


