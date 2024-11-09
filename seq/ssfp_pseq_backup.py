"""
Created on Tuesday, Nov 6th 2024
@author: Kaisheng Lin, School of Electronics, Peking University, China
@Summary: GRE (non-balanced SSFP) sequence coded with pypulseq compatible with MaSeq
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
import math
import pypulseq as pp
import numpy as np
import seq.mriBlankSeq as blankSeq   
import configs.units as units
import scipy.signal as sig
import experiment as ex
import configs.hw_config_pseq as hw
from flocra_pulseq.interpreter_pseq import PseqInterpreter
from pypulseq.convert import convert

class SSFPPSEQ(blankSeq.MRIBLANKSEQ):
    def __init__(self):
        super(SSFPPSEQ, self).__init__()
        # Input the parameters
        self.files = None
        self.output = None
        self.nScans = None
        self.shimming = None
        self.expt = None
        self.larmorFreq = None
        self.addParameter(key='seqName', string='PulseqReader', val='PulseqReader')
        self.addParameter(key='nScans', string='Number of scans', val=1, field='IM')
        self.addParameter(key='larmorFreq', string='Larmor frequency (MHz)', val=3.0, units=units.MHz, field='IM')
        self.addParameter(key='shimming', string='Shimming', val=[0, 0, 0], field='IM', units=units.sh)
        self.addParameter(key='files', string='Files', val="/home/lks/MaSeq_pack/MaSeq/pseq_file/ssfp.seq", field='IM', tip='List .seq files')

    def sequenceInfo(self):
        print("Pulseq Reader")
        print("Author: PhD. J.M. Algarín")
        print("Contact: josalggui@i3m.upv.es")
        print("mriLab @ i3M, CSIC, Spain")
        print("Run a list of .seq files\n")
        

    def sequenceTime(self):
        return 0  # minutes, scanTime

    def sequenceAtributes(self):
        super().sequenceAtributes()

        # Convert files to a list
        self.files = self.files.strip('[]').split(',')
        self.files = [s.strip() for s in self.files]

    def sequenceRun(self, plotSeq=0, demo=False, standalone=False):
        init_gpa = False
        self.demo = demo
        max_grad_Hz = convert(from_value=hw.max_grad, from_unit='mT/m', gamma=hw.gammaB, to_unit='Hz/m')
        max_rf_Hz = hw.max_rf * 1e-6 * hw.gammaB
        # Step 1: Define the interpreter for FloSeq/PSInterpreter.
        # The interpreter is responsible for converting the high-level pulse sequence description into low-level
        # instructions for the scanner hardware. You will typically update the interpreter during scanner calibration.
        self.flo_interpreter = PseqInterpreter(
            tx_warmup=hw.blkTime,  # Transmit chain warm-up time (us)
            rf_center=hw.larmorFreq * 1e6,  # Larmor frequency (Hz)
            rf_amp_max=max_rf_Hz,  # Maximum RF amplitude (Hz)
            grad_max=max_grad_Hz,  # Maximum gradient amplitude (Hz/m)
            grad_t=10,  # Gradient raster time (us)
        )

        # Function to get the dwell time
        def get_seq_info(file_path):
            dwell_time = None
            with open(file_path, 'r') as file:
                lines = file.readlines()
                adc_section = False

                for line in lines:
                    if line.strip() == '[ADC]':
                        adc_section = True
                        continue

                    if adc_section:
                        # If we reach another section, stop processing ADC section
                        if line.startswith('\n'):
                            break

                        # Split the line into components
                        components = line.split()

                        # Check if the line contains the ADC event data
                        if len(components) >= 4:
                            n_readouts = int(components[1])  # Extract the number of acquired points per Rx window
                            dwell_time = int(components[2])  # Extract the dwell time (3rd component)

            return n_readouts, dwell_time

        data_over = []  # To save oversampled data
        for file in self.files:
            print("Running " + file + "...")
            # Get the dwell time and n_readouts
            n_readouts, dwell = get_seq_info(file)  # dwell is in ns

            # Create experiment
            if not self.demo:
                self.expt = ex.Experiment(lo_freq=self.larmorFreq * 1e-6,  # MHz
                                          rx_t=dwell * 1e-3,  # us
                                          init_gpa=init_gpa,
                                          gpa_fhdo_offset_time=(1 / 0.2 / 3.1),
                                          )
                dwell = self.expt.get_rx_ts()[0]
            bw = 1/dwell * 1e9  # Hz
            self.mapVals['samplingPeriod'] = dwell * 1e-9  # s
            self.mapVals['bw'] = bw  # Hz

            # Run the interpreter to get the waveforms
            waveforms, param_dict = self.flo_interpreter.interpret(file)

            # Get number of Rx windows
            n_rx_windows = int(np.sum(waveforms['rx0_en'][1][:]))

            # Convert waveform to mriBlankSeq tools (just do it)
            self.pypulseq2mriblankseq(waveforms=waveforms, shimming=self.shimming)

            if not self.demo:
                if self.floDict2Exp():
                    print("Sequence waveforms loaded successfully")
                    pass
                else:
                    print("ERROR: sequence waveforms out of hardware bounds")
                    return False

            # Run the experiment
            if not plotSeq:
                for scan in range(self.nScans):
                    print("Scan %i running..." % (scan + 1))
                    if not self.demo:
                        rxd, msgs = self.expt.run()
                        rxd['rx0'] = hw.adcFactor * (np.real(rxd['rx0']) - 1j * np.imag(rxd['rx0']))
                    else:
                        rxd = {'rx0': np.random.randn(n_readouts * n_rx_windows) +
                                      1j * np.random.randn(n_readouts * n_rx_windows)}
                    data_over = np.concatenate((data_over, rxd['rx0']), axis=0)
                    print("Acquired points = %i" % np.size([rxd['rx0']]))
                    print("Expected points = %i" % n_readouts * n_rx_windows)
                    print("Scan %i ready!" % (scan + 1))
                    self.mapVals['data_over'] = data_over
            elif plotSeq and standalone:
                self.sequencePlot(standalone=standalone)
                return True

            # Close the experiment
            if not self.demo:
                self.expt.__del__()
        # Process data to be plotted
        if not plotSeq:
            self.mapVals['data_over'] = data_over
            data_full = data_over# sig.decimate(data_over, hw.oversamplingFactor, ftype='fir', zero_phase=True)
            self.mapVals['data_full'] = data_full
        return True

    def sequenceAnalysis(self, mode=None):
        self.mode = mode
        self.etl = 1 # for ssfp
        self.axesOrientation = [0,1,2] # for ssfp
        self.unlock_orientation = 0 # for ssfp
        resolution = np.array([1.0,1.0,1.0]) #self.fov / self.nPoints
        self.mapVals['resolution'] = resolution

        # Get data
        data_full = self.mapVals['data_full']
        nRD, nPH, nSL = 256, 2, 1 # self.nPoints
        nRD = nRD + 2 * hw.addRdPoints
        n_batches = 1 # self.mapVals['n_batches']

        # Reorganize data_full
        data_prov = np.zeros([self.nScans, nRD * nPH * nSL * self.etl], dtype=complex)
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
        data_full = np.reshape(data_prov, -1)

        # Average data
        data_full = np.reshape(data_full, newshape=(self.nScans, -1))
        data = np.average(data_full, axis=0)
        self.mapVals['data'] = data

        # Generate different k-space data
        data_ind = np.zeros(shape=(self.etl, nSL, nPH, nRD), dtype=complex)
        data = np.reshape(data, newshape=(nSL, nPH, self.etl, nRD))
        for echo in range(self.etl):
            data_ind[echo, :, :, :] = np.squeeze(data[:, :, echo, :])

        # Remove added data in readout direction
        data_ind = data_ind[:, :, :, hw.addRdPoints: nRD - hw.addRdPoints]
        self.mapVals['kSpace'] = data_ind

        # Get images
        image_ind = np.zeros_like(data_ind)
        for echo in range(self.etl):
            image_ind[echo] = np.fft.ifftshift(np.fft.ifftn(np.fft.ifftshift(data_ind[echo])))
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
        k_space = np.zeros((self.etl * nSL, nPH, nRD - 2 * hw.addRdPoints))
        image = np.zeros((self.etl * nSL, nPH, nRD - 2 * hw.addRdPoints))
        n = 0
        for slice in range(nSL):
            for echo in range(self.etl):
                k_space[n, :, :] = np.abs(data_ind[echo, slice, :, :])
                image[n, :, :] = np.abs(image_ind[echo, slice, :, :])
                n += 1
        image = image / np.max(image) * 100

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
                   'data': np.log10(k_space),
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
        # self.meta_data["RepetitionTime"] = self.mapVals['repetitionTime']
        # self.meta_data["EchoTime"] = self.mapVals['echoSpacing']
        # self.meta_data["EchoTrainLength"] = self.mapVals['etl']

        # create self.out to run in iterative mode
        self.output = [result1, result2]

        # save data once self.output is created
        self.saveRawData()

        # Plot result in standalone execution
        if self.mode == 'Standalone':
            self.plotResults()

        return self.output
 

if __name__ == '__main__':
    seq = SSFPPSEQ()
    seq.sequenceAtributes()
    seq.sequenceRun(plotSeq=False, demo=False, standalone=True)
    seq.sequenceAnalysis(mode='Standalone')




