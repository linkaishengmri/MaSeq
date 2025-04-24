

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


class blankSeqAnalysis(blankSeq.MRIBLANKSEQ):    
    def __init__(self, mat_contents, seq_filename):
        super(blankSeqAnalysis, self).__init__()
        
        # transfer mat_contents to mapVals
        self.meta_data = {}
        self.mapVals = {}
        self.mapVals['data_full'] = mat_contents['data_full']
        self.mapVals['nScans'] = mat_contents['nScans'][0][0]
        self.mapVals['larmorFreq'] = mat_contents['larmorFreq'][0][0]
        self.mapVals['rfExFA'] = mat_contents['rfExFA'][0][0]
        self.mapVals['rfSincExTime'] = mat_contents['rfSincExTime'][0][0]
        self.mapVals['repetitionTime'] = mat_contents['repetitionTime'][0][0]
        # self.mapVals['echoTime'] = mat_contents['echoTime'][0][0]
        self.mapVals['fovInPlane'] = mat_contents['fovInPlane'][0]
        self.mapVals['thickness'] = mat_contents['thickness'][0][0]
        self.mapVals['sliceGap'] = mat_contents['sliceGap'][0][0]
        self.mapVals['dfov'] = mat_contents['dfov'][0]
        self.mapVals['nPoints'] = mat_contents['nPoints'][0]
        self.mapVals['axesOrientation'] = mat_contents['axesOrientation'][0]
        #self.mapVals['dummyPulses'] = mat_contents['dummyPulses'][0][0]
        self.mapVals['bandwidth'] = mat_contents['bandwidth'][0][0]
        self.mapVals['DephTime'] = mat_contents['DephTime'][0][0]
        self.mapVals['shimming'] = mat_contents['shimming'][0]
        # self.mapVals['gradSpoil'] = mat_contents['gradSpoil'][0][0]
        # self.mapVals['RFSpoilPhase'] = mat_contents['RFSpoilPhase'][0][0]
        # self.mapVals['phaseGradSpoilMode'] = mat_contents['phaseGradSpoilMode'][0][0]
        self.mapVals['sliceIdx'] = mat_contents['sliceIdx'][0]
        self.mapVals['bw_MHz'] = mat_contents['bw_MHz'][0][0]
        self.mapVals['sampling_period_us'] = mat_contents['sampling_period_us'][0][0]
        self.mapVals['n_readouts'] = mat_contents['n_readouts'][0]
        self.mapVals['n_batches'] = mat_contents['n_batches'][0][0]
        self.mapVals['resolution'] = mat_contents['resolution'][0]
        self.mapVals['data'] = mat_contents['data'][0]
        self.mapVals['kSpace'] = mat_contents['kSpace']
        self.mapVals['iSpace'] = mat_contents['iSpace']
        self.mapVals['name_string'] = mat_contents['name_string'][0]
        self.mapVals['fileName'] = mat_contents['fileName'][0]
        self.mapVals['fileNameIsmrmrd'] = mat_contents['fileNameIsmrmrd'][0]
         
    def sequenceAnalysis(self, mode=None):
        if False:
            self.mode = mode
            
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
            expiangle = self.flo_interpreter.get_rx_phase_dict()['rx0']
            raw_data = np.reshape(data_prov, newshape=(1, self.nScans, -1, nRD))
            for scan in range(self.nScans):
                for line in range(raw_data.shape[2]):
                    raw_data[0, scan, line, :] = raw_data[0, scan, line, :] * expiangle[line]
            data_full = np.reshape(raw_data, -1)
            
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

            # for s_i in range(nSL):
            #     for ex_i in range(n_ex):
            #         data_arrange_slice[slice_idx[s_i], ex_i, :, :] = data_shape[ex_i, s_i, :, :]

            # # Generate different k-space data
            # data_ind = np.reshape(data_arrange_slice, newshape=(1, nSL, nPH, nRD))

        
            # chNum = 1 
            # kspace_single_slice = np.zeros([nSL, nPH, nRD], dtype=complex)
            # for s_i in range(nSL):
            #     data_ind = np.reshape(data_ind[0, s_i, :, :], newshape=(chNum, nPH, nRD))
            #     kspace_single_slice[s_i, : ,:] = sort_data_implicit(kdata=data_ind, seq=self.lastseq, shape=(1, nPH, nRD)) 

            # data_ind = np.reshape(kspace_single_slice, newshape=(1, nSL,nPH, nRD))
            self.mapVals['kSpace'] = data_ind

            # plot #0 slice: #####################################
            # first_slice_kspace = np.reshape(data_ind[0, 0, :, :], newshape=(nPH, nRD))
            # plot_nd(first_slice_kspace, vmax=10)
            # plt.title('K-space')
            ######################################################

            
            # Get images
            image_ind = np.zeros_like(data_ind)
            im = ifft_2d(data_ind[0])
            image_ind[0] = im

            # for echo in range(1):
            #     image_ind[echo] = np.fft.ifftshift(np.fft.ifftn(np.fft.ifftshift(data_ind[echo])))
            self.mapVals['iSpace'] = image_ind
        else:
            self.axesOrientation = self.mapVals['axesOrientation']
            image_ind = self.mapVals['iSpace']
            data_ind = self.mapVals['kSpace']
            nSL = self.mapVals['nPoints'][2]
            nPH = self.mapVals['nPoints'][1]
            nRD = self.mapVals['nPoints'][0] 

            
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
        if not False:  # Image orientation
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
        

        # create self.out to run in iterative mode
        self.output = [result1, result2]
 


        # save data once self.output is created
        self.session["subject_id"] = 'abc'
        self.session["study_id"] = 'abc'
        self.session["scanner"] = 'abc'
        self.session["seriesNumber"] = 0
        self.image2Dicom(fileName='generated_dicom32.dcm')
        
        #self.saveRawData()

        # Plot result in standalone execution
        if True:
            self.plotResults()

        return self.output
 
def mat2dicom(mat_filename, seq_filename, dicom_filename = None):
    # load mat file in python
    import scipy.io as sio
    mat_contents = sio.loadmat(mat_filename)
    #print(mat_contents)
    print(mat_contents['data_full'].shape)

    seq = blankSeqAnalysis(mat_contents, seq_filename)
    seq.sequenceAnalysis()
    
if __name__ == '__main__':
    mat2dicom(mat_filename = '/home/lks/MaSeq_pack/MaSeq/seq/experiments/acquisitions/2025.04.22/mat/raw_data_20250422_203949.mat',
              seq_filename = '/home/lks/MaSeq_pack/MaSeq/seq/experiments/acquisitions/2024.04.22/seq/raw_data_20250422_203949_1.seq',
              # dicom_filename is based on current time
              )
    




