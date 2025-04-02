

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
        self.mapVals['echoTime'] = mat_contents['echoTime'][0][0]
        self.mapVals['fovInPlane'] = mat_contents['fovInPlane'][0]
        self.mapVals['thickness'] = mat_contents['thickness'][0][0]
        self.mapVals['sliceGap'] = mat_contents['sliceGap'][0][0]
        self.mapVals['dfov'] = mat_contents['dfov'][0]
        self.mapVals['nPoints'] = mat_contents['nPoints'][0]
        self.mapVals['axesOrientation'] = mat_contents['axesOrientation'][0]
        self.mapVals['dummyPulses'] = mat_contents['dummyPulses'][0][0]
        self.mapVals['bandwidth'] = mat_contents['bandwidth'][0][0]
        self.mapVals['DephTime'] = mat_contents['DephTime'][0][0]
        self.mapVals['shimming'] = mat_contents['shimming'][0]
        self.mapVals['gradSpoil'] = mat_contents['gradSpoil'][0][0]
        self.mapVals['RFSpoilPhase'] = mat_contents['RFSpoilPhase'][0][0]
        self.mapVals['phaseGradSpoilMode'] = mat_contents['phaseGradSpoilMode'][0][0]
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
        self.image2Dicom(fileName='generated_dicom02.dcm')
        
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
    # mat_contents_format_like this:
    '''
    {'__header__': b'MATLAB 5.0 MAT-file Platform: posix, Created on: Mon Dec  2 17:53:56 2024', '__version__': '1.0', '__globals__': [], 'seqName': array(['flash'], dtype='<U5'), 'nScans': array([[3]]), 'larmorFreq': array([[10.35723]]), 'rfExFA': array([[90]]), 'rfSincExTime': array([[3.]]), 'repetitionTime': array([[300.]]), 'echoTime': array([[8.]]), 'fovInPlane': array([[100, 100]]), 'thickness': array([[5]]), 'sliceGap': array([[1]]), 'dfov': array([[ 0.,  0., 12.]]), 'nPoints': array([[256, 256,   1]]), 'axesOrientation': array([[1, 2, 0]]), 'dummyPulses': array([[0]]), 'bandwidth': array([[40]]), 'DephTime': array([[2.]]), 'shimming': array([[0., 0., 0.]]), 'gradSpoil': array([[6]]), 'RFSpoilPhase': array([[0]]), 'phaseGradSpoilMode': array([[1]]), 'sliceIdx': array([[0]]), 'bw_MHz': array([[0.04]]), 'sampling_period_us': array([[25.]]), 'data_full': array([[-8.37244942e-08+0.j        , -4.73880637e-05+0.00010223j,
        -2.21292211e-03+0.00324123j, ...,  1.67959708e-03-0.03840744j,
         2.68094203e-03-0.00885052j, -1.68780208e-02-0.00138355j]]), 'data_over': array([[-8.37244942e-08+0.j        , -4.73880637e-05+0.00010223j,
        -2.21292211e-03+0.00324123j, ...,  1.67959708e-03-0.03840744j,
         2.68094203e-03-0.00885052j, -1.68780208e-02-0.00138355j]]), 'n_readouts': array([[65536]]), 'n_batches': array([[1]]), 'resolution': array([[0.00039063, 0.00039063, 0.005     ]]), 'data': array([[-5.58163294e-08-2.79081647e-08j, -3.76202060e-05-2.67918381e-06j,
        -2.06272036e-03+1.91143020e-04j, ...,
        -1.22432002e-02-2.67983966e-02j,  1.39261742e-04-3.31591975e-02j,
        -1.30691145e-03-1.51312767e-02j]]), 'kSpace': array([[[[-5.58163293e-08-2.7908165e-08j,
          -3.76202042e-05-2.6791838e-06j,
          -2.06272025e-03+1.9114302e-04j, ...,
          -8.51313490e-03+1.4185580e-02j,
           3.09802964e-03+7.4896584e-03j,
           1.64597332e-02-1.7135334e-03j],
         [ 1.41021907e-02+4.6734456e-03j,
           1.44572388e-02+2.0589694e-02j,
           5.93651319e-03+5.2767084e-03j, ...,
          -8.80463514e-03-5.7656034e-03j,
           1.47572234e-02-1.1958369e-02j,
           2.93017309e-02-1.2508998e-02j],
         [ 1.73974205e-02-1.6699826e-02j,
          -1.85486034e-03-1.2407803e-02j,
           7.07564084e-03+5.9612119e-03j, ...,
           1.88368943e-03-2.1025676e-02j,
           7.23943394e-03-1.5446611e-02j,
          -3.42073175e-03-6.4415671e-03j],
         ...,
         [ 1.24128815e-02-8.7895365e-03j,
           1.39383981e-02-1.5540633e-02j,
           1.20541221e-02+7.7780057e-03j, ...,
          -2.79743061e-03-1.1370791e-02j,
           6.79321028e-03-2.2275822e-02j,
           1.37494318e-02-2.0337377e-02j],
         [ 1.08670769e-02-5.3738845e-03j,
          -2.48139864e-03+4.2717354e-03j,
          -2.18958538e-02-9.4731472e-04j, ...,
           2.08404213e-02+5.1580151e-03j,
           9.59750637e-03-1.0279275e-02j,
           1.56889651e-02+5.0728670e-03j],
         [ 3.58993886e-03+1.6995067e-02j,
          -1.07645420e-02+1.1585851e-02j,
          -1.51329795e-02+4.5567336e-03j, ...,
          -1.22432001e-02-2.6798397e-02j,
           1.39261741e-04-3.3159196e-02j,
          -1.30691146e-03-1.5131277e-02j]]]], dtype=complex64), 'iSpace': array([[[[-3.07287314e-06+8.08322238e-06j,
           5.59873934e-06+7.20612343e-06j,
           9.66056268e-06-6.11905898e-06j, ...,
           1.10564194e-07-6.11523546e-06j,
           6.70456166e-06-2.96171402e-06j,
          -1.42767722e-05-5.91005255e-06j],
         [ 2.36909204e-06-3.06110314e-06j,
          -8.41972451e-06+3.56341980e-06j,
          -9.38407823e-07-1.35996595e-06j, ...,
          -1.14743434e-05+1.77930524e-05j,
           9.01263775e-06-1.49809794e-05j,
          -7.94721018e-06+1.39602944e-05j],
         [-2.93601647e-06-5.68412315e-06j,
           1.99786223e-06+1.00394282e-05j,
          -4.23136089e-06+1.16889669e-05j, ...,
           8.52200810e-06+5.19396463e-06j,
           3.54156577e-06-5.72163299e-06j,
           2.99470571e-06-4.70011992e-06j],
         ...,
         [ 5.51994788e-07-8.24447795e-07j,
           1.26192635e-05-4.02757951e-06j,
           8.75765818e-06-1.02563072e-05j, ...,
          -6.75840556e-06+3.03775732e-06j,
          -5.58140698e-07-3.91865160e-06j,
           1.13669785e-05+4.13568932e-06j],
         [-6.79885716e-07-1.19749775e-05j,
           1.47421429e-06+5.53288055e-06j,
           4.89689137e-06-1.17546715e-05j, ...,
          -6.09199378e-06+1.73607950e-05j,
           7.95166034e-06-1.80811276e-06j,
           2.13083945e-06+7.65380378e-07j],
         [ 6.76865739e-07+2.03408717e-06j,
          -1.26153591e-05+2.46621852e-07j,
          -1.39352851e-05-8.80840435e-06j, ...,
           8.01801798e-06+7.81551171e-06j,
          -1.23054656e-06+1.83235761e-06j,
          -6.09409608e-06+5.12480767e-07j]]]], dtype=complex64), 'name_string': array(['20241202_175356'], dtype='<U15'), 'fileName': array(['raw_data_20241202_175356.mat'], dtype='<U28'), 'fileNameIsmrmrd': array(['raw_data_20241202_175356.h5'], dtype='<U27')}    '''
    
if __name__ == '__main__':
    mat2dicom(mat_filename = '/home/lks/MaSeq_pack/MaSeq/seq/experiments/acquisitions/2024.12.02/mat/raw_data_20241202_171822.mat',
              seq_filename = '/home/lks/MaSeq_pack/MaSeq/seq/experiments/acquisitions/2024.12.02/seq/raw_data_20241202_171822_1.seq',
              # dicom_filename is based on current time
              )
    




