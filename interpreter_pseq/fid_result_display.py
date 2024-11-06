


import numpy as np
import matplotlib.pyplot as plt

import configs.hw_config_pseq as hw
import dict_utils

import numpy as np
import matplotlib.pyplot as plt
from scipy.fftpack import fft,fftshift

def plot_complex_signal(arr, hide_time_phase_curve=True, hide_freq_phase_curve=True):
    freq_domain = np.fft.fft(arr)
    
    fig, axs = plt.subplots(2, 2, figsize=(12, 8), sharex='all')
    
    axs[0, 0].plot(arr.real, label='Real')
    axs[0, 0].plot(arr.imag, label='Imaginary', linestyle='--')
    axs[0, 0].set_title('Time Domain (Real and Imaginary)')
    axs[0, 0].legend()

    
    axs[0, 1].plot(np.abs(arr), label='Magnitude')
    if not hide_time_phase_curve:
        axs[0, 1].plot(np.angle(arr), label='Phase', linestyle='--')
    axs[0, 1].set_title('Time Domain (Magnitude and Phase)')
    axs[0, 1].legend()


    axs[1, 0].plot(freq_domain.real, label='Real')
    axs[1, 0].plot(freq_domain.imag, label='Imaginary', linestyle='--')
    axs[1, 0].set_title('Frequency Domain (Real and Imaginary)')
    axs[1, 0].legend()

    axs[1, 1].plot(np.abs(freq_domain), label='Magnitude')
    if not hide_freq_phase_curve:
        axs[1, 1].plot(np.angle(freq_domain), label='Phase', linestyle='--')
    axs[1, 1].set_title('Frequency Domain (Magnitude and Phase)')
    axs[1, 1].legend()

    plt.tight_layout()
    plt.show()


def plot_complex_signal_with_dwell(arr, dwell = 1e-6, hide_time_phase_curve=True, hide_freq_phase_curve=True):

    N = len(arr)
    fft_data = fft(arr)
    list0_shift = np.array(range(0, N))
    sample_freq = 1 / dwell
    freq0_shift = sample_freq * list0_shift / N - sample_freq / 2  

    fft_amp0 = np.array(np.abs(fft_data) / N * 2)   
    fft_amp0[0] = 1 * fft_amp0[0]
    fft_amp0_shift = fftshift(fft_amp0)

    fft_phase0 = np.array(np.angle(fft_data))  
    fft_phase0_shift = fftshift(fft_phase0)

    fft_real0 = np.array(np.real(fft_data))  
    fft_real0_shift = fftshift(fft_real0)

    fft_imag0 = np.array(np.angle(fft_data))  
    fft_imag0_shift = fftshift(fft_imag0)

    time_axis = np.arange(N) * dwell
    freq_axis = np.fft.fftfreq(N, d=dwell) 

    fig, axs = plt.subplots(2, 2, figsize=(12, 8), sharex='row')
    
    axs[0, 0].plot(time_axis, arr.real, label='Real')
    axs[0, 0].plot(time_axis, arr.imag, label='Imaginary', linestyle='--')
    axs[0, 0].set_title('Time Domain (Real and Imaginary)')
    axs[0, 0].set_xlabel('Time (s)')
    axs[0, 0].legend()

    
    axs[0, 1].plot(time_axis, np.abs(arr), label='Magnitude')
    if not hide_time_phase_curve:
        axs[0, 1].plot(time_axis, np.angle(arr), label='Phase', linestyle='--')
    axs[0, 1].set_title('Time Domain (Magnitude and Phase)')
    axs[0, 1].set_xlabel('Time (s)')
    axs[0, 1].legend()

    axs[1, 0].plot(freq0_shift, fft_real0_shift, label='Real')
    axs[1, 0].plot(freq0_shift, fft_imag0_shift, label='Imaginary', linestyle='--')
    axs[1, 0].set_title('Frequency Domain (Real and Imaginary)')
    axs[1, 0].set_xlabel('Frequency (Hz)')
    axs[1, 0].legend()

    axs[1, 1].plot(freq0_shift, fft_amp0_shift, label='Magnitude')
    if not hide_freq_phase_curve:
        axs[1, 1].plot(freq_axis, fft_phase0_shift, label='Phase', linestyle='--')
    axs[1, 1].set_title('Frequency Domain (Magnitude and Phase)')
    axs[1, 1].set_xlabel('Frequency (Hz)')
    axs[1, 1].legend()

    # find central freq
    max_index = np.argmax(fft_amp0_shift)
    central_freq = freq0_shift[max_index]
    print('[PLOT INFO] Central frequency is',central_freq ,'Hz')

    plt.tight_layout()
    plt.show()

    



def fid_result_display_from_files(filename, drop_data_num = 5, dwell = 0, hide_time_phase_curve=True, hide_freq_phase_curve=True):
    dict=dict_utils.load_dict(filename)
    len_data=len(dict['rx0'])
    print(f'rx0_len: {len_data}')
    rx0_data = dict['rx0'][drop_data_num:]
    if dwell == 0:
        plot_complex_signal(arr = rx0_data,
                            hide_time_phase_curve=hide_time_phase_curve, 
                            hide_freq_phase_curve=hide_freq_phase_curve)
    else:
        plot_complex_signal_with_dwell(arr = rx0_data,
                                       dwell = dwell,
                                       hide_time_phase_curve=hide_time_phase_curve, 
                                       hide_freq_phase_curve=hide_freq_phase_curve)

def fid_result_display(data, drop_data_num = 5, dwell = 3e-6, hide_time_phase_curve=True, hide_freq_phase_curve=True):
    
    print(f'rx0_len: {len(data)}')
    rx0_data = data[drop_data_num:]
    if dwell == 0:
        plot_complex_signal(arr = rx0_data, 
                            dwell = dwell, 
                            hide_time_phase_curve=hide_time_phase_curve, 
                            hide_freq_phase_curve=hide_freq_phase_curve)
    else:
        plot_complex_signal_with_dwell(arr = rx0_data, 
                                       dwell = dwell, 
                                       hide_time_phase_curve=hide_time_phase_curve, 
                                       hide_freq_phase_curve=hide_freq_phase_curve)

if __name__ == '__main__':
    central_freq = fid_result_display_from_files(
                                  filename='./output/fid_20241015_163221.txt',
                                  #filename='./output/fid_20241012_105140.txt',
                                  dwell = 3e-6, 
                                  hide_time_phase_curve=True, 
                                  hide_freq_phase_curve=True)
    
