# import numpy as np
# import matplotlib.pyplot as plt
# from scipy.signal import lfilter

# # 5.1 Pre-enhancement compensation function test
# # Custom parameters (used in the paper for simulating eddy current compensation)

# # Q-direction direct terms
# A = np.array([18, 12, 6, 3])  # Percentages
# A = A / 100
# T = np.array([0.5, 5, 50, 500])  # Time constants in ms
# T = T * 1E3  # Convert to microseconds
# ALPHA = np.array([0.383494796, 0.159428847, 0.06601789, 0.03040273])  # ALPHA values
# TAU = np.array([0.384543433, 4.35301123, 46.94852793, 485.1239174])  # TAU values
# TAU = TAU * 1E3  # Convert to microseconds

# # 5.2 Set up the input gradient: trapezoidal gradient waveform
# DT = 1  # Simulation time step in microseconds
# DNUM = 100  # Number of points for gradient rise and fall

# TRISE = DT * DNUM  # Total rise time of the gradient
# TFLAT = 5 * TRISE  # Duration of the flat top region (5 times the rise time)
# TFALL = DT * DNUM  # Total fall time (set equal to rise time for now)
# AMP = 1  # Maximum gradient amplitude

# TSTART = 0  # Start time of waveform
# M = 2.0  # Total duration multiplier for the gradient waveform
# TMAX = M * (TRISE + TFLAT + TFALL)  # Maximum simulation time
# T = np.arange(TSTART, TMAX + DT, DT)  # Time range for simulation

# # Generate trapezoidal waveform
# def trapezoidal_waveform(amp, t, dt, m, dnum):
#     x_tz = np.zeros(len(t))
#     for i in range(len(t)):
#         if t[i] < TRISE:
#             x_tz[i] = amp * (t[i] / TRISE)
#         elif t[i] < (TRISE + TFLAT):
#             x_tz[i] = amp
#         elif t[i] < (TRISE + TFLAT + TFALL):
#             x_tz[i] = amp * (1 - (t[i] - (TRISE + TFLAT)) / TFALL)
#         else:
#             x_tz[i] = 0
#     return x_tz

# X_TZ = trapezoidal_waveform(AMP, T, DT, M, DNUM)

# # Test 10: Check trapezoidal gradient
# plt.figure(10)
# plt.grid(True)
# plt.plot(T, X_TZ, '.')
# plt.show()

# # 5.3 Build pre-enhancement system: apply 1D digital filter FILTER
# N = len(ALPHA)
# unit = np.ones(N)
# BETA_TAU = np.exp(-DT / TAU)  # Discretized exponential time constants
# SUMYPE = np.zeros(X_TZ.shape)

# for i in range(N):
#     para_xpe = [ALPHA[i], -ALPHA[i]]
#     para_ype = [unit[i], -BETA_TAU[i]]
#     ype = lfilter(para_xpe, para_ype, X_TZ)
#     SUMYPE += ype

# SUMINPUT = X_TZ + SUMYPE

# # Test 11: Check total input after pre-enhancement
# plt.figure(11)
# plt.grid(True)
# plt.plot(T, SUMINPUT, '.')
# plt.show()

# # 5.4 Eddy current simulation system: apply 1D digital filter FILTER
# beta_T = np.exp(-DT / T)  # Discretized exponential time constants
# sumyEC = np.zeros(X_TZ.shape)
# for i in range(N):
#     para_xEC = [-A[i], A[i]]
#     para_yEC = [unit[i], -beta_T[i]]
#     yEC = lfilter(para_xEC, para_yEC, X_TZ)
#     sumyEC += yEC

# sumOutput_WithoutPE = X_TZ + sumyEC

# # Test 12: Check original Eddy current system output
# plt.figure(12)
# plt.grid(True)
# plt.plot(T, sumOutput_WithoutPE, '.')
# plt.show()

# # 5.5 Pre-enhancement + Eddy current system simulation
# sumyPEEC = np.zeros(X_TZ.shape)
# for i in range(N):
#     para_xEC = [-A[i], A[i]]
#     para_yEC = [unit[i], -beta_T[i]]
#     yEC_PE = lfilter(para_xEC, para_yEC, SUMINPUT)
#     sumyPEEC += yEC_PE

# sumOutput_WithPE = SUMINPUT + sumyPEEC

# # Test 13: Check output of pre-enhancement + Eddy current system
# plt.figure(13)
# plt.grid(True)
# step = 25
# plt.plot(T[0::step], sumyEC[0::step] * 100, 'bv-')  # Eddy current cross-term
# plt.plot(T[0::step], SUMINPUT[0::step] * 100, 'r^-')  # Pre-enhanced input signal
# plt.plot(T[0::step], sumOutput_WithPE[0::step] * 100, 'ko-')  # Final output with pre-enhancement and Eddy current
# plt.xlabel('Time (μs)')
# plt.ylabel('Amplitude (%)')
# plt.legend(['Eddy Current Cross-term', 'Cross-term Pre-enhancement', 'Eddy Current Compensation'])
# plt.show()

# # Optional: Testing for additional parameters and fitting
# # Fitting process and other advanced calculations could go here if needed.
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import lfilter

def pre_emphasis_grad_directterm(grad_data, pre_emphasis_params):
    # Unpack gradient data tuple
    time_list, amp_list = grad_data
    # Unpack pre-emphasis parameters tuple
    alpha, tau = pre_emphasis_params
    
    # Step 1: Calculate DT (time step)
    DT = np.diff(time_list)[10]  # Assuming uniform time steps
    
    X_TZ = amp_list
    
    # Step 3: Apply pre-emphasis compensation (using filter)
    N = len(alpha)
    unit = np.ones(N)
    BETA_TAU = np.exp(-DT / tau)  # Discretized exponential time constants
    SUMYPE = np.zeros(X_TZ.shape)

    # Applying the filter for each element of alpha and tau
    for i in range(N):
        para_xpe = [alpha[i], -alpha[i]]
        para_ype = [unit[i], -BETA_TAU[i]]
        ype = lfilter(para_xpe, para_ype, X_TZ)
        SUMYPE += ype
    
    # Step 4: Compute the total input with pre-enhancement
    SUMINPUT = X_TZ + SUMYPE
    
    # Return the final output
    return SUMINPUT


def interpolate_grad_waveform(gradTime, gradAmp, rasterTime):
    # Generate an empty list for the new gradTime and gradAmp
    new_gradTime = []
    new_gradAmp = []

    # Iterate over gradTime to generate the new gradTime and gradAmp
    for i in range(1, len(gradTime)):
        start_time = gradTime[i-1]
        end_time = gradTime[i]
        start_amp = gradAmp[i-1]
        end_amp = gradAmp[i]
        
        # Add the starting time and amplitude to the new lists
        if i == 1 or start_time != gradTime[i-1]:
            new_gradTime.append(start_time)
            new_gradAmp.append(start_amp)

        # If the amplitudes are the same, no interpolation is needed
        if start_amp == end_amp:
            # Add all the intermediate time points with the same amplitude
            intermediate_times = np.arange(start_time, end_time, rasterTime)
            new_gradTime.extend(intermediate_times)
            new_gradAmp.extend([start_amp] * len(intermediate_times))
        else:
            # Perform linear interpolation if the amplitudes differ
            for t in np.arange(start_time, end_time, rasterTime):
                # Linear interpolation formula
                gradAmp_interpolated = start_amp + (end_amp - start_amp) * (t - start_time) / (end_time - start_time)
                new_gradTime.append(t)
                new_gradAmp.append(gradAmp_interpolated)
                
    # Append the last time and amplitude values
    new_gradTime.append(gradTime[-1])
    new_gradAmp.append(gradAmp[-1])

    
    return np.array(new_gradTime), np.array(new_gradAmp)
import numpy as np

def fill_grad_waveform(gradTime, gradAmp, rasterTime):
    # Generate new gradTime based on rasterTime
    new_gradTime = []
    new_gradAmp = []
    
    # Start by adding the first gradTime and gradAmp
    new_gradTime.append(gradTime[0])
    new_gradAmp.append(gradAmp[0])
    
    # Iterate over gradTime to fill the missing time intervals
    for i in range(1, len(gradTime)):
        start_time = gradTime[i-1]
        end_time = gradTime[i]
        start_amp = gradAmp[i-1]
        
        # Add intermediate gradTimes with the same gradAmp until the next gradTime
        intermediate_times = np.arange(start_time + rasterTime, end_time, rasterTime)
        new_gradTime.extend(intermediate_times)
        new_gradAmp.extend([start_amp] * (len(intermediate_times)))
        
        # Add the current gradTime and gradAmp
        new_gradTime.append(end_time)
        new_gradAmp.append(gradAmp[i])

    return np.array(new_gradTime), np.array(new_gradAmp)
import numpy as np

def reduce_grad_waveform(gradTime, gradAmp, threshold=1e-4):
    # Initialize the reduced gradTime and gradAmp lists
    new_gradTime = [gradTime[0]]
    new_gradAmp = [gradAmp[0]]

    # Iterate through the gradTime and gradAmp to remove redundant points
    for i in range(1, len(gradTime)):
        if abs(gradAmp[i] - new_gradAmp[-1]) >= threshold:  # Check if the amplitude change is significant
            new_gradTime.append(gradTime[i])
            new_gradAmp.append(gradAmp[i])
    
    return np.array(new_gradTime), np.array(new_gradAmp)


if __name__ == "__main__":
# Example usage:
    if False:
        gradTime = np.array([0, 10, 40, 60, 100, 120, 140, 160])
        gradAmp = np.array([0, 1, 1, 0, 0, 1, 1, 0])
        new_gradTime, new_gradAmp = reduce_grad_waveform(gradTime, gradAmp)

        print("new_gradTime:", new_gradTime)
        print("new_gradAmp:", new_gradAmp)


    if False:
        # Example usage:
        gradTime = np.array([0, 10, 40, 60, 100, 120, 140, 160])
        gradAmp = np.array([0, 1, 1, 0, 0, 1, 1, 0])
        rasterTime = 5

        new_gradTime, new_gradAmp = fill_grad_waveform(gradTime, gradAmp, rasterTime)

        print("new_gradTime:", new_gradTime)
        print("new_gradAmp:", new_gradAmp)

        # q: plot original and interpolated gradient waveform
        plt.figure(1)
        plt.plot(gradTime, gradAmp, 'o-', label='Original')
        plt.plot(new_gradTime, new_gradAmp, 'x-', label='Interpolated')
        plt.xlabel('Time (ms)')
        plt.ylabel('Amplitude')
        plt.legend()
        plt.show()

    if True:
        DT = 10  # Simulation time step in microseconds
        DNUM = 100  # Number of points for gradient rise and fall

        TRISE = DT * DNUM  # Total rise time of the gradient
        TFLAT = 5 * TRISE  # Duration of the flat top region (5 times the rise time)
        TFALL = DT * DNUM  # Total fall time (set equal to rise time for now)
        AMP = 1  # Maximum gradient amplitude

        TSTART = 0  # Start time of waveform
        M = 2.0  # Total duration multiplier for the gradient waveform
        TMAX = M * (TRISE + TFLAT + TFALL)  # Maximum simulation time
        T = np.arange(TSTART, TMAX + DT, DT)  # Time range for simulation

        # Generate trapezoidal waveform
        def trapezoidal_waveform(amp, t, dt, m, dnum):
            x_tz = np.zeros(len(t))
            for i in range(len(t)):
                if t[i] < TRISE:
                    x_tz[i] = amp * (t[i] / TRISE)
                elif t[i] < (TRISE + TFLAT):
                    x_tz[i] = amp
                elif t[i] < (TRISE + TFLAT + TFALL):
                    x_tz[i] = amp * (1 - (t[i] - (TRISE + TFLAT)) / TFALL)
                else:
                    x_tz[i] = 0
            return x_tz

        X_TZ = trapezoidal_waveform(AMP, T, DT, M, DNUM)
        # Example usage
        grad_data = (T, X_TZ)  # Example time and amplitude arrays
        pre_emphasis_params = (np.array([0.383494796, 0.159428847, 0.06601789, 0.03040273]), 
                            np.array([0.384543433, 4.35301123, 46.94852793, 485.1239174] )* 1e3)  # Example alpha and tau arrays

        # Call the function
        SUMINPUT = pre_emphasis_grad_directterm(grad_data, pre_emphasis_params)

        # Plot the result
        plt.figure(1)
        plt.grid(True)
        plt.plot(grad_data[0], X_TZ, '.')

        plt.grid(True)
        plt.plot(grad_data[0], SUMINPUT, '.')
        plt.show()


