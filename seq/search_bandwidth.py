
import numpy as np


nPoint = 32
acq_time = np.zeros(5000)
acq_dw = []
for reg in range(5000):
    acq_time[reg] = reg/122.88 * nPoint #us
    acq_dw.append ((acq_time[reg], reg/122.88))
print(acq_time)


