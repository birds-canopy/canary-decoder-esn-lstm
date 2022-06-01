import glob
import numpy as np
import matplotlib.pyplot as plt

s = np.load("reports/marron1/figure2/mfcc_only/esn0/417-states.npy")

print(s.shape)
plt.plot(s[:20].T)
plt.show()

s = np.load("reports/marron1/figure2/mfcc+delta/esn0/417-states.npy")

print(s.shape)
plt.plot(s[:20].T)
plt.show()

s = np.load("reports/marron1/figure2/delta+delta2/esn0/417-states.npy")

print(s.shape)
plt.plot(s[:20].T)
plt.show()
