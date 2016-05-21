import array
import struct
import sys
import time
import os

from subprocess import call

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

from time import sleep

### python plot_mov_avg.py <folder> <num files>

if len(sys.argv) < 4:
    print("Usage: python plot_mov_avg.py <folder> <num files> <num measurements>")
    print("Example: python plot_mov_avg.py Test1 10 16")
    sys.exit(1)

plt.ion()
fig = plt.figure()
ax = fig.add_subplot(111)   
scatter, = ax.plot([0,int(sys.argv[3])], [-135, 10], 'r')
plt.show()
plt.grid(True)
        
def plot_mov_avg(values):
    scatter.set_xdata(range(len(values)))
    scatter.set_ydata(values)
    fig.canvas.draw()

for l in range(int(sys.argv[2])):
    raw_input("Hit enter to continue:")
    mov_avg0_0 = np.loadtxt(sys.argv[1]+"/moving_average_0_" + str(l) + ".csv", delimiter=',')
    for k in range(217):
        plot_mov_avg(mov_avg0_0[k])
        sleep(0.1)





