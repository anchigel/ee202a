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
    print("Usage: python plot_mov_avg.py <folder> <windowsize> <num measurements> <folder> <folder>")
    print("Example: python plot_mov_avg.py Test1 10 16")
    sys.exit(1)

plt.ion()
fig = plt.figure()
ax = fig.add_subplot(111)   
scatter, = ax.plot([0,int(sys.argv[3])], [-135, 10], 'r')
scatter2, = ax.plot([], [], 'b')
scatter3, = ax.plot([], [], 'g')
plt.show()
plt.grid(True)
        
def plot_mov_avg(values1,values2=0,values3=0):
    scatter.set_xdata(range(len(values1)))
    scatter.set_ydata(values1)
    if len(sys.argv) > 4:
        scatter2.set_xdata(range(len(values2)))
        scatter2.set_ydata(values2)
    if len(sys.argv) > 5:
        scatter3.set_xdata(range(len(values3)))
        scatter3.set_ydata(values3)
    fig.canvas.draw()

for l in range(50):
    raw_input("Hit enter to continue:")
    mov_avg0_0 = np.loadtxt(sys.argv[1]+"/moving_average_" + sys.argv[2] +"_" + str(l) + ".csv", delimiter=',')
    if len(sys.argv) > 4:
        mov_avg0_1 = np.loadtxt(sys.argv[4]+"/moving_average_" + sys.argv[2] + "_" + str(l) + ".csv", delimiter=',')
    if len(sys.argv) > 5:
        mov_avg0_2 = np.loadtxt(sys.argv[5]+"/moving_average_" + sys.argv[2] + "_" + str(l) + ".csv", delimiter=',')
    for k in range(217):
        if len(sys.argv) > 5:
            plot_mov_avg(mov_avg0_0[k],mov_avg0_1[k],mov_avg0_2[k])
        elif len(sys.argv) > 4:
            plot_mov_avg(mov_avg0_0[k],mov_avg0_1[k])
        else:
            plot_mov_avg(mov_avg0_0[k])
        sleep(0.1)

