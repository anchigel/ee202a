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
from mpl_toolkits.mplot3d import Axes3D

from time import sleep


if len(sys.argv) < 4:
    print("Usage: python plot_mov_avg.py <folder> <filename num> <num measurements>")
    print("Example: python plot_mov_avg.py Test1 0 32")
    sys.exit(1)

plt.ion()
fig = plt.figure()
ax = fig.gca(projection='3d')
#ax = fig.add_subplot(111, projection='3d')   
#scatter, = ax.plot([0,int(sys.argv[3])], [2400, 2480], [-135, 10], 'r')
ax.set_zlim([-135, 10])
ax.set_ylim([0,int(sys.argv[3])])
ax.set_xlim([2400, 2480])
ax.set_ylabel('Time')
ax.set_xlabel('Frequency (MHz)')
ax.set_zlabel('dBm')
#scatter2, = ax.plot([], [], 'b')
#scatter3, = ax.plot([], [], 'g')
plt.show()
plt.grid(True)
        
def plot_mov_avg(values1,values2=0,values3=0):
    freq = []
    for i in range(217):
        freq.append(i*0.3125+2403.25)
    time = range(len(values1[0]))
    time,freq = np.meshgrid(time,freq)
    #print zaxis
    #print xaxis
    plt.clf()
    sleep(0.5)
    ax = fig.gca(projection='3d')
    ax.set_zlim([-135, 10])
    ax.set_ylim([0,int(sys.argv[3])])
    ax.set_xlim([2400, 2480])
    ax.set_ylabel('Time')
    ax.set_xlabel('Frequency (MHz)')
    ax.set_zlabel('dBm')

    ax.plot_surface(freq, time, values1,color='c')
    """
    scatter.set_xdata(range(len(values1)))
    scatter.set_ydata(values1)
    if len(sys.argv) > 4:
        scatter2.set_xdata(range(len(values2)))
        scatter2.set_ydata(values2)
    if len(sys.argv) > 5:
        scatter3.set_xdata(range(len(values3)))
        scatter3.set_ydata(values3)
    """
    fig.canvas.draw()
raw_input("Hit enter to continue:")
for l in range(50):
    #raw_input("Hit enter to continue:")
    mov_avg0_0 = np.loadtxt(sys.argv[1]+"/moving_average_" + sys.argv[2] +"_" + str(l) + ".csv", delimiter=',')
    #if len(sys.argv) > 4:
    #    mov_avg0_1 = np.loadtxt(sys.argv[4]+"/moving_average_" + sys.argv[2] + "_" + str(l) + ".csv", delimiter=',')
    #if len(sys.argv) > 5:
    #    mov_avg0_2 = np.loadtxt(sys.argv[5]+"/moving_average_" + sys.argv[2] + "_" + str(l) + ".csv", delimiter=',')
    #for k in range(217):
    #if len(sys.argv) > 5:
    #    plot_mov_avg(mov_avg0_0,mov_avg0_1,mov_avg0_2)
    #elif len(sys.argv) > 4:
    #    plot_mov_avg(mov_avg0_0,mov_avg0_1)
    #else:
    plot_mov_avg(mov_avg0_0)

