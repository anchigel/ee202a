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


if len(sys.argv) < 4:
    print("Usage: python plot_subcarrier.py <folder> <num files> <num measurements> <folder> <folder>")
    print("Example: python plot_subcarrier.py Test1 10 16")
    sys.exit(1)

numFiles = int(sys.argv[2])

plt.ion()
fig = plt.figure()
ax = fig.add_subplot(111)   
scatter, = ax.plot([0,int(sys.argv[3])], [-135, 10], 'r')
scatter2, = ax.plot([], [], 'b')
scatter3, = ax.plot([], [], 'g')
plt.show()
plt.grid(True)

def process_samples(folder,i):
    with open(folder + "samples_" + str(i), "rb") as file:

            ###Binary file -> each 76 bytes corresponds to one frame/channel
            data = file.read(76)

            x = []
            y = []
            now = time.time()

            while data != "":
                t, length = struct.unpack(">BH", data[0:3])

                if t != 1 or length != 73:
                    print "only 20MHz supported atm"
                    sys.exit(1)

                ### metadatanumpy.nan
                max_exp, freq, rssi, noise, max_magnitude, max_index, bitmap_weight, tsf = struct.unpack('>BHbbHBBQ', data[3:20])

                ### measurements
                measurements = array.array("B")
                measurements.fromstring(data[20:])

                squaresum = sum([(m << max_exp)**2 for m in measurements])
                if squaresum == 0:
                    data = file.read(76)
                    continue

                for i, m in enumerate(measurements):
                    if m == 0 and max_exp == 0:
                        m = 1

                    v = 10.0**((noise + rssi + 20.0 * np.log10(m << max_exp) - 10.0 * np.log10(squaresum))/10.0)
                    
                    ###20MHz channel with 56 subcarriers used but 64 total subcarriers -> subcarrier freq spaced by 0.3125MHz
                    ###Calculate subcarrier frequencies
                    if i < 28:
                        f = freq - (20.0 / 64) * (28 - i)
                    else:
                        f = freq + (20.0 / 64) * (i - 27)

                    x.append(f)
                    y.append(v)
                    #print str(now) + "," + str(f) + "," + str(v)

                data = file.read(76)
    return x,y
    
def determine_index(freq):
    return (freq-2403.25)/0.3125                
        
def plot_subcarrier(values1,values2=0,values3=0):
    scatter.set_xdata(range(len(values1)))
    scatter.set_ydata(10.0 * np.log10(values1))
    
    if len(sys.argv) > 4:
        scatter2.set_xdata(range(len(values2)))
        scatter2.set_ydata(10.0 * np.log10(values2))
    if len(sys.argv) > 5:
        scatter3.set_xdata(range(len(values3)))
        scatter3.set_ydata(10.0 * np.log10(values3))
    fig.canvas.draw()
   
def fill_subcarrier_data(x,y):
    subcarrier_data = [[] for k in xrange(217)]
    for i,(frequency,value) in enumerate(zip(x,y)):
            ind = determine_index(frequency)
            subcarrier_data[int(ind)].append(value)
    return subcarrier_data

for i in range(numFiles):
    raw_input("Hit enter to continue:")
    
    x1,y1 = process_samples(sys.argv[1],i)
    subcarrier_data1 = fill_subcarrier_data(x1,y1)
        
    if len(sys.argv) > 4:
        x2,y2 = process_samples(sys.argv[4],i)
        subcarrier_data2 = fill_subcarrier_data(x2,y2)
    if len(sys.argv) > 5:  
        x3,y3 = process_samples(sys.argv[5],i)
        subcarrier_data3 = fill_subcarrier_data(x3,y3)
           
    for i in range(217):
        if len(sys.argv) > 5:
            plot_subcarrier(subcarrier_data1[i],subcarrier_data2[i],subcarrier_data3[i])
        elif len(sys.argv) > 4:
            plot_subcarrier(subcarrier_data1[i],subcarrier_data2[i])
        else:
            plot_subcarrier(subcarrier_data1[i])
        sleep(0.1)
        
        
        
    
###End of while loop
