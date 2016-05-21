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

################################################################################################
### USAGE:
### Arguments: 
### <folder>: folder in current directory that contains the samples_X files
### <output folder>: folder in which output files will be saved
### python mov_avg_processing.py <input folder> <output folder> <num files> <num measurements> <window size>
### Example: python mov_avg_processing.py Test1 Test2 50 32
#################################################################################################

if len(sys.argv) < 6:
    print("Usage: python mov_avg_processing.py <input folder> <output folder> <num files> <num measurements> <window size>")
    print("Example: python mov_avg_processing.py Test1 Test2 50 32 5")
    sys.exit(1)

plt.ion()
fig = plt.figure()
ax = fig.add_subplot(111)

scatter, = ax.plot([2401, 2472], [-150, 0], 'r')

scatter_min, = ax.plot([], [], 'b')
scatter_max, = ax.plot([], [], 'g')

plt.show()
plt.grid(True)

if not os.path.exists(sys.argv[2]):
    os.makedirs(sys.argv[2])

numFiles = int(sys.argv[3])
numSamples = int(sys.argv[4])
window_size = int(sys.argv[5])
for loop in range(numFiles):

    with open(sys.argv[1] + "samples_" + str(loop), "rb") as file:

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
            
        ###End of processing all frames in 'samples'
        subcarrier_data = [[] for k in xrange(217)]
        
        ###Functions
        def determine_index(freq):
            return (freq-2403.25)/0.3125
        
        def determine_freq_from_index(ind):
            return ind*0.3125+2403.25
                
        def moving_average(a, n):
            ret = np.cumsum(a, dtype=float)
            ret[n:] = ret[n:] - ret[:-n]
            return ret[n - 1:] / n
        
        def plot_subcarrier(values):
            scatter.set_xdata(range(len(values)))
            scatter.set_ydata(10.0 * np.log10(values))
            fig.canvas.draw()
        
        def plot_mov_avg(values,n,numSamples):
            mov_avg = moving_average(values,n)
            #scatter.set_xdata(range(len(mov_avg)))
            #scatter.set_ydata(10.0 * np.log10(mov_avg))
            #fig.canvas.draw()
            
            for i,val in enumerate(mov_avg):
                if i < numSamples-(n-1):
                    if i == numSamples-n:
                        fd.write(str(10.0 * np.log10(val)))
                        break
                    else:
                        fd.write(str(10.0 * np.log10(val)) + ',')
                else:
                    break
            fd.write('\n')
            #sleep(0.1)
            #raw_input("Hit enter to continue:")
        ################################################################
        
        ###Place RSS values in row corresponding to its freq
        for i,(frequency,value) in enumerate(zip(x,y)):
            ind = determine_index(frequency)
            subcarrier_data[int(ind)].append(value)      
        
        ###Write moving average values into file
        fd = open(sys.argv[2] + "/moving_average_" + str(loop) + ".csv",'w')
                   
        for i in range(217):
            plot_mov_avg(subcarrier_data[i],window_size,numSamples)  
        fd.close()
      
      
        ###Plot overall averaged spectrum
        df = pd.DataFrame(np.matrix([x, y]).T, columns = ["freq", "rssi"])
        group = df.groupby('freq')
        
        spectrum = group.mean()
        spectrum_min = group.min()
        spectrum_max = group.max()

        scatter.set_xdata(spectrum.index)
        TestData = [10.0 * np.log10(val) for val in spectrum['rssi']]
        #TestData = spectrum['rssi']
        scatter.set_ydata(TestData)
        
        scatter_min.set_xdata(spectrum_min.index)
        scatter_min.set_ydata([10.0 * np.log10(val) for val in spectrum_min['rssi']])

        scatter_max.set_xdata(spectrum_max.index)
        scatter_max.set_ydata([10.0 * np.log10(val) for val in spectrum_max['rssi']])
        fig.canvas.draw()
    ###End of processing one 'samples' file
    
###End of while loop
