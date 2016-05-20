#!/usr/bin/env python
#
# Copyright 2015 Bastian Bloessl <bloessl@ccs-labs.org>
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
# 
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# GNU General Public License for more details.
# 
# You should have received a copy of the GNU General Public License
# along with this program. If not, see <http://www.gnu.org/licenses/>.


##########################################################################################################
### USAGE:
### For training purposes: <ending filename>: string to be appended to the filename 
###                        <num samples>: number of samples to be collected
### python spectrum.py <folder> <ending filename> <num samples> 
### Example: python spectrum.py Exp1 0 50
###     Output files will include moving_average_X.csv with 50 samples within the file, samples_X
###         X: 0 to (num samples)
###     All located in folder Exp1
##########################################################################################################

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

import paramiko, base64
from scp import SCPClient
import scipy.io as sio
from time import sleep
from shutil import copyfile

plt.ion()
fig = plt.figure()
ax = fig.add_subplot(111)

scatter, = ax.plot([2401, 2472], [-135, 15], 'r')

scatter_min, = ax.plot([], [], 'b')
scatter_max, = ax.plot([], [], 'g')

#fig2 = plt.figure()
#ax2 = fig2.add_subplot(111)
#scatter2, = ax2.plot([0,16], [-135, 15], 'r')

plt.show()
plt.grid(True)

#print "time,freq,signal"

###SSH into router
client = paramiko.SSHClient()
client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
client.connect("192.168.8.1", username='root', password='myRouter')

###Output file name
if len(sys.argv) < 4:
    print("python spectrum.py <folder> <ending filename> <num samples>")
    print("Example: python spectrum.py Exp1 0 50")
    sys.exit(1)

if not os.path.exists(sys.argv[1]):
    os.makedirs(sys.argv[1])

file_name = sys.argv[1] + '/testFeatures_' + sys.argv[2] + '.csv'
open(file_name, 'w').close()

###Run commands on router to began spectral scanning
stdin, stdout, stderr = client.exec_command('echo 16 > /sys/kernel/debug/ieee80211/phy0/ath9k/spectral_count')
stdin, stdout, stderr = client.exec_command('echo chanscan > /sys/kernel/debug/ieee80211/phy0/ath9k/spectral_scan_ctl')

###Determine number of samples to collect
loops = 0
n = int(sys.argv[3])

while loops < n:
    ### do measurement
    
    ###Run commands on router to get channel info and transfer file using scp
    #stdin, stdout, stderr = client.exec_command('./run.sh')
    stdin, stdout, stderr = client.exec_command('iw dev wlan0 scan')
    stdin, stdout, stderr = client.exec_command('cat /sys/kernel/debug/ieee80211/phy0/ath9k/spectral_scan0 > samples')
    #stdin, stdout, stderr = client.exec_command('echo disable > /sys/kernel/debug/ieee80211/phy0/ath9k/spectral_scan_ctl')    
    sleep(1)
    with SCPClient(client.get_transport()) as scp:
        scp.get('samples')

    with open("samples", "rb") as file:

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
            scatter2.set_xdata(range(len(values)))
            scatter2.set_ydata(10.0 * np.log10(values))
            fig2.canvas.draw()
            raw_input("Hit enter to continue:")
        
        def plot_mov_avg(values,n):
            mov_avg = moving_average(values,n)
            #scatter2.set_xdata(range(len(mov_avg)))
            #scatter2.set_ydata(10.0 * np.log10(mov_avg))
            #fig2.canvas.draw()
            
            for i,val in enumerate(mov_avg):
                if i < numSamples-(n-1):
                    if i < numSamples-(n-2):
                        fd.write(str(10.0 * np.log10(val)) + ',')
                    else:
                        fd.write(str(10.0 * np.log10(val)))
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
        
        ###Check number of samples in each subcarrier freq
        ###If less than numSamples, throw out this 'samples' file
        numSamples = 16
        not_enough_samples = False
        for i in range(217):
            if len(subcarrier_data[i]) < numSamples:
                print(str(loops) + ": Subcarrier freq length: " + str(len(subcarrier_data[i])) + " at freq: " + str(determine_freq_from_index(i)))
                not_enough_samples = True
                break
        if not_enough_samples:
            continue
        
        ###Write moving average values into file
        fd = open(sys.argv[1] + "/moving_average_" + sys.argv[2] + "_" + str(loops) + ".csv",'w')
                   
        window_size = 3
        for i in range(217):
            plot_mov_avg(subcarrier_data[i],window_size)  
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
    
    ###Append data from 'samples' to csv file
	tDfile = open(file_name, 'a')
	    
    y = len(TestData)
    print(str(loops+1) + ": " + str(y))
    
    ###If there are less than 217 values in TestData, discard this sample
    if y == 217:
        ###If there are large values in data, discard this sample
        flag = False
        for x in TestData:
            if (x > 10 or x == float('Inf') or x == -float('Inf')):
                flag = True
                break            
        if flag:
            continue
                
        ###Data should be ok, write into file
        for z, item in enumerate(TestData):
            tDfile.write(str(item))
            if z < y-1:
                tDfile.write(',')
        tDfile.write('\n')	
        
        ###Save the samples file
        copyfile('samples', sys.argv[1] + '/samples_' + str(loops))
        loops = loops+1    
    ###End of processing one 'samples' file
    
###End of while loop

tDfile.close()
scp.close()
client.close()
