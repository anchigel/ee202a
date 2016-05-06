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


###########USAGE#########
# For training purposes: First argument: number of people in the room, Second argument: number of samples wanted
# python spectrum.py <# people in room (0-2)> <# samples> 
#
# Without any arguments, output file will be testFeatures.csv with at most 100 samples
# python spectrum.py 
#

import array
import struct
import sys
import time

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

plt.ion()
fig = plt.figure()
ax = fig.add_subplot(111)

scatter, = ax.plot([2401, 2472], [-135, 15], 'r')

scatter_min, = ax.plot([], [], 'b')
scatter_max, = ax.plot([], [], 'g')
plt.show()
plt.grid(True)


#print "time,freq,signal"

####SSH into router
client = paramiko.SSHClient()
client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
client.connect("192.168.8.1", username='root', password='myRouter')

if len(sys.argv) >= 2:
    file_name = 'testFeatures' + sys.argv[1] + '.csv'
    open(file_name, 'w').close()
else:
    open('testFeatures.csv', 'w').close()

stdin, stdout, stderr = client.exec_command('echo 1 > /sys/kernel/debug/ieee80211/phy0/ath9k/spectral_count')
stdin, stdout, stderr = client.exec_command('echo chanscan > /sys/kernel/debug/ieee80211/phy0/ath9k/spectral_scan_ctl')

loops = 0
if len(sys.argv) < 3:
    n = 100
else:
    n = int(sys.argv[2])
#print(n)
while loops < n:
    #print(loops)
    ### do measurement
    
    ###Run script to get channel info and transfer file using scp
    #stdin, stdout, stderr = client.exec_command('./run.sh')
    
    stdin, stdout, stderr = client.exec_command('iw dev wlan0 scan')
    stdin, stdout, stderr = client.exec_command('cat /sys/kernel/debug/ieee80211/phy0/ath9k/spectral_scan0 > samples')
    #stdin, stdout, stderr = client.exec_command('echo disable > /sys/kernel/debug/ieee80211/phy0/ath9k/spectral_scan_ctl')    
    #sleep(1)
    with SCPClient(client.get_transport()) as scp:
        scp.get('samples')

    with open("samples", "rb") as file:

        #
        data = file.read(76)

        x = []
        y = []
        now = time.time()

        fo = open("data.txt", "w+")
        fo.write("max_exp,freq,rssi,noise,max_magnitude,max_index,bitmap_weight,tsf\n")
        data_array = np.array([0,0,0,0,0,0,0,0])

        while data != "":
            t, length = struct.unpack(">BH", data[0:3])

            if t != 1 or length != 73:
                print "only 20MHz supported atm"
                sys.exit(1)

            ### metadata
            max_exp, freq, rssi, noise, max_magnitude, max_index, bitmap_weight, tsf = struct.unpack('>BHbbHBBQ', data[3:20])

            #print "max_exp: "       + str(max_exp)
            #print "freq: "          + str(freq)
            #print "rssi: "          + str(rssi)
            #print "noise: "         + str(noise)
            #print "max_magnitude: " + str(max_magnitude)
            #print "max_index: "     + str(max_index)
            #print "bitmap_weight: " + str(bitmap_weight)
            #print "tsf: "           + str(tsf)

            tmp = np.array([max_exp, freq, rssi, noise, max_magnitude, max_index, bitmap_weight, tsf])
            data_array = np.vstack((data_array,tmp))            

            fo.write(str(max_exp))
            fo.write(',')
            fo.write(str(freq))
            fo.write(',')
            fo.write(str(rssi))
            fo.write(',')
            fo.write(str(noise))
            fo.write(',')
            fo.write(str(max_magnitude))
            fo.write(',')
            fo.write(str(max_index))
            fo.write(',')
            fo.write(str(bitmap_weight))
            fo.write(',')
            fo.write(str(tsf))
            fo.write('\n')


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
                
                #v = 10.0**((rssi + 20.0 * np.log10(m << max_exp) - 10.0 * np.log10(squaresum))/10.0)
                v = 10.0**((noise + rssi + 20.0 * np.log10(m << max_exp) - 10.0 * np.log10(squaresum))/10.0)
                #v = noise + rssi + 20.0 * np.log10(m << max_exp) - 10.0 * np.log10(squaresum)
                
                ###20MHz channel with 56 subcarriers used but 64 total subcarriers
                ###Calculate subcarrier frequencies
                if i < 28:
                    f = freq - (20.0 / 64) * (28 - i)
                else:
                    f = freq + (20.0 / 64) * (i - 27)

                x.append(f)
                y.append(v)
                #print str(now) + "," + str(f) + "," + str(v)

            data = file.read(76)
            
        ###End of processing one frame in 'samples'

        df = pd.DataFrame(np.matrix([x, y]).T, columns = ["freq", "rssi"])
        group = df.groupby('freq')
        spectrum = group.mean()
        spectrum_min = group.min()
        spectrum_max = group.max()

        ### print output
        #sys.stdout.write(str(time.time()))
        #for freq, row in spectrum.iterrows():
        #    sys.stdout.write("," + str(freq) + ":" + str(row['rssi']))
        #sys.stdout.write("\n")
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
    
    ###Append data from 'samples' to csv file
    if len(sys.argv) >= 2:
	    tDfile = open(file_name, 'a')
    else:
	    tDfile = open('testFeatures.csv', 'a')
	    
    y = len(TestData)
    print(y)
    if y == 217:
        ###Check for large values in data and discard entire dataset if true
        flag = 0
        for x in TestData:
            if (x > 10 or x == float('Inf') or x == -float('Inf')):
                #print(x)
                flag = 1
                break            
        if flag == 1:
            continue
                
        ###Data should be ok, write into file
        z = 0
        for x in TestData:
            z = z + 1
            tDfile.write(str(x))
            if z != y:
                tDfile.write(',')
        tDfile.write('\n')	
        loops = loops+1

	#print(TestData)
    
    ###Save data to .mat file
    #sio.savemat('data.mat', mdict={'data_array': data_array})
    
###End of while loop

fo.close()
scp.close()
client.close()


