import numpy as np
from sklearn import svm, neighbors
from sklearn.metrics import accuracy_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.cross_validation import cross_val_score
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import sys
import cv2
from numpy import linalg as LA
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from time import sleep

################################################################################################
### USAGE:
### Arguments: <type>: type of classifier algorithm, e.g. svm, knn, dt, kmeans
### <folder>: folder in current directory that contains the moving_average_x.csv files
### python training_data.py <type> <folder> <file num1> <folder> <file num2> <folder> <file num3>
### Example: python training_data.py svm Test1 0 Test2 1 Test3 2
#################################################################################################

if len(sys.argv) < 6:
    print("Usage: python training_data.py <classifier_type> <folder> <file num1> <folder> <file num2> <folder> <file num3>")
    print("Example: python training_data.py svm Test1 0 Test2 1 Test3 2")
    sys.exit(1)

      
def take_difference(tmp_data):
    for i,data in enumerate(tmp_data):
        if i == 1:
            difference = np.subtract(tmp_data[i-1],tmp_data[i])
        elif i > 1:
            diff = np.subtract(tmp_data[i-1],tmp_data[i])
            difference = np.vstack((difference,diff))
    #print difference
    return difference

def plot_mov_avg(values):
    scatter.set_xdata(range(len(values)))
    scatter.set_ydata(values)
    fig.canvas.draw()
    sleep(0.1)
    #raw_input("Hit enter to continue:")
    
"""    
plt.ion()
fig = plt.figure()
ax = fig.add_subplot(111)   
scatter, = ax.plot([0,16], [-135, 10], 'r')
plt.show()
plt.grid(True)
"""
#loop = 0
#while loop < 1:
    #loop = loop+1
for samples in range(44):
    samples = samples+5
    print "\nSamples: " + str(samples)
    """
    for l in range(5):
        raw_input("Hit enter to continue:")
        mov_avg0_0 = np.loadtxt(sys.argv[4]+"/moving_average_" + sys.argv[5] + "_" + str(l) + ".csv", delimiter=',')
        for k in range(217):
            plot_mov_avg(mov_avg0_0[k])
    """
    ###Load in files and calculate covariance & max eigenvalue for same freq across samples, append to features 2D array
    #samples = 49
    for k in range(samples):
        print k
        mov_avg0_0 = np.loadtxt(sys.argv[2]+"/moving_average_" + sys.argv[3] + "_" + str(k) + ".csv", delimiter=',')
        mov_avg0_1 = np.loadtxt(sys.argv[2]+"/moving_average_" + sys.argv[3] + "_" + str(k+1) + ".csv", delimiter=',')
        
        mov_avg1_0 = np.loadtxt(sys.argv[4]+"/moving_average_" + sys.argv[5] + "_" + str(k) + ".csv", delimiter=',')
        mov_avg1_1 = np.loadtxt(sys.argv[4]+"/moving_average_" + sys.argv[5] + "_" + str(k+1) + ".csv", delimiter=',')
        
        if(len(sys.argv) > 7):
            mov_avg2_0 = np.loadtxt(sys.argv[6]+"/moving_average_" + sys.argv[7] + "_" + str(k) + ".csv", delimiter=',')
            mov_avg2_1 = np.loadtxt(sys.argv[6]+"/moving_average_" + sys.argv[7] + "_" + str(k+1) + ".csv", delimiter=',')

        max_eig_movavg_time_0 = []
        max_eig_movavg_time_1 = []
        max_eig_movavg_time_2 = []
        for n in range(len(mov_avg0_0)-1):
            cov0 = np.cov(mov_avg0_0[n],mov_avg0_1[n])
            eig0 = LA.eigvals(cov0)
            #print eig0
            #eig0.sort(reverse=True)
            max_eig_movavg_time_0.append(eig0[0])
            max_eig_movavg_time_0.append(eig0[1])
                        
            cov1 = np.cov(mov_avg1_0[n],mov_avg1_1[n])
            eig1 = LA.eigvals(cov1)
            #eig1.sort(reverse=True)
            max_eig_movavg_time_1.append(eig1[0])
            max_eig_movavg_time_1.append(eig1[1])
            
            if(len(sys.argv) > 7):
                cov2 = np.cov(mov_avg2_0[n],mov_avg2_1[n])
                eig2 = LA.eigvals(cov2)
                #eig2.sort(reverse=True)
                max_eig_movavg_time_2.append(eig2[0])
                max_eig_movavg_time_2.append(eig2[1])
                
        if k == 0:
            features0_time = [max_eig_movavg_time_0]
            features1_time = [max_eig_movavg_time_1]
            if(len(sys.argv) > 7):
                features2_time = [max_eig_movavg_time_2]
        else:
            features0_time.append(max_eig_movavg_time_0)
            features1_time.append(max_eig_movavg_time_1)
            if(len(sys.argv) > 7):
                features2_time.append(max_eig_movavg_time_2)

    ###Load in files and calculate covariance & max eigenvalue for different freq in same sample, append to features 2D array
    for k in range(samples+1):
        mov_avg0_0 = np.loadtxt(sys.argv[2]+"/moving_average_" + sys.argv[3] + "_" + str(k) + ".csv", delimiter=',')
        mov_avg1_0 = np.loadtxt(sys.argv[4]+"/moving_average_" + sys.argv[5] + "_" + str(k) + ".csv", delimiter=',')
        if(len(sys.argv) > 7):
            mov_avg2_0 = np.loadtxt(sys.argv[6]+"/moving_average_" + sys.argv[7] + "_" + str(k) + ".csv", delimiter=',')

        max_eig_movavg_time_0 = []
        max_eig_movavg_time_1 = []
        max_eig_movavg_time_2 = []
        for n in range(len(mov_avg0_0)-1):
            cov0 = np.cov(mov_avg0_0[n],mov_avg0_0[n+1])
            eig0 = LA.eigvals(cov0)
            #eig0.sort(reverse=True)
            max_eig_movavg_time_0.append(eig0[0])
            max_eig_movavg_time_0.append(eig0[1])
            
            cov1 = np.cov(mov_avg1_0[n],mov_avg1_0[n+1])
            eig1 = LA.eigvals(cov1)
            #eig1.sort(reverse=True)
            max_eig_movavg_time_1.append(eig1[0])
            max_eig_movavg_time_1.append(eig1[1])
            
            if(len(sys.argv) > 7):
                cov2 = np.cov(mov_avg2_0[n],mov_avg2_0[n+1])
                eig2 = LA.eigvals(cov2)
                #eig2.sort(reverse=True)
                max_eig_movavg_time_2.append(eig2[0])
                max_eig_movavg_time_2.append(eig2[1])
        if k == 0:
            features0_freq = [max_eig_movavg_time_0]
            features1_freq = [max_eig_movavg_time_1]
            if(len(sys.argv) > 7):
                features2_freq = [max_eig_movavg_time_2]
        else:
            features0_freq.append(max_eig_movavg_time_0)
            features1_freq.append(max_eig_movavg_time_1)
            if(len(sys.argv) > 7):
                features2_freq.append(max_eig_movavg_time_2)
            

    ##############################################################################################################
    ###Split data into training and testing

    training_part = 0.7
    train_time = int(len(features0_time)*training_part)
    train_freq = int(len(features0_freq)*training_part)

    target_val_time = np.array(range(2))
    target_val_freq = np.array(range(2))
    if(len(sys.argv) > 7):
        target_val_time = np.array(range(3))
        target_val_freq = np.array(range(3))

    #Data from nobody in room
    training_features_time0 = features0_time[0:int(train_time)]
    testing_features_time0 = features0_time[int(train_time):]

    training_features_freq0 = features0_freq[0:int(train_freq)]
    testing_features_freq0 = features0_freq[int(train_freq):]

    #Data from 1 person
    training_features_time1 = features1_time[0:int(train_time)]
    testing_features_time1 = features1_time[int(train_time):]

    training_features_freq1 = features1_freq[0:int(train_freq)]
    testing_features_freq1 = features1_freq[int(train_freq):]
    
    #Data from 2 people
    if(len(sys.argv) > 7):
        training_features_time2 = features2_time[0:int(train_time)]
        testing_features_time2 = features2_time[int(train_time):]

        training_features_freq2 = features2_freq[0:int(train_freq)]
        testing_features_freq2 = features2_freq[int(train_freq):]

    #Data from all
    #training_features_time = training_features_time0 + training_features_time1
    #training_features_freq = training_features_freq0 + training_features_freq1
    #testing_features_time = testing_features_time0 + testing_features_time1
    #testing_features_freq = testing_features_freq0 + testing_features_freq1
    #if(len(sys.argv) > 7):
    #    training_features_time = training_features_time + training_features_time2
    #    training_features_freq = training_features_freq + training_features_freq2
    #    testing_features_time = testing_features_time + testing_features_time2
    #    testing_features_freq = testing_features_freq + testing_features_freq2
        
    #Data from all
    training_features = training_features_time0 + training_features_freq0 + training_features_time1 + training_features_freq1
    testing_features = testing_features_time0 + testing_features_freq0 + testing_features_time1 + testing_features_freq1
    if(len(sys.argv) > 7):
        training_features = training_features + training_features_time2 + training_features_freq2
        testing_features = testing_features + testing_features_time2  + testing_features_freq2

    #print len(training_features_time)
    #print len(training_features_time[0])

    #print len(training_features_freq)
    #print len(training_features_freq[0])
    #Target
    target = np.repeat(target_val_time,int(len(training_features_time0 + training_features_freq0)))
    gnd_truth_target = np.repeat(target_val_time,int(len(testing_features_time0 + testing_features_freq0)))
    #print len(testing_features)
    #print len(target)
    #target_freq = np.repeat(target_val_time,int(len(training_features_freq0)))
    #gnd_truth_target_freq = np.repeat(target_val_time,int(len(testing_features_freq0)))
        
    #######################################################################################################
    ###Determine algorithm to use & Fit training data to target & Make prediction on testing data
    
    def check_error(gnd_truth,x):
        ###Print ground truth, prediction, and accuracy
        print("Ground Truth: " + str(gnd_truth))

        print("Prediction  : " + str(x))
        accuracy = accuracy_score(gnd_truth,x)
        print("Accuracy: " + str(accuracy))
    
    if sys.argv[1] == 'svm':
        ###SKLEARN
        #print "Using Time:"
        sklearn_clf = svm.SVC(gamma=0.0001)
        sklearn_clf.fit(training_features,target)
        x = sklearn_clf.predict(testing_features)
        check_error(gnd_truth_target,x)
        
        #print "Using Freq:"
        #sklearn_clf = svm.SVC(gamma=0.0001)
        #sklearn_clf.fit(training_features_freq,target_freq)
        #x2 = sklearn_clf.predict(testing_features_freq)
        #check_error(gnd_truth_target_freq,x2)
 
    elif sys.argv[1] == 'knn':
        ###SKLEARN
        neighbor = 5
        #print "Using Time:"
        sklearn_clf = neighbors.KNeighborsClassifier(weights='uniform')
        sklearn_clf.fit(training_features,target)
        x = sklearn_clf.predict(testing_features)
        check_error(gnd_truth_target,x)
            
        #print "Using Freq:"
        #sklearn_clf = neighbors.KNeighborsClassifier(weights='uniform')
        #sklearn_clf.fit(training_features_freq,target_freq)
        #x2 = sklearn_clf.predict(testing_features_freq)
        #check_error(gnd_truth_target_freq,x2)
        
    elif sys.argv[1] == 'dt':
        clf = DecisionTreeClassifier(random_state = 0)
        #print(cross_val_score(clf, training_features_time,target_time, cv=10))
        clf.fit(training_features,target)
        x = clf.predict(testing_features)
        check_error(gnd_truth_target,x)
        
        #clf = DecisionTreeClassifier(random_state = 0)
        #print(cross_val_score(clf, training_features_freq,target_freq, cv=10))
        #clf.fit(training_features_freq,target_freq)
        #x = clf.predict(testing_features_freq)
        #check_error(gnd_truth_target_freq,x)
        
    elif sys.argv[1] == 'kmeans':
        numfiles = len(sys.argv)-4
        kmeans = KMeans(n_clusters=numfiles)
        kmeans.fit(training_features_time,target_time)
        x = kmeans.predict(testing_features_time)
        check_error(gnd_truth_target_time,x)
        labels = kmeans.labels_
        centroids = kmeans.cluster_centers_
        
        for i in range(numfiles):
            # select only data observations with cluster label == i
            t_data = np.array(training_features_time)
            ds = t_data[np.where(labels==i)]
            # plot the data observations
            if i==0:
                dot1, = plt.plot(ds[:,0],ds[:,1],'o')
            elif i==1:
                dot2, = plt.plot(ds[:,0],ds[:,1],'o')
            elif i==2:
                dot3, = plt.plot(ds[:,0],ds[:,1],'o')
            # plot the centroids
            lines = plt.plot(centroids[i,0],centroids[i,1],'kx')
            # make the centroid x's bigger
            plt.setp(lines,ms=15.0)
            plt.setp(lines,mew=2.0)
        if (numfiles == 3):
            leg = plt.legend([dot1, dot2, dot3], ['0','1','2'])
        else:
            leg = plt.legend([dot1, dot2], ['0','1'])
        ax = plt.gca().add_artist(leg)
        plt.show()
        
    else:
        print("Current algorithms available: svm, knn, dt, kmeans")
        print("Usage: python training_data.py <classifier_type> <folder> <filenum0> <folder> <filenum1> <folder> <filenum2>")
        print("Example: python training_data.py svm Test1 0 Test2 1 Test3 2")
        sys.exit(1)

