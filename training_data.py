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

################################################################################################
### USAGE:
### Arguments: <type>: type of classifier algorithm, e.g. svm, knn
### <folder>: folder in current directory that contains the testFeatures_x.csv files
### python training_data.py <type> <folder> <file num1> <file num2> <optional: file num3>
### Example: python training_data.py svm Test1 0 1
#################################################################################################

if len(sys.argv) < 5:
    print("Usage: python training_data.py <classifier_type> <folder> <file num1> <folder> <file num2>")
    print("Example: python training_data.py svm Test1 0 Test2 1")
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
    
###Magnitudes
#tmp_data = np.loadtxt(sys.argv[2]+"/testFeatures_" + sys.argv[3] + ".csv", delimiter=',')
#tmp_data1 = np.loadtxt(sys.argv[2]+"/testFeatures_" + sys.argv[4] + ".csv", delimiter=',')
#if len(sys.argv) > 5:
#    tmp_data2 = np.loadtxt(sys.argv[2]+"/testFeatures_" + sys.argv[5] + ".csv", delimiter=',')

###Load in files and calculate covariance & max eigenvalue, append to features 2D array
###Time, nobody in room + 1 person static
def moving_average(a, n=3):
    ret = np.cumsum(a, dtype=float)
    ret[n:] = ret[n:] - ret[:-n]
    return ret[n - 1:] / n

def plot_mov_avg(values):
    scatter.set_xdata(range(len(values)))
    scatter.set_ydata(values)
    fig.canvas.draw()
    #sleep(0.1)
    #raw_input("Hit enter to continue:")
    
plt.ion()
fig = plt.figure()
ax = fig.add_subplot(111)   
scatter, = ax.plot([0,16], [-135, 15], 'r')
plt.show()
plt.grid(True)
    
loop = 0
while loop < 1:
    loop = loop+1
    """
    for l in range(5):
        mov_avg0_0 = np.loadtxt(sys.argv[2]+"/moving_average_" + sys.argv[3] + "_" + str(l) + ".csv", delimiter=',')
        for k in range(217):
            plot_mov_avg(mov_avg0_0[k])
        raw_input("Hit enter to continue:")
    """
    samples = 15
    #print "\nNumber of samples: " + str(samples) + "\n"
    for k in range(samples):
        
        mov_avg0_0 = np.loadtxt(sys.argv[2]+"/moving_average_" + sys.argv[3] + "_" + str(k) + ".csv", delimiter=',')
        mov_avg0_1 = np.loadtxt(sys.argv[2]+"/moving_average_" + sys.argv[3] + "_" + str(k+1) + ".csv", delimiter=',')
        
        mov_avg1_0 = np.loadtxt(sys.argv[4]+"/moving_average_" + sys.argv[5] + "_" + str(k) + ".csv", delimiter=',')
        mov_avg1_1 = np.loadtxt(sys.argv[4]+"/moving_average_" + sys.argv[5] + "_" + str(k+1) + ".csv", delimiter=',')
        
        #print "mov_avg_0 " + str(len(mov_avg0_0))
        #print "mov_avg_1 " + str(len(mov_avg0_1))
        max_eig_movavg_time_0 = []
        max_eig_movavg_time_1 = []
        for n in range(len(mov_avg0_0)):
            cov0 = np.cov(mov_avg0_0[n],mov_avg0_1[n])
            max_eig_movavg_time_0.append(max(LA.eigvals(cov0)))
            
            cov1 = np.cov(mov_avg1_0[n],mov_avg1_1[n])
            max_eig_movavg_time_1.append(max(LA.eigvals(cov1)))
        if k == 0:
            features0_time = [max_eig_movavg_time_0]
            features1_time = [max_eig_movavg_time_1]
        else:
            features0_time.append(max_eig_movavg_time_0)
            features1_time.append(max_eig_movavg_time_1)

    ###Load in files and calculate covariance & max eigenvalue, append to features 2D array
    ###Freq, nobody in room
    for k in range(samples+1):
        mov_avg0_0 = np.loadtxt(sys.argv[2]+"/moving_average_" + sys.argv[3] + "_" + str(k) + ".csv", delimiter=',')
        mov_avg1_0 = np.loadtxt(sys.argv[4]+"/moving_average_" + sys.argv[5] + "_" + str(k) + ".csv", delimiter=',')
        #print k
        #print "mov_avg_0 " + str(len(mov_avg0_0))
        #print "mov_avg_1 " + str(len(mov_avg0_1))
        max_eig_movavg_time_0 = []
        max_eig_movavg_time_1 = []
        for n in range(len(mov_avg0_0)-1):
            cov0 = np.cov(mov_avg0_0[n],mov_avg0_0[n+1])
            max_eig_movavg_time_0.append(max(LA.eigvals(cov0)))
            
            cov1 = np.cov(mov_avg1_0[n],mov_avg1_0[n+1])
            max_eig_movavg_time_1.append(max(LA.eigvals(cov1)))
        if k == 0:
            features0_freq = [max_eig_movavg_time_0]
            features1_freq = [max_eig_movavg_time_1]
        else:
            features0_freq.append(max_eig_movavg_time_0)
            features1_freq.append(max_eig_movavg_time_1)
            
    #print len(features0_time[0])
    #print len(features1_time[0])
    #print len(features0_freq[0])
    #print len(features1_freq[0])

    ##############################################################################################################
    ###Split data into training and testing

    training_part = 0.7
    train_time = int(len(features0_time)*training_part)
    train_freq = int(len(features0_freq)*training_part)

    target_val_time = np.array(range(2))
    target_val_freq = np.array(range(2))

    #Data from nobody in room
    training_features_time0 = features0_time[0:int(train_time)]
    testing_features_time0 = features0_time[int(train_time):]

    training_features_freq0 = features0_freq[0:int(train_freq)]
    testing_features_freq0 = features0_freq[int(train_freq):]

    #Data from 1 person static
    training_features_time1 = features1_time[0:int(train_time)]
    testing_features_time1 = features1_time[int(train_time):]

    training_features_freq1 = features1_freq[0:int(train_freq)]
    testing_features_freq1 = features1_freq[int(train_freq):]

    #Data from both
    training_features_time = training_features_time0 + training_features_time1
    training_features_freq = training_features_freq0 + training_features_freq1
    testing_features_time = testing_features_time0 + testing_features_time1
    testing_features_freq = testing_features_freq0 + testing_features_freq1

    #print len(training_features_time)
    #print len(testing_features_time)
    #print len(training_features_freq)
    #print len(testing_features_freq)

    #Target
    target_time = np.repeat(target_val_time,int(len(training_features_time0)))
    gnd_truth_target_time = np.repeat(target_val_time,int(len(testing_features_time0)))

    target_freq = np.repeat(target_val_time,int(len(training_features_freq0)))
    gnd_truth_target_freq = np.repeat(target_val_time,int(len(testing_features_freq0)))

    #print len(target_time)
    #print len(gnd_truth_target_time)
    #print len(target_freq)
    #print len(gnd_truth_target_freq)

    #print (target_time)
    #print (gnd_truth_target_time)
    #print (target_freq)
    #print (gnd_truth_target_freq)

    """

    ##############################################################################################################
    ###Percentage of data used for training
    train = int(len(tmp_data)*training_part)

    testing_data = np.vstack((tmp_data[int(train):],tmp_data1[int(train):]))
    if len(sys.argv) > 5:
        testing_data = np.vstack((testing_data,tmp_data2[int(train):]))

    ###Difference of the magnitudes
    take_diff = False
    if take_diff:
        difference = take_difference(tmp_data[0:int(train)])
        tmp_data = np.vstack((tmp_data[0:int(train)],difference))
        difference1 = take_difference(tmp_data1[0:int(train)])
        tmp_data1 = np.vstack((tmp_data1[0:int(train)],difference1))
        if len(sys.argv) > 5:
            difference2 = take_difference(tmp_data2[0:int(train)])
            tmp_data2 = np.vstack((tmp_data2[0:int(train)],difference2))
    else:
        tmp_data = tmp_data[0:int(train)]
        tmp_data1 = tmp_data1[0:int(train)]
        if len(sys.argv) > 5:
            tmp_data2 = tmp_data2[0:int(train)]
        
    training_data = np.vstack((tmp_data,tmp_data1))
    if len(sys.argv) > 5:
        training_data = np.vstack((training_data,tmp_data2))
    #print len(training_data)
    ###Target values
        sklearn_clf = svm.SVC(gamma=0.0001)
        sklearn_clf.fit(training_features_freq,target_freq)
        x = sklearn_clf.predict(testing_features_freq)
        check_error(gnd_truth_target_freq,x)
        ###OPENCV
        #opencv_clf = cv2.ml.SVM_create()
        #opencv_clf.setKernel(cv2.ml.SVM_RBF)
        #opencv_clf.setType(cv2.ml.SVM_NU_SVC)
        #opencv_clf.setNu(0.1)
        #opencv_clf.setGamma(0.0001)
        #opencv_clf.train(training_data, cv2.ml.ROW_SAMPLE, target)
        #result = opencv_clf.predict(testing_data)
        #opencv_result = result[1].ravel().astype(int)
        
    elif sys.argv[1] == 'knn':
        ###SKLEARN
        for neighbors in range(14):
            neighbors = neighbors+1
            print "Using Time:"
            sklearn_clf = neighbors.KNeighborsClassifier(weights='uniform', n_neighbors=neighbors)
            sklearn_clf.fit(training_features_time,target_time)
            x = sklearn_clf.predict(testing_features_time)
            check_error(gnd_truth_target_time,x)
            
            print "Using Freq:"
            sklearn_clf = neighbors.KNeighborsClassifier(weights='uniform', n_neighbors=neighbors)
            sklearn_clf.fit(training_features_freq,target_freq)
            x = sklearn_clf.predict(testing_features_freq)
            check_error(gnd_truth_target_freq,x)
        
        ###OPENCV
        #opencv_clf = cv2.ml.KNearest_create()
        #opencv_clf.train(training_data, cv2.ml.ROW_SAMPLE, target)
        #ret,result,neighbours,dist = opencv_clf.findNearest(testing_data,k=8)
        #opencv_result = result.ravel().astype(int)
    elif sys.argv[1] == 'dt':
        clf = DecisionTreeClassifier(random_state = 0)
        print(cross_val_score(clf, training_data, target, cv=10))
        clf.fit(training_data,target)
        x = clf.predict(testing_data)
    elif sys.argv[1] == 'kmeans':
        kmeans = KMeans(n_clusters=len(sys.argv)-3)
        kmeans.fit(training_data,target)
        x = kmeans.predict(testing_data)
        
        labels = kmeans.labels_
        centroids = kmeans.cluster_centers_
        
        for i in range(len(sys.argv)-3):
            # select only data observations with cluster label == i
            ds = training_data[np.where(labels==i)]
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
        if (len(sys.argv)-3) == 3:
            leg = plt.legend([dot1, dot2, dot3], [sys.argv[3],sys.argv[4],sys.argv[5]])
        else:
            leg = plt.legend([dot1, dot2], [sys.argv[3],sys.argv[4]])
        ax = plt.gca().add_artist(leg)
        plt.show()
        
    else:
        print("Current algorithms available: svm, knn, dt, kmeans")
        print("Usage: python training_data.py <classifier_type> <folder>")
        print("Example: python training_data.py svm Test1")
        sys.exit(1)  target_val = np.array(range(len(sys.argv)-3))
    target = np.repeat(target_val,int(len(tmp_data)))
    """

    def check_error(gnd_truth,x):
        ###Print ground truth, prediction, and accuracy
        print("Ground Truth: " + str(gnd_truth))

        print("Prediction  : " + str(x))
        accuracy = accuracy_score(gnd_truth,x)
        print("Accuracy: " + str(accuracy))
    #######################################################################################################
    ###Determine algorithm to use & Fit training data to target & Make prediction on testing data
    if sys.argv[1] == 'svm':
        ###SKLEARN
        print "Using Time:"
        sklearn_clf = svm.SVC(gamma=0.0001)
        sklearn_clf.fit(training_features_time,target_time)
        x = sklearn_clf.predict(testing_features_time)
        check_error(gnd_truth_target_time,x)
        
        print "Using Freq:"
        sklearn_clf = svm.SVC(gamma=0.0001)
        sklearn_clf.fit(training_features_freq,target_freq)
        x = sklearn_clf.predict(testing_features_freq)
        check_error(gnd_truth_target_freq,x)
        ###OPENCV
        #opencv_clf = cv2.ml.SVM_create()
        #opencv_clf.setKernel(cv2.ml.SVM_RBF)
        #opencv_clf.setType(cv2.ml.SVM_NU_SVC)
        #opencv_clf.setNu(0.1)
        #opencv_clf.setGamma(0.0001)
        #opencv_clf.train(training_data, cv2.ml.ROW_SAMPLE, target)
        #result = opencv_clf.predict(testing_data)
        #opencv_result = result[1].ravel().astype(int)
        
    elif sys.argv[1] == 'knn':
        ###SKLEARN
        neighbor = 7
        #print "\n"+ str(neighbor)
        print "Using Time:"
        sklearn_clf = neighbors.KNeighborsClassifier(weights='uniform', n_neighbors=neighbor)
        sklearn_clf.fit(training_features_time,target_time)
        x = sklearn_clf.predict(testing_features_time)
        check_error(gnd_truth_target_time,x)
            
        print "Using Freq:"
        sklearn_clf = neighbors.KNeighborsClassifier(weights='uniform', n_neighbors=neighbor)
        sklearn_clf.fit(training_features_freq,target_freq)
        x = sklearn_clf.predict(testing_features_freq)
        check_error(gnd_truth_target_freq,x)
        
        ###OPENCV
        #opencv_clf = cv2.ml.KNearest_create()
        #opencv_clf.train(training_data, cv2.ml.ROW_SAMPLE, target)
        #ret,result,neighbours,dist = opencv_clf.findNearest(testing_data,k=8)
        #opencv_result = result.ravel().astype(int)
    elif sys.argv[1] == 'dt':
        clf = DecisionTreeClassifier(random_state = 0)
        print(cross_val_score(clf, training_data, target, cv=10))
        clf.fit(training_data,target)
        x = clf.predict(testing_data)
    elif sys.argv[1] == 'kmeans':
        kmeans = KMeans(n_clusters=len(sys.argv)-3)
        kmeans.fit(training_data,target)
        x = kmeans.predict(testing_data)
        
        labels = kmeans.labels_
        centroids = kmeans.cluster_centers_
        
        for i in range(len(sys.argv)-3):
            # select only data observations with cluster label == i
            ds = training_data[np.where(labels==i)]
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
        if (len(sys.argv)-3) == 3:
            leg = plt.legend([dot1, dot2, dot3], [sys.argv[3],sys.argv[4],sys.argv[5]])
        else:
            leg = plt.legend([dot1, dot2], [sys.argv[3],sys.argv[4]])
        ax = plt.gca().add_artist(leg)
        plt.show()
        
    else:
        print("Current algorithms available: svm, knn, dt, kmeans")
        print("Usage: python training_data.py <classifier_type> <folder>")
        print("Example: python training_data.py svm Test1")
        sys.exit(1)

