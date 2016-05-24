import numpy as np
from sklearn import svm, neighbors
from sklearn.metrics import accuracy_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.cross_validation import cross_val_score
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import sys
from numpy import linalg as LA
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from time import sleep

################################################################################################
### USAGE:
### Arguments: <type>: type of classifier algorithm, e.g. svm, knn, dt, kmeans
### <folder>: folder in current directory that contains the moving_average_0_x.csv files
###           each folder will represent one case/target
###           e.g. Test0 -> no one in room, Test1 -> one person in room, etc.
### python machine_learning <type> <folder> <folder> ... <folder>
### Example: python machine_learning svm Test0 Test1 Test2
#################################################################################################

if len(sys.argv) < 4:
    print("Usage: python machine_learning.py <classifier_type> <samples> <folder> <folder> ... <folder>")
    print("Example: python machine_learning.py svm 10 Test1 Test2 Test3")
    sys.exit(1)

###Constants
###If runnining_tests == True, samples will loop through 5 to 45
###Otherwise, samples will be the 2nd argument to the program
running_tests = True
samples = int(sys.argv[2])
training_part = 0.7
numFolders = len(sys.argv)-3
numFeatures = 1
target_val = np.array(range(numFolders))

###Split data into two parts based on the given percentage
def split_data(data, percentage):
    part = int(len(data)*percentage)
    train = data[:part]
    test = data[part:]
    return (train,test)

def check_error(gnd_truth,x):
    ###Print ground truth, prediction, and accuracy
    print("Actual    : " + str(gnd_truth))
    print("Prediction: " + str(x))
    accuracy = accuracy_score(gnd_truth,x)
    print("Accuracy  : " + str(accuracy))

if running_tests:
    num_loops = 40
else:
    num_loops = 1

for loops in range(num_loops):
    if running_tests:
        samples = loops + 5
        print "\nSamples: " + str(samples)
    
    training_target = np.repeat(target_val,int(samples*training_part*numFeatures))
    testing_target = np.repeat(target_val,samples*numFeatures - int(samples*training_part*numFeatures))
    
    ###Process files in each folder
    for index in range(len(sys.argv)):
        if index > 2:
            ###Load in files and process them
            for k in range(samples):
                mov_avg0_0 = np.loadtxt(sys.argv[index]+"/moving_average_0_" + str(k) + ".csv", delimiter=',')
                mov_avg0_1 = np.loadtxt(sys.argv[index]+"/moving_average_0_" + str(k+1) + ".csv", delimiter=',')                
                
                features_arr = []
                for n in range(len(mov_avg0_0)-1):
                    ###Features: 
                    
                    ###Eigenvalues of the covariance of the moving average of the RSS of each subcarrier
                    ###Taking the covariance of the mov avg RSS of the same subcarrier freq of consecutive samples -> changes in time
                    cov0 = np.cov(mov_avg0_0[n],mov_avg0_1[n])
                    eig0 = LA.eigvals(cov0)
                    features_arr.append(eig0[0])
                    features_arr.append(eig0[1])
                    
                    ###Correlation
                    corr = np.correlate(mov_avg0_0[n],mov_avg0_1[n], "same")
                    list_corr = corr.tolist()
                    features_arr = features_arr + list_corr
                                       
                ##features is a list of lists: [ [], [],..., [] ]
                ##features_arr is a list [x1,x2...,x3]
                if k == 0:
                    features = [features_arr]
                    #features.append(max_eig_movavg_1)
                    #features.append(correlation)
                else:
                    features.append(features_arr)
                    #features.append(max_eig_movavg_1)
                    #features.append(correlation)
                    
            training_data0,testing_data0 = split_data(features,training_part)
            if index == 3:
                training_data = training_data0
                testing_data = testing_data0
            elif index > 3:
                training_data = training_data + training_data0
                testing_data = testing_data + testing_data0
        
    #print len(training_data)
    #print len(testing_data)    
    #print len(training_data[0])
    #print len(testing_data[0])  
    #print len(training_target)
    #print len(testing_target)
    #print  (training_data[11])

    if sys.argv[1] == 'svm':
        ###SKLEARN
        #print "Using Time:"
        sklearn_clf = svm.SVC(gamma=0.0001)
        sklearn_clf.fit(training_data,training_target)
        x = sklearn_clf.predict(testing_data)
        check_error(testing_target,x)
     
    elif sys.argv[1] == 'knn':
        ###SKLEARN
        sklearn_clf = neighbors.KNeighborsClassifier(n_neighbors=5)
        sklearn_clf.fit(training_data,training_target)
        x = sklearn_clf.predict(testing_data)
        check_error(testing_target,x)
        """
        ###Use another data set as test data
        for k in range(samples):
            f = "Exp2/person1_moving32/moving_average_0_"
            mov_avg_0 = np.loadtxt(f + str(k) + ".csv", delimiter=',')
            mov_avg_1 = np.loadtxt(f + str(k+1) + ".csv", delimiter=',')
                    
            max_eig_movavg_time = []

            for n in range(len(mov_avg_0)-1):
                cov = np.cov(mov_avg_0[n],mov_avg_1[n])
                eig = LA.eigvals(cov)
                max_eig_movavg_time.append(eig[0])
                max_eig_movavg_time.append(eig[1])
                        
            if k == 0:
                test_features = [max_eig_movavg_time]
            else:
                test_features.append(max_eig_movavg_time)
       
        #print len(test_features)
        x = sklearn_clf.predict(test_features)
        testing_target_2 = np.repeat([0],samples)
        check_error(testing_target_2,x)
        """

    elif sys.argv[1] == 'dt':
        sklearn_clf = DecisionTreeClassifier()
        sklearn_clf.fit(training_data,training_target)
        x = sklearn_clf.predict(testing_data)
        check_error(testing_target,x)
            
    elif sys.argv[1] == 'kmeans':
            kmeans = KMeans(n_clusters=numFolders)
            kmeans.fit(training_data,training_target)
            x = kmeans.predict(testing_data)
            check_error(testing_target,x)
            
            labels = kmeans.labels_
            centroids = kmeans.cluster_centers_
            
            for i in range(numFolders):
                # select only data observations with cluster label == i
                t_data = np.array(training_data)
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
            if (numFolders == 3):
                leg = plt.legend([dot1, dot2, dot3], ['0','1','2'])
            else:
                leg = plt.legend([dot1, dot2], ['0','1'])
            ax = plt.gca().add_artist(leg)
            plt.show()
            
    else:
        print("Current classifier algorithms available: svm, knn, dt, kmeans")
        sys.exit(1)

