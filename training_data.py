import numpy as np
from sklearn import svm, neighbors
from sklearn.metrics import accuracy_score
import sys
import cv2

################################################################################################
### USAGE:
### Arguments: <type>: type of classifier algorithm, e.g. svm, knn
### <folder>: folder in current directory that contains the testFeatures_x.csv files
### python training_data.py <type> <folder>
### Example: python training_data.py svm Test1
#################################################################################################

if len(sys.argv) < 3:
    print("Usage: python training_data.py <classifier_type> <folder>")
    print("Example: python training_data.py svm Test1")
    sys.exit(1)

###Load data from files, Merge/Split data into 2D arrays
###Assuming equal number of samples for each dataset representing 0 and 1 person; data for 1 person is split into 5 files
###testFeatures_0.csv contains 5 samples
###testFeatures_(1-5).csv contains 10 samples each for a total of 50 samples (one person in room, standing in 5 different locations)
data0 = np.loadtxt(sys.argv[2]+"/testFeatures_0.csv", delimiter=',')

###Percentage of data used for training
training_part = 0.7
train = int(len(data0)*training_part)

###Target values
target_val = np.array(range(2))
target = np.repeat(target_val,int(len(data0)*training_part))

for i in range(2):
    tmp_data = np.loadtxt(sys.argv[2]+"/testFeatures_" + str(i+1) + ".csv", delimiter=',')
    if i == 0:
        training_data = np.vstack((data0[0:int(train)],tmp_data[0:int(train)]))
        #training_data = np.vstack((data0[0:int(train)],tmp_data[0:int(train)/5]))
        testing_data = np.vstack((data0[int(train):],tmp_data[int(train):]))
        #testing_data = np.vstack((data0[int(train):],tmp_data[int(train)/5:]))
    else:
        training_data = np.vstack((training_data,tmp_data[0:int(train)]))
        #training_data = np.vstack((training_data,tmp_data[0:int(train)/5]))
        testing_data = np.vstack((testing_data,tmp_data[int(train):]))
        #testing_data = np.vstack((testing_data,tmp_data[int(train)/5:]))
        
training_data = training_data.astype(np.float32)
testing_data = testing_data.astype(np.float32)

###Determine algorithm to use & Fit training data to target & Make prediction on testing data
if sys.argv[1] == 'svm':
    ###SKLEARN
    sklearn_clf = svm.SVC(gamma=0.0001)
    sklearn_clf.fit(training_data,target)
    x = sklearn_clf.predict(testing_data)
    
    ###OPENCV
    opencv_clf = cv2.ml.SVM_create()
    opencv_clf.setKernel(cv2.ml.SVM_RBF)
    opencv_clf.setType(cv2.ml.SVM_NU_SVC)
    opencv_clf.setNu(0.1)
    opencv_clf.setGamma(0.0001)
    opencv_clf.train(training_data, cv2.ml.ROW_SAMPLE, target)
    result = opencv_clf.predict(testing_data)
    opencv_result = result[1].ravel().astype(int)
    
elif sys.argv[1] == 'knn':
    ###SKLEARN
    sklearn_clf = neighbors.KNeighborsClassifier(weights='distance',n_neighbors=8)
    sklearn_clf.fit(training_data,target)
    x = sklearn_clf.predict(testing_data)
    
    ###OPENCV
    opencv_clf = cv2.ml.KNearest_create()
    opencv_clf.train(training_data, cv2.ml.ROW_SAMPLE, target)
    ret,result,neighbours,dist = opencv_clf.findNearest(testing_data,k=8)
    opencv_result = result.ravel().astype(int)
else:
    print("Current algorithms available: svm, knn")
    print("Usage: python training_data.py <num_samples> <classifier_type> <folder>")
    print("Example: python training_data.py 50 svm Test1")
    sys.exit(1)

###Print ground truth, prediction, and accuracy
num_test = len(data0)-int(len(data0)*training_part)
gnd_truth = np.repeat(target_val,num_test)
print("Ground Truth:         " + str(gnd_truth))

print("Prediction (sklearn): " + str(x))
accuracy = accuracy_score(gnd_truth,x)
print("Accuracy (sklearn): " + str(accuracy))

print("Prediction (opencv):  " + str(opencv_result))
accuracy1 = accuracy_score(gnd_truth,opencv_result)
print("Accuracy (opencv): " + str(accuracy1))
