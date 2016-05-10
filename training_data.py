import numpy as np
from sklearn import svm, neighbors
from sklearn.metrics import accuracy_score
import sys
import cv2

################################################################################################
### USAGE:
### Arguments: <num>: number of samples in each dataset, <type>: type of classifier algorithm, e.g. svm, knn
### <folder>: folder in current directory that contains the testFeatures_x.csv files
### python training_data.py <num> <type> <folder>
### Example: python training_data.py 50 svm Test1
#################################################################################################

if len(sys.argv) < 4:
    print("Usage: python training_data.py <num_samples> <classifier_type> <folder>")
    print("Example: python training_data.py 50 svm Test1")
    sys.exit(1)

###Percentage of data used for training
training_part = 0.7
train = int(sys.argv[1])*training_part
#print(train)

###Target values
tmp = np.array([0, 1])
target = np.repeat(tmp,int(int(sys.argv[1])*training_part))
#print(target)

###Load data from files
data0 = np.loadtxt(sys.argv[3]+"/testFeatures_0.csv", delimiter=',')
data1_1 = np.loadtxt(sys.argv[3]+"/testFeatures_1.csv", delimiter=',')
data1_2 = np.loadtxt(sys.argv[3]+"/testFeatures_2.csv", delimiter=',')
data1_3 = np.loadtxt(sys.argv[3]+"/testFeatures_3.csv", delimiter=',')
data1_4 = np.loadtxt(sys.argv[3]+"/testFeatures_4.csv", delimiter=',')
data1_5 = np.loadtxt(sys.argv[3]+"/testFeatures_5.csv", delimiter=',')

###Merge training data into a single 2D array
tmp1 = np.vstack((data0[0:int(train)],data1_1[0:int(train)/5]))
tmp2 = np.vstack((data1_2[0:int(train)/5],data1_3[0:int(train)/5]))
tmp3 = np.vstack((data1_4[0:int(train)/5],data1_5[0:int(train)/5]))
tmp4 = np.vstack((tmp1,tmp2))
training_data = np.vstack((tmp4,tmp3)).astype(np.float32)

###Merge testing data into a single 2D array
tmp1 = np.vstack((data0[int(train):],data1_1[int(train)/5:]))
tmp2 = np.vstack((data1_2[int(train)/5:],data1_3[int(train)/5:]))
tmp3 = np.vstack((data1_4[int(train)/5:],data1_5[int(train)/5:]))
tmp4 = np.vstack((tmp1,tmp2))
testing_data = np.vstack((tmp4,tmp3)).astype(np.float32)


###Determine algorithm to use & Fit training data to target & Make prediction on testing data
if sys.argv[2] == 'svm':
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
    
elif sys.argv[2] == 'knn':
    ###SKLEARN
    sklearn_clf = neighbors.KNeighborsClassifier(n_neighbors=8)
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
num_test = int(sys.argv[1])-int(int(sys.argv[1])*training_part)
gnd_truth = np.repeat(tmp,num_test)
print("Ground Truth:         " + str(gnd_truth))

print("Prediction (sklearn): " + str(x))
accuracy = accuracy_score(gnd_truth,x)
print("Accuracy (sklearn): " + str(accuracy))

print("Prediction (opencv):  " + str(opencv_result))
accuracy1 = accuracy_score(gnd_truth,opencv_result)
print("Accuracy (opencv): " + str(accuracy1))
