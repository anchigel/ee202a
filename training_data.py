import numpy as np
from sklearn import svm, neighbors
from sklearn.metrics import accuracy_score
import sys

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
training_data = np.vstack((tmp4,tmp3))

###Merge testing data into a single 2D array
tmp1 = np.vstack((data0[int(train):],data1_1[int(train)/5:]))
tmp2 = np.vstack((data1_2[int(train)/5:],data1_3[int(train)/5:]))
tmp3 = np.vstack((data1_4[int(train)/5:],data1_5[int(train)/5:]))
tmp4 = np.vstack((tmp1,tmp2))
testing_data = np.vstack((tmp4,tmp3))

###Determine algorithm to use
if sys.argv[2] == 'svm':
    clf = svm.SVC(gamma=0.0004)
elif sys.argv[2] == 'knn':
    clf = neighbors.KNeighborsClassifier(weights='distance',n_neighbors=10)
else:
    print("Current algorithms available: svm, knn")
    print("Usage: python training_data.py <num samples in each dataset> <classifier type>")
    print("Example: python training_data.py 50 svm")
    sys.exit(1)

###Fit training data to target
clf.fit(training_data,target)

###Make prediction on testing data
x = clf.predict(testing_data)

###Print ground truth, prediction, and accuracy
num_test = int(sys.argv[1])-int(int(sys.argv[1])*training_part)
gnd_truth = np.repeat(tmp,num_test)
print("Ground Truth: " + str(gnd_truth))
print("Prediction:   " + str(x))
accuracy = accuracy_score(gnd_truth,x)
print("Accuracy: " + str(accuracy))
