import numpy as np
from sklearn import svm, neighbors
from sklearn.metrics import accuracy_score
import sys

#######USAGE########
# a: number of samples in each dataset
# python training_data.py <a>  
#
#

#target = np.loadtxt("data_target.csv", delimiter=",")
#print(target)

if len(sys.argv) < 3:
    print("Usage: python training_data.py <num samples in each dataset> <classifier type>")
    print("Example: python training_data.py 50 svm")
    exit(1)

tmp = np.array([0, 1])
target = np.repeat(tmp,int(int(sys.argv[1])*0.7))
#print(target)
train = int(sys.argv[1])*0.7
#print(train)

data0 = np.loadtxt("testFeatures0.csv", delimiter=',')
data1_1 = np.loadtxt("testFeatures1.csv", delimiter=',')
data1_2 = np.loadtxt("testFeatures2.csv", delimiter=',')
data1_3 = np.loadtxt("testFeatures3.csv", delimiter=',')
data1_4 = np.loadtxt("testFeatures4.csv", delimiter=',')
data1_5 = np.loadtxt("testFeatures5.csv", delimiter=',')

tmp1 = np.vstack((data0[0:int(train)],data1_1[0:int(train)/5]))
tmp2 = np.vstack((data1_2[0:int(train)/5],data1_3[0:int(train)/5]))
tmp3 = np.vstack((data1_4[0:int(train)/5],data1_5[0:int(train)/5]))
tmp4 = np.vstack((tmp1,tmp2))
training_data = np.vstack((tmp4,tmp3))
#print(len(training_data))

tmp1 = np.vstack((data0[int(train):],data1_1[int(train)/5:]))
tmp2 = np.vstack((data1_2[int(train)/5:],data1_3[int(train)/5:]))
tmp3 = np.vstack((data1_4[int(train)/5:],data1_5[int(train)/5:]))
tmp4 = np.vstack((tmp1,tmp2))
testing_data = np.vstack((tmp4,tmp3))
#print(training_data)
#print("-------------------------")
#print(testing_data)
#print(len(training_data))

if sys.argv[2] == 'svm':
    clf = svm.SVC(gamma=0.001, C=100.)
elif sys.argv[2] == 'knn':
    clf = neighbors.KNeighborsClassifier()
else:
    print("Current algorithms available: svm, knn")
    print("Usage: python training_data.py <num samples in each dataset> <classifier type>")
    print("Example: python training_data.py 50 svm")
    exit(1)
    exit(1)

clf.fit(training_data,target)

num_test = int(sys.argv[1])-int(int(sys.argv[1])*0.7)
gnd_truth = np.repeat(tmp,num_test)
print(gnd_truth)
x = clf.predict(testing_data)
print(x)
accuracy = accuracy_score(gnd_truth,x)
print("Accuracy: " + str(accuracy))
