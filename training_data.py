import numpy as np
from sklearn import svm, neighbors
from sklearn.metrics import accuracy_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.cross_validation import cross_val_score
import sys
import cv2

################################################################################################
### USAGE:
### Arguments: <type>: type of classifier algorithm, e.g. svm, knn
### <folder>: folder in current directory that contains the testFeatures_x.csv files
### python training_data.py <type> <folder> <file num1> <file num2> <optional: file num3>
### Example: python training_data.py svm Test1 0 1
#################################################################################################

if len(sys.argv) < 5:
    print("Usage: python training_data.py <classifier_type> <folder> <file num1> <file num2> <optional: file num3>")
    print("Example: python training_data.py svm Test1 0 1")
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

###Load data from files, Merge/Split data into 2D arrays

###Magnitudes
tmp_data = np.loadtxt(sys.argv[2]+"/testFeatures_" + sys.argv[3] + ".csv", delimiter=',')
tmp_data1 = np.loadtxt(sys.argv[2]+"/testFeatures_" + sys.argv[4] + ".csv", delimiter=',')
if len(sys.argv) > 5:
    tmp_data2 = np.loadtxt(sys.argv[2]+"/testFeatures_" + sys.argv[5] + ".csv", delimiter=',')

###Percentage of data used for training
training_part = 0.7
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
target_val = np.array(range(len(sys.argv)-3))
target = np.repeat(target_val,int(len(tmp_data)))

###Determine algorithm to use & Fit training data to target & Make prediction on testing data
if sys.argv[1] == 'svm':
    ###SKLEARN
    sklearn_clf = svm.SVC(gamma=0.0001)
    sklearn_clf.fit(training_data,target)
    x = sklearn_clf.predict(testing_data)
    
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
    sklearn_clf = neighbors.KNeighborsClassifier(weights='uniform', n_neighbors=7)
    sklearn_clf.fit(training_data,target)
    x = sklearn_clf.predict(testing_data)
    
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
else:
    print("Current algorithms available: svm, knn")
    print("Usage: python training_data.py <classifier_type> <folder>")
    print("Example: python training_data.py svm Test1")
    sys.exit(1)

###Print ground truth, prediction, and accuracy
num_test = int(len(testing_data)/(len(sys.argv)-3))
gnd_truth = np.repeat(target_val,num_test)
print("Ground Truth: " + str(gnd_truth))

print("Prediction  : " + str(x))
accuracy = accuracy_score(gnd_truth,x)
print("Accuracy: " + str(accuracy))

#print("Prediction (opencv):  " + str(opencv_result))
#accuracy1 = accuracy_score(gnd_truth,opencv_result)
#print("Accuracy (opencv): " + str(accuracy1))
