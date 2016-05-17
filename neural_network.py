import numpy as np
import sys
from pybrain.datasets            import ClassificationDataSet
from pybrain.structure.modules   import LSTMLayer, SoftmaxLayer, SigmoidLayer
from pybrain.supervised          import BackpropTrainer
from pybrain.tools.shortcuts     import buildNetwork
from pybrain.utilities           import percentError

################################################################################################
### USAGE:
### Arguments: <folder>: folder in current directory that contains the testFeatures_x.csv files
###            <numX>: file number
### python training_data.py <folder> <num1> <num2> <optional: num3>
### Example: python training_data.py TestFeatures_16_0_smples 0 1
#################################################################################################

if len(sys.argv) < 4:
    print("Usage: python training_data.py <folder> <num1> <num2> <optional: num3>")
    print("Example: python training_data.py TestFeatures_16_0_smples 0 1")
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

tmp_data = np.loadtxt(sys.argv[1]+"/testFeatures_" + sys.argv[2] + ".csv", delimiter=',')
tmp_data1 = np.loadtxt(sys.argv[1]+"/testFeatures_" + sys.argv[3] + ".csv", delimiter=',')
if len(sys.argv) > 4:
    tmp_data2 = np.loadtxt(sys.argv[1]+"/testFeatures_" + sys.argv[4] + ".csv", delimiter=',')

take_diff_once = False
take_diff_twice = False

if take_diff_once:
    difference = take_difference(tmp_data)
    tmp_data = np.vstack((tmp_data,difference))
    difference1 = take_difference(tmp_data1)
    tmp_data1 = np.vstack((tmp_data1,difference1))
    if len(sys.argv) > 4:
        difference2 = take_difference(tmp_data2)
        tmp_data2 = np.vstack((tmp_data2,difference2))
        
if take_diff_once and take_diff_twice:
    diff_2_1 = take_difference(difference)
    tmp_data = np.vstack((tmp_data,diff_2_1))
    diff_2_2 = take_difference(difference1)
    tmp_data1 = np.vstack((tmp_data1,diff_2_2))
    if len(sys.argv) > 4:
        diff_2_3 = take_difference(difference2)
        tmp_data2 = np.vstack((tmp_data2,diff_2_3))
        
training_data = np.vstack((tmp_data,tmp_data1))
if len(sys.argv) > 4:
    training_data = np.vstack((training_data,tmp_data2))
    
use_diff_only = False

if use_diff_only:
    training_data = np.vstack((difference,difference1))
if use_diff_only and len(sys.argv) > 4:
    diff_stack = np.vstack((diff_2_1,diff_2_2))  
    training_data = np.vstack((training_data,diff_stack))    

if use_diff_only:
    target = np.repeat(np.array(range(len(sys.argv)-2)),len(difference))
    target = np.vstack(target)
else:
    target = np.repeat(np.array(range(len(sys.argv)-2)),len(tmp_data))
    target = np.vstack(target)



DS = ClassificationDataSet(217)
assert(training_data.shape[0] == target.shape[0])
DS.setField('input', training_data)
DS.setField('target', target)
tstdata, trndata = DS.splitWithProportion(0.25)
hidden_layer_neurons = (DS.indim+DS.outdim)/2
rnn = buildNetwork(DS.indim, hidden_layer_neurons, DS.outdim, hiddenclass=LSTMLayer, outclass=SigmoidLayer, outputbias=False, recurrent=True)
#print hidden_layer_neurons
# define a training method
trainer = BackpropTrainer(rnn,dataset=trndata, verbose=True) 
trainer.trainUntilConvergence(verbose = True, validationProportion = 0.3, maxEpochs = 1000, continueEpochs = 10)
ret = rnn.activateOnDataset(tstdata)
#print ret.transpose()
#print(tstdata['target'].transpose())
#print(tstdata['input'][0])
#print (np.rint(rnn.activateOnDataset(tstdata).transpose()))
#print(tstdata['target'].transpose())
#print (trainer.testOnClassData())
#print 'Percent Error on Test dataset: ' , percentError(rnn.activateOnDataset(tstdata),tstdata['target'])
print 'Percent Error on Test dataset: ' , percentError(trainer.testOnClassData(tstdata, verbose=True), tstdata['target'] )
print 'Percent Error on Training dataset: ' , percentError(trainer.testOnClassData(trndata), trndata['target'] )
exit(0)
