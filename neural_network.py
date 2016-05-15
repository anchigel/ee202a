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
### python training_data.py <folder>
### Example: python training_data.py TestFeatures_16_0_smples
#################################################################################################

if len(sys.argv) < 2:
    print("Usage: python training_data.py <folder>")
    print("Example: python training_data.py TestFeatures_16_0_smples")
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
    
tmp_data = np.loadtxt(sys.argv[1]+"/testFeatures_0.csv", delimiter=',')
tmp_data1 = np.loadtxt(sys.argv[1]+"/testFeatures_1.csv", delimiter=',')
#tmp_data2 = np.loadtxt(sys.argv[1]+"/testFeatures_2.csv", delimiter=',')
#tmp_data3 = np.loadtxt(sys.argv[1]+"/testFeatures_3.csv", delimiter=',')


difference = take_difference(tmp_data)
diff_2 = take_difference(difference)
tmp_data = np.vstack((tmp_data,difference))
tmp_data = np.vstack((tmp_data,diff_2))

difference = take_difference(tmp_data1)
diff_2 = take_difference(difference)
tmp_data1 = np.vstack((tmp_data1,difference))
tmp_data1 = np.vstack((tmp_data1,diff_2))

training_data = np.vstack((tmp_data,tmp_data1))
#training_data = np.vstack((training_data,tmp_data2))
#training_data = np.vstack((training_data,tmp_data3))

target = np.repeat(np.array(range(2)),len(tmp_data))
target = np.vstack(target)

DS = ClassificationDataSet(217)
assert(training_data.shape[0] == target.shape[0])
DS.setField('input', training_data)
DS.setField('target', target)
#tstdata, trndata = DS.splitWithProportion(0.25)
hidden_layer_neurons = (DS.indim+DS.outdim)/2
rnn = buildNetwork(DS.indim, hidden_layer_neurons, DS.outdim, hiddenclass=LSTMLayer, outclass=SigmoidLayer, outputbias=False, recurrent=True)
#print hidden_layer_neurons
# define a training method
trainer = BackpropTrainer(rnn,dataset=DS, verbose=True) 
trainer.trainUntilConvergence(verbose = True, validationProportion = 0.3, maxEpochs = 1000, continueEpochs = 10)

#print(tstdata['input'][0])
#print (np.rint(rnn.activateOnDataset(tstdata).transpose()))
#print(tstdata['target'].transpose())
#print (trainer.testOnClassData())
#print 'Percent Error on Test dataset: ' , percentError(np.rint(rnn.activateOnDataset(tstdata)), tstdata['target'] )
#print 'Percent Error on Test dataset: ' , percentError(map(int,trainer.testOnClassData(tstdata)), tstdata['target'] )
exit(0)
