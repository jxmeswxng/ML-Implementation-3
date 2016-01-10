import numpy as np
from sklearn import datasets
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from sklearn.cross_validation import KFold

from boostedDT import BoostedDT

filename = 'challengeTrainLabeled.dat'
data = np.loadtxt(filename, delimiter=',')

Xdata = data[:,0:10]
ydata = data[:,10]

n,d = Xdata.shape
nTrain = 0.5*n

idx = np.arange(n)
np.random.seed(22)
np.random.shuffle(idx)
Xdata = Xdata[idx]
ydata = ydata[idx]

boost_iter_list = []
for i in [100,1000,10000]:
    depth_list = []
    for j in [1,2,3]:
        test_accuracy_list = []
        modelBoostedDT = BoostedDT(numBoostingIters=i, maxTreeDepth=j)
        kf = KFold(2000, n_folds=10)
        for train_index, test_index in kf:
            Xtrain, Xtest = Xdata[train_index], Xdata[test_index]
            ytrain, ytest = ydata[train_index], ydata[test_index]
            modelBoostedDT.fit(Xtrain,ytrain) 
            test_ypred_BoostedDT = modelBoostedDT.predict(Xtest)
            test_accuracyBoostedDT = accuracy_score(ytest, test_ypred_BoostedDT)
            test_accuracy_list.append(test_accuracyBoostedDT)
        depth_list.append(np.mean(test_accuracy_list))
    boost_iter_list.append(depth_list)
'''
output
[[0.031000000000000007, 0.016499999999999997, 0.017000000000000005],
 [0.027000000000000003, 0.019000000000000003, 0.013500000000000002],
 [0.026500000000000003, 0.020500000000000001, 0.013000000000000001]] 
'''

filename1 = 'challengeTestUnlabeled.dat'
data1 = np.loadtxt(filename1, delimiter=',')

unlabeledModelBoostedDT = BoostedDT(numBoostingIters=100, maxTreeDepth=1)
unlabeledModelBoostedDT.fit(Xdata,ydata) 
unlabeledypred_BoostedDT = unlabeledModelBoostedDT.predict(data1)
np.set_printoptions(threshold='nan')












