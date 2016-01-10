import numpy as np
from sklearn import datasets
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from sklearn.cross_validation import KFold
from sklearn.ensemble import RandomForestClassifier

filename = 'challengeTrainLabeled.dat'
data = np.loadtxt(filename, delimiter=',')
filename1 = 'challengeTestUnlabeled.dat'
data1 = np.loadtxt(filename1, delimiter=',')

Xdata = data[:,0:10]
ydata = data[:,10]

n,d = Xdata.shape

idx = np.arange(n)
np.random.seed(88)
np.random.shuffle(idx)
Xdata = Xdata[idx]
ydata = ydata[idx]

oob_score_matrix = []
for trial in range(100):
    oob_score_list = []
    for i in [10,100,1000,10000]:
        rf = RandomForestClassifier(n_estimators=i,max_features="sqrt",n_jobs=-1,
                                    bootstrap=True,oob_score=True)
        rf.fit(Xdata,ydata)
        oob_score_list.append(rf.oob_score_)
    oob_score_matrix.append(oob_score_list)
    
oob_score_2D = oob_score_matrix[0]
for i in range(1,100):
    oob_score_2D = np.vstack((oob_score_2D,oob_score_matrix[i]))
oob_score_means = np.mean(oob_score_2D,axis=0)    
'''    
  10         100        1000      10000
0.90374 ,  0.964825,  0.96817 ,  0.96925
'''   
#diagnostics for n_estimators = 100
rf = RandomForestClassifier(n_estimators=100,max_features="sqrt",n_jobs=-1,
                                    bootstrap=True,oob_score=True)
rf.fit(Xdata,ydata)
feature_importance_rank = rf.feature_importances_

#predictions
rf_predictions = rf.predict(data1)
commaseparated = ','.join(map(str, rf_predictions)) 

'''
- X | output predictions into .dat file
- fill out README
    - accuracy score
    - predictions
    - 3-4 sentence write-up
- update pdf
    - 3-4 sentence write-up
'''