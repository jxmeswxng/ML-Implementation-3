import numpy as np
from sklearn import tree

class BoostedDT:

    def __init__(self, numBoostingIters=100, maxTreeDepth=3):
        self.numBoostingIters=numBoostingIters
        self.maxTreeDepth=maxTreeDepth
    
    def fit(self, X, y):
        count = X.shape[0]
        w = np.ones((count))/count
        k = len(np.unique(y))
        clf_list = []
        clf_weights_list = []
        for i in range(self.numBoostingIters):
            clf = tree.DecisionTreeClassifier(max_depth=self.maxTreeDepth)
            fit = clf.fit(X,y,sample_weight=w)
            pred = fit.predict(X)
            correct = pred-y
            for i in range(X.shape[0]):
                if correct[i] != 0:
                    correct[i] = 1
            err_sum = sum(w*correct)/sum(w)
            alpha = np.log((1-err_sum)/err_sum) + np.log(k-1)
            w = w*np.exp(alpha*correct)            
            w = w/sum(w)
            clf_list.append(fit)
            clf_weights_list.append(alpha)   
        self.clf_list = clf_list
        self.clf_weights_list = clf_weights_list
        self.labels = np.unique(y)
        
    def predict(self, X):
        clf_pred = np.zeros((X.shape[0],self.numBoostingIters))
        for i in range(self.numBoostingIters):
            clf_pred[:,i] = self.clf_list[i].predict(X)
        boost_dict = {}            
        ensemble = []
        for i in range(X.shape[0]):
            for m in self.labels:
                boost_dict[m] = 0
            for j in range(self.numBoostingIters):
                for k in self.labels:
                    if clf_pred[i,j] == k:
                        boost_dict[k] = boost_dict[k] + self.clf_weights_list[j]
            for n in range(len(boost_dict.values())):
                if boost_dict.values()[n] == max(boost_dict.values()):
                    ensemble.append(n)
        ensemble = np.asarray(ensemble)
        return ensemble      