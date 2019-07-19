# models and prediction code goes here.
from sklearn.tree import DecisionTreeClassifier
import numpy as np
import pandas as pd

class Random_forest_implementation():
    
    def __init__(self, k_trees, max_depth):
        self.k_trees = k_trees
        self.depth = max_depth
        self.models = [self.create_trees() for i in range(k_trees)]

    def create_trees(self):
        tree = DecisionTreeClassifier(splitter = "random", max_depth = self.depth)
        
        return tree

    def fit(self, X, y):
        for tree in self.models:
            tree = tree.fit(X, y)

    def predict(self, x):
        predictions = pd.DataFrame()
        for m in self.models:
            prediction = m.predict(x).astype(int)
            predictions = predictions.append(pd.Series(prediction), ignore_index=True)
        predictions = predictions.transpose() 
        result = predictions.apply(np.bincount, axis = 1)
        
        return np.array(result.apply(np.argmax))
    
    def score(self, x, y):
        out = np.where(self.predict(x) == y)
        return len(out[0]) / len(y)

class Adaboost_implementation():

    def __init__(self, k_estimators):
        # self.X = x
        # self.y = np.sign(y - (0.5 * np.ones(y.shape)))
        # self.y_0_1 =y
        self.k_estimators = k_estimators
        self.stumps = [DecisionTreeClassifier(max_depth=1) for k in range(k_estimators)]
        self.a = np.zeros(k_estimators)

    def fit(self, X, y):
        y = np.sign(y - (0.5 * np.ones(y.shape)))
        self.weights = np.ones(X.shape[0])/X.shape[0]
        
        for k in range(self.k_estimators):
            stump = self.stumps[k]
            stump.fit(X, y, sample_weight = self.weights)
            pred = stump.predict(X)
            #change pred from 0/1 to -1 /1
            pred = np.sign(pred - (0.5 * np.ones(pred.shape)))
            incorrect = (pred != y)
            training_err = np.sum(np.abs(incorrect) * self.weights)/ np.sum(self.weights)
            a = np.log(np.divide((1 - training_err), training_err))
            self.weights = self.weights * np.exp(a * np.abs(incorrect))
            self.stumps[k] = stump
            self.a[k] = a

    def predict(self, x):
        predictions = pd.DataFrame()
        for k in range(self.k_estimators):
            t = self.stumps[k]
            prediction = t.predict(x).astype(int) * self.a[k]
            prediction = np.heaviside(prediction, 1)
            predictions = predictions.append(pd.Series(prediction), ignore_index=True)
        predictions = predictions.transpose() 
        result = predictions.apply(np.bincount, axis = 1)
        
        return np.array(result.apply(np.argmax))
    

    def score(self, x, y):
        out = np.where(self.predict(x) == y)
        return len(out[0]) / len(y)
