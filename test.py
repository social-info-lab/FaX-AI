
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression as LR
import shap
import math

np.random.seed(0)
X = np.random.random((100, 3))
w = np.array([1.0, -2.0, 3.0])
y = 1/(1 + np.exp(-X @ w))
Y = np.array(y >= 0.5, dtype=np.int32)
model = LR().fit(X, Y)

class SHAP():

    def influence(self, method, X, repeat=1000):
        out = np.empty(X.shape)
        for i in range(X.shape[1]):
            out[:, i] = self.shap_feature(method, X, i, repeat)
        # out is ii per-sample
        return out

    def shap_feature(self, method, X, curr, repeat=10):
        dims = X.shape[1]
        out = np.empty((2**(dims-1), X.shape[0]))
        # print("shape: ", out.shape)
        count = 0
        for i in range(2**dims):
            marginal = [int(x) for x in bin(i)[2:]]
            length = dims - len(marginal)
            marginal = [0] * length + marginal
            if marginal[curr] == 0:
                w = 1 / (dims * math.comb(dims-1, sum(marginal)))
                out[count] = w * self.gii_feature(method, X, curr, marginal, repeat)
                count += 1
        return np.sum(out, 0)

    # curr is int, marginal is array
    def gii_feature(self, method, X, curr, marginal, repeat=10):
        marginal1 = marginal.copy()
        marginal2 = marginal.copy()
        marginal2[curr] = 1
        return self.gii(method, X, marginal1, marginal2, repeat)

    def gii(self, method, X, marginal1, marginal2, repeat=10):
        order = np.empty((repeat, X.shape[0]), dtype=np.int64)
        for i in range(repeat):
            order[i] = np.random.permutation(X.shape[0])
        X_first = self.impute(X, marginal1, order)
        X_second = self.impute(X, marginal2, order)
        out = np.empty((repeat, X.shape[0]))
        for i in range(repeat):
            Y_first = method.predict_proba(X_first[i])[:, 0]
            Y_second = method.predict_proba(X_second[i])[:, 0]
            # in shap they are using (-Y_first + Y_second)
            out[i] = - Y_first + Y_second
        return out.mean(0)
    
    def impute(self, data, marginal, order):
        dims = np.nonzero(marginal)[0]
        new_data = np.tile(data, (order.shape[0], 1, 1))
        # new_data = data.repeat(order.shape[0], 1, 1)
        for i in range(order.shape[0]):
            new_data[i][:, dims] = new_data[i][:, dims][order[i]]
        return new_data

val1 = shap.Explainer(model, X)(X).values
val2 = shap.explainers.Exact(model.predict_proba, X)(X).values
val3 = SHAP().influence(model, X)

print("val1: ", val1[:5])
print("val2: ", val2[:5,:,1])
print("val3: ", val3[:5])
