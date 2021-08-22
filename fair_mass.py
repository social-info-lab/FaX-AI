from sklearn.linear_model import LogisticRegression as LR
from scipy.optimize import minimize
import torch.nn as nn
import torch
import numpy as np
import math


class Optimization():
    """
    Optimization approach for binary Y that removes the influence of the
    protected attributes Z, while preserving the influence of non-protected
    attributes X by minimizing L^{IND}_{ATE}(X) or L^{IND}_{ATE}.

    During initialization the model trains (calls self.fit) and the base
    estimator is LogisticRegression() below.

    Args:
        X: numpy array of the non-protected attributes
        Z: numpy array of the non-protected attributes of
            the protected attribute (must be binary)
        Y: numpy array of the non-protected of the target value (must be binary)
        influence: string 'shap' or 'ate' to indicate which influence measure
            to be used
        params: a dictionary of the parameters to be used
    """
    def __init__(self, X, Z, Y, influence='shap', params=None):
        # model
        partial, full, XZ = self.get_sklearn_models(X, Z, Y)
        self.model = LogisticRegression(X.shape[1])
        self.model.set_params(partial.coef_, partial.intercept_)

        # target
        XZ = torch.from_numpy(XZ).type(torch.FloatTensor)
        X = torch.from_numpy(X).type(torch.FloatTensor)
        full_torch = LogisticRegression(X.shape[1] + 1)
        full_torch.set_params(full.coef_, full.intercept_)
        if influence == 'shap':
            self.ii = SHAPTorch()
            target = self.ii.influence(full_torch, XZ)[:-1]
        elif influence == 'ate':
            self.ii = ObjBITorch(XZ, full=full_torch)
            target = torch.zeros(XZ.shape[1])
        # fit
        if params is None:
            params = {'num_epochs': 100, 'learning_rate': 1e-3}
        self.fit(X, target, params)

    def get_sklearn_models(self, X, Z, Y):
        """
        Trains a logisic regression model for full and partial model.

        Args:
            X: numpy array of the non-protected attributes
            Z: numpy array of the non-protected attributes of
                the protected attribute (must be binary)
            Y: numpy array of the non-protected of the target value (must be binary)
        """
        partial = LR().fit(X, Y)
        XZ = np.hstack((X, Z))
        full = LR().fit(XZ, Y)
        return partial, full, XZ

    def fit(self, X, target, params):
        """
        Trains the optimization approach

        Args:
            X: torch.FloatTensor of the data for non-protected attributes
            target: torch tensor of the target input influence values
            params: a dictionary of the parameters to be used
        """
        optimizer = self.model.optimizer(self.model.parameters(), lr=params['learning_rate'])
        for epoch in range(params['num_epochs']):
            optimizer.zero_grad()
            error = self.error(X, target)
            # Todo: retain graph
            error.backward(retain_graph=True)
            optimizer.step()

    def error(self, X, target):
        """
        Error with respect to the influence.

        Args:
            X: torch.FloatTensor of the non-protected attributes
            target: torch tensor of the target input influence values
        """
        inf = self.ii.influence(self, X)
        return nn.MSELoss()(target, inf)

    def predict_proba(self, X, grad=False):
        """
        Returns the probabilities of the predicted class labels

        Args:
            X: numpy array or torch tensor of the non-protected attributes
        """
        if grad:
            return self.model.predict_proba(X, True)
        with torch.no_grad():
            return self.model.predict_proba(X)

    def predict(self, X):
        """
        Returns the predicted class labels

        Args:
            X: numpy array or torch tensor of the non-protected attributes
        """
        with torch.no_grad():
            return self.model.predict(X)


class LogisticRegression(nn.Module):
    """
    Logisitc Regression model using PyTorch

    Args:
        input_size: size of the dataset that will be passed in
    """
    def __init__(self, input_size):
        super().__init__()
        self.input_size = input_size
        self.linear = nn.Linear(input_size, 1)
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = torch.optim.Adam

    def set_params(self, coef, intercept):
        with torch.no_grad():
            self.linear.weight = nn.Parameter(torch.from_numpy(coef).type(torch.FloatTensor))
            self.linear.bias = nn.Parameter(torch.from_numpy(intercept).type(torch.FloatTensor))

    def forward(self, X):
        out = self.linear(X)
        return out

        # only binary labels: returns the probability of label = 1
    def predict_proba(self, X, grad=False):
        if not grad:
            X = torch.from_numpy(X).type(torch.FloatTensor)
        forward = self.forward(X)
        out = torch.sigmoid(forward)
        if grad:
            return out
        return out.numpy()

    def predict(self, X):
        out = self.predict_proba(X)[:, 0]
        return np.array(out > 0.5, dtype=np.int32)


class SHAPTorch():
    """
    SHAP influence measure using PyTorch. Needed for optimzation approach.

    Args:
        method: size of the dataset that will be passed in
        X: torch.FloatTensor of the data being evaluated
        repeat: number of times to repeat the SHAP value calculation approach
            to later take the mean of
    """
    def influence(self, method, X, repeat=10):
        out = torch.empty(X.shape)
        for i in range(X.shape[1]):
            out[:, i] = self.shap_feature(method, X, i, repeat)
        # out is ii per-sample
        return out.mean(0)

    def shap_feature(self, method, X, curr, repeat=10):
        dims = X.shape[1]
        out = torch.empty((2**(dims-1), X.shape[0]))
        count = 0
        for i in range(2**dims):
            marginal = [int(x) for x in bin(i)[2:]]
            length = dims - len(marginal)
            marginal = [0] * length + marginal
            if marginal[curr] == 0:
                w = 1 / (dims * math.comb(dims-1, sum(marginal)))
                out[count] = w * self.gii_feature(method, X, curr, marginal, repeat)
                count += 1
        return torch.sum(out, 0)

    # curr is int, marginal is array
    def gii_feature(self, method, X, curr, marginal, repeat=10):
        marginal1 = marginal.copy()
        marginal2 = marginal.copy()
        marginal2[curr] = 1
        return self.gii(method, X, marginal1, marginal2, repeat)

    def gii(self, method, X, marginal1, marginal2, repeat=10):
        order = torch.empty((repeat, X.shape[0]), dtype=torch.int64)
        for i in range(repeat):
            order[i] = torch.randperm(X.shape[0])
        X_first = self.impute(X, marginal1, order)
        X_second = self.impute(X, marginal2, order)
        out = torch.empty((repeat, X.shape[0]))
        for i in range(repeat):
            Y_first = method.predict_proba(X_first[i], True)[:, 0]
            Y_second = method.predict_proba(X_second[i], True)[:, 0]
            # reduction = 'none'
            # in shap they are using Y_first - Y_second
            out[i] = nn.L1Loss()(Y_first, Y_second)
        return out.mean(0)

    def impute(self, data, marginal, order):
        dims = np.nonzero(marginal)[0]
        new_data = data.repeat(order.shape[0], 1, 1)
        for i in range(order.shape[0]):
            new_data[i][:, dims] = new_data[i][:, dims][order[i]]
        return new_data


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
            # in shap exact they are using (-Y_first + Y_second)
            out[i] = np.abs(-Y_first + Y_second)
        return out.mean(0)

    def impute(self, data, marginal, order):
        dims = np.nonzero(marginal)[0]
        new_data = np.tile(data, (order.shape[0], 1, 1))
        # new_data = data.repeat(order.shape[0], 1, 1)
        for i in range(order.shape[0]):
            new_data[i][:, dims] = new_data[i][:, dims][order[i]]
        return new_data


class ObjBITorch():
    """
    ATE influence measure using PyTorch. Needed for optimzation approach.

    Args:
        XZ: torch.FloatTensor of the data being evaluated
        full: the base estimator model (see Optimization method init)
        M: number of samples for the set to randomly pick replacments from
        N: number of random samples to choose to replace
    """
    def __init__(self, XZ, full, M=400, N=400):
        self.m, self.n = XZ.shape
        self.M, self.N = M, N
        self.full = full
        indices_AM = np.random.choice(self.m, self.M)
        indices_BM = np.random.choice(self.m, self.M)
        indices_AN = np.random.choice(self.m, self.N)
        indices_BN = np.random.choice(self.m, self.N)

        XZ_A = XZ[indices_AM, :].repeat(self.n, self.N, 1, 1)
        XZ_B = XZ[indices_BM, :].repeat(self.n, self.N, 1, 1)
        sample_A = XZ[indices_AN, :]
        sample_B = XZ[indices_BN, :]

        for i in range(N):
            XZ_A[np.arange(self.n), i, :, np.arange(self.n)] = sample_A[i:i+1,:].T.repeat((1, self.M))
            XZ_B[np.arange(self.n), i, :, np.arange(self.n)] = sample_B[i:i+1,:].T.repeat((1, self.M))

        self.XZ_A = torch.reshape(XZ_A, (-1, self.n))
        self.XZ_B = torch.reshape(XZ_B, (-1, self.n))

        self.Y = self.full.predict_proba(self.XZ_A, True)[:, 0]
        self.Y_ = self.full.predict_proba(self.XZ_B, True)[:, 0]

    def influence(self, method, X=None):
        # we can infer is_z from X
        if self.n == X.shape[1]:
            is_z = True
        else:
            is_z = False
        if is_z == False:
            inpA, inpB = self.XZ_A[:, :-1], self.XZ_B[:, :-1]
        else:
            inpA, inpB = self.XZ_A, self.XZ_B

        Yhat = method.predict_proba(inpA, True)[:, 0]
        Yhat_ = method.predict_proba(inpB, True)[:, 0]
        dY = self.Y - self.Y_ - Yhat + Yhat_
        dY_ = torch.reshape(dY, (self.n, self.N, self.M)).mean(2)
        # L1 Norm
        out = torch.mean(torch.abs(dY_), 1)
        return out


class ObjBI():
    def __init__(self, XZ, M=400, N=400):
        self.m, self.n = XZ.shape
        self.M, self.N = M, N
        indices_AM = np.random.choice(self.m, self.M)
        indices_BM = np.random.choice(self.m, self.M)
        indices_AN = np.random.choice(self.m, self.N)
        indices_BN = np.random.choice(self.m, self.N)

        XZ_A = np.tile(XZ[indices_AM, :], (self.n, self.N, 1, 1))
        XZ_B = np.tile(XZ[indices_BM, :], (self.n, self.N, 1, 1))
        sample_A = XZ[indices_AN, :]
        sample_B = XZ[indices_BN, :]

        for i in range(N):
            XZ_A[np.arange(self.n), i, :, np.arange(self.n)] = np.tile(sample_A[i:i+1,:].T, (1, self.M))
            XZ_B[np.arange(self.n), i, :, np.arange(self.n)] = np.tile(sample_B[i:i+1,:].T, (1, self.M))

        self.XZ_A = np.reshape(XZ_A, (-1, self.n))
        self.XZ_B = np.reshape(XZ_B, (-1, self.n))

    def influence(self, method, X):
        # we can infer is_z from X
        if self.n == X.shape[1]:
            is_z = True
        else:
            is_z = False
        if is_z == False:
            inpA, inpB = self.XZ_A[:, :-1], self.XZ_B[:, :-1]
        else:
            inpA, inpB = self.XZ_A, self.XZ_B
        Yhat = method.predict_proba(inpA)[:, 0]
        Yhat_ = method.predict_proba(inpB)[:, 0]
        dY = Yhat - Yhat_
        dY_ = np.reshape(dY, (self.n, self.N, self.M)).mean(2)
        # L1 Norm
        out = np.mean(np.abs(dY_), 1)
        return out


class OIM():
    """
    OIM using PyTorch. Needed for experiments in the inpute influence paper.
    See full details about the approach in
    https://github.com/social-info-lab/discrimination-prevention/tree/master/src

    Args:
        X: numpy array of the non-protected attributes
        Z: numpy array of the non-protected attributes of
            the protected attribute (must be binary)
        Y: numpy array of the non-protected of the target value (must be binary)
    """
    def __init__(self, X, Z, Y):
        XZ = np.hstack((X, Z))
        self.model = LR().fit(XZ, Y)
        self.Zs = np.unique(Z)
        mixXZ = self.mix_input(X)
        cons = [{'type': 'eq', 'fun': lambda w: np.sum(w) - 1, 'jac': lambda w: np.ones(w.shape[0])},
                {'type': 'ineq', 'fun': lambda w: w, 'jac': lambda w: np.eye(w.shape[0])}]
        res = minimize(self.objective, np.random.random(self.Zs.shape[0]), args=(mixXZ, Y), method='SLSQP', constraints=cons,
                        options={'disp': True})
        self.Ws = res.x

    def objective(self, w, X, Y):
        out = np.matmul(X, w)
        return np.mean((out - Y)**2)

    def mix_input(self, X):
        mixXZ = np.empty((X.shape[0], self.Zs.shape[0]))
        for idx, Zi in enumerate(self.Zs):
            XZi = np.hstack((X, torch.ones(X.shape[0], 1) * Zi))
            mixXZ[:, idx] = self.model.predict_proba(XZi)
        return mixXZ

    def predict_proba(self, X):
        mixXZ = self.mix_input(X)
        out = np.matmul(mixXZ, self.Ws)
        out = np.array([out, 1 - out])
        return out

    def predict(self, X):
        out = self.predict_proba(X)[:, 0]
        return np.array(out > 0.5, dtype=np.int32)


class MIM():
    """
    MIM using PyTorch. Needed for experiments in the inpute influence paper.

    Args:
        X: numpy array of the non-protected attributes
        Z: numpy array of the non-protected attributes of
            the protected attribute (must be binary)
        Y: numpy array of the non-protected of the target value (must be binary)
    """
    def __init__(self, X, Z, Y):
        XZ = np.hstack((X, Z))
        self.model = LR().fit(XZ, Y)
        self.Zs, self.Ws = np.unique(Z, return_counts=True)
        self.Ws = self.Ws / torch.sum(self.Ws)

    def mix_input(self, X):
        mixXZ = np.empty((X.shape[0], self.Zs.shape[0]))
        for idx, Zi in enumerate(self.Zs):
            XZi = np.hstack((X, torch.ones(X.shape[0], 1) * Zi))
            mixXZ[:, idx] = self.model.predict_proba(XZi)
        return mixXZ

    def predict_proba(self, X):
        mixXZ = self.mix_input(X)
        out = np.matmul(mixXZ, self.Ws)
        out = np.array([out, 1 - out])
        return out


    def predict(self, X):
        out = self.predict_proba(X)[:, 0]
        return np.array(out > 0.5, dtype=np.int32)
