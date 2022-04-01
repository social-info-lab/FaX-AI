from sklearn.linear_model import LogisticRegression as LR
from scipy.special import comb
from scipy.optimize import minimize
import torch.nn as nn
import torch
from torch.autograd import Variable
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
            self.ii = SHAPTorch(XZ)
            target = self.ii.influence(full_torch, XZ)[:,:-1]
        elif influence == 'mde':
            self.ii = MDETorch(XZ)
            target = self.ii.influence(full_torch, XZ)
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
        min_error = torch.tensor(float('inf'))
        best_params = None
        for epoch in range(params['num_epochs']):
            optimizer.zero_grad()
            error = self.error(X, target)
            if error < min_error:
                min_error = error
                best_params = self.model.linear.weight.data, self.model.linear.bias.data
                # print(error, best_params)
            # Todo: retain graph
            error.backward(retain_graph=True)
            optimizer.step()
        self.model.set_params(best_params[0], best_params[1])

    def error(self, X, target):
        """
        Error with respect to the influence.
        Args:
            X: torch.FloatTensor of the non-protected attributes
            target: torch tensor of the target input influence values
        """
        inf = self.ii.influence(self, X)
        # inf = Variable(inf, requires_grad = True)
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
        if type(coef) is np.ndarray:
            coef = torch.from_numpy(coef)
        if type(intercept) is np.ndarray:
            intercept = torch.from_numpy(intercept)
        with torch.no_grad():
            self.linear.weight = nn.Parameter(coef.type(torch.FloatTensor))
            self.linear.bias = nn.Parameter(intercept.type(torch.FloatTensor))

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
        return out.detach().numpy()

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
    def __init__(self, XZ, repeat=10):
        self.N, self.d = XZ.shape
        # shuffle Z
        XZ_in = XZ.numpy().copy()
        np.random.shuffle(XZ_in[:, -1])
        powd = 2**self.d
        # used for shuffling
        order = np.zeros((repeat, self.N), dtype=int)
        index = np.tile(np.arange(repeat), (self.N, 1)).T
        # shap = weights * (mode.predict(X_shuffled[first]) - model.predict(X_shuffled[second]))
        self.weights = [[] for i in range(self.d)]
        self.first = [[] for i in range(self.d)]
        self.second = [[] for i in range(self.d)]
        self.X_shuffled = np.tile(XZ_in, (repeat, powd, 1, 1))

        for i in range(repeat):
            order[i] = np.random.permutation(self.N)
        
        for i in range(powd):
            dims = []
            # i = 13 => marginal = [1 1 0 1]
            marginal = [int(x) for x in bin(i)[2:]]
            marginal = [0] * (self.d - len(marginal)) + marginal
            if i != powd-1:
                w = 1 / (self.d * comb(self.d-1, sum(marginal)))
            for idx, j in enumerate(marginal):
                if j == 0:
                    self.weights[idx].append(w)
                    self.first[idx].append(i)
                    self.second[idx].append(i + 2**(self.d-idx-1))
                elif j == 1:
                    dims.append(idx)
            # shuffling
            if len(dims):
                self.X_shuffled[:, i][:, :, dims] = self.X_shuffled[index, i, order][:, :, dims]
        self.X_shuffled = torch.tensor(self.X_shuffled.reshape(-1, self.d))

    def influence(self, method, X):
        # if no z in input then we can drop last column from X_shuffled
        # hopefully it won't effect the result
        # this is output
        out = torch.empty(X.shape)
        assert(self.N == X.shape[0])
        if X.shape[1] < self.d:
            X_shuffled = self.X_shuffled[:, :-1]
        else:
            X_shuffled = self.X_shuffled
        Yhat = method.predict_proba(X_shuffled, True)[:, 0]
        Yhat = Yhat.reshape((-1, 2**self.d, self.N))

        for i in range(X.shape[1]):
            Y_first = Yhat[:, self.first[i]]
            Y_second = Yhat[:, self.second[i]]
            # absolute diff
            temp = nn.L1Loss()(Y_first, Y_second).mean(0)
            out[:, i] = (temp * torch.tensor(self.weights[i])).sum(0)
        return out


class SHAP():
    def __init__(self, XZ, repeat=10):
        self.N, self.d = XZ.shape
        # shuffle Z
        XZ_in = XZ.copy()
        np.random.shuffle(XZ_in[:, -1])
        powd = 2**self.d
        # used for shuffling
        order = np.zeros((repeat, self.N), dtype=int)
        index = np.tile(np.arange(repeat), (self.N, 1)).T
        # shap = weights * (mode.predict(X_shuffled[first]) - model.predict(X_shuffled[second]))
        self.weights = [[] for i in range(self.d)]
        self.first = [[] for i in range(self.d)]
        self.second = [[] for i in range(self.d)]
        self.X_shuffled = np.tile(XZ_in, (repeat, powd, 1, 1))

        for i in range(repeat):
            order[i] = np.random.permutation(self.N)
        
        for i in range(powd):
            dims = []
            # i = 13 => marginal = [1 1 0 1]
            marginal = [int(x) for x in bin(i)[2:]]
            marginal = [0] * (self.d - len(marginal)) + marginal
            if i != powd-1:
                w = 1 / (self.d * comb(self.d-1, sum(marginal)))
            for idx, j in enumerate(marginal):
                if j == 0:
                    self.weights[idx].append(w)
                    self.first[idx].append(i)
                    self.second[idx].append(i + 2**(self.d-idx-1))
                elif j == 1:
                    dims.append(idx)
            # shuffling
            if len(dims):
                self.X_shuffled[:, i][:, :, dims] = self.X_shuffled[index, i, order][:, :, dims]
        self.X_shuffled = self.X_shuffled.reshape(-1, self.d)

    def influence(self, method, X):
        # if no z in input then we can drop last column from X_shuffled
        # hopefully it won't effect the result
        # this is output
        out = np.empty(X.shape)
        assert(self.N == X.shape[0])
        if X.shape[1] < self.d:
            X_shuffled = self.X_shuffled[:, :-1]
        else:
            X_shuffled = self.X_shuffled
        Yhat = method.predict_proba(X_shuffled)[:, 0]
        Yhat = Yhat.reshape((-1, 2**self.d, self.N))

        for i in range(X.shape[1]):
            out[:, i] = (np.abs(Yhat[:, self.first[i]] - Yhat[:, self.second[i]]).mean(0) * np.array(self.weights[i])[:, None]).sum(0)
        return out.mean(0)



# class MDETorch_pool():
#     def __init__(self, XZ, full, N=400, M=400):
#         m, n = XZ.shape
#         idx_XA = np.random.choice(m, N)
#         idx_XB = np.random.choice(m, N)
#         """
#         (X1, X2, ..., XN, X1, X2, ...) -> M times
#         dim : (NXM, n-1)
#         """
#         XA = XZ[idx_XA, :-1].repeat(M, 1)
#         XB = XZ[idx_XB, :-1].repeat(M, 1)

#         idx_Z = np.random.choice(m, N)
#         """
#         (Z1, Z1, ..(N times).., Z2, Z2, ..(N times).. .., ZM, ZM, ..(N times)..)
#         dim : (NXM, 1)
#         """
#         Z = XZ[idx_Z, -1:].repeat_interleave(M, axis=0)
#         self.XZ_A = torch.cat((XA.T, Z.T)).T
#         self.XZ_B = torch.cat((XB.T, Z.T)).T
#         self.N, self.M = N, M
#         self.n, self.m = n, m

#     def influence(self, method, X):
#         # we can infer is_z from X
#         if self.n == X.shape[1]:
#             is_z = True
#         else:
#             is_z = False
#         if is_z == False:
#             inpA, inpB = self.XZ_A[:, :-1], self.XZ_B[:, :-1]
#         else:
#             inpA, inpB = self.XZ_A, self.XZ_B
       
#         Yhat = method.predict_proba(inpA, True)[:, 0]
#         Yhat_ = method.predict_proba(inpB, True)[:, 0]
#         #dY = self.Y - self.Y_ - Yhat + Yhat_
#         dY = torch.reshape(Yhat - Yhat_, (self.N, self.M)).mean(1)
#         # L1 Norm
#         # todo: don't take the mean 
#         # out = dY.mean(1)
#         return dY


class MDE_pool():
    """
    1. Draw N samples of X'' and X from the full set of values of X. This gives two vectors X'' and X
    2. For each of the N samples in each of the two arrays X'' and X, draw M samples of Z'. We end up with two big matrices: A = (X, Z') and B = (X'', Z')
    """
    def __init__(self, XZ, N=400, M=400):
        m, n = XZ.shape
        idx_XA = np.random.choice(m, N)
        idx_XB = np.random.choice(m, N)
        """
        (X1, X2, ..., XN, X1, X2, ...) -> M times
        dim : (NXM, n-1)
        """
        XA = np.tile(XZ[idx_XA,:-1], (M,1)) 
        XB = np.tile(XZ[idx_XB,:-1], (M,1))

        idx_Z = np.random.choice(m, N)
        """
        (Z1, Z1, ..(N times).., Z2, Z2, ..(N times).. .., ZM, ZM, ..(N times)..)
        dim : (NXM, 1)
        """
        Z = np.repeat(XZ[idx_Z,-1:], M, axis=0)
        self.XZ_A = np.hstack((XA, Z))
        self.XZ_B = np.hstack((XB, Z))
        self.N, self.M = N, M
        self.n, self.m = n, m
    """
    3. Now, Yhat = eval_model.predict_proba(A), Yhat' = eval_model.predict_proba(B)
    4. Compute dY=Yhat-Yhat', then dY.reshape((N,M)) , followed up by mean over the M columns. This gives us a NX1 size vector.
    5. We take abs of this and average over N.
    """
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
        dY = np.reshape(Yhat - Yhat_, (self.N, self.M)).mean(1)
        # L1 Norm
        out = np.mean(np.abs(dY), 0)
        return out



class MDE_ind():
    """
    1. Draw N samples of Xi'' and Xi from the full set of values of Xi. This gives two vectors Xi'' and Xi
    2. For each of the N samples in each of the two vectors Xi'' and Xi, draw jointly M samples of Z' and all Xj' (all j!=i)
       from the full set of (X,Z) tuples. We end up with two big matrices: A = (Xi, all Xj',Z') and B = (Xi'', all Xj', Z'). 
       Size of A and B is (nXN)XM
    """
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
    """    
    3. Now, Yhat = eval_model.predict_proba(A), Yhat' = eval_model.predict_proba(B)
    4. Compute dY=Yhat-Yhat', then dY.reshape((n,N,M)) , followed up by mean over the M columns. This gives us a nXN size vector.
    5. We take abs of this and average over N.
    """
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

    
# this is the ind version
class MDETorch():
    """
    1. Draw N samples of Xi'' and Xi from the full set of values of Xi. This gives two vectors Xi'' and Xi
    2. For each of the N samples in each of the two vectors Xi'' and Xi, draw jointly M samples of Z' and all Xj' (all j!=i)
       from the full set of (X,Z) tuples. We end up with two big matrices: A = (Xi, all Xj',Z') and B = (Xi'', all Xj', Z'). 
       Size of A and B is (nXN)XM
    """
    def __init__(self, XZ, M=400, N=400):
        self.m, self.n = XZ.shape
        self.M, self.N = M, N
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

        # self.Y = self.full.predict_proba(self.XZ_A, True)[:, 0]
        # self.Y_ = self.full.predict_proba(self.XZ_B, True)[:, 0]
    """    
    3. Now, Y = full_model.predict_proba(A), Y' = full_model.predict_proba(B), Yhat = eval_model.predict_proba(A), Yhat' = eval_model.predict_proba(B)
    4. Compute dY=Y-Y', then dY.reshape((n,N,M)), followed up by mean over the M columns. This gives us a nXN size vector.
    5. Compute dYhat=Yhat-Yhat', then dYhat.reshape((n,N,M)), followed up by mean over the M columns. This gives us a nXN size vector.
    6. Next we compute the loss b/w dY and dYhat as L = (dY - dYhat).mean(2). We can combine the resulting nX1 vector by
        a. np.mean(L^2)
        b. np.mean(np.abs(L))
    """
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
        dY = Yhat - Yhat_
        dY_ = torch.reshape(dY, (self.n, self.N, self.M)).mean(2)
        # L1 Norm
        # out = torch.mean(torch.abs(dY_), 1)
        return dY_.T


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
            mixXZ[:, idx] = self.model.predict_proba(XZi)[:, 0]
        return mixXZ

    def predict_proba(self, X):
        mixXZ = self.mix_input(X)
        out = np.matmul(mixXZ, self.Ws)
        out = np.array([out, 1 - out])
        return out.T

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
        self.Ws = self.Ws / np.sum(self.Ws)

    def mix_input(self, X):
        mixXZ = np.empty((X.shape[0], self.Zs.shape[0]))
        for idx, Zi in enumerate(self.Zs):
            XZi = np.hstack((X, torch.ones(X.shape[0], 1) * Zi))
            mixXZ[:, idx] = self.model.predict_proba(XZi)[:, 0]
        return mixXZ

    def predict_proba(self, X):
        mixXZ = self.mix_input(X)
        out = np.matmul(mixXZ, self.Ws)
        out = np.array([out, 1 - out])
        return out.T


    def predict(self, X):
        out = self.predict_proba(X)[:, 1]
        return np.array(out > 0.5, dtype=np.int32)

