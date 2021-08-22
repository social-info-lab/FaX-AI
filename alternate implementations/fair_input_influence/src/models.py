import numpy as np
import pandas as pd
import cvxpy
import statsmodels.api as sm

import fairlearn.reductions as red
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.preprocessing import LabelEncoder

from aif360.sklearn.utils import check_inputs

class ExponentiatedGradientReductionFix(BaseEstimator, ClassifierMixin):
    """Fixed for our use-case copy of exponentiated gradient reduction for
    fair classification from AIF360 library.
    Exponentiated gradient reduction is an in-processing technique that reduces
    fair classification to a sequence of cost-sensitive classification problems,
    returning a randomized classifier with the lowest empirical error subject to
    fair classification constraints [#agarwal18]_.
    References:
        .. [#agarwal18] `A. Agarwal, A. Beygelzimer, M. Dudik, J. Langford, and
           H. Wallach, "A Reductions Approach to Fair Classification,"
           International Conference on Machine Learning, 2018.
           <https://arxiv.org/abs/1803.02453>`_
    """
    def __init__(self,
                 prot_attr,
                 estimator,
                 constraints,
                 eps=0.01,
                 T=50,
                 nu=None,
                 eta_mul=2.0,
                 drop_prot_attr=True):
        """
        Args:
            prot_attr: String or array-like column indices or column names of
                protected attributes.
            estimator: An estimator implementing methods ``fit(X, y,
                sample_weight)`` and ``predict(X)``, where ``X`` is the matrix
                of features, ``y`` is the vector of labels, and
                ``sample_weight`` is a vector of weights; labels ``y`` and
                predictions returned by ``predict(X)`` are either 0 or 1 -- e.g.
                scikit-learn classifiers.
            constraints (str or fairlearn.reductions.Moment): If string, keyword
                denoting the :class:`fairlearn.reductions.Moment` object
                defining the disparity constraints -- e.g., "DemographicParity"
                or "EqualizedOdds". For a full list of possible options see
                `self.model.moments`. Otherwise, provide the desired
                :class:`~fairlearn.reductions.Moment` object defining the
                disparity constraints.
            eps: Allowed fairness constraint violation; the solution is
                guaranteed to have the error within ``2*best_gap`` of the best
                error under constraint eps; the constraint violation is at most
                ``2*(eps+best_gap)``.
            T: Maximum number of iterations.
            nu: Convergence threshold for the duality gap, corresponding to a
                conservative automatic setting based on the statistical
                uncertainty in measuring classification error.
            eta_mul: Initial setting of the learning rate.
            drop_prot_attr: Boolean flag indicating whether to drop protected
                attributes from training data.
        """
        self.prot_attr = prot_attr
        self.moments = {
                "DemographicParity": red.DemographicParity,
                "EqualizedOdds": red.EqualizedOdds,
                "TruePositiveRateDifference": red.TruePositiveRateDifference,
                "ErrorRateRatio": red.ErrorRateRatio
        }

        if isinstance(constraints, str):
            if constraints not in self.moments:
                raise ValueError(f"Constraint not recognized: {constraints}")

            self.moment = self.moments[constraints]()
        elif isinstance(constraints, red.Moment):
            self.moment = constraints
        else:
            raise ValueError("constraints must be a string or Moment object.")

        self.estimator = estimator
        self.eps = eps
        self.T = T
        self.nu = nu
        self.eta_mul = eta_mul
        self.drop_prot_attr = drop_prot_attr

        self.model = red.ExponentiatedGradient(self.estimator, self.moment,
            self.eps, self.T, self.nu, self.eta_mul)

    def fit(self, X, y):
        """Learns randomized model with less bias
        Args:
            X (pandas.DataFrame): Training samples.
            y (array-like): Training labels.
        Returns:
            self
        """
        A = X[self.prot_attr]

        if self.drop_prot_attr:
            X = X.drop(self.prot_attr, axis=1)

        le = LabelEncoder()
        y = le.fit_transform(y)
        self.classes_ = le.classes_

        self.model.fit(X, y, sensitive_features=A)

        return self


    def predict(self, X):
        """Predict class labels for the given samples.
        Args:
            X (pandas.DataFrame): Test samples.
        Returns:
            numpy.ndarray: Predicted class label per sample.
        """
        if self.drop_prot_attr:
            X = X.drop(self.prot_attr, axis=1)

        return self.classes_[self.model.predict(X)]


    def predict_proba(self, X):
        """Probability estimates.
        The returned estimates for all classes are ordered by the label of
        classes.
        Args:
            X (pandas.DataFrame): Test samples.
        Returns:
            numpy.ndarray: Returns the probability of the sample for each class
            in the model, where classes are ordered as they are in
            ``self.classes_``.
        """
        if self.drop_prot_attr:
            X = X.drop(self.prot_attr, axis=1)

        return self.model._pmf_predict(X)

class marginal_mixture_model():
    """ The MIM proposed in the paper.
        Provided implementation is for binary labels and protected attribute. 
    """
    def __init__(self, prot_att, estimator):
        """
        Args:
            prot_att: String or array-like column indices or column names of
                protected attributes.
            estimator: An estimator implementing methods ``fit(X, y,
                sample_weight)`` and ``predict(X)``, where ``X`` is the matrix
                of features, ``y`` is the vector of labels, and
                ``sample_weight`` is a vector of weights; labels ``y`` and
                predictions returned by ``predict(X)`` are either 0 or 1 -- e.g.
                scikit-learn classifiers.
        """
        self.prot_att = prot_att
        self.estimator = estimator
        self.weights_mm = None
        self.Zsup = None

    def fit(self, x, y):
        """Learns the marginal distribution and base estimator model
        Args:
            X (pandas.DataFrame): Training samples.
            y (array-like): Training labels.
        Returns:
            self
        """
        x=x.copy()
        y=y.copy()
        x = x[ [self.prot_att] + [ col for col in x.columns if col != self.prot_att ] ]

        Xt= x.values
        Y = y.values

        Z0 = Xt[:,0].reshape(-1, 1)
        Xt = Xt[:,1:]

        ztype = "flt"
        if len(set(Z0.flatten().tolist())) <= 2:
            ztype = "bin"
        Xpre = np.hstack([ Z0, Xt ])
        nparams = Xt.shape[1]

        self.estimator.fit(Xpre,Y)

        # config
        nzbins = 20
        nsamples = Xt.shape[0]

        # support of the protected attribute z
        if ztype == "flt":
            self.Zsup = np.linspace(-5, 5, nzbins)
        elif ztype == "bin":
            self.Zsup = np.array(set(Z0.flatten().tolist()))
        marg_dist = []
        for zval in self.Zsup.tolist():
            marg_dist.append((Z0.flatten() == zval).mean())
        self.weights_mm = np.array(marg_dist).reshape(-1, 1)

    def predict_proba(self, x):
        """Probability estimates.
        The returned estimates for all classes are ordered by the label of
        classes.
        Args:
            X (pandas.DataFrame): Test samples.
        Returns:
            numpy.ndarray: Returns the probability of the sample for both
            classes in the model.
        """
        x=x.copy()
        x = x[ [self.prot_att] + [ col for col in x.columns if col != self.prot_att ] ]
        Xt= x.values
        nsamples = Xt.shape[0]
        Z0 = Xt[:,0].reshape(-1, 1)
        Xt = Xt[:,1:]

        preds_z = []
        for zval in self.Zsup.tolist():
            preds_z.append(self.estimator.predict_proba(np.hstack([zval * np.ones((nsamples, 1)), Xt]))[:,1].reshape(-1, 1))

        Rho = np.hstack(preds_z)
        pred1 = np.dot(Rho, self.weights_mm).reshape(-1, 1)
        pred0 = np.ones((nsamples, 1))-pred1
        return np.hstack([pred0,pred1])

    def predict(self, x):
        """Predict class labels for the given samples.
        Args:
            X (pandas.DataFrame): Test samples.
        Returns:
            numpy.ndarray: Predicted class label per sample.
        """
        pred = self.predict_proba(x)[:,1].flatten()
        return np.where(pred > 0.5, 1, 0)
