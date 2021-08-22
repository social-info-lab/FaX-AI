from aif360.sklearn.preprocessing import ReweighingMeta
from aif360.sklearn.inprocessing import AdversarialDebiasing, ExponentiatedGradientReduction, GridSearchReduction
from aif360.sklearn.postprocessing import CalibratedEqualizedOdds, PostProcessingMeta
from aif360.sklearn.datasets import fetch_adult, fetch_compas, fetch_german

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
import torch
import pandas as pd

from methods import Method
import helper
        

class AIF(Method):
    def __init__(self, XZ, Y):
        self.input = helper.dataframe(XZ)
        self.labels = s = pd.Series(Y, index=pd.Index(XZ[:,-1], name='Z'))
        self.prot_attr = 'Z'
        self.estimator = LogisticRegression(solver='lbfgs', max_iter=10000)

    def predict_proba(self, X):  # this is XZ
        input = helper.dataframe(X)
        out = self.solver.predict_proba(input)
        return out

    def predict(self, X):  # this is XZ
        input = helper.dataframe(X)
        out = self.solver.predict(input)
        return out

class AdvDeb(AIF):
    def __init__(self, XZ, Y):
        super().__init__(XZ, Y)
        self.solver = AdversarialDebiasing(prot_attr=self.prot_attr, random_state=1234567, num_epochs=10)
        self.solver.fit(self.input, self.labels)


class EOCal(AIF):
    def __init__(self, XZ, Y):
        super().__init__(XZ, Y)
        cal_eq_odds = CalibratedEqualizedOdds(prot_attr=self.prot_attr, cost_constraint='fnr', random_state=1234567)
        self.solver = PostProcessingMeta(estimator=self.estimator, postprocessor=cal_eq_odds, random_state=1234567)
        self.solver.fit(self.input, self.labels)

class ReWeigh(AIF):
    def __init__(self, XZ, Y):
        super().__init__(XZ, Y)
        rew = ReweighingMeta(estimator=self.estimator)
        params = {'estimator__C': [1, 10], 'reweigher__prot_attr': [self.prot_attr]}
        self.solver = GridSearchCV(rew, params, scoring='accuracy', cv=5)
        self.solver.fit(self.input, self.labels)


class DPExp(AIF):
    def __init__(self, XZ, Y):
        super().__init__(XZ, Y)
        self.solver =  ExponentiatedGradientReduction(prot_attr=self.prot_attr, estimator=self.estimator, constraints='DemographicParity')
        self.solver.fit(self.input, self.labels)


class EOExp(AIF):
    def __init__(self, XZ, Y):
        super().__init__(XZ, Y)
        self.solver =  ExponentiatedGradientReduction(prot_attr=self.prot_attr, estimator=self.estimator, constraints='EqualizedOdds')
        self.solver.fit(self.input, self.labels)




class TPRExp(AIF):
    def __init__(self, XZ, Y):
        super().__init__(XZ, Y)
        self.solver =  ExponentiatedGradientReduction(prot_attr=self.prot_attr, estimator=self.estimator, constraints='TruePositiveRateDifference')
        self.solver.fit(self.input, self.labels)


class ERExp(AIF):
    def __init__(self, XZ, Y):
        super().__init__(XZ, Y)
        self.solver =  ExponentiatedGradientReduction(prot_attr=self.prot_attr, estimator=self.estimator, constraints='ErrorRateRatio')
        self.solver.fit(self.input, self.labels)

        

        