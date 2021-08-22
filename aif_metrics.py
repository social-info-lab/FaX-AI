from abc import ABC, abstractmethod
from aif_methods import AIF
from aif360.sklearn.metrics import disparate_impact_ratio, average_odds_error, generalized_fpr
from aif360.sklearn.metrics import generalized_fnr, difference, statistical_parity_difference,equal_opportunity_difference
import pandas as pd


class Metric(ABC):

    @abstractmethod
    def metric(self, method, X, Z, Y):
        pass


class DisparateImpactRatio(Metric):

    def metric(self, method, X, Z, Y):
        pred = method.predict(X)
        s = pd.Series(Y, index=pd.Index(Z[:, 0], name='Z'))
        prot_attr = 'Z'
        di = disparate_impact_ratio(s, pred, prot_attr=prot_attr)
        return di

class StatisticalParityDifference(Metric):

    def metric(self, method, X, Z, Y):
        pred = method.predict(X)
        s = pd.Series(Y, index=pd.Index(Z[:, 0], name='Z'))
        prot_attr = 'Z'
        dd = statistical_parity_difference(s, pred, prot_attr=prot_attr)
        return dd


class EqualOpportunityDifference(Metric):

    def metric(self, method, X, Z, Y):
        pred = method.predict(X)
        s = pd.Series(Y, index=pd.Index(Z[:, 0], name='Z'))
        prot_attr = 'Z'
        eqopp = equal_opportunity_difference(s, pred, prot_attr=prot_attr)
        return eqopp


class AverageOddsError(Metric):

    def metric(self, method, X, Z, Y):
        pred = method.predict(X)
        s = pd.Series(Y, index=pd.Index(Z[:, 0], name='Z'))
        prot_attr = 'Z'
        eqodd = average_odds_error(s, pred, prot_attr=prot_attr)
        return eqodd
