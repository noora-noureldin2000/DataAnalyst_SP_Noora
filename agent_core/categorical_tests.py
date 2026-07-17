import numpy as np
import pandas as pd
from scipy import stats

class TwoByTwoTable:
    def __init__(self, a, b, c, d):
        self.a = a
        self.b = b
        self.c = c
        self.d = d
        
    @classmethod
    def from_dataframe(cls, df, exposure, outcome):
        tab = pd.crosstab(df[exposure], df[outcome])
        if tab.shape == (2, 2):
            return cls(tab.iloc[1, 1], tab.iloc[1, 0], tab.iloc[0, 1], tab.iloc[0, 0])
        return cls(0, 0, 0, 0)
        
    def compute(self):
        a, b, c, d = self.a, self.b, self.c, self.d
        n1 = a + b
        n0 = c + d
        
        rr = (a / n1) / (c / n0) if c != 0 and n0 != 0 and n1 != 0 else np.nan
        or_val = (a * d) / (b * c) if b != 0 and c != 0 else np.nan
        ar = (a / n1) - (c / n0) if n1 != 0 and n0 != 0 else np.nan
        nnt = 1 / ar if ar != 0 and not np.isnan(ar) else np.nan
        
        return {"RR": rr, "OR": or_val, "AR": ar, "NNT": nnt}

class McNemarTest:
    def __init__(self, b, c):
        self.b = b
        self.c = c
        
    def compute(self, continuity_correction=True, exact=False):
        b, c = self.b, self.c
        if exact or (b + c) < 25:
            p = stats.binomtest(min(b, c), b + c, 0.5).pvalue if (b+c)>0 else 1.0
            stat = np.nan
        else:
            if continuity_correction:
                stat = (abs(b - c) - 1)**2 / (b + c) if (b+c)>0 else np.nan
            else:
                stat = (b - c)**2 / (b + c) if (b+c)>0 else np.nan
            p = stats.chi2.sf(stat, 1) if not np.isnan(stat) else 1.0
            
        or_matched = c / b if b != 0 else np.nan
        return {"chi2": stat, "p": p, "OR_matched": or_matched}

class CochranQTest:
    def __init__(self, data_matrix):
        pass
        
    def compute(self):
        pass

class MantelHaenszelTest:
    def __init__(self, tables):
        pass
        
    def compute(self, measure='OR'):
        pass

class ChiSquareExtended:
    @staticmethod
    def test(observed_table, correction=True, expected_warning_threshold=5):
        stat, p, dof, expected = stats.chi2_contingency(observed_table, correction=correction)
        return {"chi2": stat, "p": p, "expected": expected}
