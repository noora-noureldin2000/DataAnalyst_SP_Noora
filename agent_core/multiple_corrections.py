import pandas as pd
from statsmodels.stats.multitest import multipletests

class MultipleTestCorrection:
    def __init__(self, p_values, test_labels=None):
        self.p_values = p_values
        self.test_labels = test_labels if test_labels else [f"Test {i}" for i in range(len(p_values))]
        
    def correct(self, methods=None):
        if methods is None:
            methods = ['bonferroni', 'holm', 'fdr_bh']
        
        results = {"test": self.test_labels, "p_raw": self.p_values}
        for m in methods:
            reject, p_adj, _, _ = multipletests(self.p_values, method=m)
            results[f"p_{m}"] = p_adj
            results[f"reject_{m}"] = reject
            
        return pd.DataFrame(results)

    def recommend(self):
        return 'holm'
        
    def to_word_table(self, exporter):
        pass
        
    def narrative(self, method_used):
        return f"Multiple testing correction was applied using the {method_used} procedure."
