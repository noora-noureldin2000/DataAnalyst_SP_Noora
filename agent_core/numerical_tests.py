import numpy as np
import pandas as pd
from scipy import stats
from statsmodels.stats.multicomp import pairwise_tukeyhsd
import statsmodels.api as sm
from statsmodels.formula.api import ols

class PairedTests:
    @staticmethod
    def paired_t_test(x, y, alpha=0.05):
        x = np.asarray(x)
        y = np.asarray(y)
        diff = x - y
        n = len(diff)
        mean_diff = np.mean(diff)
        sd_diff = np.std(diff, ddof=1)
        
        stat, p_val = stats.ttest_rel(x, y)
        d_z = mean_diff / sd_diff if sd_diff != 0 else 0
        se = sd_diff / np.sqrt(n)
        ci_low, ci_high = stats.t.interval(1 - alpha, n - 1, loc=mean_diff, scale=se)
        
        narrative = f"A paired t-test showed a mean difference of {mean_diff:.2f} (95% CI [{ci_low:.2f}, {ci_high:.2f}]), t({n-1}) = {stat:.2f}, p = {p_val:.3f}, d_z = {d_z:.2f}."
        return {"t": stat, "p": p_val, "d_z": d_z, "ci_lower": ci_low, "ci_upper": ci_high, "narrative": narrative}

    @staticmethod
    def wilcoxon_signed_rank(x, y, alpha=0.05, zero_method='wilcox'):
        stat, p_val = stats.wilcoxon(x, y, zero_method=zero_method)
        
        diff = np.asarray(x) - np.asarray(y)
        n = len(diff[diff != 0])
        r = stat / (n * (n + 1) / 2) if n > 0 else 0
        
        narrative = f"A Wilcoxon signed-rank test was conducted, W = {stat:.2f}, p = {p_val:.3f}, rank-biserial r = {r:.2f}."
        return {"W": stat, "p": p_val, "r": r, "narrative": narrative}

    @staticmethod
    def sign_test(x, y):
        diff = np.asarray(x) - np.asarray(y)
        pos = np.sum(diff > 0)
        neg = np.sum(diff < 0)
        n = pos + neg
        p_val = stats.binomtest(min(pos, neg), n, 0.5).pvalue if n > 0 else 1.0
        return {"positives": pos, "negatives": neg, "p": p_val}

class MultiGroupTests:
    @staticmethod
    def one_way_anova(data, groups, alpha=0.05, posthoc='tukey'):
        unique_groups = np.unique(groups)
        group_data = [data[groups == g] for g in unique_groups]
        
        stat, p_val = stats.f_oneway(*group_data)
        
        total_mean = np.mean(data)
        ss_total = np.sum((data - total_mean)**2)
        ss_between = np.sum([len(g) * (np.mean(g) - total_mean)**2 for g in group_data])
        eta_squared = ss_between / ss_total if ss_total != 0 else 0
        
        return {"F": stat, "p": p_val, "eta_squared": eta_squared}
        
    @staticmethod
    def kruskal_wallis(data, groups, alpha=0.05, posthoc='dunn'):
        unique_groups = np.unique(groups)
        group_data = [data[groups == g] for g in unique_groups]
        stat, p_val = stats.kruskal(*group_data)
        return {"H": stat, "p": p_val}

    @staticmethod
    def welch_anova(data, groups, alpha=0.05):
        pass

class RepeatedMeasuresTests:
    @staticmethod
    def mauchly_sphericity(data):
        return {"W": 1.0, "p": 1.0, "gg_epsilon": 1.0, "hf_epsilon": 1.0}
        
    @staticmethod
    def repeated_measures_anova(data, subject_col, within_col, dv_col, alpha=0.05):
        pass

    @staticmethod
    def friedman_test(data, subject_col, within_col, dv_col, alpha=0.05):
        df = pd.DataFrame({subject_col: data[subject_col], within_col: data[within_col], dv_col: data[dv_col]})
        wide = df.pivot(index=subject_col, columns=within_col, values=dv_col)
        stat, p_val = stats.friedmanchisquare(*[wide[c] for c in wide.columns])
        return {"chi2_r": stat, "p": p_val}

class FactorialANOVA:
    @staticmethod
    def two_way_anova(data, dv, factor_a, factor_b, alpha=0.05, ss_type=3):
        formula = f"{dv} ~ C({factor_a}) + C({factor_b}) + C({factor_a}):C({factor_b})"
        model = ols(formula, data=data).fit()
        aov_table = sm.stats.anova_lm(model, typ=ss_type)
        return aov_table

class PostHocRouter:
    @staticmethod
    def route(data, groups, normality_result, homoscedasticity_result, alpha=0.05):
        return {}
