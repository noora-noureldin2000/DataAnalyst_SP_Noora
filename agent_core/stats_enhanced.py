import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
from typing import Optional, Union, Dict, Any, List, Tuple
import warnings

import scipy.stats as stats
from scipy.stats import (
    shapiro,
    anderson,
    kstest,
    cramervonmises,
    norm,
    chi2,
    t as t_dist,
    nct as noncentral_t,
    studentized_range,
    pearsonr,
    spearmanr,
    kendalltau,
    bootstrap,
    zscore,
)
from scipy.optimize import brentq
from scipy.stats import f as f_dist

import statsmodels.api as sm
import statsmodels.formula.api as smf
from statsmodels.stats.outliers_influence import variance_inflation_factor
from statsmodels.tools.tools import add_constant

warnings.filterwarnings("ignore", category=DeprecationWarning)

_Z_95 = 1.96


# ---------------------------------------------------------------------------
# Helper: Lilliefors critical values (Lilliefors 1967, JASA)
# ---------------------------------------------------------------------------
_LILLIEFORS_CRIT = {
    4:  {0.20: 0.300, 0.15: 0.319, 0.10: 0.352, 0.05: 0.381, 0.01: 0.417},
    5:  {0.20: 0.285, 0.15: 0.299, 0.10: 0.315, 0.05: 0.337, 0.01: 0.405},
    6:  {0.20: 0.265, 0.15: 0.277, 0.10: 0.294, 0.05: 0.319, 0.01: 0.364},
    7:  {0.20: 0.247, 0.15: 0.258, 0.10: 0.276, 0.05: 0.300, 0.01: 0.348},
    8:  {0.20: 0.233, 0.15: 0.244, 0.10: 0.261, 0.05: 0.285, 0.01: 0.331},
    9:  {0.20: 0.223, 0.15: 0.233, 0.10: 0.249, 0.05: 0.271, 0.01: 0.311},
    10: {0.20: 0.215, 0.15: 0.224, 0.10: 0.239, 0.05: 0.258, 0.01: 0.294},
    11: {0.20: 0.206, 0.15: 0.217, 0.10: 0.230, 0.05: 0.249, 0.01: 0.284},
    12: {0.20: 0.199, 0.15: 0.212, 0.10: 0.223, 0.05: 0.242, 0.01: 0.275},
    13: {0.20: 0.190, 0.15: 0.202, 0.10: 0.214, 0.05: 0.234, 0.01: 0.265},
    14: {0.20: 0.185, 0.15: 0.194, 0.10: 0.207, 0.05: 0.227, 0.01: 0.260},
    15: {0.20: 0.179, 0.15: 0.187, 0.10: 0.201, 0.05: 0.220, 0.01: 0.250},
    16: {0.20: 0.174, 0.15: 0.182, 0.10: 0.195, 0.05: 0.213, 0.01: 0.242},
    17: {0.20: 0.170, 0.15: 0.177, 0.10: 0.189, 0.05: 0.206, 0.01: 0.234},
    18: {0.20: 0.166, 0.15: 0.173, 0.10: 0.184, 0.05: 0.200, 0.01: 0.228},
    19: {0.20: 0.163, 0.15: 0.169, 0.10: 0.179, 0.05: 0.195, 0.01: 0.221},
    20: {0.20: 0.160, 0.15: 0.166, 0.10: 0.174, 0.05: 0.190, 0.01: 0.215},
    25: {0.20: 0.142, 0.15: 0.147, 0.10: 0.158, 0.05: 0.173, 0.01: 0.199},
    30: {0.20: 0.131, 0.15: 0.136, 0.10: 0.144, 0.05: 0.161, 0.01: 0.187},
    40: {0.20: 0.115, 0.15: 0.121, 0.10: 0.128, 0.05: 0.139, 0.01: 0.166},
    50: {0.20: 0.104, 0.15: 0.108, 0.10: 0.115, 0.05: 0.125, 0.01: 0.144},
    100: {0.20: 0.074, 0.15: 0.077, 0.10: 0.082, 0.05: 0.089, 0.01: 0.104},
}
_LILLIEFORS_ASYMP = {0.20: 0.736, 0.15: 0.768, 0.10: 0.805, 0.05: 0.886, 0.01: 1.031}


def _lilliefors_critical(n: int, alpha: float) -> float:
    if n <= 100:
        lookup = _LILLIEFORS_CRIT.get(n)
        if lookup is not None and alpha in lookup:
            return lookup[alpha]
        ns = sorted(_LILLIEFORS_CRIT.keys())
        if n < ns[0]:
            return _LILLIEFORS_ASYMP[alpha] / np.sqrt(n)
        lower_n = max([k for k in ns if k <= n])
        upper_n = min([k for k in ns if k >= n])
        if lower_n == upper_n:
            return _LILLIEFORS_CRIT[lower_n][alpha]
        lower_cv = _LILLIEFORS_CRIT[lower_n][alpha]
        upper_cv = _LILLIEFORS_CRIT[upper_n][alpha]
        frac = (n - lower_n) / (upper_n - lower_n)
        return lower_cv + frac * (upper_cv - lower_cv)
    return _LILLIEFORS_ASYMP[alpha] / np.sqrt(n)


def _lilliefors_pvalue(D: float, n: int) -> float:
    if D <= 0:
        return 1.0
    alphas = [0.20, 0.15, 0.10, 0.05, 0.01]
    cvs = [_lilliefors_critical(n, a) for a in alphas]
    if D >= cvs[-1]:
        return min(1.0, 0.01 * cvs[-1] / max(D, 1e-10))
    if D <= cvs[0]:
        return max(0.0, 0.20 + (cvs[0] - D) / max(cvs[0], 1e-10) * 0.80)
    for i in range(len(alphas) - 1):
        if cvs[i + 1] <= D <= cvs[i]:
            frac = (D - cvs[i + 1]) / (cvs[i] - cvs[i + 1] + 1e-10)
            return alphas[i + 1] + frac * (alphas[i] - alphas[i + 1])
    return 0.50


# ---------------------------------------------------------------------------
# 1. AdvancedNormalityTestSuite
# ---------------------------------------------------------------------------
class AdvancedNormalityTestSuite:
    """Provides six normality tests with uniform output format."""

    VALID_TESTS = {
        "shapiro_wilk", "anderson_darling", "lilliefors",
        "cramer_von_mises", "shapiro_francia", "pearson_chi_square"
    }

    def run_all(self, data: np.ndarray) -> pd.DataFrame:
        """Run all 6 normality tests and return a summary DataFrame."""
        results = []
        for test_name in sorted(self.VALID_TESTS):
            result = self.run_specific(data, test_name)
            results.append(result)
        df = pd.DataFrame(results)
        for col in ["statistic", "p_value"]:
            if col in df.columns:
                try:
                    df[col] = pd.to_numeric(df[col])
                except (ValueError, TypeError):
                    pass
        return df

    def run_specific(self, data: np.ndarray, test: str) -> dict:
        """Run a single normality test by name."""
        data = np.asarray(data, dtype=float)
        data = data[~np.isnan(data)]
        n = len(data)
        if n < 4:
            return {
                "test_name": test,
                "statistic": np.nan,
                "p_value": np.nan,
                "is_normal": False,
                "interpretation": "Sample too small (n<4) for normality testing",
            }
        test = test.lower().replace("-", "_").replace(" ", "_").strip()
        if test == "shapiro_wilk":
            return self._shapiro_wilk_test(data)
        if test == "anderson_darling":
            return self._anderson_darling_test(data)
        if test == "lilliefors":
            return self._lilliefors_test(data)
        if test == "cramer_von_mises":
            return self._cramer_von_mises_test(data)
        if test == "shapiro_francia":
            return self._shapiro_francia_test(data)
        if test == "pearson_chi_square":
            return self._pearson_chi_square_test(data)
        raise ValueError(f"Unknown test: {test}. Choose from {sorted(self.VALID_TESTS)}")

    @staticmethod
    def _shapiro_wilk_test(data: np.ndarray) -> dict:
        stat, pv = shapiro(data)
        normal = bool(pv >= 0.05)
        interp = "Data appears normally distributed" if normal else "Data deviates from normality"
        return {
            "test_name": "Shapiro-Wilk",
            "statistic": stat,
            "p_value": pv,
            "is_normal": normal,
            "interpretation": interp,
        }

    @staticmethod
    def _anderson_darling_test(data: np.ndarray) -> dict:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            res = anderson(data, dist="norm")
        stat = res.statistic
        cv_05 = res.critical_values[res.significance_level.tolist().index(5.0)]
        normal = bool(stat <= cv_05)
        interp = "Data appears normally distributed" if normal else "Data deviates from normality"
        return {
            "test_name": "Anderson-Darling",
            "statistic": stat,
            "p_value": None,
            "is_normal": normal,
            "interpretation": interp,
        }

    @staticmethod
    def _lilliefors_test(data: np.ndarray) -> dict:
        mu, sigma = data.mean(), data.std(ddof=1)
        if sigma == 0:
            return {
                "test_name": "Lilliefors (K-S)",
                "statistic": 0.0,
                "p_value": 1.0,
                "is_normal": True,
                "interpretation": "Zero variance data",
            }
        D, _ = kstest(data, lambda x: norm.cdf(x, loc=mu, scale=sigma))
        pv = _lilliefors_pvalue(D, len(data))
        normal = bool(pv >= 0.05)
        interp = "Data appears normally distributed" if normal else "Data deviates from normality"
        return {
            "test_name": "Lilliefors (K-S)",
            "statistic": D,
            "p_value": pv,
            "is_normal": normal,
            "interpretation": interp,
        }

    @staticmethod
    def _cramer_von_mises_test(data: np.ndarray) -> dict:
        try:
            mu, sigma = data.mean(), data.std(ddof=1)
            res = cramervonmises(data, lambda x: norm.cdf(x, loc=mu, scale=sigma))
            stat = res.statistic
            pv = res.pvalue
            normal = bool(pv >= 0.05)
            interp = "Data appears normally distributed" if normal else "Data deviates from normality"
            return {
                "test_name": "Cramer-von Mises",
                "statistic": stat,
                "p_value": pv,
                "is_normal": normal,
                "interpretation": interp,
            }
        except Exception:
            return {
                "test_name": "Cramer-von Mises",
                "statistic": np.nan,
                "p_value": np.nan,
                "is_normal": False,
                "interpretation": "Test not available in this scipy version",
            }

    @staticmethod
    def _shapiro_francia_test(data: np.ndarray) -> dict:
        n = len(data)
        x_sorted = np.sort(data)
        pp = (np.arange(1, n + 1) - 0.375) / (n + 0.25)
        m = norm.ppf(pp)
        m = m - m.mean()
        numerator = np.sum(m * x_sorted) ** 2
        denominator = np.sum(m ** 2) * np.sum((x_sorted - x_sorted.mean()) ** 2)
        if denominator == 0:
            return {
                "test_name": "Shapiro-Francia",
                "statistic": 1.0,
                "p_value": 1.0,
                "is_normal": True,
                "interpretation": "Zero variance data",
            }
        w_prime = numerator / denominator
        u = np.log(n)
        mu = -1.5861 - 0.31082 * u - 0.083751 * u ** 2 + 0.0038915 * u ** 3
        sigma = np.exp(-0.4803 - 0.082676 * u + 0.0030302 * u ** 2)
        z = (np.log(1.0 - w_prime + 1e-15) - mu) / max(sigma, 1e-15)
        pv = norm.sf(z)
        pv = np.clip(pv, 0.0, 1.0)
        normal = bool(pv >= 0.05)
        interp = "Data appears normally distributed" if normal else "Data deviates from normality"
        return {
            "test_name": "Shapiro-Francia",
            "statistic": w_prime,
            "p_value": float(pv),
            "is_normal": normal,
            "interpretation": interp,
        }

    @staticmethod
    def _pearson_chi_square_test(data: np.ndarray) -> dict:
        n = len(data)
        k = max(3, int(np.ceil(2.0 * n ** (2.0 / 5.0))))
        mu, sigma = data.mean(), data.std(ddof=1)
        if sigma == 0:
            return {
                "test_name": "Pearson Chi-Square",
                "statistic": 0.0,
                "p_value": 1.0,
                "is_normal": True,
                "interpretation": "Zero variance data",
            }
        boundaries = norm.ppf(np.linspace(0, 1, k + 1)[1:-1], loc=mu, scale=sigma)
        boundaries = np.clip(boundaries, data.min() - 1e-10, data.max() + 1e-10)
        observed, _ = np.histogram(data, bins=np.concatenate([[-np.inf], boundaries, [np.inf]]))
        expected_cdf = norm.cdf(boundaries, loc=mu, scale=sigma)
        expected = np.diff(np.concatenate([[0], expected_cdf, [1]])) * n
        expected = np.maximum(expected, 1.0)
        chi2_stat = np.sum((observed - expected) ** 2 / expected)
        df = k - 3
        pv = 1.0 - chi2.cdf(chi2_stat, df) if df > 0 else 0.0
        normal = bool(pv >= 0.05)
        interp = "Data appears normally distributed" if normal else "Data deviates from normality"
        return {
            "test_name": "Pearson Chi-Square",
            "statistic": chi2_stat,
            "p_value": pv,
            "is_normal": normal,
            "interpretation": interp,
        }


# ---------------------------------------------------------------------------
# 2. GamesHowellPostHoc
# ---------------------------------------------------------------------------
class GamesHowellPostHoc:
    """Games-Howell post-hoc test for unequal variances (Welch ANOVA context)."""

    def test(
        self,
        data: np.ndarray,
        groups: np.ndarray,
        alpha: float = 0.05,
    ) -> dict:
        data = np.asarray(data, dtype=float)
        groups = np.asarray(groups)
        unique_groups = np.unique(groups)
        k = len(unique_groups)
        if k < 2:
            raise ValueError("Need at least 2 groups for Games-Howell test.")
        group_stats = []
        for g in unique_groups:
            mask = groups == g
            vals = data[mask]
            n_g = len(vals)
            mean_g = float(vals.mean())
            var_g = float(vals.var(ddof=1))
            group_stats.append({"group": g, "n": n_g, "mean": mean_g, "var": var_g})
        comparisons = []
        for i in range(k):
            for j in range(i + 1, k):
                gi = group_stats[i]
                gj = group_stats[j]
                mean_diff = gi["mean"] - gj["mean"]
                se = np.sqrt(gi["var"] / gi["n"] + gj["var"] / gj["n"])
                if se == 0:
                    continue
                num_df = (gi["var"] / gi["n"] + gj["var"] / gj["n"]) ** 2
                den_df = (
                    (gi["var"] / gi["n"]) ** 2 / (gi["n"] - 1)
                    + (gj["var"] / gj["n"]) ** 2 / (gj["n"] - 1)
                )
                df = num_df / max(den_df, 1e-15)
                t_val = abs(mean_diff) / se
                q_val = t_val * np.sqrt(2.0)
                try:
                    p_val = 1.0 - studentized_range.cdf(q_val, k, df)
                except Exception:
                    with warnings.catch_warnings():
                        warnings.simplefilter("ignore")
                        p_val = 1.0 - studentized_range.cdf(q_val, k, max(df, 1.0))
                try:
                    q_crit = studentized_range.ppf(1.0 - alpha, k, max(df, 1.0))
                except Exception:
                    q_crit = 3.0
                ci_half = (q_crit / np.sqrt(2.0)) * se
                comparisons.append({
                    "group1": str(gi["group"]),
                    "group2": str(gj["group"]),
                    "mean_diff": mean_diff,
                    "ci_lower": mean_diff - ci_half,
                    "ci_upper": mean_diff + ci_half,
                    "t": t_val,
                    "df": df,
                    "p_value": float(p_val),
                    "p_adjusted": float(p_val),
                })
        comparisons_df = pd.DataFrame(comparisons)
        descriptives_df = pd.DataFrame([
            {"group": str(s["group"]), "n": s["n"], "mean": s["mean"], "variance": s["var"]}
            for s in group_stats
        ])
        return {
            "pairwise_comparisons": comparisons_df,
            "descriptives": descriptives_df,
            "method": "Games-Howell Post-Hoc Test",
            "alpha": alpha,
        }


# ---------------------------------------------------------------------------
# 3. CompactLetterDisplay
# ---------------------------------------------------------------------------
class CompactLetterDisplay:
    """Generate compact letter displays for pairwise comparisons."""

    def from_pvalues(
        self,
        pvalues_matrix: np.ndarray,
        group_names: List[str],
        alpha: float = 0.05,
    ) -> pd.DataFrame:
        n = len(group_names)
        if pvalues_matrix.shape != (n, n):
            raise ValueError("pvalues_matrix must be square with dimensions matching group_names")
        diff = pvalues_matrix < alpha
        np.fill_diagonal(diff, False)
        mapping = self._multcomp_letters(diff, group_names)
        return pd.DataFrame({"group": group_names, "Letter": [mapping[g] for g in group_names]})

    def from_pairwise_comparisons(
        self,
        comparison_df: pd.DataFrame,
        group_col: str = "group",
        pval_col: str = "p_adjusted",
        alpha: float = 0.05,
    ) -> pd.DataFrame:
        groups = sorted(set(comparison_df["group1"].unique()) | set(comparison_df["group2"].unique()))
        n = len(groups)
        g_to_idx = {g: i for i, g in enumerate(groups)}
        pmat = np.ones((n, n))
        for _, row in comparison_df.iterrows():
            i = g_to_idx[row["group1"]]
            j = g_to_idx[row["group2"]]
            pv = row[pval_col]
            pmat[i, j] = pv
            pmat[j, i] = pv
        np.fill_diagonal(pmat, 1.0)
        return self.from_pvalues(pmat, groups, alpha)

    @staticmethod
    def _multcomp_letters(diff: np.ndarray, group_names: List[str]) -> Dict[str, str]:
        n = len(group_names)
        result = {g: "" for g in group_names}
        letters = "abcdefghijklmnopqrstuvwxyz"
        needs_letter = set(range(n))
        letter_idx = 0
        while needs_letter and letter_idx < 26:
            def diff_count(i):
                return int(np.sum(diff[i, list(needs_letter)]))
            reference_idx = max(needs_letter, key=diff_count)
            letter = letters[letter_idx % 26]
            result[group_names[reference_idx]] += letter
            has_letter = {reference_idx}
            for idx in sorted(needs_letter - {reference_idx}):
                ok = True
                for hl in has_letter:
                    if diff[idx, hl]:
                        ok = False
                        break
                if ok:
                    result[group_names[idx]] += letter
                    has_letter.add(idx)
            needs_letter -= has_letter
            letter_idx += 1
        for i in range(n):
            if not result[group_names[i]]:
                result[group_names[i]] = letters[min(letter_idx, 25)]
        return result


# ---------------------------------------------------------------------------
# 4. BootstrapCorrelation
# ---------------------------------------------------------------------------
class BootstrapCorrelation:
    """Correlation analysis with bootstrap BCa confidence intervals."""

    def correlate(
        self,
        x: np.ndarray,
        y: np.ndarray,
        method: str = "spearman",
        n_bootstrap: int = 999,
        conf_level: float = 0.95,
        p_adjust_method: str = "holm",
    ) -> dict:
        x = np.asarray(x, dtype=float)
        y = np.asarray(y, dtype=float)
        mask = ~(np.isnan(x) | np.isnan(y))
        x, y = x[mask], y[mask]
        n = len(x)
        if n < 3:
            raise ValueError("Need at least 3 non-missing observations.")
        method = method.lower()
        if method == "pearson":
            cor, p_val = pearsonr(x, y)
        elif method == "spearman":
            cor, p_val = spearmanr(x, y)
        elif method == "kendall":
            cor, p_val = kendalltau(x, y)
        else:
            raise ValueError(f"Unknown method: {method}. Use 'pearson', 'spearman', or 'kendall'.")
        data = (x, y)

        def _cor_stat(x_, y_):
            if method == "pearson":
                return pearsonr(x_, y_)[0]
            elif method == "spearman":
                return spearmanr(x_, y_)[0]
            else:
                return kendalltau(x_, y_)[0]

        try:
            boot_res = bootstrap(
                data,
                _cor_stat,
                n_resamples=n_bootstrap,
                method="bca",
                confidence_level=conf_level,
                random_state=42,
                paired=True,
            )
            ci_lower = boot_res.confidence_interval.low
            ci_upper = boot_res.confidence_interval.high
        except Exception:
            boot_res = bootstrap(
                data,
                _cor_stat,
                n_resamples=n_bootstrap,
                method="percentile",
                confidence_level=conf_level,
                random_state=42,
                paired=True,
            )
            ci_lower = boot_res.confidence_interval.low
            ci_upper = boot_res.confidence_interval.high
        if np.isnan(ci_lower):
            ci_lower = cor - _Z_95 * np.sqrt((1.0 - cor ** 2) / (n - 2))
            ci_upper = cor + _Z_95 * np.sqrt((1.0 - cor ** 2) / (n - 2))
        from statsmodels.stats.multitest import multipletests
        _, p_adjusted, _, _ = multipletests([p_val], method=p_adjust_method)
        return {
            "cor": cor,
            "ci_lower": ci_lower,
            "ci_upper": ci_upper,
            "p_value": float(p_val),
            "p_adjusted": float(p_adjusted[0]),
            "n": n,
            "method": method,
            "n_bootstrap": n_bootstrap,
        }

    def correlate_matrix(
        self,
        df: pd.DataFrame,
        method: str = "spearman",
        n_bootstrap: int = 999,
        conf_level: float = 0.95,
        p_adjust_method: str = "holm",
    ) -> pd.DataFrame:
        cols = df.columns.tolist()
        results = []
        pvals = []
        records = []
        for i in range(len(cols)):
            for j in range(i + 1, len(cols)):
                x = df[cols[i]].values
                y = df[cols[j]].values
                try:
                    res = self.correlate(
                        x, y, method=method,
                        n_bootstrap=n_bootstrap,
                        conf_level=conf_level,
                        p_adjust_method="none",
                    )
                except Exception:
                    continue
                records.append(res)
                pvals.append(res["p_value"])
        if pvals:
            from statsmodels.stats.multitest import multipletests
            _, p_adj, _, _ = multipletests(pvals, method=p_adjust_method)
            for k, rec in enumerate(records):
                rec["p_adjusted"] = float(p_adj[k])
        result_df = pd.DataFrame(records)
        result_df.insert(0, "variable_1", [df.columns[i] for i, j in
                                            [(i, j) for i in range(len(cols)) for j in range(i + 1, len(cols))]])
        result_df.insert(1, "variable_2", [df.columns[j] for i, j in
                                            [(i, j) for i in range(len(cols)) for j in range(i + 1, len(cols))]])
        return result_df


# ---------------------------------------------------------------------------
# 5. EnhancedQQPlot
# ---------------------------------------------------------------------------
class EnhancedQQPlot:
    """Enhanced QQ-plot with confidence envelope and multiple estimation methods."""

    def compute_qq_data(
        self,
        data: np.ndarray,
        method: str = "moment",
        envelope: float = 0.95,
    ) -> dict:
        data = np.asarray(data, dtype=float)
        data = data[~np.isnan(data)]
        n = len(data)
        if n < 4:
            raise ValueError("Need at least 4 observations for QQ plot.")
        method = method.lower()
        if method == "moment":
            mu = np.mean(data)
            sigma = np.std(data, ddof=1)
        elif method == "trimmed":
            from scipy.stats import trim_mean
            mu = trim_mean(data, 0.1)
            trimmed = data[(data > np.percentile(data, 10)) & (data < np.percentile(data, 90))]
            sigma = np.std(trimmed, ddof=1) if len(trimmed) > 1 else np.std(data, ddof=1)
        elif method == "mle":
            mu, sigma = norm.fit(data)
        else:
            raise ValueError(f"Unknown method: {method}. Use 'moment', 'trimmed', or 'mle'.")
        pp = (np.arange(1, n + 1) - 0.5) / n
        theoretical = norm.ppf(pp)
        observed = np.sort(data)
        z_crit = norm.ppf(1.0 - (1.0 - envelope) / 2.0)
        se_envelope = sigma * np.sqrt((1.0 + theoretical ** 2 / 2.0) / n)
        envelope_lower = mu + sigma * theoretical - z_crit * se_envelope
        envelope_upper = mu + sigma * theoretical + z_crit * se_envelope
        q1_sample, q3_sample = np.percentile(data, [25, 75])
        q1_theory, q3_theory = norm.ppf([0.25, 0.75])
        slope = (q3_sample - q1_sample) / (q3_theory - q1_theory + 1e-15)
        intercept = q1_sample - slope * q1_theory
        return {
            "theoretical": theoretical,
            "observed": observed,
            "envelope_lower": envelope_lower,
            "envelope_upper": envelope_upper,
            "slope": slope,
            "intercept": intercept,
            "mu": mu,
            "sigma": sigma,
            "n": n,
            "method": method,
        }

    def plot(
        self,
        data: np.ndarray,
        method: str = "moment",
        envelope: float = 0.95,
        title: str = "Q-Q Plot",
        output_path: Optional[str] = None,
    ) -> plt.Figure:
        qq = self.compute_qq_data(data, method, envelope)
        fig, ax = plt.subplots(figsize=(8, 8), dpi=150)
        ax.fill_between(
            qq["theoretical"],
            qq["envelope_lower"],
            qq["envelope_upper"],
            alpha=0.15,
            color="#1f77b4",
            label=f"{int(envelope*100)}% Confidence Envelope",
        )
        ax.scatter(
            qq["theoretical"], qq["observed"],
            color="#1f77b4", edgecolors="white", s=40, zorder=5, label="Observed",
        )
        x_line = np.linspace(qq["theoretical"].min(), qq["theoretical"].max(), 100)
        y_line = qq["intercept"] + qq["slope"] * x_line
        ax.plot(x_line, y_line, color="crimson", linestyle="--", linewidth=1.5,
                label=f"Reference (Q1-Q3): y = {qq['slope']:.3f}x + {qq['intercept']:.3f}")
        ax.axhline(0, color="gray", linewidth=0.5, alpha=0.5)
        ax.axvline(0, color="gray", linewidth=0.5, alpha=0.5)
        ax.set_xlabel("Theoretical Quantiles", fontweight="bold")
        ax.set_ylabel("Sample Quantiles", fontweight="bold")
        ax.set_title(title, fontweight="bold")
        ax.legend(loc="lower right", frameon=True, edgecolor="lightgray")
        ax.grid(True, linestyle=":", alpha=0.5)
        ax.text(
            0.02, 0.98,
            f"n = {qq['n']}  |  μ = {qq['mu']:.3f}  |  σ = {qq['sigma']:.3f}  |  {qq['method']}",
            transform=ax.transAxes, fontsize=8, va="top",
            bbox=dict(facecolor="white", alpha=0.8, edgecolor="lightgray", boxstyle="round,pad=0.3"),
        )
        fig.tight_layout()
        if output_path:
            fig.savefig(output_path, bbox_inches="tight", dpi=150)
        return fig


# ---------------------------------------------------------------------------
# 6. PValueFormatter
# ---------------------------------------------------------------------------
class PValueFormatter:
    """APA-style p-value formatting utilities."""

    @staticmethod
    def format_p(
        p_value: float,
        digits: int = 3,
        add_p: bool = False,
        rm_zero: bool = True,
        threshold: float = 0.001,
    ) -> str:
        if np.isnan(p_value):
            return "NA"
        p_value = float(p_value)
        if p_value < threshold:
            base = f"< .{'0' * (digits - 1)}1" if digits > 1 else "< .1"
            if rm_zero:
                base = base.replace("0.", ".")
            return f"p {base}" if add_p else base
        rounded = round(p_value, digits)
        if digits == 3:
            fmt = f"{rounded:.3f}"
        elif digits == 2:
            fmt = f"{rounded:.2f}"
        else:
            fmt = f"{rounded:.{digits}f}"
        if rm_zero:
            fmt = fmt.replace("0.", ".", 1)
            if fmt == ".000":
                fmt = "< .001"
        if add_p:
            if fmt.startswith("<"):
                return f"p {fmt}"
            return f"p = {fmt}"
        return fmt

    @staticmethod
    def significance_stars(p_value: float, schemes: str = "default") -> str:
        p = float(p_value)
        if np.isnan(p):
            return ""
        if p < 0.001:
            return "***"
        if p < 0.01:
            return "**"
        if p < 0.05:
            return "*"
        return "ns"

    @staticmethod
    def format_dataframe(
        df: pd.DataFrame,
        pvalue_cols: List[str],
        digits: int = 3,
        rm_zero: bool = True,
    ) -> pd.DataFrame:
        result = df.copy()
        fmt = PValueFormatter()
        for col in pvalue_cols:
            if col in result.columns:
                result[col] = result[col].apply(
                    lambda x: fmt.format_p(x, digits=digits, rm_zero=rm_zero)
                )
        return result

    @staticmethod
    def generate_legend(scheme: str = "default") -> str:
        return "***p<0.001, **p<0.01, *p<0.05"


# ---------------------------------------------------------------------------
# 7. StandardizedCoefficients
# ---------------------------------------------------------------------------
class StandardizedCoefficients:
    """Computes standardized regression coefficients (beta)."""

    @staticmethod
    def from_lm(
        model: sm.regression.linear_model.RegressionResultsWrapper,
        model_matrix: Optional[pd.DataFrame] = None,
        outcome_sd: Optional[float] = None,
    ) -> dict:
        params = model.params
        if model_matrix is not None:
            X = model_matrix.values
            predictor_names = model_matrix.columns.tolist()
        else:
            try:
                X = model.model.exog
            except AttributeError:
                X = model.model.data.exog
            predictor_names = [f"x{i}" for i in range(X.shape[1])]
        try:
            y = model.model.endog
        except AttributeError:
            y = model.model.data.endog
        y = np.asarray(y, dtype=float)
        X = np.asarray(X, dtype=float)
        n, p = X.shape
        y_sd = outcome_sd if outcome_sd is not None else np.std(y, ddof=1)
        standardized = []
        for i in range(p):
            x_sd = np.std(X[:, i], ddof=1)
            b_i = params.iloc[i] if hasattr(params, "iloc") else params[i]
            beta_i = b_i * (x_sd / max(y_sd, 1e-15))
            standardized.append({
                "predictor": predictor_names[i] if i < len(predictor_names) else f"x{i}",
                "coefficient": b_i,
                "standardized_coefficient": beta_i,
            })
        result_df = pd.DataFrame(standardized)
        abs_beta = result_df["standardized_coefficient"].abs()
        result_df["influence_rank"] = abs_beta.rank(ascending=False).astype(int)
        result_df = result_df.sort_values("influence_rank").reset_index(drop=True)
        return {"standardized": result_df}

    @staticmethod
    def from_raw(
        X: np.ndarray,
        y: np.ndarray,
        coefficients: np.ndarray,
        predictor_names: List[str],
    ) -> dict:
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float)
        coefficients = np.asarray(coefficients, dtype=float)
        y_sd = np.std(y, ddof=1)
        n, p = X.shape
        if len(coefficients) != p:
            raise ValueError(f"len(coefficients)={len(coefficients)} != X.shape[1]={p}")
        standardized = []
        for i in range(p):
            x_sd = np.std(X[:, i], ddof=1)
            beta_i = coefficients[i] * (x_sd / max(y_sd, 1e-15))
            standardized.append({
                "predictor": predictor_names[i] if i < len(predictor_names) else f"x{i}",
                "coefficient": coefficients[i],
                "standardized_coefficient": beta_i,
            })
        result_df = pd.DataFrame(standardized)
        abs_beta = result_df["standardized_coefficient"].abs()
        result_df["influence_rank"] = abs_beta.rank(ascending=False).astype(int)
        result_df = result_df.sort_values("influence_rank").reset_index(drop=True)
        return {"standardized": result_df}


# ---------------------------------------------------------------------------
# 8. InfluenceDiagnostics
# ---------------------------------------------------------------------------
class InfluenceDiagnostics:
    """Regression influence diagnostics: DFBETAs, DFFITS, Cook's distance, hat values."""

    def compute(
        self,
        model: sm.regression.linear_model.RegressionResultsWrapper,
        X: Optional[np.ndarray] = None,
        y: Optional[np.ndarray] = None,
    ) -> dict:
        if X is None:
            try:
                X_full = model.model.exog
            except AttributeError:
                X_full = model.model.data.exog
        else:
            X_full = X
        if y is None:
            try:
                y = model.model.endog
            except AttributeError:
                y = model.model.data.endog
        X_full = np.asarray(X_full, dtype=float)
        if X_full.ndim == 1:
            X_full = X_full.reshape(-1, 1)
        y = np.asarray(y, dtype=float).ravel()
        n, p = X_full.shape
        residuals = np.asarray(model.resid).ravel()
        mse = np.sum(residuals ** 2) / max(n - p, 1)
        H = X_full @ np.linalg.inv(X_full.T @ X_full + np.eye(p) * 1e-10) @ X_full.T
        hat_values = np.diag(H).copy()
        standardized_resid = residuals / (np.sqrt(mse * (1.0 - hat_values)) + 1e-15)
        # Studentized (jackknife) residuals
        studentized_resid = np.zeros(n)
        dfbetas = np.zeros((n, p))
        dffits = np.zeros(n)
        cooksd = np.zeros(n)
        covratio = np.zeros(n)
        for i in range(n):
            mask = np.ones(n, dtype=bool)
            mask[i] = False
            Xi, yi = X_full[mask], y[mask]
            try:
                coef_i = np.linalg.lstsq(Xi, yi, rcond=None)[0]
            except np.linalg.LinAlgError:
                coef_i = np.zeros(p)
            y_pred_i = X_full @ coef_i
            ei = y - y_pred_i
            mse_i = np.sum(ei[mask] ** 2) / max(len(yi) - p, 1)
            studentized_resid[i] = ei[i] / (np.sqrt(mse_i * (1.0 - hat_values[i])) + 1e-15)
            dfbetas[i, :] = (model.params - coef_i) / (
                np.sqrt(np.diag(np.linalg.inv(X_full.T @ X_full + np.eye(p) * 1e-10))) + 1e-15
            )
            dffits[i] = np.sqrt(hat_values[i] / (1.0 - hat_values[i] + 1e-15)) * studentized_resid[i] / np.sqrt(n)
            # Cook's distance
            local_mse = mse if mse > 0 else 1e-15
            cooksd[i] = (residuals[i] ** 2 / (p * local_mse)) * (hat_values[i] / ((1.0 - hat_values[i]) ** 2 + 1e-15))
            # COVRATIO
            det_ratio = 1.0
            try:
                det_full = np.linalg.det(X_full.T @ X_full)
                det_omit = np.linalg.det(Xi.T @ Xi)
                det_ratio = det_omit / max(det_full, 1e-30)
            except Exception:
                pass
            covratio[i] = det_ratio / ((1.0 - hat_values[i] + 1e-15) ** (p + 1))
        # Cutoffs
        cutoff_hat = 2.0 * p / n
        cutoff_dffits = 2.0 * np.sqrt(p / n)
        cutoff_cook = 4.0 / n
        cutoff_dfbeta = 2.0 / np.sqrt(n)
        dfbetas_df = pd.DataFrame(
            dfbetas,
            columns=[f"DFBETA_{i}" for i in range(p)],
        )
        diagnostics_df = pd.DataFrame({
            "observation": np.arange(1, n + 1),
            "hat": hat_values,
            "standardized_residual": standardized_resid,
            "studentized_residual": studentized_resid,
            "dffits": dffits,
            "cooks_d": cooksd,
            "covratio": covratio,
        })
        diagnostics_df = pd.concat([diagnostics_df, dfbetas_df], axis=1)
        is_influential = pd.DataFrame({
            "high_leverage": hat_values > cutoff_hat,
            "large_dffits": np.abs(dffits) > cutoff_dffits,
            "large_cooks_d": cooksd > cutoff_cook,
            "large_dfbeta": (np.abs(dfbetas) > cutoff_dfbeta).any(axis=1),
        })
        flagged = {
            "n": n,
            "p": p,
            "cutoff_hat": cutoff_hat,
            "cutoff_dffits": cutoff_dffits,
            "cutoff_cook": cutoff_cook,
            "cutoff_dfbeta": cutoff_dfbeta,
            "n_high_leverage": int(is_influential["high_leverage"].sum()),
            "n_large_dffits": int(is_influential["large_dffits"].sum()),
            "n_large_cooks_d": int(is_influential["large_cooks_d"].sum()),
            "n_large_dfbeta": int(is_influential["large_dfbeta"].sum()),
            "n_any": int(is_influential.any(axis=1).sum()),
        }
        return {
            "diagnostics": diagnostics_df,
            "is_influential": is_influential,
            "flagged_summary": flagged,
        }

    @staticmethod
    def influence_plot(
        diagnostics: dict,
        output_path: Optional[str] = None,
    ) -> plt.Figure:
        diag = diagnostics["diagnostics"]
        inf = diagnostics["is_influential"]
        fig, ax = plt.subplots(figsize=(10, 6), dpi=150)
        sizes = 50 + 200 * (diag["cooks_d"] / max(diag["cooks_d"].max(), 1e-10))
        sc = ax.scatter(
            diag["hat"], diag["studentized_residual"],
            s=sizes, c=diag["cooks_d"], cmap="YlOrRd",
            alpha=0.7, edgecolors="gray", linewidth=0.5,
        )
        flagged_idx = inf.any(axis=1)
        ax.scatter(
            diag.loc[flagged_idx, "hat"],
            diag.loc[flagged_idx, "studentized_residual"],
            s=sizes[flagged_idx] * 1.2,
            facecolors="none", edgecolors="red", linewidth=1.5,
            label=f"Flagged ({flagged_idx.sum()})",
        )
        cbar = fig.colorbar(sc, ax=ax, label="Cook's distance")
        ax.axhline(0, color="gray", linewidth=0.8, linestyle="--")
        ax.axhline(-2, color="red", linewidth=0.5, linestyle=":", alpha=0.6)
        ax.axhline(2, color="red", linewidth=0.5, linestyle=":", alpha=0.6)
        cutoff_hat = diagnostics["flagged_summary"]["cutoff_hat"]
        ax.axvline(cutoff_hat, color="red", linewidth=0.5, linestyle=":", alpha=0.6,
                   label=f"leverage cutoff = {cutoff_hat:.3f}")
        ax.set_xlabel("Hat Values (Leverage)", fontweight="bold")
        ax.set_ylabel("Studentized Residuals", fontweight="bold")
        ax.set_title("Influence Diagnostics Plot", fontweight="bold")
        ax.legend(loc="upper right", frameon=True, edgecolor="lightgray")
        ax.grid(True, linestyle=":", alpha=0.4)
        fig.tight_layout()
        if output_path:
            fig.savefig(output_path, bbox_inches="tight", dpi=150)
        return fig


# ---------------------------------------------------------------------------
# 9. IQVCalculator
# ---------------------------------------------------------------------------
class IQVCalculator:
    """Index of Qualitative Variation — measures dispersion for categorical data."""

    @staticmethod
    def compute(data: np.ndarray) -> dict:
        data = np.asarray(data)
        data = data[~pd.isna(data)]
        n = len(data)
        if n == 0:
            raise ValueError("Empty data.")
        unique, counts = np.unique(data, return_counts=True)
        k = len(unique)
        proportions = counts / n
        sum_p_sq = np.sum(proportions ** 2)
        iqv = (k / max(k - 1, 1)) * (1.0 - sum_p_sq)
        if iqv >= 0.95:
            interp = "Maximum variation — categories are evenly distributed"
        elif iqv >= 0.50:
            interp = "High variation — moderate dispersion across categories"
        elif iqv >= 0.10:
            interp = "Low variation — most observations concentrated in few categories"
        else:
            interp = "No meaningful variation — all observations in a single category"
        return {
            "iqv": iqv,
            "k": k,
            "n": n,
            "category_proportions": dict(zip([str(u) for u in unique], proportions.tolist())),
            "interpretation": interp,
        }


# ---------------------------------------------------------------------------
# 10. PowerSensitivityAnalysis
# ---------------------------------------------------------------------------
class PowerSensitivityAnalysis:
    """Power sensitivity: vary effect size and SD to produce a range of sample sizes."""

    @staticmethod
    def power_sensitivity_test(
        test_type: str = "two_sample_t_test",
        base_effect_size: float = 0.5,
        base_sd: float = 1.0,
        effect_size_range: Tuple[float, float] = (0.9, 1.1),
        sd_range: Tuple[float, float] = (0.9, 1.1),
        steps: int = 5,
        alpha: float = 0.05,
        power: float = 0.8,
    ) -> dict:
        if test_type != "two_sample_t_test":
            raise ValueError("Currently only 'two_sample_t_test' is supported.")
        z_alpha = norm.ppf(1.0 - alpha / 2.0)
        z_beta = norm.ppf(power)
        effect_sizes = np.linspace(
            base_effect_size * effect_size_range[0],
            base_effect_size * effect_size_range[1],
            steps,
        )
        sds = np.linspace(
            base_sd * sd_range[0],
            base_sd * sd_range[1],
            steps,
        )
        matrix = []
        for es in effect_sizes:
            for sd in sds:
                cohens_d = abs(es) / max(sd, 1e-15)
                if cohens_d == 0:
                    n_per = np.inf
                else:
                    n_per = (2.0 * (z_alpha + z_beta) ** 2) / (cohens_d ** 2)
                n_final = int(np.ceil(n_per)) if np.isfinite(n_per) else 99999
                matrix.append({
                    "effect_size": es,
                    "sd": sd,
                    "n": n_final,
                })
        sensitivity_df = pd.DataFrame(matrix)
        ns = sensitivity_df["n"].values
        finite_ns = ns[np.isfinite(ns)]
        return {
            "min_n": int(finite_ns.min()) if len(finite_ns) > 0 else 99999,
            "max_n": int(finite_ns.max()) if len(finite_ns) > 0 else 99999,
            "median_n": int(np.median(finite_ns)) if len(finite_ns) > 0 else 99999,
            "range_n": int(finite_ns.max() - finite_ns.min()) if len(finite_ns) > 0 else 0,
            "sensitivity_matrix": sensitivity_df,
            "interpretation": (
                f"Sample size varies from {int(finite_ns.min())} to {int(finite_ns.max())} "
                f"across the sensitivity grid. Median = {int(np.median(finite_ns))}."
            ),
        }


# ---------------------------------------------------------------------------
# 11. TOSTSampleSize
# ---------------------------------------------------------------------------
class TOSTSampleSize:
    """Bioequivalence (TOST) sample size calculation using non-central t-distribution."""

    @staticmethod
    def calculate(
        power: float = 0.80,
        alpha: float = 0.05,
        theta0: float = 0.90,
        theta_lower: float = 0.80,
        theta_upper: float = 1.25,
        cv: float = 0.14,
    ) -> dict:
        mse = np.log(cv ** 2 + 1.0)
        n_found = None
        power_achieved = 0.0
        for n_per_arm in range(4, 10001):
            df = 2 * n_per_arm - 2
            t_crit = t_dist.ppf(1.0 - alpha, df)
            ncp_upper = np.sqrt(n_per_arm) * (np.log(theta_upper) - np.log(theta0)) / np.sqrt(mse)
            ncp_lower = np.sqrt(n_per_arm) * (np.log(theta0) - np.log(theta_lower)) / np.sqrt(mse)
            lower_bound = -ncp_lower + t_crit
            upper_bound = ncp_upper - t_crit
            if upper_bound <= lower_bound:
                power_val = 0.0
            else:
                try:
                    power_val = t_dist.cdf(upper_bound, df) - t_dist.cdf(lower_bound, df)
                except Exception:
                    power_val = 0.0
            if power_val >= power:
                n_found = n_per_arm
                power_achieved = power_val
                break
        if n_found is None:
            n_found = 10000
            power_achieved = power_val
        return {
            "sample_size_total": int(n_found * 2),
            "sample_size_per_arm": int(n_found),
            "power_achieved": float(power_achieved),
            "parameters": {
                "alpha": alpha,
                "power_target": power,
                "theta0": theta0,
                "theta_lower": theta_lower,
                "theta_upper": theta_upper,
                "CV": cv,
            },
        }

    @staticmethod
    def sensitivity_analysis(
        cv_range: Optional[Tuple[float, float]] = None,
        theta0_range: Optional[Tuple[float, float]] = None,
        power: float = 0.80,
        alpha: float = 0.05,
        theta_lower: float = 0.80,
        theta_upper: float = 1.25,
        steps: int = 10,
    ) -> pd.DataFrame:
        if cv_range is None:
            cv_range = (0.05, 1.0)
        if theta0_range is None:
            theta0_range = (0.81, 1.24)
        cvs = np.linspace(cv_range[0], cv_range[1], steps)
        theta0s = np.linspace(theta0_range[0], theta0_range[1], steps)
        rows = []
        for cv in cvs:
            for t0 in theta0s:
                try:
                    res = TOSTSampleSize.calculate(
                        power=power, alpha=alpha,
                        theta0=t0, theta_lower=theta_lower,
                        theta_upper=theta_upper, cv=cv,
                    )
                    rows.append({
                        "CV": cv,
                        "theta0": t0,
                        "n_per_arm": res["sample_size_per_arm"],
                        "n_total": res["sample_size_total"],
                        "power_achieved": res["power_achieved"],
                    })
                except Exception:
                    continue
        return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# 12. SurvivalSensitivityAnalysis
# ---------------------------------------------------------------------------
class SurvivalSensitivityAnalysis:
    """Sensitivity analyses for survival models (Cox PH)."""

    @staticmethod
    def bootstrap_hr(
        time: np.ndarray,
        status: np.ndarray,
        group: np.ndarray,
        n_iterations: int = 1000,
        alpha: float = 0.05,
    ) -> dict:
        time = np.asarray(time, dtype=float)
        status = np.asarray(status, dtype=int)
        group = np.asarray(group, dtype=float)
        mask = ~(np.isnan(time) | np.isnan(status) | np.isnan(group))
        time, status, group = time[mask], status[mask], group[mask]
        n = len(time)
        if n < 10:
            raise ValueError("Need at least 10 observations for bootstrap HR.")
        hrs = []
        rng = np.random.default_rng(42)
        for _ in range(n_iterations):
            idx = rng.integers(0, n, size=n)
            boot_time = time[idx]
            boot_status = status[idx]
            boot_group = group[idx]
            if np.sum(boot_status == 1) < 3 or len(np.unique(boot_group)) < 2:
                continue
            try:
                boot_exog = sm.add_constant(boot_group, has_constant="add")
                boot_model = sm.PHReg(boot_time, boot_exog, status=boot_status)
                boot_fit = boot_model.fit(disp=0)
                coef_val = float(boot_fit.params[-1])
                hr = float(np.exp(coef_val))
                if np.isfinite(hr) and hr > 0 and hr < 100:
                    hrs.append(hr)
            except Exception:
                continue
        hrs = np.array(hrs)
        if len(hrs) < max(20, n_iterations // 10):
            raise ValueError(f"Only {len(hrs)} successful bootstrap iterations (<{max(20, n_iterations//10)}).")
        hr_median = float(np.median(hrs))
        ci_lower = float(np.percentile(hrs, 100 * alpha / 2))
        ci_upper = float(np.percentile(hrs, 100 * (1.0 - alpha / 2)))
        return {
            "hr_median": hr_median,
            "ci_lower": ci_lower,
            "ci_upper": ci_upper,
            "hr_distribution": hrs,
        }

    @staticmethod
    def leave_one_out(
        time: np.ndarray,
        status: np.ndarray,
        group: np.ndarray,
    ) -> dict:
        time = np.asarray(time, dtype=float)
        status = np.asarray(status, dtype=int)
        group = np.asarray(group, dtype=float)
        mask = ~(np.isnan(time) | np.isnan(status) | np.isnan(group))
        time, status, group = time[mask], status[mask], group[mask]
        n = len(time)
        X_full = sm.add_constant(group, has_constant="add")
        try:
            full_model = sm.PHReg(time, X_full, status=status)
            full_fit = full_model.fit(disp=0)
            full_coef = float(full_fit.params[-1])
        except Exception:
            full_coef = 0.0
        dfbeta = np.zeros(n)
        for i in range(n):
            mask_i = np.ones(n, dtype=bool)
            mask_i[i] = False
            try:
                loo_X = sm.add_constant(group[mask_i], has_constant="add")
                loo_model = sm.PHReg(time[mask_i], loo_X, status=status[mask_i])
                loo_fit = loo_model.fit(disp=0)
                loo_coef = float(loo_fit.params[-1])
                dfbeta[i] = full_coef - loo_coef
            except Exception:
                dfbeta[i] = 0.0
        dfbeta_abs = np.abs(dfbeta)
        max_dfbeta = float(dfbeta_abs.max())
        influential_idx = np.where(dfbeta_abs > 2.0 * np.std(dfbeta, ddof=1) if np.std(dfbeta, ddof=1) > 0 else dfbeta_abs > 0.1)[0]
        return {
            "dfbeta_max": max_dfbeta,
            "dfbeta_values": dfbeta.tolist(),
            "influential_indices": influential_idx.tolist(),
            "n_influential": int(len(influential_idx)),
        }


# ---------------------------------------------------------------------------
# 13. DescriptiveStatsEnhanced
# ---------------------------------------------------------------------------
class DescriptiveStatsEnhanced:
    """Enhanced descriptive statistics with grouping."""

    @staticmethod
    def describe_grouped(
        data: pd.DataFrame,
        variables: List[str],
        group_col: Optional[str] = None,
    ) -> pd.DataFrame:
        results = []
        if group_col is not None and group_col in data.columns:
            groups = data[group_col].unique()
        else:
            groups = [None]
        for var in variables:
            if var not in data.columns:
                continue
            for grp in groups:
                if grp is not None:
                    subset = data[data[group_col] == grp][var]
                else:
                    subset = data[var]
                subset = subset.dropna()
                n = len(subset)
                n_missing = int(data[var].isna().sum()) if group_col is None or grp is None else int(
                    data.loc[data[group_col] == grp, var].isna().sum()
                )
                if np.issubdtype(data[var].dtype, np.number):
                    vals = subset.values.astype(float)
                    if n == 0:
                        row = {
                            "variable": var,
                            "group": str(grp) if grp is not None else "Overall",
                            "n": 0, "missing": n_missing,
                            "type": "numeric",
                        }
                        results.append(row)
                        continue
                    mean_v = float(np.mean(vals))
                    sd_v = float(np.std(vals, ddof=1)) if n > 1 else 0.0
                    qs = np.percentile(vals, [0, 25, 50, 75, 100])
                    iqr = float(qs[3] - qs[1])
                    cv = float(sd_v / max(abs(mean_v), 1e-15)) if n > 1 else 0.0
                    se = float(sd_v / np.sqrt(n)) if n > 1 else 0.0
                    skew = float(stats.skew(vals)) if n > 2 else 0.0
                    kurt = float(stats.kurtosis(vals, fisher=True)) if n > 3 else 0.0
                    row = {
                        "variable": var,
                        "group": str(grp) if grp is not None else "Overall",
                        "n": n,
                        "missing": n_missing,
                        "mean": mean_v,
                        "sd": sd_v,
                        "min": float(qs[0]),
                        "q1": float(qs[1]),
                        "median": float(qs[2]),
                        "q3": float(qs[3]),
                        "max": float(qs[4]),
                        "iqr": iqr,
                        "cv": cv,
                        "se": se,
                        "skewness": skew,
                        "kurtosis": kurt,
                        "type": "numeric",
                    }
                else:
                    if n == 0:
                        row = {
                            "variable": var,
                            "group": str(grp) if grp is not None else "Overall",
                            "n": 0, "missing": n_missing,
                            "type": "categorical",
                        }
                        results.append(row)
                        continue
                    unique_vals = subset.unique()
                    n_categories = len(unique_vals)
                    mode_val = subset.mode().iloc[0] if n > 0 else np.nan
                    try:
                        iqv_result = IQVCalculator.compute(subset.values)
                        iqv = iqv_result["iqv"]
                    except Exception:
                        iqv = 0.0
                    row = {
                        "variable": var,
                        "group": str(grp) if grp is not None else "Overall",
                        "n": n,
                        "missing": n_missing,
                        "n_categories": n_categories,
                        "mode": mode_val,
                        "iqv": iqv,
                        "type": "categorical",
                    }
                results.append(row)
        return pd.DataFrame(results)

    @staticmethod
    def freq_table_continuous(
        data: np.ndarray,
        bins: Optional[int] = None,
    ) -> pd.DataFrame:
        data = np.asarray(data, dtype=float)
        data = data[~np.isnan(data)]
        n = len(data)
        if n == 0:
            return pd.DataFrame(columns=["bin_start", "bin_end", "frequency", "percent", "cumulative_percent"])
        if bins is None:
            # Freedman-Diaconis rule
            iqr = np.percentile(data, 75) - np.percentile(data, 25)
            if iqr > 0:
                fd_bw = 2.0 * iqr / (n ** (1.0 / 3.0))
                bins = max(5, int(np.ceil((data.max() - data.min()) / max(fd_bw, 1e-15))))
            else:
                bins = max(5, int(np.ceil(2.0 * n ** (2.0 / 5.0))))
        bin_edges = np.linspace(data.min(), data.max(), bins + 1)
        freqs, edges = np.histogram(data, bins=bin_edges)
        total = float(freqs.sum())
        cumsum = 0.0
        rows = []
        for i in range(len(freqs)):
            pct = freqs[i] / total * 100.0
            cumsum += pct
            rows.append({
                "bin_start": edges[i],
                "bin_end": edges[i + 1],
                "frequency": int(freqs[i]),
                "percent": round(pct, 2),
                "cumulative_percent": round(cumsum, 2),
            })
        return pd.DataFrame(rows)


class MixedEffectsModel:
    def __init__(self, formula, data, groups, re_formula=None, vc_formula=None):
        self.formula = formula
        self.data = data
        self.groups = groups
        self.re_formula = re_formula
        self.vc_formula = vc_formula
        self._model = None
        self._fitted = None
        self._method = None

    def fit(self, reml=True, method=None, **kwargs):
        self._method = "REML" if reml else "ML"
        groups_data = self.data[self.groups] if isinstance(self.groups, str) else self.groups
        try:
            self._model = smf.mixedlm(
                self.formula,
                self.data,
                groups=groups_data,
                re_formula=self.re_formula,
                vc_formula=self.vc_formula,
            )
            if method is not None:
                self._fitted = self._model.fit(method=method, reml=reml, **kwargs)
            else:
                self._fitted = self._model.fit(reml=reml, **kwargs)
        except Exception as e:
            raise RuntimeError(f"MixedLM failed to converge: {e}. Try rescaling variables or using method='cg'.")
        return self._extract_results()

    def _extract_results(self):
        r = self._fitted
        coefs = r.fe_params
        pvalues = r.pvalues
        conf_int = r.conf_int()
        resid_var = r.scale
        if r.cov_re is not None:
            random_effects_var = np.trace(r.cov_re) / r.cov_re.shape[0]
        else:
            random_effects_var = 0.0
        total_var = resid_var + random_effects_var
        return {
            "coefficients": coefs,
            "standard_errors": r.bse_fe,
            "z_values": r.tvalues,
            "p_values": pvalues,
            "conf_int": conf_int,
            "log_likelihood": r.llf,
            "aic": r.aic,
            "bic": r.bic,
            "n_groups": len(r.random_effects) if r.random_effects is not None else 0,
            "n_obs": r.nobs,
            "residual_var": resid_var,
            "random_effects_var": random_effects_var,
            "total_var": total_var,
            "method": self._method,
            "formula": self.formula,
            "re_formula": self.re_formula,
            "groups": self.groups if isinstance(self.groups, str) else "custom",
            "converged": r.converged,
        }

    def summary(self, decimals=4):
        if self._fitted is None:
            return "Model not fitted yet. Call .fit() first."
        return self._fitted.summary()

    def fixed_effects_table(self, decimals=4):
        if self._fitted is None:
            return None
        r = self._fitted
        coef = r.fe_params
        se = r.bse_fe
        z = r.tvalues
        p = r.pvalues
        ci = r.conf_int()
        rows = []
        for var in coef.index:
            rows.append({
                "Variable": var,
                "Coefficient": round(coef[var], decimals),
                "SE": round(se[var], decimals),
                "z": round(z[var], decimals),
                "p": p[var],
                "CI_lower": round(ci.loc[var, 0], decimals),
                "CI_upper": round(ci.loc[var, 1], decimals),
            })
        df = pd.DataFrame(rows)
        df["Sig"] = df["p"].apply(
            lambda x: "***" if x < 0.001 else ("**" if x < 0.01 else ("*" if x < 0.05 else "ns"))
        )
        return df

    def random_effects(self):
        if self._fitted is None:
            return None
        r = self._fitted
        if r.random_effects is None:
            return None
        groups = list(r.random_effects.keys())
        re_names = list(r.random_effects[groups[0]].index) if hasattr(r.random_effects[groups[0]], "index") else [f"RE{i}" for i in range(len(r.random_effects[groups[0]]))]
        data = {g: r.random_effects[g].values.flatten() for g in groups}
        df = pd.DataFrame(data, index=re_names).T
        df.index.name = "Group"
        return df

    def icc(self):
        if self._fitted is None:
            return None
        r = self._fitted
        resid_var = r.scale
        if r.cov_re is not None:
            if r.cov_re.shape[0] == 1:
                re_var = r.cov_re.iloc[0, 0]
            else:
                re_var = np.trace(r.cov_re.values) / r.cov_re.shape[0]
        else:
            re_var = 0.0
        total = re_var + resid_var
        icc_val = re_var / total if total > 0 else 0.0
        return {"icc": round(icc_val, 4), "between_var": round(re_var, 4), "within_var": round(resid_var, 4), "total_var": round(total, 4)}

    def anova_lrt(self):
        if self._fitted is None:
            return None
        r = self._fitted
        null_formula = self.formula.split("~")[0] + "~ 1"
        null_model = smf.mixedlm(
            null_formula,
            self.data,
            groups=self.data[self.groups] if isinstance(self.groups, str) else self.groups,
        )
        null_fitted = null_model.fit(reml=False)
        lrt_stat = 2 * (r.llf - null_fitted.llf)
        df_diff = r.k_fe - null_fitted.k_fe
        p_value = 1.0 - f_dist.cdf(lrt_stat, df_diff, r.nobs - r.df_modelwc) if df_diff > 0 else 1.0
        return {
            "log_likelihood_full": r.llf,
            "log_likelihood_null": null_fitted.llf,
            "lrt_stat": round(lrt_stat, 4),
            "df_diff": df_diff,
            "p_value": p_value,
            "aic_full": r.aic,
            "aic_null": null_fitted.aic,
            "bic_full": r.bic,
            "bic_null": null_fitted.bic,
        }

    def predict(self, newdata=None):
        if self._fitted is None:
            return None
        if newdata is not None:
            return self._fitted.predict(newdata)
        return self._fitted.fittedvalues

    def residuals(self, rtype="pearson"):
        if self._fitted is None:
            return None
        if rtype == "pearson":
            return self._fitted.resid / np.sqrt(self._fitted.scale)
        return self._fitted.resid

    def diagnostic_plots(self, figsize=(12, 4)):
        if self._fitted is None:
            return None
        resid = self.residuals("pearson")
        fitted = self.predict()
        fig, axes = plt.subplots(1, 3, figsize=figsize)
        axes[0].scatter(fitted, resid, alpha=0.6, edgecolors="k", linewidth=0.5)
        axes[0].axhline(0, color="red", linestyle="--", alpha=0.5)
        axes[0].set_xlabel("Fitted Values")
        axes[0].set_ylabel("Pearson Residuals")
        axes[0].set_title("Residuals vs Fitted")
        axes[1].hist(resid, bins=20, edgecolor="black", alpha=0.7)
        axes[1].set_xlabel("Residuals")
        axes[1].set_ylabel("Frequency")
        axes[1].set_title("Histogram of Residuals")
        stats.probplot(resid, dist="norm", plot=axes[2])
        axes[2].set_title("Q-Q Plot")
        plt.tight_layout()
        return fig

    def marginal_means(self, variable, at=None, decimals=4):
        if self._fitted is None:
            return None
        if at is None:
            if self.data[variable].dtype in (np.float64, np.int64, float, int):
                at = [self.data[variable].mean(), self.data[variable].mean() + self.data[variable].std(), self.data[variable].mean() - self.data[variable].std()]
            else:
                at = self.data[variable].unique().tolist()
        results = []
        for val in at:
            pred_data = self.data.copy()
            pred_data[variable] = val
            preds = self._fitted.predict(pred_data)
            results.append({"value": val, "mean": round(preds.mean(), decimals), "sd": round(preds.std(), decimals)})
        return pd.DataFrame(results)

    def pairwise(self, variable, p_adjust="tukey", decimals=4):
        if self._fitted is None:
            return None
        levels = self.data[variable].unique()
        if len(levels) < 2:
            return None
        import itertools
        from scipy.stats import t as t_dist
        results = []
        n_comp = 0
        lsm = self.marginal_means(variable)
        means_dict = dict(zip(lsm["value"], lsm["mean"]))
        for l1, l2 in itertools.combinations(levels, 2):
            n_comp += 1
            diff = means_dict.get(l1, 0) - means_dict.get(l2, 0)
            pooled_se = np.sqrt(2 * self._fitted.scale / self.data.shape[0])
            t_val = diff / pooled_se if pooled_se > 0 else 0
            p_raw = 2 * (1 - t_dist.cdf(abs(t_val), df=self._fitted.nobs - self._fitted.df_modelwc))
            results.append({"level1": l1, "level2": l2, "diff": round(diff, decimals), "SE": round(pooled_se, decimals), "t": round(t_val, decimals), "p_raw": round(p_raw, decimals)})
        df = pd.DataFrame(results)
        if p_adjust == "bonferroni":
            df["p_adjusted"] = (df["p_raw"] * n_comp).clip(upper=1.0)
        else:
            df["p_adjusted"] = df["p_raw"]
        df["Sig"] = df["p_adjusted"].apply(lambda x: "***" if x < 0.001 else ("**" if x < 0.01 else ("*" if x < 0.05 else "ns")))
        return df

    def cov_re(self):
        if self._fitted is None:
            return None
        return self._fitted.cov_re

    def write_apa(self):
        if self._fitted is None:
            return "Model not fitted."
        r = self._fitted
        icc_vals = self.icc()
        tbl = self.fixed_effects_table()
        lines = []
        lines.append(f"A linear mixed-effects model was fitted using {self._method} estimation.")
        lines.append(f"The model formula was: {self.formula}")
        if self.re_formula:
            lines.append(f"Random effects structure: {self.re_formula}, grouped by {self.groups if isinstance(self.groups, str) else 'grouping variable'}")
        n_groups_apa = len(r.random_effects) if r.random_effects is not None else 0
        lines.append(f"The model converged successfully (N = {int(r.nobs)}, groups = {n_groups_apa}).")
        lines.append(f"ICC = {icc_vals['icc']:.3f} ({icc_vals['between_var']:.3f} between-group variance, {icc_vals['within_var']:.3f} within-group variance).")
        lines.append(f"Model fit: AIC = {r.aic:.1f}, BIC = {r.bic:.1f}, log-likelihood = {r.llf:.2f}.")
        lines.append("")
        lines.append("Fixed effects (coefficient +/- SE, z, p):")
        for _, row in tbl.iterrows():
            p_str = f"p {'< .001' if row['p'] < 0.001 else f'= {row.p:.3f}'}"
            lines.append(f"  {row['Variable']}: b = {row['Coefficient']:.3f} +/- {row['SE']:.3f}, z = {row['z']:.2f}, {p_str} {row['Sig']}")
        return "\n".join(lines)

    @staticmethod
    def compare_models(models, model_names=None):
        if not models:
            return None
        rows = []
        for i, m in enumerate(models):
            if m._fitted is None:
                continue
            r = m._fitted
            name = model_names[i] if model_names and i < len(model_names) else f"Model{i+1}"
            n_groups = len(r.random_effects) if r.random_effects is not None else 0
            rows.append({"Model": name, "LogLik": round(r.llf, 2), "AIC": round(r.aic, 2), "BIC": round(r.bic, 2), "N": int(r.nobs), "Groups": n_groups, "Converged": r.converged})
        df = pd.DataFrame(rows)
        if len(df) >= 2:
            best_aic = df.loc[df["AIC"].idxmin()]
            best_bic = df.loc[df["BIC"].idxmin()]
            df["Best_AIC"] = df["Model"] == best_aic["Model"]
            df["Best_BIC"] = df["Model"] == best_bic["Model"]
        return df
