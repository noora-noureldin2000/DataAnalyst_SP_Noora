import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from typing import Optional, Union, Dict, Any, List, Tuple
import warnings

from scipy.stats import norm, chi2, t as t_dist, f as f_dist, pearsonr, beta
from scipy.optimize import brentq

warnings.filterwarnings("ignore", category=DeprecationWarning)

_Z_95 = 1.96


def _wilson_ci(n_success: int, n_total: int, alpha: float = 0.05) -> Tuple[float, float, float]:
    if n_total == 0:
        return np.nan, np.nan, np.nan
    p = n_success / n_total
    z = norm.ppf(1.0 - alpha / 2.0)
    denom = 1.0 + z ** 2 / n_total
    center = (p + z ** 2 / (2.0 * n_total)) / denom
    half = z * np.sqrt((p * (1.0 - p) + z ** 2 / (4.0 * n_total)) / n_total) / denom
    return p, center - half, center + half


def _logit(p: np.ndarray) -> np.ndarray:
    p = np.clip(p, 1e-15, 1.0 - 1e-15)
    return np.log(p / (1.0 - p))


def _invlogit(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))



class ROC_Analysis:
    """Receiver Operating Characteristic (ROC) curve analysis for binary classifiers."""

    def __init__(self, y_true: np.ndarray, y_score: np.ndarray):
        self.y_true = np.asarray(y_true, dtype=int).ravel()
        self.y_score = np.asarray(y_score, dtype=float).ravel()
        mask = ~(np.isnan(self.y_true) | np.isnan(self.y_score))
        self.y_true = self.y_true[mask]
        self.y_score = self.y_score[mask]
        self._n_pos = int(np.sum(self.y_true == 1))
        self._n_neg = int(np.sum(self.y_true == 0))
        self._n = len(self.y_true)
        self._scores = None
        self._tpr = None
        self._fpr = None
        self._thresholds = None
        self._auc = None
        # Cache for bootstrap CI results computed in compute()
        self._boot_ci = None        # (lower_ci, upper_ci) for AUC
        self._tpr_band = None       # (fpr_grid, tpr_lower, tpr_upper) for CI band

    def _compute_curve(self):
        if self._scores is not None:
            return
        unique = np.unique(self.y_score)
        if len(unique) == 1:
            self._scores = np.array([unique[0]])
            self._tpr = np.array([1.0])
            self._fpr = np.array([1.0])
            self._thresholds = np.array([unique[0]])
            self._auc = 0.5
            return
        thresholds = np.sort(np.unique(self.y_score))[::-1]
        thresholds = np.concatenate([thresholds, [thresholds[-1] - 1.0]])
        tpr = np.zeros(len(thresholds))
        fpr = np.zeros(len(thresholds))
        for i, th in enumerate(thresholds):
            pred = (self.y_score >= th).astype(int)
            tp = np.sum((pred == 1) & (self.y_true == 1))
            fn = np.sum((pred == 0) & (self.y_true == 1))
            fp = np.sum((pred == 1) & (self.y_true == 0))
            tn = np.sum((pred == 0) & (self.y_true == 0))
            tpr[i] = tp / max(tp + fn, 1)
            fpr[i] = fp / max(fp + tn, 1)
        self._thresholds = thresholds
        self._tpr = tpr
        self._fpr = fpr
        self._scores = self.y_score

    def _compute_auc(self) -> float:
        if self._auc is not None:
            return self._auc
        self._compute_curve()
        sorted_idx = np.argsort(self._fpr)
        fpr_sorted = self._fpr[sorted_idx]
        tpr_sorted = self._tpr[sorted_idx]
        self._auc = float(np.trapezoid(tpr_sorted, fpr_sorted))
        return self._auc

    def _delong_se(self, auc: float) -> float:
        n1 = self._n_pos
        n0 = self._n_neg
        if n1 == 0 or n0 == 0:
            return 0.25
        y_true = self.y_true
        y_score = self.y_score
        idx_pos = y_true == 1
        idx_neg = y_true == 0
        x1 = y_score[idx_pos]
        x0 = y_score[idx_neg]

        def _theta(x_i, x_j):
            return np.mean((x_i.reshape(-1, 1) > x_j.reshape(1, -1)).astype(float) +
                           0.5 * (x_i.reshape(-1, 1) == x_j.reshape(1, -1)).astype(float))

        V10 = np.zeros(n1)
        for i in range(n1):
            V10[i] = np.mean((x1[i] > x0).astype(float) + 0.5 * (x1[i] == x0).astype(float))
        V01 = np.zeros(n0)
        for i in range(n0):
            V01[i] = np.mean((x1 > x0[i]).astype(float) + 0.5 * (x1 == x0[i]).astype(float))
        S10 = np.var(V10, ddof=1)
        S01 = np.var(V01, ddof=1)
        se = np.sqrt(S10 / n1 + S01 / n0)
        return se

    def compute(self, n_boot: int = 2000) -> dict:
        """Compute ROC statistics including bootstrap CI for the AUC.

        Parameters
        ----------
        n_boot : int
            Number of bootstrap iterations for the AUC confidence interval.
            Set to 0 to skip bootstrap (uses DeLong SE instead).
        """
        self._compute_curve()
        auc = self._compute_auc()
        n1 = self._n_pos
        n0 = self._n_neg
        if n1 == 0 or n0 == 0:
            return {
                "auc": np.nan, "auc_ci": (np.nan, np.nan),
                "youden_index": np.nan, "youden_threshold": np.nan,
                "sensitivity": np.nan, "specificity": np.nan,
                "ppv": np.nan, "npv": np.nan,
                "lr_plus": np.nan, "lr_minus": np.nan,
            }

        # --- Bootstrap CI (computed once and cached) ---
        if self._boot_ci is None and n_boot > 0:
            n = len(self.y_true)
            rng = np.random.default_rng(42)
            boot_aucs = []
            fpr_grid = np.linspace(0, 1, 200)
            tpr_boot_rows = []
            for _ in range(n_boot):
                idx = rng.integers(0, n, size=n)
                yt = self.y_true[idx]
                ys = self.y_score[idx]
                if np.sum(yt == 1) == 0 or np.sum(yt == 0) == 0:
                    continue
                roc_boot = ROC_Analysis(yt, ys)
                roc_boot._compute_curve()
                boot_aucs.append(roc_boot._compute_auc())
                tpr_boot_rows.append(
                    np.interp(fpr_grid, roc_boot._fpr, roc_boot._tpr, left=0, right=1)
                )
            boot_aucs = np.array(boot_aucs)
            if len(boot_aucs) > 10:
                lower_ci = float(np.percentile(boot_aucs, 2.5))
                upper_ci = float(np.percentile(boot_aucs, 97.5))
                tpr_mat = np.array(tpr_boot_rows)
                tpr_lower = np.percentile(tpr_mat, 2.5, axis=0)
                tpr_upper = np.percentile(tpr_mat, 97.5, axis=0)
                self._boot_ci = (lower_ci, upper_ci)
                self._tpr_band = (fpr_grid, tpr_lower, tpr_upper)
            else:
                # Fall back to DeLong SE
                se_auc = self._delong_se(auc)
                self._boot_ci = (
                    max(0.0, auc - _Z_95 * se_auc),
                    min(1.0, auc + _Z_95 * se_auc),
                )
                self._tpr_band = None
        elif self._boot_ci is None:
            # n_boot=0: use DeLong
            se_auc = self._delong_se(auc)
            self._boot_ci = (
                max(0.0, auc - _Z_95 * se_auc),
                min(1.0, auc + _Z_95 * se_auc),
            )
            self._tpr_band = None

        auc_lower, auc_upper = self._boot_ci

        youden_idx = np.argmax(self._tpr - self._fpr)
        youden_th = self._thresholds[youden_idx]
        pred_youden = (self.y_score >= youden_th).astype(int)
        tp = np.sum((pred_youden == 1) & (self.y_true == 1))
        fn = np.sum((pred_youden == 0) & (self.y_true == 1))
        fp = np.sum((pred_youden == 1) & (self.y_true == 0))
        tn = np.sum((pred_youden == 0) & (self.y_true == 0))
        sens = tp / max(tp + fn, 1)
        spec = tn / max(tn + fp, 1)
        ppv = tp / max(tp + fp, 1)
        npv = tn / max(tn + fn, 1)
        lr_plus = sens / max(1.0 - spec, 1e-15)
        lr_minus = (1.0 - sens) / max(spec, 1e-15)

        return {
            "auc": auc,
            "auc_ci": (auc_lower, auc_upper),
            "youden_index": float(self._tpr[youden_idx] - self._fpr[youden_idx]),
            "youden_threshold": float(youden_th),
            "sensitivity": float(sens),
            "specificity": float(spec),
            "ppv": float(ppv),
            "npv": float(npv),
            "lr_plus": float(lr_plus),
            "lr_minus": float(lr_minus),
        }

    def optimal_cutoff(self, method: str = 'youden') -> dict:
        self._compute_curve()
        if method == 'youden':
            idx = np.argmax(self._tpr - self._fpr)
        else:
            idx = np.argmin(np.sqrt((1.0 - self._tpr) ** 2 + self._fpr ** 2))
        th = self._thresholds[idx]
        pred = (self.y_score >= th).astype(int)
        tp = np.sum((pred == 1) & (self.y_true == 1))
        fn = np.sum((pred == 0) & (self.y_true == 1))
        fp = np.sum((pred == 1) & (self.y_true == 0))
        tn = np.sum((pred == 0) & (self.y_true == 0))
        sens = tp / max(tp + fn, 1)
        spec = tn / max(tn + fp, 1)
        ppv = tp / max(tp + fp, 1)
        npv = tn / max(tn + fn, 1)
        return {
            "threshold": float(th),
            "sensitivity": float(sens),
            "specificity": float(spec),
            "ppv": float(ppv),
            "npv": float(npv),
            "lr_plus": float(sens / max(1.0 - spec, 1e-15)),
            "lr_minus": float((1.0 - sens) / max(spec, 1e-15)),
            "youden_index": float(sens + spec - 1.0),
            "method": method,
        }

    def threshold_table(self, thresholds: Optional[np.ndarray] = None) -> pd.DataFrame:
        self._compute_curve()
        if thresholds is not None:
            ths = np.sort(np.asarray(thresholds, dtype=float))[::-1]
        else:
            n_vals = min(20, len(self._thresholds))
            ths = np.linspace(self.y_score.min(), self.y_score.max(), n_vals)[::-1]
        rows = []
        for th in ths:
            pred = (self.y_score >= th).astype(int)
            tp = np.sum((pred == 1) & (self.y_true == 1))
            fn = np.sum((pred == 0) & (self.y_true == 1))
            fp = np.sum((pred == 1) & (self.y_true == 0))
            tn = np.sum((pred == 0) & (self.y_true == 0))
            sens = tp / max(tp + fn, 1)
            spec = tn / max(tn + fp, 1)
            ppv = tp / max(tp + fp, 1)
            npv = tn / max(tn + fn, 1)
            lrp = sens / max(1.0 - spec, 1e-15)
            lrm = (1.0 - sens) / max(spec, 1e-15)
            rows.append({
                "threshold": float(th),
                "sensitivity": float(sens),
                "specificity": float(spec),
                "ppv": float(ppv),
                "npv": float(npv),
                "lr+": float(lrp),
                "lr-": float(lrm),
            })
        return pd.DataFrame(rows)

    def plot_roc(self, figsize: Tuple[float, float] = (8, 6), annotate_cutoff: bool = False,
                 optimal_cutoff: Optional[dict] = None) -> plt.Figure:
        """Plot the ROC curve using pre-computed bootstrap CI from compute().

        Call compute() first to run the (expensive) bootstrap CI computation.
        If compute() has not been called, this method calls it with default n_boot=2000.
        """
        self._compute_curve()
        auc = self._compute_auc()

        # Ensure bootstrap CI is available (runs compute if not already done)
        if self._boot_ci is None:
            self.compute()

        fig, ax = plt.subplots(figsize=figsize, dpi=150)

        # Draw bootstrap CI band from cached results
        if self._tpr_band is not None:
            fpr_grid, tpr_lower, tpr_upper = self._tpr_band
            ax.fill_between(fpr_grid, tpr_lower, tpr_upper, alpha=0.15, color="#1f77b4",
                            label="95% CI (bootstrap)")

        lower_ci, upper_ci = self._boot_ci if self._boot_ci else (None, None)
        ax.plot(self._fpr, self._tpr, color="#00468B", linewidth=2, zorder=3,
                label=f"ROC (AUC = {auc:.3f})")
        ax.plot([0, 1], [0, 1], "k--", linewidth=0.8, alpha=0.5, label="Chance")

        if annotate_cutoff and optimal_cutoff:
            th = optimal_cutoff.get('threshold', 0)
            sens = optimal_cutoff.get('sensitivity', 0)
            spec = optimal_cutoff.get('specificity', 0)
            fpr_at = 1.0 - spec
            ax.plot(fpr_at, sens, 'o', color='#ED0000', markersize=10, zorder=4, label=f'Optimal cutoff')
            ax.annotate(f'Cutoff = {th:.2f}\nSens = {sens:.3f}, Spec = {spec:.3f}',
                        xy=(fpr_at, sens), xytext=(fpr_at + 0.15, sens - 0.15),
                        fontsize=8, color='#333333',
                        arrowprops=dict(arrowstyle='->', color='gray', lw=0.8),
                        bbox=dict(facecolor='white', alpha=0.8, edgecolor='lightgray', boxstyle='round,pad=0.3'))

        ax.set_xlabel("1 - Specificity (False Positive Rate)", fontweight="bold")
        ax.set_ylabel("Sensitivity (True Positive Rate)", fontweight="bold")
        ax.set_title("Receiver Operating Characteristic (ROC) Curve", fontweight="bold")
        ax.legend(loc="lower right", frameon=True, edgecolor="lightgray")
        ax.set_xlim(-0.02, 1.02)
        ax.set_ylim(-0.02, 1.02)
        ax.grid(True, linestyle=":", alpha=0.4)
        auc_text = f"AUC = {auc:.3f}" + (
            f" (95% CI: {lower_ci:.3f}-{upper_ci:.3f})"
            if lower_ci is not None and upper_ci is not None else ""
        )
        ax.text(0.02, 0.02, f"n = {self._n} (pos = {self._n_pos}, neg = {self._n_neg})\n{auc_text}",
                transform=ax.transAxes, fontsize=8, va="bottom",
                bbox=dict(facecolor="white", alpha=0.8, edgecolor="lightgray", boxstyle="round,pad=0.3"))
        fig.tight_layout()
        return fig

    def plot_performance_metrics(self, figsize: Tuple[float, float] = (10, 6)) -> plt.Figure:
        self._compute_curve()
        ths = self._thresholds
        tpr = self._tpr
        tnr = 1.0 - self._fpr
        n = len(self.y_true)
        ppv_vals = np.zeros(len(ths))
        npv_vals = np.zeros(len(ths))
        for i, th in enumerate(ths):
            pred = (self.y_score >= th).astype(int)
            tp = np.sum((pred == 1) & (self.y_true == 1))
            fp = np.sum((pred == 1) & (self.y_true == 0))
            fn = np.sum((pred == 0) & (self.y_true == 1))
            tn = np.sum((pred == 0) & (self.y_true == 0))
            ppv_vals[i] = tp / max(tp + fp, 1)
            npv_vals[i] = tn / max(tn + fn, 1)
        fig, ax = plt.subplots(figsize=figsize, dpi=150)
        ax.plot(ths, tpr, label="Sensitivity", color="#1f77b4", linewidth=2)
        ax.plot(ths, tnr, label="Specificity", color="#ff7f0e", linewidth=2)
        ax.plot(ths, ppv_vals, label="PPV", color="#2ca02c", linewidth=2, linestyle="--")
        ax.plot(ths, npv_vals, label="NPV", color="#d62728", linewidth=2, linestyle="--")
        ax.set_xlabel("Threshold", fontweight="bold")
        ax.set_ylabel("Metric Value", fontweight="bold")
        ax.set_title("Performance Metrics vs. Threshold", fontweight="bold")
        ax.legend(loc="best", frameon=True, edgecolor="lightgray")
        ax.set_xlim(ths.max(), ths.min())
        ax.set_ylim(-0.02, 1.02)
        ax.grid(True, linestyle=":", alpha=0.4)
        fig.tight_layout()
        return fig

    def report(self) -> str:
        res = self.compute()
        lines = []
        lines.append("ROC Analysis")
        lines.append("=" * 60)
        lines.append(f"Area Under the Curve (AUC): {res['auc']:.3f}")
        lines.append(f"95% CI (DeLong): [{res['auc_ci'][0]:.3f}, {res['auc_ci'][1]:.3f}]")
        lines.append(f"Youden Index J = {res['youden_index']:.3f} at threshold = {res['youden_threshold']:.3f}")
        lines.append("")
        lines.append("At Youden-optimal threshold:")
        lines.append(f"  Sensitivity (TPR): {res['sensitivity']:.3f}")
        lines.append(f"  Specificity (TNR): {res['specificity']:.3f}")
        lines.append(f"  Positive Predictive Value (PPV): {res['ppv']:.3f}")
        lines.append(f"  Negative Predictive Value (NPV): {res['npv']:.3f}")
        lines.append(f"  Positive Likelihood Ratio (LR+): {res['lr_plus']:.3f}")
        lines.append(f"  Negative Likelihood Ratio (LR-): {res['lr_minus']:.3f}")
        lines.append("")
        lines.append(f"Sample: n = {self._n} ({self._n_pos} positive, {self._n_neg} negative)")
        return "\n".join(lines)



class BlandAltman:
    """Bland-Altman method comparison analysis for paired measurements."""

    def __init__(self, method1: np.ndarray, method2: np.ndarray):
        self.m1 = np.asarray(method1, dtype=float).ravel()
        self.m2 = np.asarray(method2, dtype=float).ravel()
        mask = ~(np.isnan(self.m1) | np.isnan(self.m2))
        self.m1 = self.m1[mask]
        self.m2 = self.m2[mask]
        self._n = len(self.m1)
        self._diff = self.m1 - self.m2
        self._mean = (self.m1 + self.m2) / 2.0
        self._log_transformed = False

    def compute(self) -> dict:
        n = self._n
        if n < 3:
            return {"bias": np.nan, "sd_of_differences": np.nan,
                    "lower_loa": np.nan, "upper_loa": np.nan,
                    "loa_ci": (np.nan, np.nan, np.nan, np.nan),
                    "correlation": np.nan, "proportional_bias_p": np.nan,
                    "log_transformed": False}

        diff = self._diff
        mean_val = self._mean
        bias = float(np.mean(diff))
        sd_diff = float(np.std(diff, ddof=1))

        slope, intercept, r_val, p_val, se_slope = self._proportional_bias_test()
        proportional_p = float(p_val)

        if proportional_p < 0.05 and np.all(self.m1 > 0) and np.all(self.m2 > 0):
            log_m1 = np.log(self.m1)
            log_m2 = np.log(self.m2)
            log_diff = log_m1 - log_m2
            log_mean = (log_m1 + log_m2) / 2.0
            log_bias = float(np.mean(log_diff))
            log_sd = float(np.std(log_diff, ddof=1))
            lower_loa_log = log_bias - _Z_95 * log_sd
            upper_loa_log = log_bias + _Z_95 * log_sd

            ratio_bias = float(np.exp(log_bias))
            lower_loa = float(np.exp(lower_loa_log))
            upper_loa = float(np.exp(upper_loa_log))
            bias = ratio_bias
            sd_diff = log_sd
            self._log_transformed = True

            lower_loa_se = np.sqrt(3.0 * log_sd ** 2 / n)
            upper_loa_se = lower_loa_se
            t_crit = t_dist.ppf(0.975, n - 1)
            loa_ci_lower = (float(np.exp(lower_loa_log - t_crit * lower_loa_se)),
                            float(np.exp(lower_loa_log + t_crit * lower_loa_se)))
            loa_ci_upper = (float(np.exp(upper_loa_log - t_crit * upper_loa_se)),
                            float(np.exp(upper_loa_log + t_crit * upper_loa_se)))
            loa_ci = (loa_ci_lower[0], loa_ci_lower[1], loa_ci_upper[0], loa_ci_upper[1])

            corr, _ = pearsonr(log_m1, log_m2)
        else:
            self._log_transformed = False
            lower_loa = bias - _Z_95 * sd_diff
            upper_loa = bias + _Z_95 * sd_diff

            se_loa = np.sqrt(sd_diff ** 2 * (1.0 / n + _Z_95 ** 2 / (2.0 * (n - 1))))
            t_crit = t_dist.ppf(0.975, n - 1)
            loa_ci_lower = (lower_loa - t_crit * se_loa, lower_loa + t_crit * se_loa)
            loa_ci_upper = (upper_loa - t_crit * se_loa, upper_loa + t_crit * se_loa)
            loa_ci = (loa_ci_lower[0], loa_ci_lower[1], loa_ci_upper[0], loa_ci_upper[1])

            corr, _ = pearsonr(self.m1, self.m2)

        return {
            "bias": bias,
            "sd_of_differences": sd_diff,
            "lower_loa": lower_loa,
            "upper_loa": upper_loa,
            "loa_ci": loa_ci,
            "correlation": float(corr),
            "proportional_bias_p": proportional_p,
            "log_transformed": self._log_transformed,
        }

    def _proportional_bias_test(self) -> Tuple[float, float, float, float, float]:
        x = self._mean
        y = self._diff
        n = len(x)
        if n < 3:
            return 0.0, 0.0, 0.0, 1.0, 0.0
        sx = np.sum(x)
        sy = np.sum(y)
        sxx = np.sum(x ** 2)
        sxy = np.sum(x * y)
        slope = (n * sxy - sx * sy) / max(n * sxx - sx ** 2, 1e-15)
        intercept = (sy - slope * sx) / n
        y_pred = intercept + slope * x
        resid = y - y_pred
        mse = np.sum(resid ** 2) / max(n - 2, 1)
        se_slope = np.sqrt(mse / max(sxx - sx ** 2 / n, 1e-15))
        t_val = slope / max(se_slope, 1e-15)
        p_val = 2.0 * (1.0 - t_dist.cdf(abs(t_val), max(n - 2, 1)))
        _, r_val = pearsonr(x, y) if n > 2 else (0.0, 1.0)
        return slope, intercept, r_val, p_val, se_slope

    def plot(self, figsize: Tuple[float, float] = (8, 6)) -> plt.Figure:
        res = self.compute()
        n = self._n
        if self._log_transformed:
            x = (np.log(self.m1) + np.log(self.m2)) / 2.0
            y = np.log(self.m1) - np.log(self.m2)
            bias = float(np.mean(y))
            sd = float(np.std(y, ddof=1))
            loa_low = bias - _Z_95 * sd
            loa_high = bias + _Z_95 * sd
            ylabel = "log(Method1) - log(Method2)"
            xlabel = "Mean of log(Method1) and log(Method2)"
        else:
            x = self._mean
            y = self._diff
            bias = res["bias"]
            sd = res["sd_of_differences"]
            loa_low = res["lower_loa"]
            loa_high = res["upper_loa"]
            ylabel = "Method1 - Method2"
            xlabel = "Mean of Method1 and Method2"

        fig, ax = plt.subplots(figsize=figsize, dpi=150)
        ax.scatter(x, y, alpha=0.6, edgecolors="k", s=40, zorder=3)
        ax.axhline(bias, color="#1f77b4", linewidth=1.5, linestyle="-", label=f"Bias = {bias:.3f}")
        ax.axhline(loa_low, color="#d62728", linewidth=1.2, linestyle="--",
                   label=f"Lower LOA = {loa_low:.3f}")
        ax.axhline(loa_high, color="#d62728", linewidth=1.2, linestyle="--",
                   label=f"Upper LOA = {loa_high:.3f}")

        slope, intercept, _, p_val, _ = self._proportional_bias_test()
        if p_val < 0.05 and not self._log_transformed:
            x_line = np.linspace(x.min(), x.max(), 100)
            y_line = intercept + slope * x_line
            ax.plot(x_line, y_line, color="#2ca02c", linewidth=1, linestyle=":",
                    label=f"Prop. bias (p={p_val:.3f})")

        y_lim = ax.get_ylim()
        ax.fill_between([-1e10, 1e10], loa_low, loa_high, alpha=0.05, color="#2ca02c")
        ax.set_xlabel(xlabel, fontweight="bold")
        ax.set_ylabel(ylabel, fontweight="bold")
        ax.set_title("Bland-Altman Plot", fontweight="bold")
        ax.legend(loc="best", frameon=True, edgecolor="lightgray")
        ax.grid(True, linestyle=":", alpha=0.4)
        ax.text(0.02, 0.98, f"n = {n}",
                transform=ax.transAxes, fontsize=9, va="top",
                bbox=dict(facecolor="white", alpha=0.8, edgecolor="lightgray", boxstyle="round,pad=0.3"))
        fig.tight_layout()
        return fig

    def report(self) -> str:
        res = self.compute()
        lines = []
        lines.append("Bland-Altman Method Comparison")
        lines.append("=" * 60)
        if res["log_transformed"]:
            lines.append("Log-transformed analysis (proportional bias detected)")
            lines.append(f"  Ratio of geometric means (bias): {res['bias']:.3f}")
            lines.append(f"  SD of log-differences: {res['sd_of_differences']:.3f}")
            lines.append(f"  Limits of Agreement (ratio scale):")
            lines.append(f"    Lower: {res['lower_loa']:.3f}")
            lines.append(f"    Upper: {res['upper_loa']:.3f}")
        else:
            lines.append(f"  Bias (mean difference): {res['bias']:.3f}")
            lines.append(f"  SD of differences: {res['sd_of_differences']:.3f}")
            lines.append(f"  Limits of Agreement:")
            lines.append(f"    Lower: {res['lower_loa']:.3f}")
            lines.append(f"    Upper: {res['upper_loa']:.3f}")
        lines.append(f"  95% CI for Lower LOA: [{res['loa_ci'][0]:.3f}, {res['loa_ci'][1]:.3f}]")
        lines.append(f"  95% CI for Upper LOA: [{res['loa_ci'][2]:.3f}, {res['loa_ci'][3]:.3f}]")
        lines.append(f"  Pearson r (methods): {res['correlation']:.3f}")
        lines.append(f"  Proportional bias p: {res['proportional_bias_p']:.4f} "
                     f"{'(significant)' if res['proportional_bias_p'] < 0.05 else '(not significant)'}")
        lines.append(f"  n = {self._n}")
        return "\n".join(lines)



class IntraclassCorrelation:
    """Intraclass correlation coefficient (ICC) per Shrout & Fleiss (1979)."""

    def __init__(self, data: pd.DataFrame, raters: Optional[str] = None, scores: Optional[str] = None):
        if raters is not None and scores is not None:
            self._data = data.pivot_table(index=data.index if raters not in data.columns else None,
                                          columns=raters, values=scores, aggfunc="first")
            if isinstance(self._data.index, pd.MultiIndex) or self._data.index.name is None:
                self._data = self._data.reset_index(drop=True)
        else:
            self._data = data.copy()
        self._matrix = self._data.values.astype(float)
        self._n = self._matrix.shape[0]
        self._k = self._matrix.shape[1]

    def compute(self, icc_type: str = 'ICC2k') -> dict:
        n = self._n
        k = self._k
        X = self._matrix
        if n < 2 or k < 2:
            return {"icc": np.nan, "icc_ci": (np.nan, np.nan), "F_value": np.nan,
                    "F_df1": np.nan, "F_df2": np.nan, "F_p": np.nan,
                    "icc_type": icc_type, "interpretation": "Insufficient data"}

        grand_mean = np.nanmean(X)
        subj_means = np.nanmean(X, axis=1)
        rater_means = np.nanmean(X, axis=0)

        SS_total = np.nansum((X - grand_mean) ** 2)
        SS_subjects = k * np.nansum((subj_means - grand_mean) ** 2)
        SS_raters = n * np.nansum((rater_means - grand_mean) ** 2)
        SS_within = np.nansum((X - subj_means.reshape(-1, 1)) ** 2)
        SS_error = SS_total - SS_subjects - SS_raters

        BMS = SS_subjects / max(n - 1, 1)
        WMS = SS_within / max(n * (k - 1), 1)
        if k > 1 and n > 1:
            JMS = SS_raters / (k - 1)
            EMS = SS_error / max((n - 1) * (k - 1), 1)
        else:
            JMS = 0.0
            EMS = 0.0

        alpha = 0.05
        icc_type = icc_type.upper()

        if icc_type == 'ICC1':
            icc = (BMS - WMS) / max(BMS + (k - 1) * WMS, 1e-15)
            F_val = BMS / max(WMS, 1e-15)
            df1 = n - 1
            df2 = n * (k - 1)
        elif icc_type == 'ICC1K':
            icc = (BMS - WMS) / max(BMS, 1e-15)
            F_val = BMS / max(WMS, 1e-15)
            df1 = n - 1
            df2 = n * (k - 1)
        elif icc_type == 'ICC2':
            icc = (BMS - EMS) / max(BMS + (k - 1) * EMS + k * (JMS - EMS) / max(n, 1), 1e-15)
            F_val = BMS / max(EMS, 1e-15)
            df1 = n - 1
            df2 = max((n - 1) * (k - 1), 1)
        elif icc_type == 'ICC2K':
            icc = (BMS - EMS) / max(BMS + (JMS - EMS) / max(n, 1), 1e-15)
            F_val = BMS / max(EMS, 1e-15)
            df1 = n - 1
            df2 = max((n - 1) * (k - 1), 1)
        elif icc_type == 'ICC3':
            icc = (BMS - EMS) / max(BMS + (k - 1) * EMS, 1e-15)
            F_val = BMS / max(EMS, 1e-15)
            df1 = n - 1
            df2 = max((n - 1) * (k - 1), 1)
        elif icc_type == 'ICC3K':
            icc = (BMS - EMS) / max(BMS, 1e-15)
            F_val = BMS / max(EMS, 1e-15)
            df1 = n - 1
            df2 = max((n - 1) * (k - 1), 1)
        else:
            raise ValueError(f"Unknown ICC type: {icc_type}. Choose from ICC1, ICC2, ICC3, ICC1k, ICC2k, ICC3k")

        F_p = 1.0 - f_dist.cdf(F_val, df1, df2)
        F_p = max(F_p, 1e-15)

        F_crit_lower = f_dist.ppf(1.0 - alpha / 2.0, df1, df2)
        F_L = F_val / max(F_crit_lower, 1e-15)
        F_U = F_val * f_dist.ppf(1.0 - alpha / 2.0, df2, df1)

        if icc_type in ('ICC1', 'ICC2', 'ICC3'):
            icc_lower = (F_L - 1.0) / max(F_L + k - 1.0, 1e-15)
            icc_upper = (F_U - 1.0) / max(F_U + k - 1.0, 1e-15)
        else:
            icc_lower = 1.0 - 1.0 / max(F_L, 1e-15)
            icc_upper = 1.0 - 1.0 / max(F_U, 1e-15)

        icc_lower = max(-1.0, icc_lower)
        icc_upper = min(1.0, icc_upper)

        abs_icc = abs(icc)
        if abs_icc < 0.5:
            interp = "Poor"
        elif abs_icc < 0.75:
            interp = "Moderate"
        elif abs_icc < 0.9:
            interp = "Good"
        else:
            interp = "Excellent"

        return {
            "icc": icc,
            "icc_ci": (icc_lower, icc_upper),
            "F_value": F_val,
            "F_df1": df1,
            "F_df2": df2,
            "F_p": F_p,
            "icc_type": icc_type,
            "interpretation": interp,
        }

    def report(self) -> str:
        res = self.compute()
        lines = []
        lines.append("Intraclass Correlation Coefficient (ICC)")
        lines.append("=" * 60)
        lines.append(f"Type: {res['icc_type']} (Shrout & Fleiss, 1979)")
        lines.append(f"ICC = {res['icc']:.4f}")
        lines.append(f"95% CI: [{res['icc_ci'][0]:.4f}, {res['icc_ci'][1]:.4f}]")
        lines.append(f"F({res['F_df1']}, {res['F_df2']}) = {res['F_value']:.3f}, p = {res['F_p']:.4f}")
        lines.append(f"Interpretation (Koo & Li, 2016): {res['interpretation']}")
        lines.append(f"Subjects: {self._n}, Raters: {self._k}")
        return "\n".join(lines)



class DiagnosticAccuracy:
    """Binary diagnostic accuracy metrics with confidence intervals."""

    def __init__(self, y_true: np.ndarray, y_pred: np.ndarray):
        self.y_true = np.asarray(y_true, dtype=int).ravel()
        self.y_pred = np.asarray(y_pred, dtype=int).ravel()
        mask = ~(np.isnan(self.y_true) | np.isnan(self.y_pred))
        self.y_true = self.y_true[mask]
        self.y_pred = self.y_pred[mask]
        self._tp = np.sum((self.y_pred == 1) & (self.y_true == 1))
        self._fp = np.sum((self.y_pred == 1) & (self.y_true == 0))
        self._fn = np.sum((self.y_pred == 0) & (self.y_true == 1))
        self._tn = np.sum((self.y_pred == 0) & (self.y_true == 0))
        self._n = len(self.y_true)

    def compute(self) -> dict:
        tp, fp, fn, tn = self._tp, self._fp, self._fn, self._tn
        n = self._n
        accuracy = (tp + tn) / max(n, 1)
        sensitivity = tp / max(tp + fn, 1)
        specificity = tn / max(tn + fp, 1)
        ppv = tp / max(tp + fp, 1)
        npv = tn / max(tn + fn, 1)
        lr_plus = sensitivity / max(1.0 - specificity, 1e-15)
        lr_minus = (1.0 - sensitivity) / max(specificity, 1e-15)
        prevalence = (tp + fn) / max(n, 1)
        f1 = 2 * ppv * sensitivity / max(ppv + sensitivity, 1e-15)

        num = (tp * tn - fp * fn)
        den = np.sqrt(float((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn)))
        mcc = num / max(den, 1e-15)

        _, acc_l, acc_u = _wilson_ci(int(tp + tn), n)
        _, sens_l, sens_u = _wilson_ci(int(tp), int(tp + fn))
        _, spec_l, spec_u = _wilson_ci(int(tn), int(tn + fp))
        _, ppv_l, ppv_u = _wilson_ci(int(tp), int(tp + fp))
        _, npv_l, npv_u = _wilson_ci(int(tn), int(tn + fn))

        return {
            "accuracy": float(accuracy),
            "sensitivity": float(sensitivity),
            "specificity": float(specificity),
            "ppv": float(ppv),
            "npv": float(npv),
            "lr_plus": float(lr_plus),
            "lr_minus": float(lr_minus),
            "prevalence": float(prevalence),
            "accuracy_ci": (float(acc_l), float(acc_u)),
            "sensitivity_ci": (float(sens_l), float(sens_u)),
            "specificity_ci": (float(spec_l), float(spec_u)),
            "ppv_ci": (float(ppv_l), float(ppv_u)),
            "npv_ci": (float(npv_l), float(npv_u)),
            "f1_score": float(f1),
            "matthews_corrcoef": float(mcc),
            "tp": int(tp), "fp": int(fp), "fn": int(fn), "tn": int(tn),
            "n": n,
        }

    def confusion_matrix_plot(self, figsize: Tuple[float, float] = (5, 5)) -> plt.Figure:
        cm = np.array([[self._tp, self._fp], [self._fn, self._tn]])
        fig, ax = plt.subplots(figsize=figsize, dpi=150)
        im = ax.imshow(cm, cmap="Blues", aspect="auto")
        cbar = fig.colorbar(im, ax=ax, shrink=0.8)
        cbar.set_label("Count")
        labels = [["TP", "FP"], ["FN", "TN"]]
        for i in range(2):
            for j in range(2):
                color = "white" if cm[i, j] > cm.max() / 2 else "black"
                ax.text(j, i, f"{labels[i][j]}\n{cm[i, j]}", ha="center", va="center",
                        fontsize=14, fontweight="bold", color=color)
        ax.set_xticks([0, 1])
        ax.set_yticks([0, 1])
        ax.set_xticklabels(["Predicted Pos", "Predicted Neg"])
        ax.set_yticklabels(["Actual Pos", "Actual Neg"])
        ax.set_xlabel("Predicted", fontweight="bold")
        ax.set_ylabel("Actual", fontweight="bold")
        ax.set_title("Confusion Matrix", fontweight="bold")
        fig.tight_layout()
        return fig

    def report(self) -> str:
        res = self.compute()
        lines = []
        lines.append("Diagnostic Accuracy")
        lines.append("=" * 60)
        lines.append(f"Accuracy: {res['accuracy']:.3f} (95% CI: [{res['accuracy_ci'][0]:.3f}, {res['accuracy_ci'][1]:.3f}])")
        lines.append(f"Sensitivity: {res['sensitivity']:.3f} (95% CI: [{res['sensitivity_ci'][0]:.3f}, {res['sensitivity_ci'][1]:.3f}])")
        lines.append(f"Specificity: {res['specificity']:.3f} (95% CI: [{res['specificity_ci'][0]:.3f}, {res['specificity_ci'][1]:.3f}])")
        lines.append(f"PPV: {res['ppv']:.3f} (95% CI: [{res['ppv_ci'][0]:.3f}, {res['ppv_ci'][1]:.3f}])")
        lines.append(f"NPV: {res['npv']:.3f} (95% CI: [{res['npv_ci'][0]:.3f}, {res['npv_ci'][1]:.3f}])")
        lines.append(f"LR+: {res['lr_plus']:.3f}")
        lines.append(f"LR-: {res['lr_minus']:.3f}")
        lines.append(f"Prevalence: {res['prevalence']:.3f}")
        lines.append(f"F1 Score: {res['f1_score']:.3f}")
        lines.append(f"Matthews r: {res['matthews_corrcoef']:.3f}")
        lines.append("")
        lines.append(f"Confusion Matrix: TP={res['tp']}, FP={res['fp']}, FN={res['fn']}, TN={res['tn']} (n={res['n']})")
        return "\n".join(lines)



class DiagnosticMetaAnalysis:
    """Meta-analysis of diagnostic test accuracy with SROC curve."""

    def __init__(self, studies_df: pd.DataFrame):
        required = ["study", "tp", "fp", "fn", "tn"]
        for col in required:
            if col not in studies_df.columns:
                raise ValueError(f"Required column '{col}' not found")
        self.df = studies_df.copy()
        for col in ["tp", "fp", "fn", "tn"]:
            self.df[col] = pd.to_numeric(self.df[col], errors="coerce").fillna(0).astype(int)
        self.df = self.df[self.df["tp"] + self.df["fn"] > 0]
        self.df = self.df[self.df["fp"] + self.df["tn"] > 0]
        self._n_studies = len(self.df)

    def compute(self) -> dict:
        df = self.df
        n = self._n_studies
        if n == 0:
            return {"pooled_sensitivity": np.nan, "pooled_specificity": np.nan,
                    "pooled_dor": np.nan, "heterogeneity_I2": np.nan,
                    "cochran_Q": np.nan, "Q_p": np.nan}

        tpr = df["tp"].values / (df["tp"].values + df["fn"].values).astype(float)
        fpr = df["fp"].values / (df["fp"].values + df["tn"].values).astype(float)
        tpr = np.clip(tpr, 1e-15, 1.0 - 1e-15)
        fpr = np.clip(fpr, 1e-15, 1.0 - 1e-15)

        logit_tpr = _logit(tpr)
        logit_fpr = _logit(fpr)

        fp_safe = np.maximum(df["fp"].values, 1e-15)
        fn_safe = np.maximum(df["fn"].values, 1e-15)
        or_val = (df["tp"].values * df["tn"].values) / (fp_safe * fn_safe)
        or_val = np.clip(or_val, 1e-15, 1e15)
        ln_dor = np.log(or_val)

        var_ln_dor = (1.0 / df["tp"].values + 1.0 / df["fp"].values +
                      1.0 / df["fn"].values + 1.0 / df["tn"].values)
        var_ln_dor = np.clip(var_ln_dor, 1e-15, None)
        w = 1.0 / var_ln_dor

        w_sum = np.sum(w)
        ln_dor_fe = np.sum(w * ln_dor) / max(w_sum, 1e-15)

        Q = np.sum(w * (ln_dor - ln_dor_fe) ** 2)
        Q_p = 1.0 - chi2.cdf(Q, n - 1) if n > 1 else 1.0

        tau2 = max((Q - (n - 1)) / max(Q - n + 1 + w_sum - np.sum(w ** 2) / max(w_sum, 1e-15), 1e-15), 0.0)

        w_star = 1.0 / (var_ln_dor + tau2)
        w_star_sum = np.sum(w_star)
        ln_dor_re = np.sum(w_star * ln_dor) / max(w_star_sum, 1e-15)
        se_ln_dor_re = np.sqrt(1.0 / max(w_star_sum, 1e-15))
        pooled_dor = np.exp(ln_dor_re)
        dor_ci = (np.exp(ln_dor_re - _Z_95 * se_ln_dor_re),
                  np.exp(ln_dor_re + _Z_95 * se_ln_dor_re))

        I2 = max((Q - (n - 1)) / max(Q, 1e-15) * 100.0, 0.0) if n > 1 else 0.0

        w_sens = 1.0 / (tpr * (1.0 - tpr))
        w_sens = np.clip(w_sens, 1e-15, None)
        w_spec = 1.0 / (fpr * (1.0 - fpr))
        w_spec = np.clip(w_spec, 1e-15, None)

        logit_tpr_pooled = np.sum(w_sens * logit_tpr) / max(np.sum(w_sens), 1e-15)
        logit_fpr_pooled = np.sum(w_spec * logit_fpr) / max(np.sum(w_spec), 1e-15)
        pooled_tpr = _invlogit(logit_tpr_pooled)
        pooled_fpr = _invlogit(logit_fpr_pooled)

        return {
            "pooled_sensitivity": float(pooled_tpr),
            "pooled_specificity": float(1.0 - pooled_fpr),
            "pooled_dor": float(pooled_dor),
            "dor_ci": (float(dor_ci[0]), float(dor_ci[1])),
            "heterogeneity_I2": float(I2),
            "cochran_Q": float(Q),
            "Q_p": float(Q_p),
            "tau2": float(tau2),
            "n_studies": n,
        }

    def _sroc_params(self):
        df = self.df
        tpr = df["tp"].values / (df["tp"].values + df["fn"].values).astype(float)
        fpr = df["fp"].values / (df["fp"].values + df["tn"].values).astype(float)
        tpr = np.clip(tpr, 1e-15, 1.0 - 1e-15)
        fpr = np.clip(fpr, 1e-15, 1.0 - 1e-15)

        logit_tpr = _logit(tpr)
        logit_fpr = _logit(fpr)

        D = logit_tpr - logit_fpr
        S = logit_tpr + logit_fpr

        n = len(D)
        sx = np.sum(S)
        sy = np.sum(D)
        sxx = np.sum(S ** 2)
        sxy = np.sum(S * D)
        beta = (n * sxy - sx * sy) / max(n * sxx - sx ** 2, 1e-15)
        alpha = (sy - beta * sx) / n
        return alpha, beta

    def forest_plot(self, figsize: Tuple[float, float] = (10, 8)) -> plt.Figure:
        df = self.df
        n = len(df)
        fp_safe = np.maximum(df["fp"].values, 1e-15)
        fn_safe = np.maximum(df["fn"].values, 1e-15)
        or_val = (df["tp"].values * df["tn"].values) / (fp_safe * fn_safe)
        or_val = np.clip(or_val, 1e-15, 1e15)
        ln_dor = np.log(or_val)
        se_ln = np.sqrt(1.0 / df["tp"].values + 1.0 / df["fp"].values +
                        1.0 / df["fn"].values + 1.0 / df["tn"].values)
        se_ln = np.clip(se_ln, 1e-15, None)
        lower = np.exp(ln_dor - _Z_95 * se_ln)
        upper = np.exp(ln_dor + _Z_95 * se_ln)

        order = np.argsort(or_val)[::-1]
        or_sorted = or_val[order]
        lower_sorted = lower[order]
        upper_sorted = upper[order]
        names = df["study"].values[order]
        se_ln_sorted = se_ln[order]

        pooled = self.compute()
        pooled_or = pooled["pooled_dor"]
        pooled_lower = pooled["dor_ci"][0]
        pooled_upper = pooled["dor_ci"][1]

        fig, ax = plt.subplots(figsize=figsize, dpi=150)
        y_pos = np.arange(n)
        for i in range(n):
            color = "#1f77b4"
            ax.errorbar(or_sorted[i], y_pos[i], xerr=[[or_sorted[i] - lower_sorted[i]],
                                                       [upper_sorted[i] - or_sorted[i]]],
                        fmt="o", color=color, capsize=3, capthick=1, markersize=6)
            ax.text(or_sorted[i], y_pos[i] + 0.35, f"{or_sorted[i]:.2f} [{lower_sorted[i]:.2f}, {upper_sorted[i]:.2f}]",
                    ha="center", va="bottom", fontsize=7)

        ax.axvline(1, color="gray", linewidth=0.8, linestyle="--", alpha=0.6)
        ax.axvline(pooled_or, color="#d62728", linewidth=1.5, linestyle="-",
                   label=f"Pooled DOR = {pooled_or:.2f} [{pooled_lower:.2f}, {pooled_upper:.2f}]")

        diamond_x = [pooled_lower, pooled_or, pooled_upper, pooled_or]
        diamond_y = [-0.5, -0.3, -0.5, -0.7]
        ax.fill(diamond_x, diamond_y, color="#d62728", alpha=0.6, edgecolor="black")

        ax.set_yticks(y_pos)
        ax.set_yticklabels(names, fontsize=8)
        ax.invert_yaxis()
        ax.set_xscale("log")
        ax.set_xlabel("Diagnostic Odds Ratio (log scale)", fontweight="bold")
        ax.set_title("Forest Plot: Diagnostic Odds Ratio", fontweight="bold")
        ax.axhline(-0.5, color="gray", linewidth=0.5)
        ax.legend(loc="lower right", frameon=True, edgecolor="lightgray")
        ax.grid(True, axis="x", linestyle=":", alpha=0.4)
        ax.text(0.02, 0.98, f"I² = {pooled['heterogeneity_I2']:.1f}%  |  Q = {pooled['cochran_Q']:.2f}, p = {pooled['Q_p']:.4f}",
                transform=ax.transAxes, fontsize=8, va="top",
                bbox=dict(facecolor="white", alpha=0.8, edgecolor="lightgray", boxstyle="round,pad=0.3"))
        fig.tight_layout()
        return fig

    def sroc_plot(self, figsize: Tuple[float, float] = (8, 8)) -> plt.Figure:
        df = self.df
        tpr = df["tp"].values / (df["tp"].values + df["fn"].values).astype(float)
        fpr = df["fp"].values / (df["fp"].values + df["tn"].values).astype(float)
        n = df["tp"].values + df["fn"].values

        alpha, beta = self._sroc_params()

        fpr_grid = np.linspace(1e-6, 1.0 - 1e-6, 200)
        logit_fpr_grid = _logit(fpr_grid)
        sroc_logit = alpha / max(1.0 - beta, 1e-15) + logit_fpr_grid * (1.0 + beta) / max(1.0 - beta, 1e-15)
        sroc_tpr = _invlogit(sroc_logit)
        sroc_tpr = np.clip(sroc_tpr, 0, 1)

        fig, ax = plt.subplots(figsize=figsize, dpi=150)
        scatter = ax.scatter(fpr, tpr, s=n / n.max() * 150 + 30, c=n, cmap="viridis",
                             edgecolors="black", linewidth=0.5, alpha=0.8, zorder=5)
        cbar = fig.colorbar(scatter, ax=ax, label="Sample size (n)")
        ax.plot(fpr_grid, sroc_tpr, color="#d62728", linewidth=2, label="SROC curve")
        ax.plot([0, 1], [0, 1], "k--", linewidth=0.8, alpha=0.4, label="Chance")
        ax.set_xlabel("1 - Specificity (False Positive Rate)", fontweight="bold")
        ax.set_ylabel("Sensitivity (True Positive Rate)", fontweight="bold")
        ax.set_title("Summary ROC (SROC) Curve", fontweight="bold")
        ax.legend(loc="lower right", frameon=True, edgecolor="lightgray")
        ax.set_xlim(-0.02, 1.02)
        ax.set_ylim(-0.02, 1.02)
        ax.grid(True, linestyle=":", alpha=0.4)

        pooled = self.compute()
        ax.text(0.02, 0.98,
                f"Pooled Sens = {pooled['pooled_sensitivity']:.3f}\n"
                f"Pooled Spec = {pooled['pooled_specificity']:.3f}\n"
                f"Pooled DOR = {pooled['pooled_dor']:.2f}\n"
                f"I² = {pooled['heterogeneity_I2']:.1f}%\n"
                f"n studies = {pooled['n_studies']}",
                transform=ax.transAxes, fontsize=8, va="top",
                bbox=dict(facecolor="white", alpha=0.8, edgecolor="lightgray", boxstyle="round,pad=0.3"))

        for i in range(len(df)):
            ax.annotate(df["study"].values[i], (fpr[i], tpr[i]),
                        fontsize=6, xytext=(5, 5), textcoords="offset points",
                        alpha=0.7)

        fig.tight_layout()
        return fig
