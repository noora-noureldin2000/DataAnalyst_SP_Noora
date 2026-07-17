import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
from typing import Optional, Dict, Any, List, Tuple
import warnings
from io import BytesIO

warnings.filterwarnings("ignore", category=DeprecationWarning)


class ReportGenerator:
    def __init__(self, results: dict, data: Optional[pd.DataFrame] = None):
        self.results = results
        self.data = data
        self.test_name = results.get("test", results.get("test_name", "Unknown"))
        self.statistic = results.get("statistic", results.get("stat", None))
        self.p_value = results.get("p_value", results.get("p", None))
        self.effect_size = results.get("effect_size", results.get("d", results.get("r", None)))
        self.effect_size_ci = results.get("effect_size_ci", results.get("ci", None))
        self.params = results.get("params", results.get("parameters", {}))
        self.variables = results.get("variables", results.get("var", []))

    def apa_table(self, decimals: int = 4) -> str:
        return self._auto_table(decimals)

    def _auto_table(self, decimals: int) -> str:
        r = self.results
        if "coefficients" in r or "Coefficient" in str(r.keys()):
            return self._regression_table(decimals)
        if "auc" in r:
            return self._roc_table(decimals)
        if "bias" in r and "lower_loa" in r:
            return self._bland_altman_table(decimals)
        if "icc" in r and "icc_type" in r:
            return self._icc_table(decimals)
        if "accuracy" in r and "sensitivity" in r:
            return self._diagnostic_accuracy_table(decimals)
        if "bf10" in r:
            return self._bayes_factor_table(decimals)
        if "group1" in r or "level1" in r:
            return self._pairwise_table(decimals)
        return self._generic_table(decimals)

    def _regression_table(self, decimals: int) -> str:
        r = self.results
        lines = []
        lines.append("| Variable | Coefficient | SE | z/t | p | 95% CI | Sig |")
        lines.append("|----------|------------|----|-----|---|--------|-----|")
        coefs = r.get("coefficients", r.get("fe_params", {}))
        ses = r.get("standard_errors", r.get("bse_fe", {}))
        zs = r.get("z_values", r.get("tvalues", {}))
        ps = r.get("p_values", {})
        ci = r.get("conf_int", {})
        if isinstance(coefs, pd.Series):
            for var in coefs.index:
                se_val = ses.get(var, ses.loc[var]) if hasattr(ses, "loc") else ses.get(var, 0)
                z_val = zs.get(var, zs.loc[var]) if hasattr(zs, "loc") else zs.get(var, 0)
                p_val = ps.get(var, ps.loc[var]) if hasattr(ps, "loc") else ps.get(var, 1)
                ci_lower = ci.loc[var, 0] if hasattr(ci, "loc") else ci.get(var, [0, 0])[0]
                ci_upper = ci.loc[var, 1] if hasattr(ci, "loc") else ci.get(var, [0, 0])[1]
                p_str = f"< .001" if p_val < 0.001 else f"{p_val:.{decimals}f}"
                sig = "***" if p_val < 0.001 else "**" if p_val < 0.01 else "*" if p_val < 0.05 else "ns"
                lines.append(f"| {var} | {coefs[var]:.{decimals}f} | {se_val:.{decimals}f} | {z_val:.2f} | {p_str} | [{ci_lower:.{decimals}f}, {ci_upper:.{decimals}f}] | {sig} |")
        return "\n".join(lines)

    def _roc_table(self, decimals: int) -> str:
        r = self.results
        auc = r.get("auc", 0)
        auc_ci = r.get("auc_ci", [0, 0])
        youden = r.get("youden_index", 0)
        thr = r.get("youden_threshold", 0)
        sens = r.get("sensitivity", 0)
        spec = r.get("specificity", 0)
        ppv = r.get("ppv", 0)
        npv = r.get("npv", 0)
        lrp = r.get("lr_plus", 0)
        lrm = r.get("lr_minus", 0)
        lines = []
        lines.append("| Metric | Value | 95% CI |")
        lines.append("|--------|-------|--------|")
        lines.append(f"| AUC | {auc:.{decimals}f} | [{auc_ci[0]:.{decimals}f}, {auc_ci[1]:.{decimals}f}] |")
        lines.append(f"| Youden's J | {youden:.{decimals}f} | |")
        lines.append(f"| Optimal Threshold | {thr:.{decimals}f} | |")
        lines.append(f"| Sensitivity | {sens:.{decimals}f} | |")
        lines.append(f"| Specificity | {spec:.{decimals}f} | |")
        lines.append(f"| PPV | {ppv:.{decimals}f} | |")
        lines.append(f"| NPV | {npv:.{decimals}f} | |")
        lines.append(f"| LR+ | {lrp:.{decimals}f} | |")
        lines.append(f"| LR- | {lrm:.{decimals}f} | |")
        return "\n".join(lines)

    def _bland_altman_table(self, decimals: int) -> str:
        r = self.results
        bias = r.get("bias", 0)
        sd = r.get("sd_of_differences", 0)
        loa_l = r.get("lower_loa", 0)
        loa_u = r.get("upper_loa", 0)
        loa_ci = r.get("loa_ci", {})
        prop_p = r.get("proportional_bias_p", 1)
        lines = []
        lines.append("| Metric | Value | 95% CI |")
        lines.append("|--------|-------|--------|")
        lines.append(f"| Bias (mean diff) | {bias:.{decimals}f} | |")
        lines.append(f"| SD of differences | {sd:.{decimals}f} | |")
        lines.append(f"| Lower LoA | {loa_l:.{decimals}f} | [{loa_ci.get('lower_loa_ci', [0,0])[0]:.{decimals}f}, {loa_ci.get('lower_loa_ci', [0,0])[1]:.{decimals}f}] |")
        lines.append(f"| Upper LoA | {loa_u:.{decimals}f} | [{loa_ci.get('upper_loa_ci', [0,0])[0]:.{decimals}f}, {loa_ci.get('upper_loa_ci', [0,0])[1]:.{decimals}f}] |")
        lines.append(f"| Proportional bias p | {prop_p:.{decimals}f} | |")
        return "\n".join(lines)

    def _icc_table(self, decimals: int) -> str:
        r = self.results
        icc = r.get("icc", 0)
        icc_ci = r.get("icc_ci", [0, 0])
        icc_type = r.get("icc_type", "ICC")
        interp = r.get("interpretation", "")
        f_val = r.get("F_value", 0)
        f_p = r.get("F_p", 1)
        lines = []
        lines.append("| Metric | Value |")
        lines.append("|--------|-------|")
        lines.append(f"| {icc_type} | {icc:.{decimals}f} |")
        lines.append(f"| 95% CI | [{icc_ci[0]:.{decimals}f}, {icc_ci[1]:.{decimals}f}] |")
        lines.append(f"| F({r.get('F_df1',0)},{r.get('F_df2',0)}) | {f_val:.2f} |")
        lines.append(f"| F p-value | {f_p:.{decimals}f} |")
        lines.append(f"| Interpretation | {interp} |")
        return "\n".join(lines)

    def _diagnostic_accuracy_table(self, decimals: int) -> str:
        r = self.results
        lines = []
        lines.append("| Metric | Value | 95% CI |")
        lines.append("|--------|-------|--------|")
        for key, label in [("accuracy", "Accuracy"), ("sensitivity", "Sensitivity"), ("specificity", "Specificity"), ("ppv", "PPV"), ("npv", "NPV")]:
            val = r.get(key, 0)
            ci = r.get(f"{key}_ci", [0, 0])
            lines.append(f"| {label} | {val:.{decimals}f} | [{ci[0]:.{decimals}f}, {ci[1]:.{decimals}f}] |")
        lines.append(f"| LR+ | {r.get('lr_plus', 0):.{decimals}f} | |")
        lines.append(f"| LR- | {r.get('lr_minus', 0):.{decimals}f} | |")
        lines.append(f"| F1 Score | {r.get('f1_score', 0):.{decimals}f} | |")
        lines.append(f"| MCC | {r.get('matthews_corrcoef', 0):.{decimals}f} | |")
        return "\n".join(lines)

    def _bayes_factor_table(self, decimals: int) -> str:
        r = self.results
        lines = []
        lines.append("| Metric | Value |")
        lines.append("|--------|-------|")
        lines.append(f"| BF\u2081\u2080 | {r.get('bf10', 0):.{decimals}f} |")
        lines.append(f"| BF\u2080\u2081 | {r.get('bf01', 0):.{decimals}f} |")
        lines.append(f"| log(BF) | {r.get('log_bf', 0):.{decimals}f} |")
        lines.append(f"| Interpretation | {r.get('interpretation', '')} |")
        return "\n".join(lines)

    def _pairwise_table(self, decimals: int) -> str:
        r = self.results
        groups = r.get("pairwise_comparisons", r.get("groups", []))
        if not groups and isinstance(r, dict):
            keys_to_check = ["level1", "level2", "diff", "p_adjusted"]
            if all(k in r for k in keys_to_check):
                groups = [r]
        if not groups:
            return self._generic_table(decimals)
        lines = []
        lines.append("| Comparison | Difference | SE | t | p (adjusted) | Sig |")
        lines.append("|------------|-----------|----|---|--------------|-----|")
        for comp in groups:
            l1 = comp.get("level1", comp.get("group1", ""))
            l2 = comp.get("level2", comp.get("group2", ""))
            diff = comp.get("diff", comp.get("difference", 0))
            se = comp.get("SE", comp.get("se", 0))
            t = comp.get("t", comp.get("statistic", 0))
            p = comp.get("p_adjusted", comp.get("p", 1))
            sig = comp.get("Sig", "ns")
            p_str = f"< .001" if p < 0.001 else f"{p:.{decimals}f}"
            lines.append(f"| {l1} vs {l2} | {diff:.{decimals}f} | {se:.{decimals}f} | {t:.2f} | {p_str} | {sig} |")
        return "\n".join(lines)

    def _generic_table(self, decimals: int) -> str:
        r = self.results
        lines = []
        lines.append(f"| {self.test_name} | Value |")
        lines.append("|--------|-------|")
        if self.statistic is not None:
            lines.append(f"| Test Statistic | {self.statistic:.{decimals}f} |")
        if self.p_value is not None:
            p_str = f"< .001" if self.p_value < 0.001 else f"{self.p_value:.{decimals}f}"
            lines.append(f"| p-value | {p_str} |")
        if self.effect_size is not None:
            lines.append(f"| Effect Size | {self.effect_size:.{decimals}f} |")
        if self.effect_size_ci is not None and len(self.effect_size_ci) == 2:
            lines.append(f"| 95% CI | [{self.effect_size_ci[0]:.{decimals}f}, {self.effect_size_ci[1]:.{decimals}f}] |")
        for k, v in r.items():
            if k not in ("test", "test_name", "statistic", "p_value", "effect_size", "effect_size_ci", "variables", "params", "pairwise_comparisons") and isinstance(v, (int, float)):
                # Both branches must have the trailing pipe for valid Markdown table syntax
                lines.append(f"| {k} | {v:.{decimals}f} |" if isinstance(v, float) else f"| {k} | {v} |")
        return "\n".join(lines)

    def figure(self, figsize=(8, 6)) -> Optional[plt.Figure]:
        r = self.results
        if "auc" in r:
            return self._plot_roc(figsize, r)
        if "bias" in r and "lower_loa" in r:
            return self._plot_bland_altman(figsize, r)
        if "icc" in r:
            return self._plot_icc(figsize, r)
        if "accuracy" in r:
            return self._plot_confusion(figsize, r)
        if "coefficients" in r or "fe_params" in r:
            return self._plot_coefficients(figsize, r)
        if "bf10" in r:
            return self._plot_bf(figsize, r)
        if self.data is not None and len(self.data.columns) >= 2:
            return self._plot_scatter(figsize, r)
        return None

    def _plot_roc(self, figsize, r):
        fig, ax = plt.subplots(1, 1, figsize=figsize)
        fpr = r.get("fpr", np.linspace(0, 1, 100))
        tpr = r.get("tpr", np.linspace(0, 1, 100))
        auc = r.get("auc", 0)
        auc_ci = r.get("auc_ci", [0, 0])
        ax.plot(fpr, tpr, "b-", linewidth=2, label=f"AUC = {auc:.3f} [{auc_ci[0]:.3f}, {auc_ci[1]:.3f}]")
        ax.plot([0, 1], [0, 1], "k--", alpha=0.5, label="Random (AUC = 0.5)")
        ax.fill_between(fpr, tpr, alpha=0.15, color="blue")
        ax.set_xlabel("1 - Specificity (False Positive Rate)", fontsize=12)
        ax.set_ylabel("Sensitivity (True Positive Rate)", fontsize=12)
        ax.set_title("ROC Curve", fontsize=14, fontweight="bold")
        ax.legend(loc="lower right", fontsize=10)
        ax.set_xlim(-0.02, 1.02)
        ax.set_ylim(-0.02, 1.02)
        ax.set_aspect("equal")
        youden = r.get("youden_threshold", None)
        if youden and "fpr_at_threshold" in r:
            fpr_at = r["fpr_at_threshold"]
            tpr_at = r.get("tpr_at_threshold", r.get("sensitivity", 0))
            ax.plot(fpr_at, tpr_at, "ro", markersize=8, label=f"Optimal cutoff = {youden:.2f}")
            ax.legend(loc="lower right", fontsize=10)
        plt.tight_layout()
        return fig

    def _plot_bland_altman(self, figsize, r):
        fig, ax = plt.subplots(1, 1, figsize=figsize)
        means = r.get("means", np.array([0]))
        diffs = r.get("differences", np.array([0]))
        bias = r.get("bias", 0)
        sd = r.get("sd_of_differences", 0)
        loa_l = r.get("lower_loa", 0)
        loa_u = r.get("upper_loa", 0)
        ax.scatter(means, diffs, alpha=0.6, edgecolors="k", linewidth=0.5)
        ax.axhline(bias, color="blue", linestyle="-", linewidth=2, label=f"Bias = {bias:.2f}")
        ax.axhline(loa_l, color="red", linestyle="--", linewidth=1.5, label=f"Lower LoA = {loa_l:.2f}")
        ax.axhline(loa_u, color="red", linestyle="--", linewidth=1.5, label=f"Upper LoA = {loa_u:.2f}")
        ax.fill_between([means.min(), means.max()], loa_l, loa_u, alpha=0.08, color="green")
        prop_p = r.get("proportional_bias_p", 1)
        if prop_p < 0.05 and "regression_line" in r:
            ax.plot(means, r["regression_line"], "g--", linewidth=1, alpha=0.7, label=f"Prop. bias (p={prop_p:.3f})")
        ax.set_xlabel("Mean of Measurements", fontsize=12)
        ax.set_ylabel("Difference", fontsize=12)
        ax.set_title("Bland-Altman Plot", fontsize=14, fontweight="bold")
        ax.legend(loc="upper right", fontsize=9)
        plt.tight_layout()
        return fig

    def _plot_icc(self, figsize, r):
        fig, ax = plt.subplots(1, 1, figsize=figsize)
        icc = r.get("icc", 0)
        icc_ci = r.get("icc_ci", [0, 0])
        icc_type = r.get("icc_type", "ICC")
        interp = r.get("interpretation", "")
        categories = ["Poor", "Fair", "Good", "Excellent"]
        colors = ["#d73027", "#fee08b", "#a6d96a", "#1a9850"]
        thresholds = [0, 0.5, 0.75, 0.9, 1.0]
        for i in range(len(categories)):
            ax.barh(0, thresholds[i + 1] - thresholds[i], left=thresholds[i], color=colors[i], alpha=0.4, edgecolor="gray", linewidth=0.5, label=categories[i] if i < 2 else "")
        ax.errorbar(icc, 0, xerr=[[icc - icc_ci[0]], [icc_ci[1] - icc]], fmt="ko", markersize=10, capsize=5, linewidth=2)
        ax.set_xlim(0, 1.05)
        ax.set_yticks([])
        ax.set_xlabel("ICC Value", fontsize=12)
        ax.set_title(f"{icc_type} = {icc:.3f} [{icc_ci[0]:.3f}, {icc_ci[1]:.3f}] ({interp})", fontsize=13, fontweight="bold")
        ax.legend(loc="lower right", fontsize=9, ncol=2)
        plt.tight_layout()
        return fig

    def _plot_confusion(self, figsize, r):
        fig, ax = plt.subplots(1, 1, figsize=figsize)
        tp = r.get("tp", 0)
        fp = r.get("fp", 0)
        fn = r.get("fn", 0)
        tn = r.get("tn", 0)
        cm = np.array([[tn, fp], [fn, tp]])
        ax.imshow(cm, interpolation="nearest", cmap=plt.cm.Blues)
        for i in range(2):
            for j in range(2):
                ax.text(j, i, str(int(cm[i, j])), ha="center", va="center", fontsize=16, fontweight="bold", color="white" if cm[i, j] > cm.max() / 2 else "black")
        ax.set_xticks([0, 1])
        ax.set_yticks([0, 1])
        ax.set_xticklabels(["Negative", "Positive"])
        ax.set_yticklabels(["Negative", "Positive"])
        ax.set_xlabel("Predicted", fontsize=12)
        ax.set_ylabel("Actual", fontsize=12)
        ax.set_title(f"Confusion Matrix (Accuracy = {r.get('accuracy', 0):.3f})", fontsize=13, fontweight="bold")
        plt.tight_layout()
        return fig

    def _plot_coefficients(self, figsize, r):
        fig, ax = plt.subplots(1, 1, figsize=figsize)
        coefs = r.get("coefficients", r.get("fe_params", {}))
        ses = r.get("standard_errors", r.get("bse_fe", {}))
        if isinstance(coefs, pd.Series):
            coefs = coefs.to_dict()
            ses = {k: v for k, v in ses.items()} if isinstance(ses, pd.Series) else ses
        names = list(coefs.keys())
        vals = np.array([coefs[k] for k in names])
        errs = np.array([ses.get(k, 0) for k in names])
        y_pos = np.arange(len(names))
        colors = ["#1a9850" if v > 0 else "#d73027" for v in vals]
        ax.barh(y_pos, vals, xerr=errs, color=colors, alpha=0.7, edgecolor="gray", linewidth=0.5, capsize=3)
        ax.set_yticks(y_pos)
        ax.set_yticklabels(names, fontsize=10)
        ax.axvline(0, color="black", linewidth=0.8)
        ax.set_xlabel("Coefficient Estimate", fontsize=12)
        ax.set_title("Model Coefficients with 95% CI", fontsize=13, fontweight="bold")
        for i, (v, e) in enumerate(zip(vals, errs)):
            ci_low = v - 1.96 * e
            ci_high = v + 1.96 * e
            sig = "ns"
            if ci_low > 0 or ci_high < 0:
                sig = "*"
            ax.text(max(vals) * 1.05 if v >= 0 else min(vals) * 1.05, i, sig, va="center", fontsize=12, fontweight="bold", color="red" if sig == "*" else "gray")
        plt.tight_layout()
        return fig

    def _plot_bf(self, figsize, r):
        fig, ax = plt.subplots(1, 1, figsize=figsize)
        bf = r.get("bf10", 1)
        interp = r.get("interpretation", "No evidence")
        categories = ["Strong H0", "Moderate H0", "Anecdotal H0", "Anecdotal H1", "Moderate H1", "Strong H1"]
        boundaries = [1 / 10, 1 / 3, 1, 3, 10, 30]
        log_bf = np.log(bf)
        colors_bf = ["#d73027", "#fc8d59", "#fee08b", "#a6d96a", "#1a9850", "#006837"]
        bar_colors = [colors_bf[0] if bf <= 1/10 else colors_bf[1] if bf <= 1/3 else colors_bf[2] if bf <= 1 else colors_bf[3] if bf <= 3 else colors_bf[4] if bf <= 10 else colors_bf[5]]
        ax.bar(0, log_bf, width=0.6, color=bar_colors[0], alpha=0.8, edgecolor="gray")
        ax.set_xticks([])
        ax.axhline(0, color="black", linewidth=0.8)
        ax.set_ylabel("log(BF\u2081\u2080)", fontsize=12)
        ax.set_title(f"Bayes Factor BF\u2081\u2080 = {bf:.3f}\n{interp}", fontsize=13, fontweight="bold")
        for cat, bound, color in zip(categories, boundaries, colors_bf):
            if bound == 30:
                continue
            log_bound = np.log(bound)
            if abs(log_bound) < 10:
                ax.axhline(log_bound, color=color, linestyle="--", alpha=0.3, linewidth=1)
        plt.tight_layout()
        return fig

    def _plot_scatter(self, figsize, r):
        fig, ax = plt.subplots(1, 1, figsize=figsize)
        cols = self.data.columns[:2]
        ax.scatter(self.data[cols[0]], self.data[cols[1]], alpha=0.6, edgecolors="k", linewidth=0.5)
        ax.set_xlabel(cols[0], fontsize=12)
        ax.set_ylabel(cols[1], fontsize=12)
        ax.set_title(f"{cols[0]} vs {cols[1]}", fontsize=13, fontweight="bold")
        if self.statistic is not None and self.p_value is not None:
            r_val = self.statistic if abs(self.statistic) <= 1 else None
            if r_val is not None:
                ax.text(0.05, 0.95, f"r = {r_val:.3f}" + (f", p {'< .001' if self.p_value < 0.001 else f'= {self.p_value:.3f}'}" if self.p_value is not None else ""), transform=ax.transAxes, fontsize=11, verticalalignment="top", bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5))
        plt.tight_layout()
        return fig

    def narrative(self) -> str:
        r = self.results
        if "auc" in r:
            return self._narrative_roc(r)
        if "bias" in r and "lower_loa" in r:
            return self._narrative_bland_altman(r)
        if "icc" in r:
            return self._narrative_icc(r)
        if "accuracy" in r:
            return self._narrative_diagnostic(r)
        if "bf10" in r:
            return self._narrative_bayes(r)
        if "coefficients" in r or "fe_params" in r:
            return self._narrative_regression(r)
        return self._narrative_generic(r)

    def _narrative_roc(self, r):
        auc = r.get("auc", 0)
        auc_ci = r.get("auc_ci", [0, 0])
        youden = r.get("youden_index", 0)
        thr = r.get("youden_threshold", 0)
        sens = r.get("sensitivity", 0)
        spec = r.get("specificity", 0)
        interp = "excellent" if auc >= 0.9 else "good" if auc >= 0.8 else "fair" if auc >= 0.7 else "poor"
        return (f"ROC analysis revealed {interp} discriminatory performance (AUC = {auc:.3f}, "
                f"95% CI [{auc_ci[0]:.3f}, {auc_ci[1]:.3f}]). "
                f"The optimal cutoff according to Youden's index (J = {youden:.3f}) was {thr:.2f}, "
                f"yielding sensitivity = {sens:.3f} and specificity = {spec:.3f}.")

    def _narrative_bland_altman(self, r):
        bias = r.get("bias", 0)
        sd = r.get("sd_of_differences", 0)
        loa_l = r.get("lower_loa", 0)
        loa_u = r.get("upper_loa", 0)
        prop_p = r.get("proportional_bias_p", 1)
        txt = (f"Bland-Altman analysis showed a mean difference (bias) of {bias:.3f} "
               f"(SD = {sd:.3f}), with 95% limits of agreement from {loa_l:.3f} to {loa_u:.3f}.")
        if prop_p < 0.05:
            txt += f" Proportional bias was detected (p = {prop_p:.3f}), indicating the difference varies with the magnitude of the measurement."
        else:
            txt += f" No significant proportional bias was observed (p = {prop_p:.3f})."
        return txt

    def _narrative_icc(self, r):
        icc = r.get("icc", 0)
        icc_ci = r.get("icc_ci", [0, 0])
        icc_type = r.get("icc_type", "ICC")
        interp = r.get("interpretation", "")
        return (f"The {icc_type} was {icc:.3f} (95% CI [{icc_ci[0]:.3f}, {icc_ci[1]:.3f}]), "
                f"indicating {interp} reliability per Koo and Li (2016) guidelines.")

    def _narrative_diagnostic(self, r):
        acc = r.get("accuracy", 0)
        sens = r.get("sensitivity", 0)
        spec = r.get("specificity", 0)
        ppv = r.get("ppv", 0)
        npv = r.get("npv", 0)
        acc_ci = r.get("accuracy_ci", [0, 0])
        return (f"Diagnostic accuracy: accuracy = {acc:.3f} (95% CI [{acc_ci[0]:.3f}, {acc_ci[1]:.3f}]), "
                f"sensitivity = {sens:.3f}, specificity = {spec:.3f}, "
                f"PPV = {ppv:.3f}, NPV = {npv:.3f}.")

    def _narrative_bayes(self, r):
        bf = r.get("bf10", 1)
        interp = r.get("interpretation", "")
        return (f"Bayesian analysis yielded BF\u2081\u2080 = {bf:.3f}, "
                f"indicating {interp} Per Raftery (1995) classification.")

    def _narrative_regression(self, r):
        r_obj = r
        if hasattr(r_obj, "get"):
            aic = r_obj.get("aic", None)
            if aic is None or (isinstance(aic, float) and np.isnan(aic)):
                aic = r_obj.get("AIC", None)
            log_lik = r_obj.get("log_likelihood", None)
            coefs = r_obj.get("coefficients", r_obj.get("fe_params", {}))
            n_pred = len(coefs) if isinstance(coefs, (dict, pd.Series)) else 0
            parts = [f"A regression model with {n_pred} predictor(s) was fitted."]
            if log_lik is not None and not (isinstance(log_lik, float) and np.isnan(log_lik)):
                parts[-1] = parts[-1].rstrip(".") + f" (log-likelihood = {log_lik:.2f})."
            if aic is not None and not (isinstance(aic, float) and np.isnan(aic)):
                parts.append(f" Model fit: AIC = {aic:.1f}.")
            return " ".join(parts)
        return self._narrative_generic(r)

    def _narrative_generic(self, r):
        name = self.test_name
        stat = self.statistic
        p = self.p_value
        es = self.effect_size
        parts = [f"A {name} was performed."]
        if stat is not None:
            parts.append(f" Test statistic = {stat:.3f}.")
        if p is not None:
            parts.append(f" p {'< .001' if p < 0.001 else f'= {p:.3f}'}.")
        if es is not None:
            parts.append(f" Effect size = {es:.3f}.")
        return " ".join(parts)

    def write_excel(self, filepath: str):
        table_text = self.apa_table()
        lines = [l for l in table_text.split("\n") if l.startswith("|")]
        rows = []
        for line in lines[2:]:
            cells = [c.strip() for c in line.split("|")[1:-1]]
            if cells:
                rows.append(cells)
        if rows:
            df = pd.DataFrame(rows[1:], columns=rows[0]) if len(rows) > 1 else pd.DataFrame(rows)
            df.to_excel(filepath, index=False)
            return True
        return False

    def report(self) -> Dict[str, Any]:
        return {
            "table": self.apa_table(),
            "narrative": self.narrative(),
            "figure": self.figure(),
        }
