import numpy as np
import pandas as pd
import scipy.stats as stats
import matplotlib.pyplot as plt
import statsmodels.api as sm
import re

class EffectSizeConverter:
    """Performs statistical conversions between different effect size metrics."""
    
    @staticmethod
    def continuous_to_smd(n1: float, m1: float, sd1: float, n2: float, m2: float, sd2: float) -> dict:
        """Computes Cohen's d, Hedges' g, and Hedges' g variance from continuous trial groups."""
        df = n1 + n2 - 2
        if df <= 0:
            raise ValueError("Sample sizes are too small to compute degrees of freedom.")
            
        sd_pooled = np.sqrt(((n1 - 1) * (sd1 ** 2) + (n2 - 1) * (sd2 ** 2)) / df)
        cohens_d = (m1 - m2) / sd_pooled
        
        # Hedges' g correction factor J
        j_correction = 1 - (3 / (4 * df - 1))
        hedges_g = cohens_d * j_correction
        
        # Variance of Hedges' g
        var_g = (j_correction ** 2) * ((n1 + n2) / (n1 * n2) + (cohens_d ** 2) / (2 * (n1 + n2)))
        se_g = np.sqrt(var_g)
        
        return {
            "cohens_d": cohens_d,
            "hedges_g": hedges_g,
            "variance": var_g,
            "se": se_g,
            "ci_lower": hedges_g - 1.96 * se_g,
            "ci_upper": hedges_g + 1.96 * se_g
        }

    @staticmethod
    def binary_to_log_or(e1: float, n1: float, e2: float, n2: float) -> dict:
        """Computes Log Odds Ratio and its variance from binary group events."""
        # Haldane-Anscombe correction for zero cells
        a, c = e1, e2
        b, d = n1 - e1, n2 - e2
        
        if a == 0 or b == 0 or c == 0 or d == 0:
            a += 0.5
            b += 0.5
            c += 0.5
            d += 0.5
            n1 += 1.0
            n2 += 1.0
            
        or_val = (a * d) / (b * c)
        log_or = np.log(or_val)
        var_log_or = 1/a + 1/b + 1/c + 1/d
        se_log_or = np.sqrt(var_log_or)
        
        ci_lower_log = log_or - 1.96 * se_log_or
        ci_upper_log = log_or + 1.96 * se_log_or
        
        return {
            "odds_ratio": or_val,
            "log_or": log_or,
            "variance": var_log_or,
            "se": se_log_or,
            "ci_lower": np.exp(ci_lower_log),
            "ci_upper": np.exp(ci_upper_log),
            "log_ci_lower": ci_lower_log,
            "log_ci_upper": ci_upper_log
        }

    @staticmethod
    def binary_to_log_rr(e1: float, n1: float, e2: float, n2: float) -> dict:
        """Computes Log Risk Ratio and its variance from binary group events."""
        a, c = e1, e2
        if a == 0:
            a += 0.5
            n1 += 0.5
        if c == 0:
            c += 0.5
            n2 += 0.5
            
        p1 = a / n1
        p2 = c / n2
        rr_val = p1 / p2
        log_rr = np.log(rr_val)
        var_log_rr = 1/a - 1/n1 + 1/c - 1/n2
        se_log_rr = np.sqrt(var_log_rr)
        
        ci_lower_log = log_rr - 1.96 * se_log_rr
        ci_upper_log = log_rr + 1.96 * se_log_rr
        
        return {
            "risk_ratio": rr_val,
            "log_rr": log_rr,
            "variance": var_log_rr,
            "se": se_log_rr,
            "ci_lower": np.exp(ci_lower_log),
            "ci_upper": np.exp(ci_upper_log),
            "log_ci_lower": ci_lower_log,
            "log_ci_upper": ci_upper_log
        }

    @staticmethod
    def t_to_d(t: float, n1: float, n2: float) -> dict:
        """Converts t-statistic and group sizes to Cohen's d and Hedges' g."""
        df = n1 + n2 - 2
        d = t * np.sqrt((n1 + n2) / (n1 * n2))
        j_correction = 1 - (3 / (4 * df - 1))
        g = d * j_correction
        var_g = (j_correction ** 2) * ((n1 + n2) / (n1 * n2) + (d ** 2) / (2 * (n1 + n2)))
        se_g = np.sqrt(var_g)
        
        return {
            "cohens_d": d,
            "hedges_g": g,
            "variance": var_g,
            "se": se_g,
            "ci_lower": g - 1.96 * se_g,
            "ci_upper": g + 1.96 * se_g
        }

    @staticmethod
    def r_to_d(r: float, n: float) -> dict:
        """Converts correlation coefficient r to Cohen's d."""
        d = (2 * r) / np.sqrt(1 - r**2)
        # Approximate variance
        var_d = 4 / ((n - 1) * ((1 - r**2) ** 3))
        se_d = np.sqrt(var_d)
        
        return {
            "cohens_d": d,
            "variance": var_d,
            "se": se_d,
            "ci_lower": d - 1.96 * se_d,
            "ci_upper": d + 1.96 * se_d
        }

    @staticmethod
    def chi2_to_d(chi2: float, n: float) -> dict:
        """Converts Chi-square statistic to Cohen's d (assuming 2 equal groups)."""
        d = 2 * np.sqrt(chi2 / (n - chi2))
        # Approximate variance
        var_d = 4 / n
        se_d = np.sqrt(var_d)
        
        return {
            "cohens_d": d,
            "variance": var_d,
            "se": se_d,
            "ci_lower": d - 1.96 * se_d,
            "ci_upper": d + 1.96 * se_d
        }


class PValueCombiner:
    """Implements Fisher's and Stouffer's methods for combining p-values."""

    @staticmethod
    def fishers_method(p_values: list[float]) -> dict:
        """Combines independent p-values using Fisher's chi-square sum method."""
        p_values = [p for p in p_values if 0 < p <= 1]
        k = len(p_values)
        if k == 0:
            raise ValueError("No valid p-values provided.")
            
        # Fisher's test statistic: T = -2 * sum(ln(p))
        t_stat = -2 * np.sum(np.log(p_values))
        # Degrees of freedom = 2 * k
        df = 2 * k
        combined_p = stats.chi2.sf(t_stat, df)
        
        return {
            "method": "Fisher's Method",
            "statistic": t_stat,
            "df": df,
            "combined_p": combined_p
        }

    @staticmethod
    def stouffers_method(p_values: list[float]) -> dict:
        """Combines independent p-values using Stouffer's Z-score method."""
        p_values = [p for p in p_values if 0 < p < 1]
        k = len(p_values)
        if k == 0:
            raise ValueError("No valid p-values in range (0, 1) provided.")
            
        # Calculate standard normal z-scores: Z_i = inv_norm(1 - p_i)
        z_scores = stats.norm.ppf(1 - np.array(p_values))
        z_sum = np.sum(z_scores)
        z_combined = z_sum / np.sqrt(k)
        
        # Combined p-value (one-tailed)
        combined_p = 1 - stats.norm.cdf(z_combined)
        
        return {
            "method": "Stouffer's Z-score Method",
            "statistic": z_combined,
            "combined_p": combined_p
        }


class MetaPoolingEngine:
    """Calculates pooled estimates and heterogeneity statistics for clinical meta-analysis."""

    @staticmethod
    def pool(effect_sizes: list[float], variances: list[float], method: str = "random") -> dict:
        """Performs clinical meta-analysis pooling. Returns detailed stats dict.
        
        method: 'fixed' (Inverse Variance) or 'random' (DerSimonian-Laird)
        """
        y = np.array(effect_sizes)
        v = np.array(variances)
        w = 1.0 / v
        k = len(y)
        
        if k < 2:
            raise ValueError("Meta-analysis requires at least 2 studies.")

        # 1. Fixed-Effect Model (Inverse Variance)
        sum_w = np.sum(w)
        pooled_fixed = np.sum(w * y) / sum_w
        var_fixed = 1.0 / sum_w
        se_fixed = np.sqrt(var_fixed)
        
        # Cochrane's Q (Heterogeneity statistic)
        q_stat = np.sum(w * (y - pooled_fixed) ** 2)
        df = k - 1
        q_p_value = stats.chi2.sf(q_stat, df)
        
        # I^2 (Percentage of variation due to heterogeneity)
        i_squared = max(0.0, (q_stat - df) / q_stat) * 100 if q_stat > 0 else 0.0
        
        # DerSimonian-Laird Tau^2 (Between-study variance)
        c_constant = sum_w - (np.sum(w ** 2) / sum_w)
        if c_constant > 0:
            tau_squared = max(0.0, (q_stat - df) / c_constant)
        else:
            tau_squared = 0.0
            
        # 2. Random-Effects Model (DerSimonian-Laird weights)
        v_random = v + tau_squared
        w_random = 1.0 / v_random
        sum_w_random = np.sum(w_random)
        pooled_random = np.sum(w_random * y) / sum_w_random
        var_random = 1.0 / sum_w_random
        se_random = np.sqrt(var_random)

        # Select corresponding model results
        if method == "fixed":
            pooled_effect = pooled_fixed
            se = se_fixed
            weights = w / sum_w * 100
        else:
            pooled_effect = pooled_random
            se = se_random
            weights = w_random / sum_w_random * 100
            
        z_val = pooled_effect / se
        p_val = 2 * (1 - stats.norm.cdf(abs(z_val)))
        
        return {
            "model": "Fixed-Effect" if method == "fixed" else "Random-Effects (DerSimonian-Laird)",
            "pooled_effect": pooled_effect,
            "se": se,
            "ci_lower": pooled_effect - 1.96 * se,
            "ci_upper": pooled_effect + 1.96 * se,
            "z_value": z_val,
            "p_value": p_val,
            "heterogeneity": {
                "Q": q_stat,
                "df": df,
                "Q_p_value": q_p_value,
                "I2": i_squared,
                "tau2": tau_squared
            },
            "weights": weights.tolist()
        }


class MetaVisualizer:
    """Generates Forest Plots and Funnel Plots using matplotlib."""

    @staticmethod
    def plot_forest(study_labels: list[str], effect_sizes: list[float], 
                    variances: list[float], pooled_results: dict, 
                    sm_name: str = "Effect Size", output_path: str = "forest_plot.png"):
        """Draws a clinical-grade publication-ready Forest Plot."""
        y_es = np.array(effect_sizes)
        se = np.sqrt(np.array(variances))
        k = len(y_es)
        
        ci_lower = y_es - 1.96 * se
        ci_upper = y_es + 1.96 * se
        
        fig, ax = plt.subplots(figsize=(10, k * 0.6 + 3), dpi=150)
        
        # Plot study rows
        y_pos = np.arange(k)
        
        # Invert axis so first study is at the top
        y_pos = y_pos[::-1]
        
        # Plot zero line (null effect line)
        # SMD / Difference null is 0; Risk/Odds Ratio null is 1 (plotted on log scale)
        is_ratio = sm_name.upper() in ("OR", "RR", "ODDS RATIO", "RISK RATIO")
        null_line = 1.0 if is_ratio else 0.0
        
        # If ratio, we plot on log scale internally
        plot_es = np.log(y_es) if is_ratio else y_es
        plot_ci_l = np.log(ci_lower) if is_ratio else ci_lower
        plot_ci_u = np.log(ci_upper) if is_ratio else ci_upper
        plot_pooled = np.log(pooled_results["pooled_effect"]) if is_ratio else pooled_results["pooled_effect"]
        plot_pooled_l = np.log(pooled_results["ci_lower"]) if is_ratio else pooled_results["ci_lower"]
        plot_pooled_u = np.log(pooled_results["ci_upper"]) if is_ratio else pooled_results["ci_upper"]
        plot_null = np.log(null_line) if is_ratio else null_line
        
        ax.axvline(x=plot_null, color="gray", linestyle="--", linewidth=1.2)
        
        # Render each study
        for i in range(k):
            # Error bars (95% CI)
            ax.plot([plot_ci_l[i], plot_ci_u[i]], [y_pos[i], y_pos[i]], color="black", linewidth=1.5)
            
            # Point estimate (square size proportional to weight)
            sq_size = max(5, pooled_results["weights"][i] * 12)
            ax.scatter(plot_es[i], y_pos[i], color="navy", marker="s", s=sq_size, zorder=3)
            
            # Study labels
            ax.text(plot_ci_l.min() - 0.5, y_pos[i], study_labels[i], va="center", ha="right", fontsize=9)
            
            # Stats labels on right
            val_text = f"{y_es[i]:.2f} [{ci_lower[i]:.2f}, {ci_upper[i]:.2f}]"
            weight_text = f"{pooled_results['weights'][i]:.1f}%"
            ax.text(plot_ci_u.max() + 0.5, y_pos[i], val_text, va="center", ha="left", fontsize=9)
            ax.text(plot_ci_u.max() + 2.5, y_pos[i], weight_text, va="center", ha="left", fontsize=9)
            
        # Draw pooled estimate diamond
        diamond_y = -1.2
        diamond_x = [plot_pooled_l, plot_pooled, plot_pooled_u, plot_pooled, plot_pooled_l]
        diamond_ys = [diamond_y, diamond_y + 0.2, diamond_y, diamond_y - 0.2, diamond_y]
        ax.fill(diamond_x, diamond_ys, color="darkred", edgecolor="black", zorder=4)
        
        # Pooled labels
        ax.text(plot_ci_l.min() - 0.5, diamond_y, "Pooled Estimate", va="center", ha="right", fontweight="bold", fontsize=9.5)
        pooled_text = f"{pooled_results['pooled_effect']:.2f} [{pooled_results['ci_lower']:.2f}, {pooled_results['ci_upper']:.2f}]"
        ax.text(plot_ci_u.max() + 0.5, diamond_y, pooled_text, va="center", ha="left", fontweight="bold", fontsize=9.5)
        ax.text(plot_ci_u.max() + 2.5, diamond_y, "100.0%", va="center", ha="left", fontweight="bold", fontsize=9.5)
        
        # Set axis titles and clean up spines
        ax.set_yticks([])
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['left'].set_visible(False)
        
        # Set X label
        label_suffix = " (log scale)" if is_ratio else ""
        ax.set_xlabel(f"{sm_name}{label_suffix}", fontweight="bold", fontsize=10)
        
        # Format X ticks on log scale if needed
        if is_ratio:
            # Generate ticks based on data ranges
            min_val = min(plot_ci_l.min(), plot_pooled_l)
            max_val = max(plot_ci_u.max(), plot_pooled_u)
            tick_vals = np.round(np.linspace(min_val, max_val, 5), 2)
            ax.set_xticks(tick_vals)
            ax.set_xticklabels([f"{np.exp(t):.2f}" for t in tick_vals])
            
        # Add heterogeneity subtitle at the bottom
        het = pooled_results["heterogeneity"]
        subtitle_text = (
            f"Heterogeneity: Q = {het['Q']:.2f} (df = {het['df']}), p = {het['Q_p_value']:.4f}; "
            f"I² = {het['I2']:.1f}%, τ² = {het['tau2']:.4f}\n"
            f"Model: {pooled_results['model']} (p = {pooled_results['p_value']:.4f})"
        )
        ax.text(plot_ci_l.min() - 0.5, -2.5, subtitle_text, va="center", ha="left", style="italic", fontsize=8.5)
        
        # Column headers
        ax.text(plot_ci_l.min() - 0.5, k + 0.5, "Study", va="bottom", ha="right", fontweight="bold", fontsize=10)
        ax.text(plot_ci_u.max() + 0.5, k + 0.5, f"{sm_name} [95% CI]", va="bottom", ha="left", fontweight="bold", fontsize=10)
        ax.text(plot_ci_u.max() + 2.5, k + 0.5, "Weight", va="bottom", ha="left", fontweight="bold", fontsize=10)
        
        plt.tight_layout()
        plt.savefig(output_path, bbox_inches="tight")
        plt.close()

    @staticmethod
    def plot_funnel(effect_sizes: list[float], variances: list[float], 
                    pooled_effect: float, sm_name: str = "Effect Size", 
                    output_path: str = "funnel_plot.png"):
        """Draws a clinical funnel plot for assessing publication bias."""
        y_es = np.array(effect_sizes)
        se = np.sqrt(np.array(variances))
        
        is_ratio = sm_name.upper() in ("OR", "RR", "ODDS RATIO", "RISK RATIO")
        plot_es = np.log(y_es) if is_ratio else y_es
        plot_pooled = np.log(pooled_effect) if is_ratio else pooled_effect
        
        fig, ax = plt.subplots(figsize=(7, 6), dpi=150)
        
        # Plot studies
        ax.scatter(plot_es, se, color="navy", edgecolor="black", zorder=3, alpha=0.8, label="Studies")
        
        # Invert y axis so smaller SE is at the top
        ax.set_ylim(max(se) * 1.1, 0.0)
        
        # Draw pooled line
        ax.axvline(x=plot_pooled, color="darkred", linestyle="-", linewidth=1.5, label="Pooled Effect")
        
        # Draw confidence lines (representing standard error boundary contours)
        # 95% CI limits are pooled_effect +/- 1.96 * se
        se_grid = np.linspace(0.0, max(se) * 1.1, 100)
        ci_lower = plot_pooled - 1.96 * se_grid
        ci_upper = plot_pooled + 1.96 * se_grid
        
        ax.plot(ci_lower, se_grid, color="gray", linestyle="--", linewidth=1)
        ax.plot(ci_upper, se_grid, color="gray", linestyle="--", linewidth=1)
        
        ax.set_ylabel("Standard Error (SE)", fontweight="bold")
        label_suffix = " (log scale)" if is_ratio else ""
        ax.set_xlabel(f"{sm_name}{label_suffix}", fontweight="bold")
        
        if is_ratio:
            # Transform ticks back to original scale
            min_val = min(plot_es.min(), ci_lower.min())
            max_val = max(plot_es.max(), ci_upper.max())
            tick_vals = np.round(np.linspace(min_val, max_val, 5), 2)
            ax.set_xticks(tick_vals)
            ax.set_xticklabels([f"{np.exp(t):.2f}" for t in tick_vals])
            
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.set_title("Funnel Plot for Publication Bias Assessment", fontweight="bold", pad=15)
        ax.legend(loc="upper right")
        
        plt.tight_layout()
        plt.savefig(output_path, bbox_inches="tight")
        plt.close()


class DTAMetaAnalysis:
    """Performs Diagnostic Test Accuracy (DTA) Meta-Analysis and Moses-Littenberg SROC plotting."""
    
    @staticmethod
    def calculate_metrics(tp: np.ndarray, fn: np.ndarray, fp: np.ndarray, tn: np.ndarray) -> pd.DataFrame:
        """Calculates sensitivity, specificity, DOR, +LR, -LR for each study."""
        tp = np.asarray(tp, dtype=float)
        fn = np.asarray(fn, dtype=float)
        fp = np.asarray(fp, dtype=float)
        tn = np.asarray(tn, dtype=float)
        
        # Haldane-Anscombe correction for zero cells
        for i in range(len(tp)):
            if tp[i] == 0 or fn[i] == 0 or fp[i] == 0 or tn[i] == 0:
                tp[i] += 0.5
                fn[i] += 0.5
                fp[i] += 0.5
                tn[i] += 0.5
                
        sens = tp / (tp + fn)
        spec = tn / (tn + fp)
        fpr = fp / (tn + fp)
        
        # Diagnostic Odds Ratio (DOR)
        dor = (tp * tn) / (fp * fn)
        log_dor = np.log(dor)
        var_log_dor = 1/tp + 1/fn + 1/fp + 1/tn
        se_log_dor = np.sqrt(var_log_dor)
        
        # Likelihood Ratios
        pos_lr = sens / np.clip(1.0 - spec, 1e-6, 1.0)
        neg_lr = (1.0 - sens) / np.clip(spec, 1e-6, 1.0)
        
        return pd.DataFrame({
            "tp": tp, "fn": fn, "fp": fp, "tn": tn,
            "sens": sens,
            "spec": spec,
            "fpr": fpr,
            "dor": dor,
            "log_dor": log_dor,
            "se_log_dor": se_log_dor,
            "dor_ci_lower": np.exp(log_dor - 1.96 * se_log_dor),
            "dor_ci_upper": np.exp(log_dor + 1.96 * se_log_dor),
            "pos_lr": pos_lr,
            "neg_lr": neg_lr
        })
        
    @staticmethod
    def fit_sroc(df: pd.DataFrame) -> dict:
        """Fits Moses-Littenberg regression D = a + b S.
        
        D = ln(DOR) = logit(sens) - logit(fpr)
        S = logit(sens) + logit(fpr)
        """
        sens = df["sens"].values
        fpr = df["fpr"].values
        
        sens = np.clip(sens, 0.001, 0.999)
        fpr = np.clip(fpr, 0.001, 0.999)
        
        logit_sens = np.log(sens / (1 - sens))
        logit_fpr = np.log(fpr / (1 - fpr))
        
        D = logit_sens - logit_fpr
        S = logit_sens + logit_fpr
        
        # Fit OLS: D = a + b*S
        X = sm.add_constant(S)
        model = sm.OLS(D, X)
        fit_results = model.fit()
        
        a = fit_results.params[0]
        b = fit_results.params[1] if len(fit_results.params) > 1 else 0.0
        
        a_p = fit_results.pvalues[0]
        b_p = fit_results.pvalues[1] if len(fit_results.pvalues) > 1 else 1.0
        
        # Pooled Sensitivity & Specificity using inverse variance weights
        var_logit_sens = 1/df["tp"] + 1/df["fn"]
        var_logit_spec = 1/df["fp"] + 1/df["tn"]
        
        w_sens = 1.0 / var_logit_sens
        w_spec = 1.0 / var_logit_spec
        
        pooled_logit_sens = np.sum(w_sens * logit_sens) / np.sum(w_sens)
        pooled_logit_spec = np.sum(w_spec * np.log(df["spec"] / (1 - df["spec"]))) / np.sum(w_spec)
        
        pooled_sens = 1 / (1 + np.exp(-pooled_logit_sens))
        pooled_spec = 1 / (1 + np.exp(-pooled_logit_spec))
        
        return {
            "a": a,
            "b": b,
            "a_p_value": a_p,
            "b_p_value": b_p,
            "r_squared": fit_results.rsquared,
            "pooled_sens": pooled_sens,
            "pooled_spec": pooled_spec
        }
        
    @staticmethod
    def plot_sroc(df: pd.DataFrame, sroc_results: dict, study_labels: list[str] = None, output_path: str = "sroc_plot.png"):
        """Plots Moses-Littenberg SROC curve with study points and pooled estimate."""
        a = sroc_results["a"]
        b = sroc_results["b"]
        
        fig, ax = plt.subplots(figsize=(7, 7), dpi=150)
        
        sample_sizes = df["tp"] + df["fn"] + df["fp"] + df["tn"]
        marker_sizes = 20 + (sample_sizes / sample_sizes.max()) * 200
        
        ax.scatter(df["fpr"], df["sens"], s=marker_sizes, color="teal", edgecolor="black", alpha=0.7, zorder=3, label="Studies")
        
        if study_labels:
            for i, label in enumerate(study_labels):
                ax.text(df["fpr"].iloc[i] + 0.01, df["sens"].iloc[i] + 0.01, label, fontsize=8, alpha=0.8)
                
        # Generate SROC curve line
        fpr_grid = np.linspace(0.001, 0.999, 300)
        logit_fpr = np.log(fpr_grid / (1 - fpr_grid))
        
        denom = 1.0 - b if abs(1.0 - b) > 1e-5 else 1e-5
        logit_tpr = a / denom + ((1.0 + b) / denom) * logit_fpr
        tpr_grid = 1.0 / (1.0 + np.exp(-logit_tpr))
        
        ax.plot(fpr_grid, tpr_grid, color="darkred", linestyle="-", linewidth=2, label="Moses-Littenberg SROC")
        
        pooled_sens = sroc_results["pooled_sens"]
        pooled_spec = sroc_results["pooled_spec"]
        pooled_fpr = 1.0 - pooled_spec
        
        ax.scatter(pooled_fpr, pooled_sens, marker="D", s=100, color="gold", edgecolor="black", zorder=4, label=f"Summary Point\n(Sens={pooled_sens:.2f}, Spec={pooled_spec:.2f})")
        
        ax.set_xlim(-0.02, 1.02)
        ax.set_ylim(-0.02, 1.02)
        ax.set_xlabel("1 - Specificity (False Positive Rate)", fontweight="bold")
        ax.set_ylabel("Sensitivity (True Positive Rate)", fontweight="bold")
        ax.set_title("Summary Receiver Operating Characteristic (SROC) Curve", fontweight="bold", pad=15)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.grid(True, linestyle=":", alpha=0.6)
        ax.legend(loc="lower left", frameon=True, edgecolor="lightgray")
        
        eq_text = f"SROC Equation: logit(Sens) - logit(FPR) = {a:.2f} + {b:.2f} * S\nR² = {sroc_results['r_squared']:.3f}"
        ax.text(0.95, 0.05, eq_text, transform=ax.transAxes, fontsize=8.5, ha="right", va="bottom",
                bbox=dict(facecolor="white", alpha=0.8, edgecolor="lightgray", boxstyle="round,pad=0.3"))
                
        plt.tight_layout()
        plt.savefig(output_path, bbox_inches="tight")
        plt.close()


class NetworkMetaAnalysis:
    """Performs Bucher indirect comparison calculations and draws network graphs."""
    
    @staticmethod
    def bucher_indirect(effect_AC: float, var_AC: float, effect_BC: float, var_BC: float) -> dict:
        """Calculates indirect treatment comparison A vs B using common comparator C."""
        effect_AB = effect_AC - effect_BC
        var_AB = var_AC + var_BC
        se_AB = np.sqrt(var_AB)
        
        ci_lower = effect_AB - 1.96 * se_AB
        ci_upper = effect_AB + 1.96 * se_AB
        
        z_val = effect_AB / se_AB if se_AB > 0 else 0.0
        p_val = 2 * (1 - stats.norm.cdf(abs(z_val)))
        
        return {
            "indirect_effect": effect_AB,
            "variance": var_AB,
            "se": se_AB,
            "ci_lower": ci_lower,
            "ci_upper": ci_upper,
            "z_value": z_val,
            "p_value": p_val
        }
        
    @staticmethod
    def plot_network(df: pd.DataFrame, ref_node: str = None, output_path: str = "network_graph.png"):
        """Plots trial network graph showing treatment comparisons.
        
        df columns: tx1, tx2, num_studies
        """
        import networkx as nx
        
        fig, ax = plt.subplots(figsize=(8, 7), dpi=150)
        
        G = nx.Graph()
        
        for _, row in df.iterrows():
            tx1 = str(row["tx1"]).strip()
            tx2 = str(row["tx2"]).strip()
            num_st = int(row.get("num_studies", 1))
            
            if G.has_edge(tx1, tx2):
                G[tx1][tx2]["weight"] += num_st
            else:
                G.add_edge(tx1, tx2, weight=num_st)
                
        pos = nx.circular_layout(G)
        
        weights = [G[u][v]["weight"] for u, v in G.edges()]
        max_wt = max(weights) if len(weights) > 0 else 1
        edge_widths = [1 + (w / max_wt) * 6 for w in weights]
        
        node_colors = []
        for node in G.nodes():
            if ref_node and node.lower() == ref_node.lower():
                node_colors.append("gold")
            else:
                node_colors.append("skyblue")
                
        nx.draw_networkx_nodes(G, pos, ax=ax, node_color=node_colors, node_size=1200, 
                               edgecolors="black", linewidths=1.2)
        
        nx.draw_networkx_edges(G, pos, ax=ax, width=edge_widths, edge_color="gray", alpha=0.8)
        
        nx.draw_networkx_labels(G, pos, ax=ax, font_size=10, font_weight="bold", font_color="black")
        
        edge_labels = {(u, v): f"{G[u][v]['weight']} studies" for u, v in G.edges()}
        nx.draw_networkx_edge_labels(G, pos, ax=ax, edge_labels=edge_labels, font_size=8)
        
        ax.set_title("Evidence Network of Direct Treatment Comparisons", fontweight="bold", pad=15)
        ax.axis("off")
        
        plt.tight_layout()
        plt.savefig(output_path, bbox_inches="tight")
        plt.close()
