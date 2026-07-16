import numpy as np
import pandas as pd
import scipy.stats as stats
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import statsmodels.api as sm
import statsmodels.formula.api as smf
from statsmodels.stats.outliers_influence import variance_inflation_factor
from patsy import dmatrices

class KaplanMeierEstimator:
    """Calculates Kaplan-Meier survival curves, standard errors, and confidence intervals."""
    
    @staticmethod
    def fit(time: np.ndarray, status: np.ndarray) -> pd.DataFrame:
        """Fits Kaplan-Meier estimator for a single group.
        
        Parameters:
        - time: array-like, survival times
        - status: array-like, event status (1 = event, 0 = censored)
        
        Returns:
        - pd.DataFrame containing:
          time, n_at_risk, n_events, n_censored, survival, se, ci_lower, ci_upper
        """
        time = np.asarray(time)
        status = np.asarray(status)
        
        # Sort data by time ascending
        order = np.argsort(time)
        t_sorted = time[order]
        s_sorted = status[order]
        
        unique_times = np.unique(t_sorted)
        
        # Initialize results lists
        times = [0.0]
        n_at_risk = [len(time)]
        n_events = [0]
        n_censored = [0]
        survival = [1.0]
        se = [0.0]
        ci_lower = [1.0]
        ci_upper = [1.0]
        
        current_survival = 1.0
        greenwood_sum = 0.0
        
        total_n = len(time)
        
        for ut in unique_times:
            # Number at risk right before time ut
            at_risk_mask = t_sorted >= ut
            n_i = np.sum(at_risk_mask)
            
            # Events and censored at time ut
            event_mask = (t_sorted == ut) & (s_sorted == 1)
            censored_mask = (t_sorted == ut) & (s_sorted == 0)
            
            d_i = np.sum(event_mask)
            c_i = np.sum(censored_mask)
            
            if n_i == 0:
                continue
                
            # If events occur, update survival and greenwood sum
            if d_i > 0:
                current_survival *= (1.0 - d_i / n_i)
                # Greenwood sum element
                if n_i > d_i:
                    greenwood_sum += d_i / (n_i * (n_i - d_i))
            
            current_se = current_survival * np.sqrt(greenwood_sum) if greenwood_sum > 0 else 0.0
            
            # Confidence intervals: Log-log transformation (standard in lifelines/SAS/R)
            # theta = ln(-ln(S(t)))
            # Var(theta) = greenwood_sum / (ln(S(t)))^2
            if 0 < current_survival < 1:
                log_log_var = greenwood_sum / (np.log(current_survival) ** 2)
                log_log_se = np.sqrt(log_log_var)
                z = 1.96
                # confidence interval bounds
                lower_ci = current_survival ** np.exp(z * log_log_se)
                upper_ci = current_survival ** np.exp(-z * log_log_se)
            else:
                # Fallback to linear CI if survival is 1 or 0
                lower_ci = max(0.0, current_survival - 1.96 * current_se)
                upper_ci = min(1.0, current_survival + 1.96 * current_se)
                
            times.append(ut)
            n_at_risk.append(int(n_i))
            n_events.append(int(d_i))
            n_censored.append(int(c_i))
            survival.append(current_survival)
            se.append(current_se)
            ci_lower.append(lower_ci)
            ci_upper.append(upper_ci)
            
        return pd.DataFrame({
            "time": times,
            "n_at_risk": n_at_risk,
            "n_events": n_events,
            "n_censored": n_censored,
            "survival": survival,
            "se": se,
            "ci_lower": ci_lower,
            "ci_upper": ci_upper
        })


class LogRankTest:
    """Performs the log-rank test to compare survival distributions between two groups."""
    
    @staticmethod
    def compare(time: np.ndarray, status: np.ndarray, group: np.ndarray) -> dict:
        """Compares survival between two groups (coded as 0 and 1, or two unique values)."""
        time = np.asarray(time)
        status = np.asarray(status)
        group = np.asarray(group)
        
        unique_groups = np.unique(group)
        if len(unique_groups) != 2:
            raise ValueError(f"Log-rank test requires exactly 2 groups. Found: {unique_groups}")
            
        g1, g2 = unique_groups[0], unique_groups[1]
        
        # Sort and find unique event times across both groups
        order = np.argsort(time)
        t_sorted = time[order]
        s_sorted = status[order]
        g_sorted = group[order]
        
        # Only evaluate at times where at least one event occurred
        event_times = np.unique(t_sorted[s_sorted == 1])
        
        observed_1 = 0
        expected_1 = 0
        variance = 0.0
        
        total_observed_1 = np.sum(s_sorted[(g_sorted == g1) & (s_sorted == 1)])
        total_observed_2 = np.sum(s_sorted[(g_sorted == g2) & (s_sorted == 1)])
        
        for t in event_times:
            # Risk set: patients with time >= t
            risk_g1 = (t_sorted >= t) & (g_sorted == g1)
            risk_g2 = (t_sorted >= t) & (g_sorted == g2)
            
            n_1j = np.sum(risk_g1)
            n_2j = np.sum(risk_g2)
            n_j = n_1j + n_2j
            
            if n_j == 0:
                continue
                
            # Events at time t
            event_g1 = (t_sorted == t) & (s_sorted == 1) & (g_sorted == g1)
            event_g2 = (t_sorted == t) & (s_sorted == 1) & (g_sorted == g2)
            
            d_1j = np.sum(event_g1)
            d_2j = np.sum(event_g2)
            d_j = d_1j + d_2j
            
            # Expected events in Group 1
            e_1j = d_j * (n_1j / n_j)
            
            observed_1 += d_1j
            expected_1 += e_1j
            
            # Hypergeometric variance
            if n_j > 1:
                v_j = (d_j * (n_j - d_j) * n_1j * n_2j) / (n_j**2 * (n_j - 1))
                variance += v_j
                
        if variance == 0:
            chi2_stat = 0.0
            p_value = 1.0
        else:
            chi2_stat = ((observed_1 - expected_1) ** 2) / variance
            p_value = stats.chi2.sf(chi2_stat, 1)
            
        expected_2 = total_observed_1 + total_observed_2 - expected_1
        
        return {
            "group_labels": [str(g1), str(g2)],
            "observed": [int(observed_1), int(total_observed_2)],
            "expected": [float(expected_1), float(expected_2)],
            "chi2": chi2_stat,
            "df": 1,
            "p_value": p_value
        }


class SurvivalPlotter:
    """Plots clinical Kaplan-Meier survival curves with Number-at-Risk tables."""
    
    @staticmethod
    def plot(time: np.ndarray, status: np.ndarray, group: np.ndarray = None, 
             title: str = "Kaplan-Meier Survival Curve", 
             xlabel: str = "Time (Days)", ylabel: str = "Survival Probability",
             output_path: str = "survival_curve.png"):
        """Generates a publication-quality KM plot with Number-at-Risk table at the bottom."""
        time = np.asarray(time)
        status = np.asarray(status)
        
        fig = plt.figure(figsize=(10, 7), dpi=150)
        
        # GridSpec for KM plot on top, risk table on bottom
        gs = gridspec.GridSpec(2, 1, height_ratios=[4, 1], hspace=0.3)
        ax_plot = plt.subplot(gs[0])
        ax_table = plt.subplot(gs[1])
        
        # Curated color scheme
        colors = ["#1f77b4", "#d62728", "#2ca02c", "#ff7f0e"]
        
        # Determine X ticks based on time range
        max_time = time.max()
        ticks = np.linspace(0, max_time, 6)
        ticks = np.round(ticks).astype(int)
        
        ax_plot.set_xticks(ticks)
        
        risk_data = {} # store risk lists for the table
        
        if group is None:
            # Single group
            df = KaplanMeierEstimator.fit(time, status)
            ax_plot.step(df["time"], df["survival"], where="post", color=colors[0], label="Overall Cohort", linewidth=2)
            ax_plot.fill_between(df["time"], df["ci_lower"], df["ci_upper"], color=colors[0], alpha=0.15, step="post")
            
            # Plot censored ticks
            censored_df = df[df["n_censored"] > 0]
            ax_plot.scatter(censored_df["time"], censored_df["survival"], marker="+", color=colors[0], s=60, label="Censored")
            
            # Extract number at risk at tick points
            risk_counts = []
            for t in ticks:
                # find closest time in KM output that is <= t, get its n_at_risk
                subset = df[df["time"] <= t]
                if not subset.empty:
                    # check if anyone remained after or if we take the last index
                    risk_counts.append(subset.iloc[-1]["n_at_risk"])
                else:
                    risk_counts.append(len(time))
            risk_data["Overall Cohort"] = risk_counts
            
        else:
            # Multiple groups
            group = np.asarray(group)
            unique_groups = np.unique(group)
            
            for idx, g in enumerate(unique_groups):
                g_mask = group == g
                df = KaplanMeierEstimator.fit(time[g_mask], status[g_mask])
                g_color = colors[idx % len(colors)]
                
                ax_plot.step(df["time"], df["survival"], where="post", color=g_color, label=f"Group: {g}", linewidth=2)
                ax_plot.fill_between(df["time"], df["ci_lower"], df["ci_upper"], color=g_color, alpha=0.12, step="post")
                
                # Censored ticks
                censored_df = df[df["n_censored"] > 0]
                ax_plot.scatter(censored_df["time"], censored_df["survival"], marker="+", color=g_color, s=50)
                
                # Extract risk counts
                risk_counts = []
                for t in ticks:
                    subset = df[df["time"] <= t]
                    if not subset.empty:
                        risk_counts.append(subset.iloc[-1]["n_at_risk"])
                    else:
                        risk_counts.append(np.sum(g_mask))
                risk_data[f"Group: {g}"] = risk_counts
                
            # Log-rank test if 2 groups
            if len(unique_groups) == 2:
                try:
                    lr_res = LogRankTest.compare(time, status, group)
                    p_val_text = f"Log-Rank p = {lr_res['p_value']:.4f}" if lr_res['p_value'] >= 0.0001 else "Log-Rank p < 0.0001"
                    ax_plot.text(0.05, 0.1, p_val_text, transform=ax_plot.transAxes, 
                                 bbox=dict(facecolor='white', alpha=0.8, edgecolor='lightgray', boxstyle='round,pad=0.5'),
                                 fontsize=10, fontweight='bold', color='#333333')
                except Exception:
                    pass
                    
        # Style main plot
        ax_plot.set_title(title, fontweight="bold", fontsize=12, pad=15)
        ax_plot.set_ylabel(ylabel, fontweight="bold", fontsize=10)
        ax_plot.set_ylim(-0.02, 1.05)
        ax_plot.spines["top"].set_visible(False)
        ax_plot.spines["right"].set_visible(False)
        ax_plot.grid(True, linestyle=":", alpha=0.6)
        ax_plot.legend(loc="upper right", frameon=True, edgecolor="lightgray")
        
        # Style table subplot
        ax_table.axis("off")
        ax_table.set_xlim(ax_plot.get_xlim())
        
        # Print Numbers-at-risk
        y_pos = 0.8
        ax_table.text(ticks[0] - (ticks[-1] * 0.18), y_pos, "Number at Risk", fontweight="bold", fontsize=9, va="center", ha="left")
        
        for g_label, counts in risk_data.items():
            y_pos -= 0.3
            ax_table.text(ticks[0] - (ticks[-1] * 0.18), y_pos, g_label, fontsize=9, va="center", ha="left")
            for t_idx, count in enumerate(counts):
                ax_table.text(ticks[t_idx], y_pos, str(count), fontsize=9, va="center", ha="center")
                
        plt.savefig(output_path, bbox_inches="tight")
        plt.close()


class ClinicalRegressionSuite:
    """Fits multiple linear, logistic, and Poisson regressions, reporting medical publication metrics."""
    
    @staticmethod
    def multiple_linear(formula: str, data: pd.DataFrame) -> dict:
        """Fits OLS linear regression model. Computes VIF diagnostics."""
        model = smf.ols(formula, data=data)
        fit_results = model.fit()
        
        # VIF calculation
        y, X = dmatrices(formula, data, return_type='dataframe')
        vifs = {}
        # intercept is column 0 usually, VIF only calculated for covariates
        for i in range(X.shape[1]):
            col_name = X.columns[i]
            if col_name == 'Intercept':
                continue
            vifs[col_name] = variance_inflation_factor(X.values, i)
            
        summary_df = pd.DataFrame({
            "coef": fit_results.params,
            "std_err": fit_results.bse,
            "t_value": fit_results.tvalues,
            "p_value": fit_results.pvalues,
            "ci_lower": fit_results.conf_int()[0],
            "ci_upper": fit_results.conf_int()[1]
        })
        
        return {
            "type": "Multiple Linear Regression (OLS)",
            "formula": formula,
            "summary": summary_df,
            "r_squared": fit_results.rsquared,
            "adj_r_squared": fit_results.rsquared_adj,
            "f_statistic": fit_results.fvalue,
            "f_p_value": fit_results.f_pvalue,
            "vif": vifs,
            "n": int(fit_results.nobs)
        }

    @staticmethod
    def logistic(formula: str, data: pd.DataFrame) -> dict:
        """Fits Logistic Regression using Logit. Reports Odds Ratios."""
        model = smf.logit(formula, data=data)
        fit_results = model.fit(disp=0)
        
        summary_df = pd.DataFrame({
            "coef": fit_results.params,
            "std_err": fit_results.bse,
            "z_value": fit_results.tvalues,
            "p_value": fit_results.pvalues,
            "ci_lower": fit_results.conf_int()[0],
            "ci_upper": fit_results.conf_int()[1]
        })
        
        # Compute Odds Ratios
        summary_df["Odds_Ratio"] = np.exp(summary_df["coef"])
        summary_df["OR_ci_lower"] = np.exp(summary_df["ci_lower"])
        summary_df["OR_ci_upper"] = np.exp(summary_df["ci_upper"])
        
        # McFadden's pseudo R2
        prsquared = fit_results.prsquared
        
        # Likelihood ratio test
        lr_p = fit_results.llr_pvalue
        
        return {
            "type": "Logistic Regression (Binomial Logit)",
            "formula": formula,
            "summary": summary_df,
            "pseudo_r_squared": prsquared,
            "llr_p_value": lr_p,
            "llr_stat": fit_results.llr,
            "n": int(fit_results.nobs)
        }

    @staticmethod
    def poisson(formula: str, data: pd.DataFrame) -> dict:
        """Fits Poisson Regression. Reports Incident Rate Ratios and dispersion."""
        model = smf.poisson(formula, data=data)
        fit_results = model.fit(disp=0)
        
        summary_df = pd.DataFrame({
            "coef": fit_results.params,
            "std_err": fit_results.bse,
            "z_value": fit_results.tvalues,
            "p_value": fit_results.pvalues,
            "ci_lower": fit_results.conf_int()[0],
            "ci_upper": fit_results.conf_int()[1]
        })
        
        # Compute Incident Rate Ratios (IRR)
        summary_df["IRR"] = np.exp(summary_df["coef"])
        summary_df["IRR_ci_lower"] = np.exp(summary_df["ci_lower"])
        summary_df["IRR_ci_upper"] = np.exp(summary_df["ci_upper"])
        
        # Checking for dispersion: Pearson chi2 / degrees of freedom
        predicted = fit_results.predict()
        pearson_chi2 = np.sum((model.endog - predicted) ** 2 / predicted)
        df_resid = fit_results.df_resid
        dispersion_ratio = pearson_chi2 / df_resid if df_resid > 0 else 1.0
        
        return {
            "type": "Poisson Regression (Log link)",
            "formula": formula,
            "summary": summary_df,
            "pearson_chi2": pearson_chi2,
            "df_resid": df_resid,
            "dispersion_ratio": dispersion_ratio,
            "n": int(fit_results.nobs)
        }

    @staticmethod
    def cox_ph_regression(formula: str, data: pd.DataFrame, status_col: str) -> dict:
        """Fits a Cox Proportional Hazards regression model using statsmodels PHReg."""
        # formula is usually like "time_col ~ covariate1 + covariate2"
        # We split to get covariates formula on RHS
        lhs, rhs = [x.strip() for x in formula.split("~")]
        
        model = smf.phreg(rhs, data=data, status=data[status_col])
        fit_results = model.fit(disp=0)
        
        summary_df = pd.DataFrame({
            "coef": fit_results.params,
            "std_err": fit_results.bse,
            "z_value": fit_results.tvalues, # in statsmodels PHReg, tvalues are the Wald z-statistics
            "p_value": fit_results.pvalues,
            "ci_lower": fit_results.conf_int()[0],
            "ci_upper": fit_results.conf_int()[1]
        })
        
        # Hazard Ratios
        summary_df["Hazard_Ratio"] = np.exp(summary_df["coef"])
        summary_df["HR_ci_lower"] = np.exp(summary_df["ci_lower"])
        summary_df["HR_ci_upper"] = np.exp(summary_df["ci_upper"])
        
        return {
            "type": "Cox Proportional Hazards Regression",
            "formula": formula,
            "status_col": status_col,
            "summary": summary_df,
            "n": int(fit_results.nobs),
            "n_events": int(fit_results.model.status.sum())
        }
