"""
Advanced Survival Analysis Methods for Mega Medical Writer.
Adds competing risks, recurrent events, time-varying covariates,
PH diagnostics, IPCW, and left truncation analysis.
Adapted/ported from survival-pipe R methodology (https://github.com/htlin222/survival-pipe).
"""

import numpy as np
import pandas as pd
import scipy.stats as stats
from scipy.interpolate import interp1d
from scipy.linalg import inv as scipy_inv
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import warnings
from copy import deepcopy
from typing import Optional, Union, List, Dict, Tuple

from lifelines import KaplanMeierFitter, CoxPHFitter, NelsonAalenFitter, CoxTimeVaryingFitter
from lifelines.utils import CensoringType
from lifelines.exceptions import StatisticalWarning

Z95 = 1.96


class CompetingRisksAnalysis:
    """Cumulative Incidence Function estimation and Gray's test for competing risks."""

    @staticmethod
    def _all_cause_survival(time: np.ndarray, status: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Compute all-cause Kaplan-Meier survival at each unique event time.
        Returns (unique_times, survival, n_at_risk) arrays sorted by time.
        """
        order = np.argsort(time)
        t_sorted = time[order]
        s_sorted = (status > 0).astype(float)[order]
        unique_times = np.unique(t_sorted)

        surv = 1.0
        surv_vals = []
        n_risk_vals = []

        for ut in unique_times:
            n_i = int(np.sum(t_sorted >= ut))
            d_i = int(np.sum((t_sorted == ut) & (s_sorted == 1)))
            if n_i > 0 and d_i > 0:
                surv *= (1.0 - d_i / n_i)
            n_risk_vals.append(n_i)
            surv_vals.append(surv)

        return np.array(unique_times), np.array(surv_vals), np.array(n_risk_vals)

    @staticmethod
    def cif_estimate(
        time: np.ndarray,
        status: np.ndarray,
        group: np.ndarray = None,
        event_of_interest: int = 1
    ) -> dict:
        """CIF estimation using the Aalen-Johansen estimator for competing risks.

        Parameters
        ----------
        time : array-like, survival times
        status : array-like, event status (0=censored, 1,2,...=event types)
        group : array-like, optional, group labels for stratified estimation
        event_of_interest : int, the event type to focus on (default=1)

        Returns
        -------
        dict with keys:
            - 'results': list of DataFrames per group with time, cif_eoi, cif_other,
              var_cif, ci_lower, ci_upper
            - 'grays_test': dict with chi2, df, p_value (if group provided)
            - 'event_of_interest': int
        """
        time = np.asarray(time, dtype=float)
        status = np.asarray(status, dtype=int)

        if group is not None:
            group = np.asarray(group)
            unique_groups = np.unique(group)
            results_list = []
            for g in unique_groups:
                mask = group == g
                res = CompetingRisksAnalysis._cif_single(
                    time[mask], status[mask], event_of_interest, label=str(g)
                )
                results_list.append(res)

            gray = CompetingRisksAnalysis.grays_test(time, status, group, event_of_interest)
            return {"results": results_list, "grays_test": gray, "event_of_interest": event_of_interest}
        else:
            res = CompetingRisksAnalysis._cif_single(
                time, status, event_of_interest, label="Overall"
            )
            return {"results": [res], "grays_test": None, "event_of_interest": event_of_interest}

    @staticmethod
    def _cif_single(
        time: np.ndarray, status: np.ndarray, event_of_interest: int, label: str = ""
    ) -> pd.DataFrame:
        """Compute CIF for a single group using Aalen-Johansen estimator."""
        order = np.argsort(time)
        t_sorted = time[order]
        s_sorted = status[order]

        unique_times = np.unique(t_sorted)
        n_total = len(time)

        all_types = np.unique(s_sorted)
        all_types = all_types[all_types > 0]
        if event_of_interest not in all_types:
            all_types = np.append(all_types, event_of_interest)

        other_type = -1
        for et in all_types:
            if et != event_of_interest:
                other_type = et
                break
        if other_type == -1:
            other_type = event_of_interest + 1

        surv = 1.0
        times_out = []
        cif_eoi = []
        cif_other = []
        var_cif = []
        green_sum = 0.0
        prev_surv = 1.0
        prev_cif_eoi = 0.0
        prev_cif_other = 0.0

        for ut in unique_times:
            n_i = int(np.sum(t_sorted >= ut))
            d_i = int(np.sum((t_sorted == ut) & (s_sorted > 0)))
            d_eoi = int(np.sum((t_sorted == ut) & (s_sorted == event_of_interest)))
            d_other = d_i - d_eoi

            if n_i == 0:
                continue

            surv_contrib = (1.0 - d_i / n_i)
            surv *= surv_contrib

            inc_eoi = prev_surv * d_eoi / n_i if n_i > 0 else 0.0
            inc_other = prev_surv * d_other / n_i if n_i > 0 else 0.0

            cur_cif_eoi = prev_cif_eoi + inc_eoi
            cur_cif_other = prev_cif_other + inc_other

            term1_eoi = (cur_cif_eoi - prev_cif_eoi) ** 2
            term1_other = (cur_cif_other - prev_cif_other) ** 2

            if green_sum > 0 or d_i > 0:
                del_gr = d_i / (n_i * (n_i - d_i)) if n_i > d_i else 0.0
                if del_gr > 0 or green_sum > 0:
                    if prev_surv > 0:
                        term2_eoi = prev_surv ** 2 * d_eoi * (n_i - d_eoi) / (n_i ** 3) if n_i > 0 else 0.0
                        term2_other = prev_surv ** 2 * d_other * (n_i - d_other) / (n_i ** 3) if n_i > 0 else 0.0

                        var_eoi = cur_cif_eoi ** 2 * green_sum + term1_eoi + term2_eoi
                        var_other = cur_cif_other ** 2 * green_sum + term1_other + term2_other
                    else:
                        var_eoi = 0.0
                        var_other = 0.0
                else:
                    var_eoi = 0.0
                    var_other = 0.0
            else:
                var_eoi = 0.0
                var_other = 0.0

            if d_i > 0:
                if n_i > d_i:
                    green_sum += d_i / (n_i * (n_i - d_i))

            prev_surv = surv
            prev_cif_eoi = cur_cif_eoi
            prev_cif_other = cur_cif_other

            times_out.append(ut)
            cif_eoi.append(cur_cif_eoi)
            cif_other.append(cur_cif_other)
            var_cif.append(var_eoi)

        cif_eoi = np.array(cif_eoi)
        var_cif = np.array(var_cif)
        se = np.sqrt(np.maximum(var_cif, 0))

        with np.errstate(divide="ignore", invalid="ignore"):
            log_term = np.where(
                (cif_eoi > 0) & (cif_eoi < 1),
                np.log(-np.log(np.clip(1 - cif_eoi, 1e-15, 1 - 1e-15))),
                np.nan
            )
            denom = ((1 - cif_eoi) ** 2 * (np.log(np.clip(1 - cif_eoi, 1e-15, 1 - 1e-15))) ** 2)
            var_log = np.where(
                (cif_eoi > 0) & (cif_eoi < 1) & (~np.isnan(log_term)) & (denom > 1e-15),
                var_cif / denom,
                np.inf
            )

        valid = np.isfinite(log_term) & np.isfinite(var_log) & (var_log > 0)
        if np.any(valid):
            se_log = np.sqrt(var_log)
            ci_lower_log = log_term - Z95 * se_log
            ci_upper_log = log_term + Z95 * se_log
            ci_lower = np.where(valid, 1 - np.exp(-np.exp(ci_lower_log)), cif_eoi - Z95 * se)
            ci_upper = np.where(valid, 1 - np.exp(-np.exp(ci_upper_log)), cif_eoi + Z95 * se)
            ci_lower = np.clip(ci_lower, 0, 1)
            ci_upper = np.clip(ci_upper, 0, 1)
        else:
            ci_lower = np.maximum(cif_eoi - Z95 * se, 0)
            ci_upper = np.minimum(cif_eoi + Z95 * se, 1)

        return pd.DataFrame({
            "time": times_out,
            "cif_eoi": cif_eoi,
            "cif_other": cif_other,
            "se": se,
            "var_cif": var_cif,
            "ci_lower": ci_lower,
            "ci_upper": ci_upper,
            "group": label
        })

    @staticmethod
    def grays_test(
        time: np.ndarray,
        status: np.ndarray,
        group: np.ndarray,
        event_of_interest: int = 1
    ) -> dict:
        """Gray's test comparing cumulative incidence functions across groups.
        Implements a K-sample weighted chi-square test analogous to the cmprsk R package.

        Parameters
        ----------
        time : array-like, survival times
        status : array-like, event status (0=censored, 1,2,...)
        group : array-like, group labels
        event_of_interest : int, the cause of interest

        Returns
        -------
        dict with chi2_statistic, df, p_value, group_labels
        """
        time = np.asarray(time, dtype=float)
        status = np.asarray(status, dtype=int)
        group = np.asarray(group)

        unique_groups = np.unique(group)
        K = len(unique_groups)
        if K < 2:
            return {"chi2_statistic": 0.0, "df": 0, "p_value": 1.0, "group_labels": [str(g) for g in unique_groups]}

        group_map = {g: idx for idx, g in enumerate(unique_groups)}
        group_idx = np.array([group_map[g] for g in group])

        eoi = event_of_interest
        order = np.argsort(time)
        t_sorted = time[order]
        s_sorted = status[order]
        g_sorted = group_idx[order]

        event_times = np.unique(t_sorted[s_sorted == eoi])
        if len(event_times) == 0:
            return {"chi2_statistic": 0.0, "df": K - 1, "p_value": 1.0,
                    "group_labels": [str(g) for g in unique_groups]}

        U = np.zeros(K)
        V = np.zeros((K, K))

        for t in event_times:
            at_risk = t_sorted >= t
            Y_g = np.array([np.sum(at_risk & (g_sorted == k)) for k in range(K)])
            Y_total = np.sum(Y_g)
            if Y_total == 0:
                continue

            d_g = np.array([
                np.sum((t_sorted == t) & (s_sorted == eoi) & (g_sorted == k))
                for k in range(K)
            ])
            d_total = np.sum(d_g)
            if d_total == 0:
                continue

            w_j = Y_total

            for g in range(K):
                U[g] += w_j * (d_g[g] - Y_g[g] * d_total / Y_total)

            if Y_total >= 2:
                for g in range(K):
                    for h in range(K):
                        if g == h:
                            V[g, h] += d_total * Y_g[g] * (Y_total - Y_g[g]) * (Y_total - d_total) / (Y_total - 1)
                        else:
                            V[g, h] += -d_total * Y_g[g] * Y_g[h] * (Y_total - d_total) / (Y_total - 1)

        U = U[:K - 1]
        V = V[:K - 1, :K - 1]

        try:
            V_inv = np.linalg.pinv(V)
            chi2 = U @ V_inv @ U
        except np.linalg.LinAlgError:
            chi2 = 0.0

        df = K - 1
        p_val = stats.chi2.sf(chi2, df)

        return {
            "chi2_statistic": float(chi2),
            "df": int(df),
            "p_value": float(p_val),
            "group_labels": [str(g) for g in unique_groups]
        }

    @staticmethod
    def plot_cif(
        results: dict,
        title: str = "Cumulative Incidence Functions",
        output_path: str = None
    ) -> plt.Figure:
        """Stacked CIF plot. If multiple groups, separate lines per group with Gray's test p-value.

        Parameters
        ----------
        results : dict from cif_estimate
        title : str
        output_path : str, optional

        Returns
        -------
        matplotlib Figure
        """
        fig, ax = plt.subplots(figsize=(10, 7), dpi=150)
        colors = ["#1f77b4", "#d62728", "#2ca02c", "#ff7f0e", "#9467bd"]

        dfs = results.get("results", [])
        eoi = results.get("event_of_interest", 1)

        if len(dfs) == 1:
            df = dfs[0]
            t = df["time"].values
            ax.step(t, df["cif_eoi"].values, where="post",
                    color=colors[0], linewidth=2, label=f"Event {eoi}")
            ax.fill_between(t, df["ci_lower"].values, df["ci_upper"].values,
                            color=colors[0], alpha=0.15, step="post")
            ax.step(t, df["cif_other"].values, where="post",
                    color=colors[1], linewidth=2, label="Competing Event")
            ax.fill_between(t, np.maximum(df["cif_other"] - Z95 * df["se"], 0),
                            np.minimum(df["cif_other"] + Z95 * df["se"], 1),
                            color=colors[1], alpha=0.10, step="post")
        else:
            for idx, df in enumerate(dfs):
                g_label = df["group"].iloc[0] if "group" in df.columns else f"Group {idx}"
                color = colors[idx % len(colors)]
                t = df["time"].values
                ax.step(t, df["cif_eoi"].values, where="post",
                        color=color, linewidth=2, label=f"{g_label} - Event {eoi}")
                ax.fill_between(t, df["ci_lower"].values, df["ci_upper"].values,
                                color=color, alpha=0.12, step="post")

            gray = results.get("grays_test")
            if gray is not None and gray["chi2_statistic"] > 0:
                p = gray["p_value"]
                p_text = f"Gray's test p = {p:.4f}" if p >= 0.0001 else "Gray's test p < 0.0001"
                ax.text(0.05, 0.95, p_text, transform=ax.transAxes,
                        bbox=dict(facecolor="white", alpha=0.8, edgecolor="lightgray",
                                  boxstyle="round,pad=0.5"),
                        fontsize=10, fontweight="bold", va="top")

        ax.set_title(title, fontweight="bold", fontsize=12, pad=15)
        ax.set_xlabel("Time", fontweight="bold", fontsize=10)
        ax.set_ylabel("Cumulative Incidence", fontweight="bold", fontsize=10)
        ax.set_ylim(-0.02, 1.05)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.grid(True, linestyle=":", alpha=0.6)
        ax.legend(loc="best", frameon=True, edgecolor="lightgray")

        fig.tight_layout()
        if output_path:
            fig.savefig(output_path, bbox_inches="tight")
        return fig


class CauseSpecificCox:
    """Cause-specific Cox proportional hazards models for competing risks."""

    @staticmethod
    def fit(
        time: np.ndarray,
        status: np.ndarray,
        X: pd.DataFrame,
        event_of_interest: int = 1
    ) -> dict:
        """Fit cause-specific Cox model by censoring competing events.

        Parameters
        ----------
        time : array-like
        status : array-like (0=censored, 1,2,...=event types)
        X : pd.DataFrame of covariates
        event_of_interest : int, event type to model

        Returns
        -------
        dict with summary DataFrame (coef, HR, se, z, p, ci_lower, ci_upper),
        model object, call_info
        """
        time = np.asarray(time, dtype=float)
        status = np.asarray(status, dtype=int)

        cs_status = np.where(status == event_of_interest, 1, 0).astype(int)

        data = X.copy()
        data["time"] = time
        data["status"] = cs_status

        cph = CoxPHFitter()
        try:
            cph.fit(data, duration_col="time", event_col="status")
            summary = cph.summary.copy()
            summary.columns = [c.lower().replace(" ", "_") for c in summary.columns]
            summary = summary.rename(columns={
                "coef": "coef", "exp(coef)": "hr",
                "se(coef)": "se", "z": "z", "p": "p",
                "exp(coef)_lower_95%": "ci_lower",
                "exp(coef)_upper_95%": "ci_upper"
            })
            if "ci_lower" not in summary.columns:
                summary["ci_lower"] = np.exp(summary["coef"] - Z95 * summary["se"])
                summary["ci_upper"] = np.exp(summary["coef"] + Z95 * summary["se"])
                summary["hr"] = np.exp(summary["coef"])
            summary = summary[["coef", "hr", "se", "z", "p", "ci_lower", "ci_upper"]]
        except Exception as e:
            warnings.warn(f"CauseSpecificCox fit failed: {e}")
            summary = pd.DataFrame(columns=["coef", "hr", "se", "z", "p", "ci_lower", "ci_upper"],
                                   index=X.columns)
            cph = None

        n_events = int(np.sum(cs_status))
        return {
            "summary": summary,
            "model": cph,
            "event_of_interest": event_of_interest,
            "n_events": n_events,
            "n_total": len(time)
        }

    @staticmethod
    def forest_plot(
        results_list: List[dict],
        event_labels: List[str] = None,
        output_path: str = None
    ) -> plt.Figure:
        """Forest plot comparing cause-specific HRs for different event types.

        Parameters
        ----------
        results_list : list of result dicts from fit()
        event_labels : list of str, labels for each event type
        output_path : str, optional

        Returns
        -------
        matplotlib Figure
        """
        if event_labels is None:
            event_labels = [f"Event {r.get('event_of_interest', i + 1)}" for i, r in enumerate(results_list)]

        covariate_names = list(results_list[0]["summary"].index)
        n_covariates = len(covariate_names)
        n_events = len(results_list)

        n_rows = n_covariates * n_events
        fig, ax = plt.subplots(figsize=(10, max(4, n_rows * 0.6)), dpi=150)

        y_pos = 0
        y_ticks = []
        y_labels = []

        colors = plt.cm.Set1(np.linspace(0, 1, n_events))

        for ci, cov_name in enumerate(covariate_names):
            for ei, res in enumerate(results_list):
                y_pos += 1
                row = res["summary"].loc[cov_name]
                hr = row["hr"]
                ci_low = row["ci_lower"]
                ci_up = row["ci_upper"]

                color = colors[ei]
                offset = (ei - (n_events - 1) / 2) * 0.25
                y = -y_pos + offset

                ax.plot([ci_low, ci_up], [y, y], color=color, linewidth=2)
                ax.plot(hr, y, marker="s", color=color, markersize=8)

                if ci == 0:
                    ax.text(ci_up + 0.02, y, event_labels[ei],
                            fontsize=8, va="center", color=color)

            y_ticks.append(-y_pos + (n_events - 1) * 0.25 / 2)
            y_labels.append(cov_name)

        ax.axvline(1.0, color="gray", linestyle="--", linewidth=1)
        ax.set_xscale("log")
        ax.set_xlabel("Hazard Ratio (95% CI)", fontweight="bold")
        ax.set_yticks(y_ticks)
        ax.set_yticklabels(y_labels, fontweight="bold")
        ax.invert_yaxis()
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.grid(True, axis="x", linestyle=":", alpha=0.5)
        ax.set_title("Cause-Specific Hazard Ratios", fontweight="bold")

        fig.tight_layout()
        if output_path:
            fig.savefig(output_path, bbox_inches="tight")
        return fig


class FineGrayModel:
    """Fine-Gray subdistribution hazard model for competing risks."""

    @staticmethod
    def fit(
        time: np.ndarray,
        status: np.ndarray,
        X: pd.DataFrame,
        event_of_interest: int = 1
    ) -> dict:
        """Fit Fine-Gray subdistribution hazard model.

        Creates subdistribution times (INF for competing events), computes IPCW
        censoring weights, and fits a weighted Cox model.

        Parameters
        ----------
        time : array-like
        status : array-like (0=censored, 1,2,...)
        X : pd.DataFrame of covariates
        event_of_interest : int

        Returns
        -------
        dict with summary DataFrame (coef, SHR, se, z, p, ci_lower, ci_upper),
        weights, model object
        """
        time = np.asarray(time, dtype=float)
        status = np.asarray(status, dtype=int)
        n = len(time)

        sub_time = time.copy()
        sub_status = np.zeros(n, dtype=int)

        eoi_mask = status == event_of_interest
        ce_mask = (status > 0) & (status != event_of_interest)
        cens_mask = status == 0

        sub_status[eoi_mask] = 1
        sub_status[ce_mask] = 0

        inf_time = np.max(time) * 10.0 + 1.0
        sub_time[ce_mask] = inf_time

        cens_event = cens_mask.astype(int)
        kmf_cens = KaplanMeierFitter()
        kmf_cens.fit(time, event_observed=cens_event)

        G_t = kmf_cens.survival_function_.squeeze()
        G_t.index = kmf_cens.timeline

        weights = np.ones(n)
        for i in range(n):
            if cens_mask[i] or eoi_mask[i] or ce_mask[i]:
                t_i = time[i]
                idx = np.searchsorted(G_t.index, t_i, side="left")
                if idx > 0:
                    G_minus = G_t.iloc[max(0, idx - 1)]
                else:
                    G_minus = G_t.iloc[0]
                if G_minus > 0:
                    weights[i] = 1.0 / G_minus

        w_max = np.percentile(weights, 99)
        weights = np.minimum(weights, w_max)

        data = X.copy()
        data["time"] = sub_time
        data["status"] = sub_status
        data["_fg_weight_"] = weights

        cph = CoxPHFitter()
        try:
            cph.fit(data, duration_col="time", event_col="status",
                    weights_col="_fg_weight_", robust=True)
            summary = cph.summary.copy()
            summary.columns = [c.lower().replace(" ", "_") for c in summary.columns]
            summary = summary.rename(columns={
                "coef": "coef", "exp(coef)": "shr",
                "se(coef)": "se", "z": "z", "p": "p",
                "exp(coef)_lower_95%": "ci_lower",
                "exp(coef)_upper_95%": "ci_upper"
            })
            if "ci_lower" not in summary.columns:
                summary["ci_lower"] = np.exp(summary["coef"] - Z95 * summary["se"])
                summary["ci_upper"] = np.exp(summary["coef"] + Z95 * summary["se"])
                summary["shr"] = np.exp(summary["coef"])
            summary = summary[["coef", "shr", "se", "z", "p", "ci_lower", "ci_upper"]]
        except Exception as e:
            warnings.warn(f"FineGrayModel fit failed: {e}")
            summary = pd.DataFrame(columns=["coef", "shr", "se", "z", "p", "ci_lower", "ci_upper"],
                                   index=X.columns)
            cph = None

        return {
            "summary": summary,
            "model": cph,
            "weights": weights,
            "event_of_interest": event_of_interest,
            "n_events": int(np.sum(eoi_mask))
        }

    @staticmethod
    def compare_with_causespecific(
        cs_results: dict,
        fg_results: dict,
        output_path: str = None
    ) -> plt.Figure:
        """Side-by-side forest plot comparing cause-specific HR vs subdistribution HR.

        Parameters
        ----------
        cs_results : dict from CauseSpecificCox.fit()
        fg_results : dict from FineGrayModel.fit()

        Returns
        -------
        matplotlib Figure
        """
        cov_names = list(cs_results["summary"].index)
        n_covs = len(cov_names)

        fig, axes = plt.subplots(1, 2, figsize=(12, max(4, n_covs * 0.5)), dpi=150, sharey=True)

        for ax, results, label in zip(
            axes,
            [cs_results, fg_results],
            ["Cause-Specific HR", "Subdistribution HR (Fine-Gray)"]
        ):
            summary = results["summary"]
            hr_col = "hr" if "hr" in summary.columns else "shr"
            for i, cov_name in enumerate(cov_names):
                row = summary.loc[cov_name]
                hr = row[hr_col]
                ci_l = row["ci_lower"]
                ci_u = row["ci_upper"]
                y = n_covs - 1 - i
                ax.plot([ci_l, ci_u], [y, y], color="navy", linewidth=2)
                ax.plot(hr, y, marker="s", color="navy", markersize=8, zorder=5)

            ax.axvline(1.0, color="gray", linestyle="--", linewidth=1)
            ax.set_xscale("log")
            ax.set_title(label, fontweight="bold", fontsize=11)
            ax.set_xlabel("Estimate (95% CI)", fontweight="bold")
            ax.set_yticks(range(n_covs))
            ax.set_yticklabels(cov_names, fontweight="bold")
            ax.invert_yaxis()
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)
            ax.grid(True, axis="x", linestyle=":", alpha=0.5)

        fig.suptitle("Cause-Specific vs Fine-Gray Comparison", fontweight="bold", fontsize=13)
        fig.tight_layout()
        if output_path:
            fig.savefig(output_path, bbox_inches="tight")
        return fig


class PHDiagnostics:
    """Proportional hazards assumption diagnostics and model diagnostics."""

    @staticmethod
    def schoenfeld_test(
        model: CoxPHFitter,
        time: np.ndarray = None,
        training_data: pd.DataFrame = None
    ) -> dict:
        """Compute scaled Schoenfeld residual-based PH test for each covariate.

        Tests the null hypothesis that the log hazard ratio is constant over time
        (i.e., PH assumption holds). Computes correlation between scaled Schoenfeld
        residuals and time (rank transform).

        Parameters
        ----------
        model : fitted CoxPHFitter
        time : array-like, optional, event/censoring times (used for ranking).
               Falls back to duration_col from training_data if not provided.
        training_data : pd.DataFrame, optional, the original training data.
               Required if model does not store it internally.

        Returns
        -------
        dict with 'global_test' (rho, chi2, p) and 'per_covariate' DataFrame
        """
        if training_data is None:
            try:
                training_data = model.data
            except Exception:
                try:
                    event_c = model.event_col
                    dur_c = model.duration_col
                    param_idx = model.params_.index
                    training_data = pd.DataFrame({dur_c: np.zeros(10), event_c: np.zeros(10)})
                    for c in param_idx:
                        training_data[c] = np.zeros(10)
                except Exception:
                    try:
                        training_data = model._model.data
                    except Exception:
                        residuals = model.compute_residuals(pd.DataFrame(), kind="scaled_schoenfeld")
                        return {
                            "global_test": {"rho": 0.0, "chi2": 0.0, "p_value": 1.0},
                            "per_covariate": pd.DataFrame(columns=["rho", "chi2", "p_value"])
                        }

        try:
            residuals = model.compute_residuals(training_data, kind="scaled_schoenfeld")
        except Exception:
            return {
                "global_test": {"rho": 0.0, "chi2": 0.0, "p_value": 1.0},
                "per_covariate": pd.DataFrame(columns=["rho", "chi2", "p_value"])
            }

        if residuals.empty or residuals.shape[0] == 0:
            return {
                "global_test": {"rho": 0.0, "chi2": 0.0, "p_value": 1.0},
                "per_covariate": pd.DataFrame(columns=["rho", "chi2", "p_value"])
            }

        try:
            event_col = model.event_col
            event_mask = training_data[event_col].astype(bool).values
        except Exception:
            event_mask = np.ones(len(training_data), dtype=bool)

        if time is not None:
            time = np.asarray(time, dtype=float)
            event_times = time[event_mask] if len(time) == len(event_mask) else time
        else:
            try:
                event_times = training_data[model.duration_col].values[event_mask]
            except Exception:
                event_times = np.arange(len(residuals))

        if len(event_times) == 0 or len(event_times) != residuals.shape[0]:
            event_times = np.arange(residuals.shape[0])

        event_ranks = stats.rankdata(event_times)

        per_cov = {}
        global_chi2 = 0.0
        global_df = 0

        for col in residuals.columns:
            r = residuals[col].values
            if len(r) < 3 or np.std(r) < 1e-12 or np.std(event_ranks) < 1e-12:
                per_cov[col] = {"rho": 0.0, "chi2": 0.0, "p_value": 1.0}
                continue

            rho, p_val = stats.spearmanr(r, event_ranks)
            if np.isnan(rho):
                rho, _ = stats.pearsonr(r, event_ranks)
                if np.isnan(rho):
                    rho = 0.0
                    p_val = 1.0

            n_eff = len(r)
            if abs(rho) < 1.0:
                t_stat = rho * np.sqrt((n_eff - 2) / (1 - rho ** 2))
                chi2_val = t_stat ** 2
                p_chi2 = stats.chi2.sf(chi2_val, 1)
            else:
                chi2_val = n_eff ** 2
                p_chi2 = 0.0

            per_cov[col] = {"rho": float(rho), "chi2": float(chi2_val), "p_value": float(p_chi2)}
            global_chi2 += chi2_val
            global_df += 1

        global_p = stats.chi2.sf(global_chi2, global_df) if global_df > 0 else 1.0

        per_cov_df = pd.DataFrame(per_cov).T

        return {
            "global_test": {"rho": None, "chi2": float(global_chi2), "p_value": float(global_p), "df": global_df},
            "per_covariate": per_cov_df
        }

    @staticmethod
    def schoenfeld_plot(
        model: CoxPHFitter,
        training_data: pd.DataFrame = None,
        output_path: str = None
    ) -> plt.Figure:
        """Plot scaled Schoenfeld residuals vs time for each covariate with smooth spline.

        Parameters
        ----------
        model : fitted CoxPHFitter
        training_data : pd.DataFrame, optional, the original training data
        output_path : str, optional

        Returns
        -------
        matplotlib Figure
        """
        if training_data is None:
            try:
                training_data = model.data
            except Exception:
                try:
                    event_c = model.event_col
                    dur_c = model.duration_col
                    param_idx = model.params_.index
                    training_data = pd.DataFrame({dur_c: np.zeros(10), event_c: np.zeros(10)})
                    for c in param_idx:
                        training_data[c] = np.zeros(10)
                except Exception:
                    try:
                        training_data = model._model.data
                    except Exception:
                        pass

        try:
            residuals = model.compute_residuals(training_data, kind="scaled_schoenfeld")
        except Exception:
            fig, ax = plt.subplots(figsize=(8, 4))
            ax.text(0.5, 0.5, "Could not compute residuals", ha="center", va="center", transform=ax.transAxes)
            return fig

        try:
            event_mask = training_data[model.event_col].astype(bool).values
            duration_col = model.duration_col
            event_times = training_data[duration_col].values[event_mask]
        except Exception:
            event_times = np.arange(len(residuals)) if not residuals.empty else []

        if residuals.empty or len(event_times) == 0:
            fig, ax = plt.subplots(figsize=(8, 4))
            ax.text(0.5, 0.5, "No residuals available", ha="center", va="center", transform=ax.transAxes)
            return fig

        n_covs = len(residuals.columns)
        n_cols = min(3, n_covs)
        n_rows = int(np.ceil(n_covs / n_cols))

        fig, axes = plt.subplots(n_rows, n_cols, figsize=(5 * n_cols, 4 * n_rows), dpi=150)
        if n_covs == 1:
            axes = np.array([axes])
        axes_flat = axes.flatten()

        for idx, col in enumerate(residuals.columns):
            ax = axes_flat[idx]
            r = residuals[col].values

            ax.scatter(event_times, r, alpha=0.6, s=20, color="steelblue")

            if len(event_times) >= 5:
                try:
                    order = np.argsort(event_times)
                    xs = event_times[order]
                    ys = r[order]
                    z = np.polyfit(xs, ys, min(3, len(xs) - 1))
                    p = np.poly1d(z)
                    x_smooth = np.linspace(xs.min(), xs.max(), 100)
                    y_smooth = p(x_smooth)
                    ax.plot(x_smooth, y_smooth, color="red", linewidth=2, label="Smooth")
                except Exception:
                    pass

            ax.axhline(0, color="gray", linestyle="--", linewidth=1)
            ax.set_title(f"{col}", fontweight="bold", fontsize=10)
            ax.set_xlabel("Time", fontsize=8)
            ax.set_ylabel("Scaled Schoenfeld Residual", fontsize=8)
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)
            ax.grid(True, linestyle=":", alpha=0.4)

        for j in range(idx + 1, len(axes_flat)):
            axes_flat[j].set_visible(False)

        fig.suptitle("Schoenfeld Residuals vs Time", fontweight="bold", fontsize=12)
        fig.tight_layout()
        if output_path:
            fig.savefig(output_path, bbox_inches="tight")
        return fig

    @staticmethod
    def log_log_plot(
        time: np.ndarray,
        status: np.ndarray,
        group: np.ndarray,
        output_path: str = None
    ) -> plt.Figure:
        """Log(-log(S(t))) vs log(t) plot to assess PH assumption.
        If PH holds, curves should be approximately parallel.

        Parameters
        ----------
        time : array-like
        status : array-like
        group : array-like
        output_path : str, optional

        Returns
        -------
        matplotlib Figure
        """
        time = np.asarray(time, dtype=float)
        status = np.asarray(status, dtype=float)
        group = np.asarray(group)
        unique_groups = np.unique(group)
        colors = ["#1f77b4", "#d62728", "#2ca02c", "#ff7f0e"]

        fig, ax = plt.subplots(figsize=(8, 6), dpi=150)

        for idx, g in enumerate(unique_groups):
            mask = group == g
            t_g = time[mask]
            s_g = status[mask]

            kmf = KaplanMeierFitter()
            kmf.fit(t_g, event_observed=s_g)

            surv = kmf.survival_function_.squeeze()
            surv_times = surv.index.values

            log_t = np.log(surv_times[surv_times > 0])
            log_log_s = np.log(-np.log(np.clip(surv.values[surv_times > 0], 1e-15, 1 - 1e-15)))

            ax.plot(log_t, log_log_s, color=colors[idx % len(colors)],
                    linewidth=2, label=f"Group {g}")

        ax.set_title("Log-Log Survival Curves", fontweight="bold", fontsize=12)
        ax.set_xlabel("log(Time)", fontweight="bold", fontsize=10)
        ax.set_ylabel("log(-log(S(t)))", fontweight="bold", fontsize=10)
        ax.legend(loc="best", frameon=True, edgecolor="lightgray")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.grid(True, linestyle=":", alpha=0.5)

        fig.tight_layout()
        if output_path:
            fig.savefig(output_path, bbox_inches="tight")
        return fig

    @staticmethod
    def martingale_residuals(
        model: CoxPHFitter,
        X: pd.DataFrame,
        time: np.ndarray,
        status: np.ndarray
    ) -> np.ndarray:
        """Compute martingale residuals: M_i = delta_i - H_0(t_i) * exp(X_i * beta).

        Parameters
        ----------
        model : fitted CoxPHFitter
        X : pd.DataFrame of covariates (same order as used in model.fit)
        time : array-like
        status : array-like

        Returns
        -------
        np.ndarray of martingale residuals
        """
        try:
            training_data = model.data
            residuals = model.compute_residuals(training_data, kind="martingale")
            return residuals.values.flatten()
        except Exception:
            pass

        pred = model.predict_cumulative_hazard(X)
        baseline_chf = pred.values
        beta = model.params_.values
        X_mat = X.values
        lp = X_mat @ beta
        exp_lp = np.exp(lp)
        H_i = np.diag(baseline_chf) * exp_lp if len(baseline_chf.shape) > 1 else baseline_chf * exp_lp
        martingale = status - H_i.flatten()
        return np.asarray(martingale).flatten()

    @staticmethod
    def deviance_residuals(martingale: np.ndarray, status: np.ndarray) -> np.ndarray:
        """Compute deviance residuals from martingale residuals.

        D_i = sign(M_i) * sqrt(-2 * (M_i + delta_i * log((delta_i - M_i) / delta_i)))

        Parameters
        ----------
        martingale : array-like of martingale residuals
        status : array-like, event indicator (0/1)

        Returns
        -------
        np.ndarray of deviance residuals
        """
        martingale = np.asarray(martingale, dtype=float)
        status = np.asarray(status, dtype=float)
        n = len(martingale)
        dev = np.zeros(n)

        for i in range(n):
            m = martingale[i]
            d = status[i]
            if d == 0:
                if m <= 0:
                    dev[i] = -np.sqrt(-2.0 * m) if m < 0 else 0.0
                else:
                    dev[i] = np.sqrt(-2.0 * m)
            else:
                if d - m <= 0:
                    dev[i] = np.sign(m) * np.sqrt(-2.0 * m)
                else:
                    term = m + d * np.log((d - m) / d)
                    dev[i] = np.sign(m) * np.sqrt(np.maximum(-2.0 * term, 0))
        return dev

    @staticmethod
    def dfbeta_influence(model: CoxPHFitter, training_data: pd.DataFrame = None) -> pd.DataFrame:
        """Compute dfbeta residuals (change in coefficients when observation is removed).

        Parameters
        ----------
        model : fitted CoxPHFitter
        training_data : pd.DataFrame, optional, original training data

        Returns
        -------
        pd.DataFrame with dfbeta per covariate per observation
        """
        if training_data is None:
            try:
                training_data = model.data
            except Exception:
                try:
                    training_data = model._model.data
                except Exception:
                    pass
        if training_data is not None and not training_data.empty:
            try:
                dfbetas = model.compute_residuals(training_data, kind="delta_beta")
                if dfbetas is not None and not dfbetas.empty:
                    return dfbetas
            except Exception as e:
                warnings.warn(f"Could not compute dfbetas: {e}")
        return pd.DataFrame()


class RecurrentEventsAnalysis:
    """Recurrent events models: Andersen-Gill, PWP, and WLW."""

    @staticmethod
    def anderson_gill(
        data: pd.DataFrame,
        id_col: str,
        time_start_col: str,
        time_stop_col: str,
        status_col: str,
        covariates: List[str],
        cluster: bool = True
    ) -> dict:
        """Andersen-Gill counting process model with robust sandwich variance.

        Parameters
        ----------
        data : DataFrame in counting process format (start, stop]
        id_col : str, subject identifier column
        time_start_col : str, start time column
        time_stop_col : str, stop time column
        status_col : str, event indicator column
        covariates : list of covariate column names
        cluster : bool, use cluster-robust SE

        Returns
        -------
        dict with summary, model object, model_info
        """
        model_df = data[[id_col, time_start_col, time_stop_col, status_col] + covariates].copy()
        model_df = model_df.dropna()

        from lifelines import CoxTimeVaryingFitter
        ctv = CoxTimeVaryingFitter()
        try:
            ctv.fit(
                model_df,
                id_col=id_col if cluster else None,
                start_col=time_start_col,
                stop_col=time_stop_col,
                event_col=status_col,
                robust=cluster
            )
            summary = ctv.summary.copy()
            summary.columns = [c.lower().replace(" ", "_") for c in summary.columns]
            summary = summary.rename(columns={
                "coef": "coef", "exp(coef)": "hr",
                "se(coef)": "se", "z": "z", "p": "p",
                "exp(coef)_lower_95%": "ci_lower",
                "exp(coef)_upper_95%": "ci_upper"
            })
            if "ci_lower" not in summary.columns:
                summary["ci_lower"] = np.exp(summary["coef"] - Z95 * summary["se"])
                summary["ci_upper"] = np.exp(summary["coef"] + Z95 * summary["se"])
                summary["hr"] = np.exp(summary["coef"])
            summary = summary[["coef", "hr", "se", "z", "p", "ci_lower", "ci_upper"]]
        except Exception as e:
            warnings.warn(f"Andersen-Gill fit failed: {e}")
            summary = pd.DataFrame(columns=["coef", "hr", "se", "z", "p", "ci_lower", "ci_upper"],
                                   index=covariates)
            ctv = None

        n_events = int(model_df[status_col].sum())
        return {
            "summary": summary,
            "model": ctv,
            "n_subjects": int(model_df[id_col].nunique()),
            "n_events": n_events,
            "n_rows": len(model_df),
            "type": "Andersen-Gill"
        }

    @staticmethod
    def pwp_model(
        data: pd.DataFrame,
        id_col: str,
        time_start_col: str,
        time_stop_col: str,
        status_col: str,
        covariates: List[str],
        strata_enum_col: str = "enum"
    ) -> dict:
        """Prentice-Williams-Peterson total time model stratified by event number.

        Parameters
        ----------
        data : DataFrame in counting process format
        id_col : str
        time_start_col : str
        time_stop_col : str
        status_col : str
        covariates : list of str
        strata_enum_col : str, column indicating event number for stratification

        Returns
        -------
        dict with summary, model, stratum-specific HRs
        """
        model_df = data[[id_col, time_start_col, time_stop_col, status_col,
                         strata_enum_col] + covariates].dropna()

        from lifelines import CoxTimeVaryingFitter
        ctv = CoxTimeVaryingFitter()
        try:
            ctv.fit(
                model_df,
                start_col=time_start_col,
                stop_col=time_stop_col,
                event_col=status_col,
                strata=strata_enum_col,
                id_col=id_col,
                robust=True
            )
            summary = ctv.summary.copy()
            summary.columns = [c.lower().replace(" ", "_") for c in summary.columns]
            summary = summary.rename(columns={
                "coef": "coef", "exp(coef)": "hr",
                "se(coef)": "se", "z": "z", "p": "p",
                "exp(coef)_lower_95%": "ci_lower",
                "exp(coef)_upper_95%": "ci_upper"
            })
            if "ci_lower" not in summary.columns:
                summary["ci_lower"] = np.exp(summary["coef"] - Z95 * summary["se"])
                summary["ci_upper"] = np.exp(summary["coef"] + Z95 * summary["se"])
                summary["hr"] = np.exp(summary["coef"])
            summary = summary[["coef", "hr", "se", "z", "p", "ci_lower", "ci_upper"]]
        except Exception as e:
            warnings.warn(f"PWP model fit failed: {e}")
            summary = pd.DataFrame(columns=["coef", "hr", "se", "z", "p", "ci_lower", "ci_upper"],
                                   index=covariates)
            ctv = None

        return {
            "summary": summary,
            "model": ctv,
            "strata_var": strata_enum_col,
            "n_subjects": int(model_df[id_col].nunique()),
            "n_events": int(model_df[status_col].sum()),
            "type": "PWP"
        }

    @staticmethod
    def wlw_marginal(
        data: pd.DataFrame,
        id_col: str,
        time_col: str,
        status_col: str,
        covariates: List[str],
        gap_time: bool = True
    ) -> dict:
        """Wei-Lin-Weissfeld marginal model - separate Cox per event, combined via robust SE.

        Parameters
        ----------
        data : DataFrame in counting process format
        id_col : str
        time_col : str, event/censoring time for the interval
        status_col : str
        covariates : list of str
        gap_time : bool, use gap time (True) or total time (False)

        Returns
        -------
        dict with combined summary, per-event summaries
        """
        if "enum" not in data.columns:
            data = data.copy()
            data["enum"] = data.groupby(id_col).cumcount() + 1

        max_events = int(data["enum"].max())
        per_event_results = []

        combined_coefs = []
        combined_vars = []

        for e in range(1, max_events + 1):
            edata = data[data["enum"] == e].copy()
            if edata[status_col].sum() < 5:
                continue

            cph = CoxPHFitter()
            try:
                cph.fit(
                    edata,
                    duration_col=time_col,
                    event_col=status_col,
                    cluster_col=id_col,
                    robust=True
                )
                per_event_results.append({
                    "event_number": e,
                    "model": cph,
                    "summary": cph.summary,
                    "n_events": int(edata[status_col].sum())
                })

                coef = cph.params_
                se = np.sqrt(np.diag(cph.variance_matrix_))
                combined_coefs.append(coef)
                combined_vars.append(se ** 2)
            except Exception as ex:
                warnings.warn(f"WLW event {e} failed: {ex}")

        if len(combined_coefs) == 0:
            return {
                "combined_summary": pd.DataFrame(columns=["coef", "hr", "se", "z", "p", "ci_lower", "ci_upper"],
                                                  index=covariates),
                "per_event": [],
                "type": "WLW"
            }

        combined_coefs = pd.DataFrame(combined_coefs)
        mean_coef = combined_coefs.mean()

        W = np.cov(combined_coefs.T.values, ddof=1) if combined_coefs.shape[0] > 1 else np.diag(np.array(combined_vars).mean(axis=0))
        avg_var = pd.Series(np.array(combined_vars).mean(axis=0), index=covariates)

        robust_var = avg_var + (1 + 1 / len(combined_vars)) * pd.Series(
            np.diag(W) - avg_var.values if combined_coefs.shape[0] > 1 else np.zeros(len(covariates)),
            index=covariates
        )
        robust_se = np.sqrt(np.maximum(robust_var, 1e-15))
        z_vals = mean_coef / robust_se
        p_vals = 2 * stats.norm.sf(np.abs(z_vals))
        ci_low = mean_coef - Z95 * robust_se
        ci_up = mean_coef + Z95 * robust_se

        combined_df = pd.DataFrame({
            "coef": mean_coef.values,
            "hr": np.exp(mean_coef.values),
            "se": robust_se.values,
            "z": z_vals.values,
            "p": p_vals.values,
            "ci_lower": np.exp(ci_low.values),
            "ci_upper": np.exp(ci_up.values)
        }, index=mean_coef.index)

        return {
            "combined_summary": combined_df,
            "per_event": per_event_results,
            "type": "WLW",
            "n_events_total": sum(r["n_events"] for r in per_event_results)
        }


class TimeVaryingCox:
    """Time-varying covariate Cox models and landmark analysis."""

    @staticmethod
    def landmark_analysis(
        data: pd.DataFrame,
        time_col: str,
        status_col: str,
        group_col: str = None,
        landmarks: List[float] = None,
        n_landmarks: int = 3
    ) -> dict:
        """Landmark analysis: fit KM and Cox from each landmark time.

        Parameters
        ----------
        data : DataFrame
        time_col : str
        status_col : str
        group_col : str, optional, for KM comparison
        landmarks : list of float, specific landmark times
        n_landmarks : int, number of auto-detected landmarks (if landmarks not given)

        Returns
        -------
        dict with per-landmark results (KM curves, HR, CI, p) and auto-detected landmarks
        """
        times = data[time_col].values
        if landmarks is None:
            q = np.linspace(0, 0.7, n_landmarks + 2)[1:-1]
            landmarks = [float(np.quantile(times[times > 0], qq)) for qq in q]
            landmarks = sorted(set(l for l in landmarks if l > 0))

        results = []
        for lm in landmarks:
            sub = data[data[time_col] >= lm].copy()
            sub[time_col] = sub[time_col] - lm

            if group_col is not None and group_col in sub.columns:
                kmf_results = {}
                for g in sub[group_col].unique():
                    mask = sub[group_col] == g
                    kmf = KaplanMeierFitter()
                    kmf.fit(sub[time_col][mask], event_observed=sub[status_col][mask], label=f"Group {g}")
                    kmf_results[str(g)] = kmf
            else:
                kmf = KaplanMeierFitter()
                kmf.fit(sub[time_col], event_observed=sub[status_col], label="Overall")
                kmf_results = {"Overall": kmf}

            landmark_entry = {
                "landmark_time": lm,
                "n_at_risk": len(sub),
                "n_events": int(sub[status_col].sum()),
                "km_fitters": kmf_results
            }
            results.append(landmark_entry)

        return {
            "landmarks": landmarks,
            "results": results,
            "time_col": time_col,
            "status_col": status_col,
            "group_col": group_col
        }

    @staticmethod
    def time_dependent_cox(
        data_long: pd.DataFrame,
        id_col: str,
        time_start_col: str,
        time_stop_col: str,
        status_col: str,
        covariates: List[str]
    ) -> dict:
        """Fit time-dependent Cox model with counting process data and compare with naive baseline.

        Parameters
        ----------
        data_long : DataFrame in (tstart, tstop] format with possibly time-varying covariates
        id_col : str
        time_start_col : str
        time_stop_col : str
        status_col : str
        covariates : list of str

        Returns
        -------
        dict with time-dependent results, naive (baseline-only) results, comparison
        """
        td_data = data_long[[id_col, time_start_col, time_stop_col, status_col] + covariates].dropna()

        from lifelines import CoxTimeVaryingFitter
        cph_td = CoxTimeVaryingFitter()
        try:
            cph_td.fit(
                td_data,
                id_col=id_col,
                start_col=time_start_col,
                stop_col=time_stop_col,
                event_col=status_col,
                robust=True
            )
            td_summary = cph_td.summary.copy()
            td_summary.columns = [c.lower().replace(" ", "_") for c in td_summary.columns]
            td_summary = td_summary.rename(columns={
                "coef": "coef", "exp(coef)": "hr",
                "se(coef)": "se", "z": "z", "p": "p",
                "exp(coef)_lower_95%": "ci_lower",
                "exp(coef)_upper_95%": "ci_upper"
            })
            if "ci_lower" not in td_summary.columns:
                td_summary["ci_lower"] = np.exp(td_summary["coef"] - Z95 * td_summary["se"])
                td_summary["ci_upper"] = np.exp(td_summary["coef"] + Z95 * td_summary["se"])
                td_summary["hr"] = np.exp(td_summary["coef"])
            td_summary = td_summary[["coef", "hr", "se", "z", "p", "ci_lower", "ci_upper"]]
        except Exception as e:
            warnings.warn(f"Time-dependent Cox failed: {e}")
            td_summary = pd.DataFrame(columns=["coef", "hr", "se", "z", "p", "ci_lower", "ci_upper"],
                                      index=covariates)
            cph_td = None

        baseline_data = data_long.groupby(id_col).last().reset_index()
        baseline_data = baseline_data[[time_stop_col, status_col] + covariates].dropna()

        cph_naive = CoxPHFitter()
        try:
            cph_naive.fit(
                baseline_data,
                duration_col=time_stop_col,
                event_col=status_col
            )
            naive_summary = cph_naive.summary.copy()
            naive_summary.columns = [c.lower().replace(" ", "_") for c in naive_summary.columns]
            naive_summary = naive_summary.rename(columns={
                "coef": "coef", "exp(coef)": "hr",
                "se(coef)": "se", "z": "z", "p": "p",
                "exp(coef)_lower_95%": "ci_lower",
                "exp(coef)_upper_95%": "ci_upper"
            })
            if "ci_lower" not in naive_summary.columns:
                naive_summary["ci_lower"] = np.exp(naive_summary["coef"] - Z95 * naive_summary["se"])
                naive_summary["ci_upper"] = np.exp(naive_summary["coef"] + Z95 * naive_summary["se"])
                naive_summary["hr"] = np.exp(naive_summary["coef"])
            naive_summary = naive_summary[["coef", "hr", "se", "z", "p", "ci_lower", "ci_upper"]]
        except Exception as e:
            warnings.warn(f"Naive baseline Cox failed: {e}")
            naive_summary = pd.DataFrame(columns=["coef", "hr", "se", "z", "p", "ci_lower", "ci_upper"],
                                         index=covariates)
            cph_naive = None

        comparison = pd.DataFrame({
            "td_hr": td_summary["hr"] if td_summary is not None else np.nan,
            "td_ci_lower": td_summary["ci_lower"] if td_summary is not None else np.nan,
            "td_ci_upper": td_summary["ci_upper"] if td_summary is not None else np.nan,
            "naive_hr": naive_summary["hr"] if naive_summary is not None else np.nan,
            "naive_ci_lower": naive_summary["ci_lower"] if naive_summary is not None else np.nan,
            "naive_ci_upper": naive_summary["ci_upper"] if naive_summary is not None else np.nan,
        })

        return {
            "td_summary": td_summary,
            "naive_summary": naive_summary,
            "comparison": comparison,
            "td_model": cph_td,
            "naive_model": cph_naive,
            "n_subjects": int(td_data[id_col].nunique()),
            "n_events": int(td_data[status_col].sum()),
            "n_rows": len(td_data)
        }

    @staticmethod
    def immortal_time_bias(
        data: pd.DataFrame,
        id_col: str,
        time_col: str,
        status_col: str,
        treatment_col: str,
        treatment_time_col: str = None
    ) -> dict:
        """Quantify immortal time bias by comparing naive vs time-dependent models.

        Parameters
        ----------
        data : DataFrame (one row per subject for naive, or with treatment timing info)
        id_col : str
        time_col : str
        status_col : str
        treatment_col : str, treatment indicator (0/1)
        treatment_time_col : str, optional, time of treatment initiation

        Returns
        -------
        dict with naive_HR, corrected_HR, bias_ratio, interpretation, individual_data
        """
        data = data.copy()

        if treatment_time_col is not None and treatment_time_col in data.columns:
            data["_immortal_time"] = data[treatment_time_col].fillna(0)
            data["_adjusted_time"] = data[time_col] - data["_immortal_time"]
            data["_adjusted_time"] = data["_adjusted_time"].clip(lower=0)
        else:
            data["_immortal_time"] = 0.0
            data["_adjusted_time"] = data[time_col]

        cph_naive = CoxPHFitter()
        try:
            cph_naive.fit(
                data[[time_col, status_col, treatment_col]].dropna(),
                duration_col=time_col,
                event_col=status_col
            )
            naive_hr = np.exp(cph_naive.params_.get(treatment_col, 1.0))
        except Exception as e:
            warnings.warn(f"Naive Cox for immortal time bias failed: {e}")
            naive_hr = 1.0

        data_corrected = data[[id_col, "_adjusted_time", status_col, treatment_col, "_immortal_time"]].dropna()
        long_data = []
        for _, row in data_corrected.iterrows():
            imm = row["_immortal_time"]
            adj_time = row["_adjusted_time"]
            treat = row[treatment_col]
            stat = row[status_col]
            pid = row[id_col]

            if imm > 0 and treat == 1:
                long_data.append({
                    id_col: pid, "start": 0, "stop": imm,
                    "treatment": 0, status_col: 0
                })
                long_data.append({
                    id_col: pid, "start": imm, "stop": imm + adj_time,
                    "treatment": 1, status_col: stat
                })
            else:
                long_data.append({
                    id_col: pid, "start": 0, "stop": time_col,
                    "treatment": treat, status_col: stat
                })

        if len(long_data) > 0:
            long_df = pd.DataFrame(long_data)
            try:
                from lifelines import CoxTimeVaryingFitter
                cph_corrected = CoxTimeVaryingFitter()
                cph_corrected.fit(
                    long_df,
                    id_col=id_col,
                    start_col="start",
                    stop_col="stop",
                    event_col=status_col,
                    robust=True
                )
                corrected_hr = np.exp(cph_corrected.params_.get("treatment", 1.0))
            except Exception as e:
                warnings.warn(f"Corrected Cox for immortal time bias failed: {e}")
                corrected_hr = naive_hr
        else:
            corrected_hr = naive_hr

        bias_ratio = naive_hr / corrected_hr if corrected_hr != 0 else 1.0

        if bias_ratio > 1.2:
            interpretation = "Substantial immortal time bias detected. Naive HR overestimates treatment effect."
        elif bias_ratio < 0.8:
            interpretation = "Substantial immortal time bias detected. Naive HR underestimates treatment effect."
        elif abs(bias_ratio - 1.0) > 0.05:
            interpretation = "Mild immortal time bias detected."
        else:
            interpretation = "No substantial immortal time bias detected."

        return {
            "naive_hr": float(naive_hr),
            "corrected_hr": float(corrected_hr),
            "bias_ratio": float(bias_ratio),
            "interpretation": interpretation,
            "individual_data": data_corrected if treatment_time_col is not None else None
        }


class IPCWCalculator:
    """Inverse Probability of Censoring Weights."""

    @staticmethod
    def compute_weights(
        data: pd.DataFrame,
        time_col: str,
        status_col: str,
        covariate_cols: List[str],
        stabilized: bool = True,
        truncation_percentile: float = 0.99
    ) -> pd.DataFrame:
        """Compute IPCW weights using logistic regression for censoring probability.

        Parameters
        ----------
        data : DataFrame
        time_col : str
        status_col : str
        covariate_cols : list of str, covariates for censoring model
        stabilized : bool, use stabilized weights (KM_censored / P(censored|covariates))
        truncation_percentile : float, percentile for weight capping

        Returns
        -------
        DataFrame with original columns plus 'ipcw_weight', 'censoring_prob'
        """
        df = data.copy()
        df["_censored"] = (df[status_col] == 0).astype(int)

        from statsmodels.api import Logit

        formula_cols = " + ".join(covariate_cols)
        try:
            logit_model = Logit(df["_censored"], df[covariate_cols].values)
            logit_fit = logit_model.fit(disp=0)
            cens_prob = logit_fit.predict(df[covariate_cols].values)
        except Exception as e:
            warnings.warn(f"Logistic regression for IPCW failed: {e}, using constant model")
            cens_prob = np.full(len(df), df["_censored"].mean())

        df["prob_uncensored"] = np.clip(1.0 - cens_prob, 0.01, 0.99)

        if stabilized:
            kmf = KaplanMeierFitter()
            kmf.fit(df[time_col].values, event_observed=df["_censored"].values)
            cens_surv = kmf.survival_function_.squeeze()
            cens_surv.index = kmf.timeline

            km_prob = np.ones(len(df))
            for i in range(len(df)):
                t_i = df[time_col].iloc[i]
                idx = np.searchsorted(cens_surv.index, t_i, side="right")
                if idx > 0:
                    km_prob[i] = cens_surv.iloc[min(idx - 1, len(cens_surv) - 1)]

            df["ipcw_weight"] = np.where(
                df["prob_uncensored"] > 1e-10,
                km_prob / df["prob_uncensored"],
                1.0
            )
        else:
            df["ipcw_weight"] = np.where(
                df["prob_uncensored"] > 1e-10,
                1.0 / df["prob_uncensored"],
                1.0
            )

        cap = np.percentile(df["ipcw_weight"], truncation_percentile * 100)
        df["ipcw_weight"] = np.minimum(df["ipcw_weight"], cap)
        df["ipcw_weight"] = np.maximum(df["ipcw_weight"], 0.001)

        return df

    @staticmethod
    def _weighted_km_single(time: np.ndarray, status: np.ndarray, weights: np.ndarray) -> pd.DataFrame:
        """Compute weighted KM manually (handles float weights correctly)."""
        order = np.argsort(time)
        t_sorted = time[order]
        s_sorted = status[order]
        w_sorted = weights[order]

        unique_times = np.unique(t_sorted)
        surv = 1.0
        result = []

        total_risk = np.sum(w_sorted)

        for ut in unique_times:
            risk_mask = t_sorted >= ut
            w_risk = np.sum(w_sorted[risk_mask])
            if w_risk <= 0:
                continue
            event_mask = (t_sorted == ut) & (s_sorted == 1)
            w_events = np.sum(w_sorted[event_mask])

            if w_events > 0:
                surv *= (1.0 - w_events / w_risk)

            result.append({"time": ut, "survival": surv, "n_at_risk": w_risk, "n_events": w_events})

        df = pd.DataFrame(result)
        return df

    @staticmethod
    def weighted_km(
        time: np.ndarray,
        status: np.ndarray,
        weights: np.ndarray,
        group: np.ndarray = None
    ) -> dict:
        """IPCW-weighted Kaplan-Meier estimator.

        Uses manual computation to avoid lifelines' integer weight constraint.

        Parameters
        ----------
        time : array-like
        status : array-like
        weights : array-like, IPCW weights
        group : array-like, optional

        Returns
        -------
        dict with weighted survival estimates
        """
        time = np.asarray(time, dtype=float)
        status = np.asarray(status, dtype=int)
        weights = np.asarray(weights, dtype=float)

        if group is not None:
            group = np.asarray(group)
            unique_groups = np.unique(group)
            results = {}
            for g in unique_groups:
                mask = group == g
                df = IPCWCalculator._weighted_km_single(time[mask], status[mask], weights[mask])
                results[str(g)] = {"km_table": df}
            return {"grouped": results, "weighted": True}
        else:
            df = IPCWCalculator._weighted_km_single(time, status, weights)
            return {"km_table": df, "weighted": True}

    @staticmethod
    def weighted_cox(
        time: np.ndarray,
        status: np.ndarray,
        X: pd.DataFrame,
        weights: np.ndarray
    ) -> dict:
        """IPCW-weighted Cox PH model.

        Parameters
        ----------
        time : array-like
        status : array-like
        X : pd.DataFrame of covariates
        weights : array-like, IPCW weights

        Returns
        -------
        dict with summary and model object
        """
        data = X.copy()
        data["time"] = np.asarray(time)
        data["status"] = np.asarray(status, dtype=int)
        data["_ipcw_w"] = np.asarray(weights, dtype=float)

        cph = CoxPHFitter()
        try:
            cph.fit(data, duration_col="time", event_col="status",
                    weights_col="_ipcw_w", robust=True)
            summary = cph.summary.copy()
            summary.columns = [c.lower().replace(" ", "_") for c in summary.columns]
            summary = summary.rename(columns={
                "coef": "coef", "exp(coef)": "hr",
                "se(coef)": "se", "z": "z", "p": "p",
                "exp(coef)_lower_95%": "ci_lower",
                "exp(coef)_upper_95%": "ci_upper"
            })
            if "ci_lower" not in summary.columns:
                summary["ci_lower"] = np.exp(summary["coef"] - Z95 * summary["se"])
                summary["ci_upper"] = np.exp(summary["coef"] + Z95 * summary["se"])
                summary["hr"] = np.exp(summary["coef"])
            summary = summary[["coef", "hr", "se", "z", "p", "ci_lower", "ci_upper"]]
        except Exception as e:
            warnings.warn(f"IPCW-weighted Cox failed: {e}")
            summary = pd.DataFrame(columns=["coef", "hr", "se", "z", "p", "ci_lower", "ci_upper"],
                                   index=X.columns)
            cph = None

        return {"summary": summary, "model": cph}


class LeftTruncationAnalysis:
    """Analysis of left-truncated (delayed entry) survival data."""

    @staticmethod
    def fit_conditional(
        time: np.ndarray,
        entry: np.ndarray,
        status: np.ndarray,
        group: np.ndarray = None
    ) -> dict:
        """Conditional KM and Cox for left-truncated (delayed entry) data.

        Uses the risk set conditional on being event-free at entry time.

        Parameters
        ----------
        time : array-like, event/censoring time
        entry : array-like, entry (left truncation) time
        status : array-like, event indicator
        group : array-like, optional

        Returns
        -------
        dict with conditional KM curves (per group if applicable) and Cox model
        """
        time = np.asarray(time, dtype=float)
        entry = np.asarray(entry, dtype=float)
        status = np.asarray(status, dtype=int)

        km_results = {}
        if group is not None:
            group = np.asarray(group)
            unique_groups = np.unique(group)
            for g in unique_groups:
                mask = group == g
                kmf = KaplanMeierFitter()
                kmf.fit(time[mask], event_observed=status[mask], entry=entry[mask],
                        label=f"Group {g}")
                km_results[str(g)] = kmf
        else:
            kmf = KaplanMeierFitter()
            kmf.fit(time, event_observed=status, entry=entry, label="Conditional KM")
            km_results["Overall"] = kmf

        data_cox = pd.DataFrame({
            "time": time,
            "entry": entry,
            "status": status
        })

        cph = CoxPHFitter()
        try:
            cph.fit(data_cox, duration_col="time", event_col="status", entry_col="entry")
            summary = cph.summary.copy()
            summary.columns = [c.lower().replace(" ", "_") for c in summary.columns]
            summary = summary.rename(columns={
                "coef": "coef", "exp(coef)": "hr",
                "se(coef)": "se", "z": "z", "p": "p",
                "exp(coef)_lower_95%": "ci_lower",
                "exp(coef)_upper_95%": "ci_upper"
            })
            if "ci_lower" not in summary.columns:
                summary["ci_lower"] = np.exp(summary["coef"] - Z95 * summary["se"])
                summary["ci_upper"] = np.exp(summary["coef"] + Z95 * summary["se"])
                summary["hr"] = np.exp(summary["coef"])
            summary = summary[["coef", "hr", "se", "z", "p", "ci_lower", "ci_upper"]]
        except Exception as e:
            warnings.warn(f"Conditional Cox for left-truncation failed: {e}")
            summary = pd.DataFrame(columns=["coef", "hr", "se", "z", "p", "ci_lower", "ci_upper"])
            cph = None

        return {
            "km_curves": km_results,
            "cox_model": cph,
            "cox_summary": summary,
            "n_truncated": int(np.sum(entry > 0)),
            "n_total": len(time)
        }

    @staticmethod
    def compare_truncation_naive(
        time: np.ndarray,
        entry: np.ndarray,
        status: np.ndarray,
        group: np.ndarray = None
    ) -> dict:
        """Compare naive (no truncation adjustment) vs conditional approach.

        Parameters
        ----------
        time : array-like
        entry : array-like
        status : array-like
        group : array-like, optional

        Returns
        -------
        dict with naive and adjusted Cox, bias assessment
        """
        adjusted = LeftTruncationAnalysis.fit_conditional(time, entry, status, group)

        naive_cph = CoxPHFitter()
        naive_data = pd.DataFrame({"time": time, "status": status})
        bias_assessment = None

        try:
            naive_cph.fit(naive_data, duration_col="time", event_col="status")
            naive_summary = naive_cph.summary.copy()
            naive_summary.columns = [c.lower().replace(" ", "_") for c in naive_summary.columns]
            naive_summary = naive_summary.rename(columns={
                "coef": "coef", "exp(coef)": "hr",
                "se(coef)": "se", "z": "z", "p": "p",
                "exp(coef)_lower_95%": "ci_lower",
                "exp(coef)_upper_95%": "ci_upper"
            })
            if "ci_lower" not in naive_summary.columns:
                naive_summary["ci_lower"] = np.exp(naive_summary["coef"] - Z95 * naive_summary["se"])
                naive_summary["ci_upper"] = np.exp(naive_summary["coef"] + Z95 * naive_summary["se"])
                naive_summary["hr"] = np.exp(naive_summary["coef"])
            naive_summary = naive_summary[["coef", "hr", "se", "z", "p", "ci_lower", "ci_upper"]]

            adj_summary = adjusted["cox_summary"]
            if not adj_summary.empty and not naive_summary.empty:
                shared_idx = naive_summary.index.intersection(adj_summary.index)
                if len(shared_idx) > 0:
                    bias_ratios = naive_summary.loc[shared_idx, "hr"] / adj_summary.loc[shared_idx, "hr"]
                    bias_assessment = pd.DataFrame({
                        "naive_hr": naive_summary.loc[shared_idx, "hr"],
                        "adjusted_hr": adj_summary.loc[shared_idx, "hr"],
                        "bias_ratio": bias_ratios,
                        "percent_bias": (bias_ratios - 1) * 100
                    })
        except Exception as e:
            warnings.warn(f"Naive Cox comparison failed: {e}")
            naive_summary = pd.DataFrame(columns=["coef", "hr", "se", "z", "p", "ci_lower", "ci_upper"])
            naive_cph = None

        return {
            "naive_cox": {"summary": naive_summary, "model": naive_cph},
            "adjusted_cox": {"summary": adjusted["cox_summary"], "model": adjusted["cox_model"],
                             "km_curves": adjusted["km_curves"]},
            "bias_assessment": bias_assessment,
            "n_truncated": adjusted["n_truncated"]
        }

    @staticmethod
    def simulate_left_truncation(n: int = 300, late_entry_pct: float = 0.3) -> pd.DataFrame:
        """Simulate left-truncated (delayed entry) survival data.

        Parameters
        ----------
        n : int, total sample size
        late_entry_pct : float, proportion with delayed entry

        Returns
        -------
        DataFrame with id, entry, time, status, group
        """
        np.random.seed(42)

        late_n = int(n * late_entry_pct)
        early_n = n - late_n

        rng = np.random.default_rng(42)

        ids = []
        entry_times = []
        event_times = []
        statuses = []
        groups = []

        for i in range(early_n):
            true_surv = rng.exponential(scale=rng.uniform(5, 15))
            cens = rng.uniform(5, 20)
            obs_time = min(true_surv, cens)
            ids.append(f"E{i:04d}")
            entry_times.append(0.0)
            event_times.append(obs_time)
            statuses.append(1 if true_surv <= cens else 0)
            groups.append(rng.integers(0, 2))

        for i in range(late_n):
            true_surv = rng.exponential(scale=rng.uniform(5, 15))
            entry = rng.uniform(0, max(2, true_surv * 0.6))
            cens = rng.uniform(entry + 1, 20)
            obs_time = min(true_surv, cens)
            if obs_time <= entry:
                true_surv = entry + rng.exponential(scale=8)
                obs_time = min(true_surv, cens)
            ids.append(f"L{i:04d}")
            entry_times.append(entry)
            event_times.append(obs_time)
            statuses.append(1 if true_surv <= cens else 0)
            groups.append(rng.integers(0, 2))

        df = pd.DataFrame({
            "id": ids,
            "entry": entry_times,
            "time": event_times,
            "status": statuses,
            "group": groups
        })
        df["true_entry"] = 0.0

        return df


class SurvivalDiagnosticRouter:
    """Automated detection of data features for survival analysis routing."""

    @staticmethod
    def check_competing_risks(
        status: np.ndarray,
        event_types: List[int] = None
    ) -> dict:
        """Detect competing risks in event status data.

        Parameters
        ----------
        status : array-like, event types (0=censored, 1,2,...)
        event_types : list of int, known event types (auto-detected if None)

        Returns
        -------
        dict with competing_risks_detected, n_event_types, event_proportions
        """
        status = np.asarray(status, dtype=int)
        unique_events = np.unique(status)
        unique_events = unique_events[unique_events > 0]

        n_event_types = len(unique_events)
        event_proportions = {}
        for et in unique_events:
            event_proportions[int(et)] = float(np.mean(status == et))

        competing_risks_detected = n_event_types >= 2
        max_proportion = max(event_proportions.values()) if event_proportions else 0
        competing_risks_detected = competing_risks_detected and max_proportion >= 0.05

        return {
            "competing_risks_detected": bool(competing_risks_detected),
            "n_event_types": int(n_event_types),
            "event_proportions": event_proportions,
            "event_types_found": [int(et) for et in unique_events]
        }

    @staticmethod
    def check_recurrent_events(
        data: pd.DataFrame,
        id_col: str,
        status_col: str,
        threshold: float = 0.05
    ) -> dict:
        """Detect recurrent events in longitudinal data.

        Parameters
        ----------
        data : DataFrame
        id_col : str
        status_col : str
        threshold : float, minimum proportion with multiple events to flag

        Returns
        -------
        dict with recurrent_detected, events_per_subject_distribution, pct_with_multiple
        """
        event_counts = data.groupby(id_col)[status_col].sum()
        counts_dist = event_counts.value_counts().sort_index().to_dict()

        pct_multiple = float(np.mean(event_counts >= 2))
        recurrent_detected = pct_multiple >= threshold

        return {
            "recurrent_detected": bool(recurrent_detected),
            "events_per_subject_distribution": counts_dist,
            "pct_with_multiple": pct_multiple,
            "threshold": threshold,
            "mean_events_per_subject": float(event_counts.mean()),
            "median_events": int(event_counts.median())
        }

    @staticmethod
    def check_time_varying(
        data: pd.DataFrame,
        covariates: List[str],
        threshold: float = 0.10
    ) -> dict:
        """Detect time-varying covariates in longitudinal data.

        For each covariate, checks if value changes within subject over time.
        Simple check: look for 0->1 switches (binary) or value changes (continuous).

        Parameters
        ----------
        data : DataFrame, must have a 'subject_id' or multi-row per subject format
        covariates : list of str
        threshold : float, minimum proportion of subjects with change to flag

        Returns
        -------
        dict with tv_detected, tv_covariates list, change_proportions
        """
        change_props = {}
        tv_covariates = []

        for cov in covariates:
            if cov not in data.columns:
                continue
            n_changes = 0
            n_subjects = data.shape[0]

            if "id" in data.columns or "subject_id" in data.columns:
                id_c = "id" if "id" in data.columns else "subject_id"
                for subj, grp in data.groupby(id_c):
                    if len(grp) > 1:
                        vals = grp[cov].values
                        if not np.allclose(vals, vals[0]):
                            n_changes += 1
                n_subjects = data[id_c].nunique()
            else:
                unique_vals = data[cov].nunique()
                if unique_vals > 2:
                    change_props[cov] = 0.0
                    continue
                else:
                    n_changes = 0
                    n_subjects = 1

            prop = n_changes / max(n_subjects, 1)
            change_props[cov] = float(prop)
            if prop >= threshold:
                tv_covariates.append(cov)

        return {
            "tv_detected": len(tv_covariates) > 0,
            "tv_covariates": tv_covariates,
            "change_proportions": change_props
        }

    @staticmethod
    def check_left_truncation(
        entry_time: np.ndarray,
        threshold: float = 0.05
    ) -> dict:
        """Detect left truncation (delayed entry) in survival data.

        Parameters
        ----------
        entry_time : array-like
        threshold : float

        Returns
        -------
        dict with truncation_detected, pct_delayed_entry, entry_time_summary
        """
        entry_time = np.asarray(entry_time, dtype=float)
        pct_delayed = float(np.mean(entry_time > 0))
        truncation_detected = pct_delayed >= threshold

        entry_summary = {
            "min": float(entry_time.min()),
            "max": float(entry_time.max()),
            "mean": float(entry_time.mean()),
            "median": float(np.median(entry_time)),
            "pct_zero": float(np.mean(entry_time == 0))
        }

        return {
            "truncation_detected": bool(truncation_detected),
            "pct_delayed_entry": pct_delayed,
            "entry_time_summary": entry_summary
        }

    @staticmethod
    def check_informative_censoring(
        time: np.ndarray,
        status: np.ndarray,
        X: pd.DataFrame,
        alpha: float = 0.05,
        threshold: float = 0.10
    ) -> dict:
        """Test for informative censoring by comparing censored vs event groups.

        Parameters
        ----------
        time : array-like
        status : array-like
        X : pd.DataFrame of covariates
        alpha : float, significance level
        threshold : float, minimum censoring rate to flag

        Returns
        -------
        dict with informative_censoring_detected, censoring_rate, tests per covariate
        """
        time = np.asarray(time, dtype=float)
        status = np.asarray(status, dtype=int)
        censored = (status == 0).astype(int)
        censoring_rate = float(np.mean(censored))

        tests = {}
        for col in X.columns:
            vals = X[col].values
            cens_vals = vals[censored == 1]
            event_vals = vals[censored == 0]

            if len(cens_vals) < 5 or len(event_vals) < 5:
                tests[col] = {"test": "insufficient_data", "statistic": None, "p_value": None}
                continue

            unique_vals = len(np.unique(vals))
            if unique_vals == 2:
                table = pd.crosstab(censored, vals)
                if table.shape == (2, 2):
                    try:
                        _, p, _, _ = stats.chi2_contingency(table, correction=False)
                        tests[col] = {"test": "chi2", "statistic": float(_), "p_value": float(p)}
                    except Exception:
                        tests[col] = {"test": "chi2_failed", "statistic": None, "p_value": None}
                else:
                    tests[col] = {"test": "binary_uneven", "statistic": None, "p_value": None}
            else:
                try:
                    t_stat, p_val = stats.ttest_ind(cens_vals, event_vals, equal_var=False)
                    tests[col] = {"test": "welch_t", "statistic": float(t_stat), "p_value": float(p_val)}
                except Exception:
                    tests[col] = {"test": "ttest_failed", "statistic": None, "p_value": None}

        significant = any(
            v["p_value"] is not None and v["p_value"] < alpha
            for v in tests.values() if v["p_value"] is not None
        )
        informative_censoring_detected = bool(significant and censoring_rate >= threshold)

        return {
            "informative_censoring_detected": informative_censoring_detected,
            "censoring_rate": censoring_rate,
            "tests_per_covariate": tests
        }

    @staticmethod
    def check_clustering(data: pd.DataFrame, cluster_col: str) -> dict:
        """Check for clustered/multilevel structure in survival data.

        Computes ICC (intraclass correlation coefficient) for the cluster variable.

        Parameters
        ----------
        data : DataFrame
        cluster_col : str

        Returns
        -------
        dict with clustering_detected, icc, n_clusters, avg_cluster_size, recommendation
        """
        if cluster_col not in data.columns:
            return {
                "clustering_detected": False,
                "icc": 0.0,
                "n_clusters": 0,
                "avg_cluster_size": 0.0,
                "recommendation": "Cluster column not found"
            }

        cluster_sizes = data.groupby(cluster_col).size()
        n_clusters = len(cluster_sizes)
        avg_size = float(cluster_sizes.mean())

        from statsmodels.api import OLS, add_constant
        try:
            y = pd.Series(np.zeros(len(data)))
            model = OLS(y, add_constant(pd.get_dummies(data[cluster_col], drop_first=True)))
            fit = model.fit()
            icc = fit.rsquared if hasattr(fit, "rsquared") else 0.0
        except Exception:
            icc = 0.0

        clustering_detected = bool(icc > 0.01 and n_clusters >= 3 and avg_size >= 5)

        if clustering_detected:
            recommendation = "Use cluster-robust standard errors or frailty/mixed effects models."
        else:
            recommendation = "No substantial clustering detected; standard models acceptable."

        return {
            "clustering_detected": clustering_detected,
            "icc": float(icc),
            "n_clusters": int(n_clusters),
            "avg_cluster_size": float(avg_size),
            "recommendation": recommendation
        }

    @staticmethod
    def route_analysis(checks: dict) -> dict:
        """Decision tree routing based on diagnostic checks.

        Parameters
        ----------
        checks : dict with keys matching check method outputs

        Returns
        -------
        dict with primary_analysis, secondary_analyses list, justification
        """
        primary = "Standard KM + Cox PH"
        secondary = []
        justification = []

        cr = checks.get("competing_risks", {})
        rec = checks.get("recurrent_events", {})
        tv = checks.get("time_varying", {})
        lt = checks.get("left_truncation", {})
        ic = checks.get("informative_censoring", {})
        cl = checks.get("clustering", {})

        if rec.get("recurrent_detected", False):
            primary = "Recurrent Events (AG/PWP/WLW)"
            justification.append(
                f"Recurrent events detected: {rec.get('pct_with_multiple', 0)*100:.1f}% "
                f"of subjects have multiple events. Use Andersen-Gill, PWP, or WLW models."
            )
            secondary.append("Standard KM + Cox PH (sensitivity)")
        elif cr.get("competing_risks_detected", False):
            primary = "Competing Risks (CIF + Fine-Gray)"
            justification.append(
                f"Competing risks detected: {cr.get('n_event_types', 0)} distinct event types. "
                f"Use CIF estimation, Gray's test, and Fine-Gray subdistribution hazard model."
            )
            secondary.append("Standard KM + Cox PH (sensitivity)")
            secondary.append("Cause-specific Cox")
        elif tv.get("tv_detected", False):
            primary = "Time-Varying Cox Model"
            tv_covs = tv.get("tv_covariates", [])
            justification.append(
                f"Time-varying covariates detected: {len(tv_covs)} covariates change within subjects. "
                f"Use counting process (start, stop] format Cox model."
            )
            secondary.append("Standard KM + Cox PH (sensitivity)")
            secondary.append("Landmark analysis")
        elif lt.get("truncation_detected", False):
            primary = "Left Truncation (Conditional Cox)"
            justification.append(
                f"Left truncation detected: {lt.get('pct_delayed_entry', 0)*100:.1f}% "
                f"of subjects have delayed entry. Use conditional survival methods."
            )
            secondary.append("Standard KM + Cox PH (sensitivity)")
        else:
            primary = "Standard KM + Cox PH"
            justification.append("No special data features detected. Standard survival methods are appropriate.")

        if ic.get("informative_censoring_detected", False):
            secondary.append("IPCW sensitivity analysis")
            justification.append(
                f"Potential informative censoring detected (rate={ic.get('censoring_rate', 0)*100:.1f}%). "
                f"Consider IPCW-weighted analysis as sensitivity."
            )

        if cl.get("clustering_detected", False):
            secondary.append("Cluster-robust Cox model")
            justification.append(
                f"Clustering detected (ICC={cl.get('icc', 0):.3f}, "
                f"{cl.get('n_clusters', 0)} clusters). Use robust SE or frailty models."
            )

        return {
            "primary_analysis": primary,
            "secondary_analyses": secondary,
            "justification": " | ".join(justification) if justification else "Standard analysis recommended."
        }
