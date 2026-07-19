#!/usr/bin/env python3
"""
Medical Research Data Analysis Pipeline
----------------------------------------
End-to-end statistical analysis for medical research:
  1. Read Excel data + optional study protocol
  2. Automated data cleaning
  3. Variable type detection
  4. Build statistical analysis plan (SAP)
  5. Descriptive statistics (Table 1)
  6. Bivariate analysis (tests + effect sizes)
  7. Multivariable modeling (logistic/linear regression)
  8. Publication-ready figures (ROC, boxplots, barplots, forest plot)
  9. APA-formatted Word report with tables, figures, and interpretation

Usage:
  python run_analysis.py --data path/to/data.xlsx --output path/to/output.docx
  python run_analysis.py --data path/to/data.xlsx --brief "study protocol text" --outcome outcome_var
"""

import argparse
import sys
import os
import json
import subprocess
from typing import Optional, Dict, List, Tuple

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats as scipy_stats

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from agent_core.data_cleaning import DataCleaner, MICEImputer
from agent_core.stats_enhanced import (
    DescriptiveStatsEnhanced, PValueFormatter, GamesHowellPostHoc,
    StandardizedCoefficients, InfluenceDiagnostics
)
from agent_core.analysis_planner import StatisticalPlanner
from agent_core.report_generator import ReportGenerator
from agent_core.diagnostic_toolkit import ROC_Analysis, DiagnosticAccuracy
from agent_core.word_exporter import APAWordExporter
from agent_core.visual_style import set_publication_style, get_palette, apply_figure_style
from agent_core.meta_analysis import MetaPoolingEngine, MetaVisualizer, EffectSizeConverter
from agent_core.causal_inference import PropensityScoreMatcher
from agent_core.biostats import ClinicalRegressionSuite


# Apply publication-quality style globally
set_publication_style(palette='lancet', dpi=800)


def read_excel(filepath: str) -> pd.DataFrame:
    ext = os.path.splitext(filepath)[1].lower()
    if ext == '.xlsx':
        return pd.read_excel(filepath, engine='openpyxl')
    elif ext == '.xls':
        # xlrd is required for legacy .xls; openpyxl does not support it
        return pd.read_excel(filepath, engine='xlrd')
    elif ext == '.csv':
        return pd.read_csv(filepath)
    else:
        raise ValueError(f"Unsupported file format: {ext}")


def auto_detect_outcome(df: pd.DataFrame, brief: str = "", user_outcome: str = "") -> str:
    if user_outcome and user_outcome in df.columns:
        return user_outcome
    cleaner = DataCleaner(df)
    kw = ['outcome', 'dependent', 'result', 'status', 'group', 'diagnosis', 'disease',
          'dr_status', 'retinopathy', 'death', 'event', 'response', 'class']
    if brief:
        import re
        brief_kw = re.findall(r'\b(' + '|'.join(kw) + r')\b', brief.lower())
        kw = brief_kw + kw
    return cleaner.detect_outcome_variable(kw)


def build_table1(df: pd.DataFrame, outcome: str, var_types: Dict[str, str],
                 continuous_vars: List[str], categorical_vars: List[str]) -> Tuple[List[str], List[List[str]]]:
    headers = ['Variable', f'Without {outcome}', f'With {outcome}', 'Test', 'p', 'Effect Size']
    rows = []

    for var in continuous_vars:
        if var == outcome:
            continue
        g = [g.dropna() for _, g in df.groupby(outcome)[var]]
        if len(g) < 2:
            continue
        g0, g1 = g[0].values, g[1].values
        m0, s0 = np.mean(g0), np.std(g0, ddof=1)
        m1, s1 = np.mean(g1), np.std(g1, ddof=1)

        from scipy.stats import shapiro, mannwhitneyu, ttest_ind
        use_mw = (len(g0) < 30 or shapiro(g0).pvalue < 0.05) or (len(g1) < 30 or shapiro(g1).pvalue < 0.05)

        if use_mw:
            stat, pv = mannwhitneyu(g0, g1, alternative='two-sided')
            stat_name = 'U'
            from scipy.stats import norm
            z = norm.ppf(1 - pv / 2) if pv < 1 else 0
            r = abs(z) / np.sqrt(len(g0) + len(g1))
            es = f'r = {r:.2f}'
            cell0 = f'{m0:.1f} +/- {s0:.1f}'
            cell1 = f'{m1:.1f} +/- {s1:.1f}'
        else:
            stat, pv = ttest_ind(g0, g1, equal_var=False)
            stat_name = "Welch's t"
            pooled = np.sqrt((s0**2 + s1**2) / 2)
            d = abs(m1 - m0) / pooled if pooled > 0 else 0
            es = f'd = {d:.2f}'
            cell0 = f'{m0:.1f} +/- {s0:.1f}'
            cell1 = f'{m1:.1f} +/- {s1:.1f}'

        pv_str = f'{pv:.3f}' if pv >= 0.001 else '<.001'
        rows.append([var, cell0, cell1, f'{stat_name} = {stat:.2f}', pv_str, es])

    for var in categorical_vars:
        if var == outcome:
            continue
        ct = pd.crosstab(df[var], df[outcome])
        from scipy.stats import chi2_contingency
        chi2, pv, dof, expected = chi2_contingency(ct)
        n_total = ct.values.sum()
        cramer_v = np.sqrt(chi2 / (n_total * min(ct.shape) - 1)) if n_total * (min(ct.shape) - 1) > 0 else 0
        pv_str = f'{pv:.3f}' if pv >= 0.001 else '<.001'

        for idx, row_val in enumerate(ct.index):
            vals = ct.loc[row_val].values
            total = vals.sum()
            pct0 = vals[0] / ct.iloc[:, 0].sum() * 100 if len(vals) > 0 else 0
            # Guard: crosstab may have only 1 column if outcome is constant in this subset
            pct1 = (vals[1] / ct.iloc[:, 1].sum() * 100) if len(vals) > 1 and ct.shape[1] > 1 else 0
            val1_str = f'{vals[1]} ({pct1:.1f}%)' if len(vals) > 1 else '0 (0.0%)'
            if idx == 0:
                rows.append([f'{var} ({row_val})', f'{vals[0]} ({pct0:.1f}%)', val1_str,
                             f'{chi2:.2f}', pv_str, f'V = {cramer_v:.3f}'])
            else:
                rows.append([f'{var} ({row_val})', f'{vals[0]} ({pct0:.1f}%)', val1_str, '', '', ''])

    return headers, rows


def run_logistic_regression(df: pd.DataFrame, outcome: str, predictors: List[str]) -> Dict:
    import statsmodels.api as sm
    y = df[outcome].map({'yes': 1, 'no': 0, 1: 1, 0: 0})
    if y.isna().any():
        y = pd.Categorical(df[outcome]).codes
    X = df[predictors].select_dtypes(include=np.number).dropna()
    common = X.index.intersection(y.dropna().index)
    X = sm.add_constant(X.loc[common])
    y = y.loc[common]

    if y.nunique() < 2:
        return {'error': 'Outcome has fewer than 2 levels after cleaning'}

    model = sm.Logit(y, X).fit(disp=False, maxiter=200)
    results = {
        'model': model,
        'coef': model.params,
        'pvalues': model.pvalues,
        'conf_int': model.conf_int(),
        'pseudo_r2': model.prsquared,
        'llf': model.llf,
        'aic': model.aic,
        'bic': model.bic,
        'nobs': model.nobs,
        'model_type': 'logistic',
    }
    try:
        from scipy.stats import chi2
        results['lr_stat'] = model.llf - model.llnull if hasattr(model, 'llnull') else None
        # G² = 2*(llf - llnull); use positive sign
        results['lr_pvalue'] = chi2.sf(2 * results['lr_stat'], len(predictors)) if results['lr_stat'] else None
    except Exception:
        pass
    return results


def run_linear_regression(df: pd.DataFrame, outcome: str, predictors: List[str]) -> Dict:
    """OLS regression for continuous outcomes."""
    import statsmodels.api as sm
    y = df[outcome]
    X = df[predictors].select_dtypes(include=np.number).dropna()
    common = X.index.intersection(y.dropna().index)
    X = sm.add_constant(X.loc[common])
    y = y.loc[common]

    if len(y) < len(predictors) + 2:
        return {'error': 'Insufficient observations for linear regression'}

    model = sm.OLS(y, X).fit()
    results = {
        'model': model,
        'coef': model.params,
        'pvalues': model.pvalues,
        'conf_int': model.conf_int(),
        'r2': model.rsquared,
        'adj_r2': model.rsquared_adj,
        'llf': model.llf,
        'aic': model.aic,
        'bic': model.bic,
        'nobs': model.nobs,
        'model_type': 'linear',
    }
    return results


def make_forest_plot(results: Dict, pvalues: Dict = None, title: str = "Forest Plot") -> plt.Figure:
    palette = get_palette('lancet')
    coef = results['coef']
    ci = results['conf_int']
    or_vals = np.exp(coef)
    or_ci_low = np.exp(ci[0])
    or_ci_high = np.exp(ci[1])
    if pvalues is None:
        pvalues = results.get('pvalues', {})

    params = coef.index
    params = params[params != 'const'] if 'const' in params else params
    or_vals = or_vals[params]
    or_ci_low = or_ci_low[params]
    or_ci_high = or_ci_high[params]

    fig, ax = plt.subplots(figsize=(9, max(4.5, len(params) * 0.45)))
    y_pos = np.arange(len(params))

    for i, (p, lo, hi) in enumerate(zip(params, or_ci_low, or_ci_high)):
        pv = pvalues.get(p, 1)
        if pv < 0.001:
            marker_color = palette[0]
            sig_label = '***'
        elif pv < 0.01:
            marker_color = palette[1]
            sig_label = '**'
        elif pv < 0.05:
            marker_color = palette[2]
            sig_label = '*'
        else:
            marker_color = '#BBBBBB'
            sig_label = ''
        ax.errorbar(or_vals[p], y_pos[i],
                    xerr=[[or_vals[p] - lo], [hi - or_vals[p]]],
                    fmt='o', color=marker_color, ecolor='gray', capsize=3, markersize=8, zorder=3)
        val_text = f'{or_vals[p]:.2f} [{lo:.2f}, {hi:.2f}] {sig_label}'
        ax.text(hi * 1.05, y_pos[i], val_text, va='center', fontsize=8, color='#333333')

    ax.axvline(x=1, color='red', linestyle='--', alpha=0.5, linewidth=1)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(params, fontsize=10)
    ax.set_xlabel('Adjusted Odds Ratio (95% CI)', fontsize=11)
    ax.set_title(title, fontsize=13, fontweight='bold')
    ax.set_xscale('log')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.grid(axis='x', alpha=0.3, linestyle=':')
    fig.tight_layout()
    return fig


def make_boxplots(df: pd.DataFrame, outcome: str, continuous_vars: List[str],
                  max_vars: int = 9) -> List[plt.Figure]:
    palette = get_palette('lancet')
    vars_to_plot = [v for v in continuous_vars if v != outcome][:max_vars]
    figs = []
    n_cols = 3
    n_rows = (len(vars_to_plot) + n_cols - 1) // n_cols
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(5 * n_cols, 4 * n_rows))
    axes = axes.flatten() if n_rows * n_cols > 1 else [axes]

    for i, var in enumerate(vars_to_plot):
        ax = axes[i]
        groups = df.groupby(outcome)[var]
        data = [g.dropna().values for _, g in groups]
        labels = [str(l) for l, _ in groups]
        bp = ax.boxplot(data, patch_artist=True)
        ax.set_xticklabels(labels)
        for patch, color in zip(bp['boxes'], palette[:2]):
            patch.set_facecolor(color)
            patch.set_alpha(0.7)
        ax.set_title(var, fontsize=10, fontweight='bold')
        ax.set_ylabel('Value', fontsize=9)
        ax.tick_params(labelsize=8)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

    # Hide any unused subplot panels (use len(vars_to_plot) to avoid NameError when list is empty)
    for j in range(len(vars_to_plot), len(axes)):
        axes[j].set_visible(False)

    fig.suptitle('Distribution of Key Continuous Variables by Outcome', fontsize=13, fontweight='bold', y=1.02)
    fig.tight_layout()
    figs.append(fig)
    return figs


def make_barplots(df: pd.DataFrame, outcome: str, categorical_vars: List[str],
                  max_vars: int = 6) -> List[plt.Figure]:
    palette = get_palette('lancet')
    vars_to_plot = [v for v in categorical_vars if v != outcome][:max_vars]
    figs = []
    n_cols = 2
    n_rows = (len(vars_to_plot) + n_cols - 1) // n_cols

    if n_rows == 0:
        return figs

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(5 * n_cols, 4 * n_rows))
    axes = axes.flatten() if n_rows * n_cols > 1 else [axes]

    for i, var in enumerate(vars_to_plot):
        ax = axes[i]
        ct = pd.crosstab(df[var], df[outcome], normalize='columns') * 100
        ct.plot(kind='bar', ax=ax, color=[palette[0], palette[1]], legend=False, alpha=0.85)
        ax.set_title(var, fontsize=10, fontweight='bold')
        ax.set_ylabel('Percentage (%)', fontsize=9)
        ax.tick_params(labelsize=8)
        ax.legend(title=outcome, fontsize=7)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

    # Hide any unused subplot panels (use len(vars_to_plot) to avoid NameError when list is empty)
    for j in range(len(vars_to_plot), len(axes)):
        axes[j].set_visible(False)

    fig.suptitle('Categorical Variables Stratified by Outcome', fontsize=13, fontweight='bold', y=1.02)
    fig.tight_layout()
    figs.append(fig)
    return figs


def make_scatter_plot(df: pd.DataFrame, outcome: str, x_var: str, y_var: str) -> plt.Figure:
    palette = get_palette('lancet')
    fig, ax = plt.subplots(figsize=(7, 5))
    for idx, group in enumerate(df[outcome].unique()):
        subset = df[df[outcome] == group]
        color = palette[idx % len(palette)]
        ax.scatter(subset[x_var], subset[y_var], alpha=0.6, label=group, s=25, color=color, edgecolors='white', linewidth=0.3)
        # Drop NaN jointly so both arrays stay the same length
        sub_clean = subset[[x_var, y_var]].dropna()
        if len(sub_clean) > 3:
            z = np.polyfit(sub_clean[x_var], sub_clean[y_var], 1)
            p = np.poly1d(z)
            x_line = np.linspace(sub_clean[x_var].min(), sub_clean[x_var].max(), 100)
            ax.plot(x_line, p(x_line), lw=1.5, color=color, alpha=0.7)
    ax.set_xlabel(x_var, fontsize=11)
    ax.set_ylabel(y_var, fontsize=11)
    ax.set_title(f'{y_var} vs {x_var} by {outcome}', fontsize=12, fontweight='bold')
    ax.legend(fontsize=9, frameon=True, edgecolor='lightgray')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.grid(alpha=0.3, linestyle=':')
    fig.tight_layout()
    return fig


def format_p(pv: float) -> str:
    if pv < 0.001:
        return "p < .001"
    return f"p = {pv:.3f}".replace("0.", ".")


def generate_narrative(outcome: str, table1_rows: List[List[str]],
                       logit_results: Dict, auc_val: Optional[float],
                       psm_results: Optional[Dict] = None) -> str:
    lines = []

    lines.append(f"A total of {len(table1_rows)} candidate predictors were evaluated "
                 f"for their association with {outcome}.")

    sig_vars = []
    for r in table1_rows:
        if len(r) >= 5:
            pv_text = r[4]
            try:
                pv_float = float(pv_text) if not pv_text.startswith('<') else 0.0
                if pv_text.startswith('<') or pv_float < 0.05:
                    sig_vars.append(r[0])
            except ValueError:
                pass
    if sig_vars:
        vars_str = ', '.join(sig_vars[:5])
        lines.append(
            f"Bivariate analysis revealed that the following variable(s) were significantly "
            f"associated with {outcome}: {vars_str}."
        )

    if logit_results and 'error' not in logit_results:
        model_type = logit_results.get('model_type', 'logistic')
        sig_preds = []
        for var, pv in logit_results.get('pvalues', {}).items():
            if var != 'const' and pv < 0.05:
                coef_val = logit_results['coef'][var]
                ci = logit_results['conf_int']
                ci_lo = ci[0][var]
                ci_hi = ci[1][var]
                if model_type == 'logistic':
                    or_val = np.exp(coef_val)
                    or_lo = np.exp(ci_lo)
                    or_hi = np.exp(ci_hi)
                    sig_preds.append(f"{var} (aOR = {or_val:.3f}, 95% CI [{or_lo:.3f}, {or_hi:.3f}])")
                else:
                    sig_preds.append(f"{var} (β = {coef_val:.3f}, 95% CI [{ci_lo:.3f}, {ci_hi:.3f}])")

        reg_label = 'logistic' if model_type == 'logistic' else 'linear'
        if sig_preds:
            lines.append(f"Multivariable {reg_label} regression identified {len(sig_preds)} "
                         f"independent predictor{'s' if len(sig_preds) > 1 else ''} of {outcome}: "
                         f"{'; '.join(sig_preds)}.")

        if model_type == 'logistic':
            pseudo_r2 = logit_results.get('pseudo_r2', 0)
            aic = logit_results.get('aic', 0)
            nobs = logit_results.get('nobs', 0)
            lines.append(f"The model explained {pseudo_r2*100:.1f}% of the variance "
                         f"(McFadden pseudo-R² = {pseudo_r2:.3f}, AIC = {aic:.1f}, N = {nobs}).")
            if auc_val:
                qual = 'excellent' if auc_val > 0.9 else 'good' if auc_val > 0.8 else 'fair' if auc_val > 0.7 else 'poor'
                lines.append(f"The model demonstrated {qual} discriminatory performance "
                             f"(AUC = {auc_val:.3f}).")
        else:
            r2 = logit_results.get('r2', 0)
            adj_r2 = logit_results.get('adj_r2', 0)
            aic = logit_results.get('aic', 0)
            nobs = logit_results.get('nobs', 0)
            lines.append(f"The model explained {r2*100:.1f}% of the variance "
                         f"(R² = {r2:.3f}, adjusted R² = {adj_r2:.3f}, AIC = {aic:.1f}, N = {nobs}).")

        if sig_preds:
            first_pred = sig_preds[0].split(' (')[0]
            lines.append(f"These findings suggest that {first_pred} "
                         f"may serve as an independent predictor of {outcome}, after adjusting for confounders.")

    if psm_results:
        lines.append(f"Propensity score matching yielded {psm_results.get('matched_n', 0)} "
                     f"matched observations, reducing confounding by indication.")

    return '\n'.join(lines)


def run_full_pipeline(data_path: str, output_path: str, study_brief: str = "",
                      outcome_var: str = "", generate_plots: bool = True,
                      use_mice: bool = False, run_psm: bool = False,
                      journal_format: str = "", validate_refs: bool = False) -> Dict:
    step_counter = [0]

    def _step(label: str) -> str:
        step_counter[0] += 1
        return f"[{step_counter[0]}] {label}"

    print(_step("Proposal Understanding (Parsing brief)..."))
    if study_brief:
        print(f"  Brief provided: {len(study_brief)} characters.")

    print(_step(f"Data Understanding (Reading data from {data_path})..."))
    df = read_excel(data_path)
    print(f"  Loaded {df.shape[0]} rows x {df.shape[1]} columns")

    print(_step("Data Cleaning..."))
    cleaner = DataCleaner(df)
    cleaner.clean_pipeline(use_mice=use_mice)
    df_clean = cleaner.get_cleaned_df()
    cleaning_report = cleaner.get_report()
    cleaning_audit = cleaner.get_audit()

    outcome = auto_detect_outcome(df_clean, study_brief, outcome_var)
    print(f"  Outcome variable: {outcome}")
    if outcome not in df_clean.columns:
        outcome = df_clean.columns[-1]
        print(f"  Using last column as outcome: {outcome}")

    print(_step("Determine Study Type..."))
    planner = StatisticalPlanner(df_clean, study_brief)
    planner.parse_brief()
    print(f"  Design type: {planner.design.get('design_type', 'unknown')}")

    print(_step("Determine Data Types..."))
    var_types = cleaner.detect_variable_types()
    continuous_vars = [k for k, v in var_types.items() if v in ('continuous', 'ordinal')]
    categorical_vars = [k for k, v in var_types.items() if v in ('binary', 'categorical')]
    outcome_type = var_types.get(outcome, 'binary')
    print(f"  Continuous: {len(continuous_vars)}, Categorical: {len(categorical_vars)}, Outcome type: {outcome_type}")

    print(_step("Normality Testing..."))
    planner.profile_data()
    print(f"  Assessed normality for continuous variables.")
    plan = planner.build_plan()
    print(f"  Plan steps generated: {len(plan)}")

    print(_step("Descriptive Statistics..."))
    headers, table1_rows = build_table1(df_clean, outcome, var_types,
                                         continuous_vars, categorical_vars)
    print(f"  Table 1: {len(table1_rows)} rows")

    auc_val = None
    psm_results = None

    if run_psm:
        print("  [Extra] Running Propensity Score Matching...")
        treatment_var = outcome
        psm_covariates = [v for v in continuous_vars + categorical_vars if v != outcome]
        if len(psm_covariates) >= 2 and df_clean[outcome].nunique() == 2:
            matcher = PropensityScoreMatcher(caliper=0.1)
            try:
                matched_df = matcher.match(df_clean, treatment_var, psm_covariates)
                fig_love = matcher.love_plot(df_clean, treatment_var, psm_covariates)
                print(f"  PSM: matched {len(matched_df)} observations")
                psm_results = {'matched_n': len(matched_df), 'figure': fig_love}
            except Exception as e:
                print(f"  PSM skipped: {e}")
                psm_results = None

    print(_step("Hypothesis Testing..."))
    print(f"  H0: There is no statistically significant association between predictors and {outcome}.")
    print(f"  H1: There is a statistically significant association between predictors and {outcome}.")

    print(_step("Inferential Statistics..."))
    predictors = continuous_vars + categorical_vars
    predictors = [p for p in predictors if p != outcome]
    if outcome_type in ('continuous', 'ordinal'):
        logit_results = run_linear_regression(df_clean, outcome, predictors)
        if 'error' in logit_results:
            print(f"  Warning: {logit_results['error']}")
        else:
            print(f"  R² = {logit_results.get('r2', 0):.3f}")
    else:
        logit_results = run_logistic_regression(df_clean, outcome, predictors)
        if 'error' in logit_results:
            print(f"  Warning: {logit_results['error']}")
            auc_val = None
        else:
            print(f"  Pseudo-R² = {logit_results.get('pseudo_r2', 0):.3f}")

    print(_step("Data Visualization..."))
    figs = []
    if generate_plots:
        box_figs = make_boxplots(df_clean, outcome, continuous_vars)
        for f in box_figs: figs.append(('Boxplots', f))

        bar_figs = make_barplots(df_clean, outcome, categorical_vars)
        for f in bar_figs: figs.append(('Barplots', f))

        if len(continuous_vars) >= 2:
            try:
                xv = continuous_vars[1] if len(continuous_vars) > 1 else continuous_vars[0]
                yv = continuous_vars[0]
                if xv != outcome and yv != outcome:
                    figs.append(('Scatter Plot', make_scatter_plot(df_clean, outcome, xv, yv)))
            except Exception as e:
                print(f"  Scatter plot skipped: {e}")

        if logit_results and 'error' not in logit_results:
            try:
                y_true = pd.Categorical(df_clean[outcome]).codes
                y_score = logit_results['model'].predict()
                roc = ROC_Analysis(y_true, y_score)
                roc_res = roc.compute()
                auc_val = roc_res.get('auc', roc_res.get('AUC'))
                opt = roc.optimal_cutoff(method='youden')
                figs.append(('ROC Curve', roc.plot_roc(annotate_cutoff=True, optimal_cutoff=opt)))
                print(f"  ROC AUC = {auc_val:.3f}")
            except Exception as e:
                print(f"  ROC curve skipped: {e}")
                auc_val = None

            try:
                figs.append(('Forest Plot', make_forest_plot(logit_results)))
            except Exception as e:
                print(f"  Forest plot skipped: {e}")

        if psm_results and psm_results.get('figure'):
            figs.append(('Love Plot', psm_results['figure']))

    print(_step("Statistical Interpretation..."))
    narrative = generate_narrative(outcome, table1_rows, logit_results, auc_val, psm_results)

    print(_step("APA Style Tables and Figures..."))
    exporter = APAWordExporter(f"Analysis of {outcome}")
    exporter.add_heading("1. Background and Study Design", level=1)
    if study_brief: exporter.add_paragraph(study_brief[:2000])
    exporter.add_paragraph(f"This report presents a comprehensive statistical analysis of {outcome} "
                           f"among the study population. A total of {len(df_clean)} patients were analyzed.")

    exporter.add_heading("2. Data Cleaning Methodology", level=1)
    exporter.add_paragraph(cleaning_report[:2000])

    if validate_refs:
        try:
            ref_report_path = output_path.replace('.docx', '_ref_validation.json')
            validate_references_simple(study_brief, ref_report_path)
            exporter.add_paragraph(f"References were validated. Report saved to {ref_report_path}")
        except Exception as e:
            pass

    exporter.add_heading("3. Table 1: Baseline Characteristics", level=1)
    exp_headers = ['Variable', f'Without {outcome}', f'With {outcome}', 'Test', 'p', 'Effect Size']
    exporter.add_apa_table(exp_headers, table1_rows,
                           caption=f"Baseline characteristics stratified by {outcome}. "
                                   f"Continuous: M ± SD or Mdn [IQR]. Effect sizes: Cohen's d or rank-biserial r.")

    exporter.add_heading("4. Bivariate Analysis Results", level=1)
    exporter.add_paragraph(narrative)

    exporter.add_heading("5. Multivariable Regression", level=1)
    if logit_results and 'error' not in logit_results:
        is_logistic = logit_results.get('model_type', 'logistic') == 'logistic'
        lr_headers = ['Predictor', 'aOR' if is_logistic else 'β', '95% CI', 'p', '']
        lr_rows = []
        for var in logit_results['coef'].index:
            if var == 'const': continue
            coef_val = logit_results['coef'][var]
            ci_lo = logit_results['conf_int'][0][var]
            ci_hi = logit_results['conf_int'][1][var]
            if is_logistic:
                coef_display = f'{np.exp(coef_val):.3f}'
                ci_display = f'[{np.exp(ci_lo):.3f}, {np.exp(ci_hi):.3f}]'
            else:
                coef_display = f'{coef_val:.3f}'
                ci_display = f'[{ci_lo:.3f}, {ci_hi:.3f}]'
            pv = logit_results['pvalues'][var]
            pv_str = f'{pv:.3f}' if pv >= 0.001 else '<.001'
            sig = '***' if pv < 0.001 else '**' if pv < 0.01 else '*' if pv < 0.05 else ''
            lr_rows.append([var, coef_display, ci_display, pv_str, sig])

        caption = (f"Independent predictors of {outcome} from multivariable "
                   f"{'logistic' if is_logistic else 'linear'} regression.")
        exporter.add_apa_table(lr_headers, lr_rows, caption=caption)

    exporter.add_heading("6. Figures", level=1)
    for label, fig in figs:
        exporter.add_figure(fig, caption=f"{label} for {outcome}.")

    exporter.add_heading("7. Discussion and Conclusions", level=1)
    exporter.add_paragraph(
        f"The present analysis identified key factors associated with {outcome} in the study population. "
        f"These results should be interpreted considering the study's inherent limitations."
    )

    print(_step(f"Professional Report Writing..."))
    exporter.save(output_path)
    print(f"\nReport saved to: {output_path}")
    return {'status': 'success', 'output': output_path, 'rows': len(table1_rows)}


def run_meta_analysis_simple(effect_sizes: List[float], variances: List[float],
                              study_labels: List[str], measure_name: str = "OR") -> Dict:
    engine = MetaPoolingEngine()
    visualizer = MetaVisualizer()
    result = engine.pool(effect_sizes, variances, method="random")
    result['study_labels'] = study_labels
    result['individual_effects'] = effect_sizes
    result['individual_variances'] = variances
    result['measure_name'] = measure_name
    result['forest_path'] = 'forest_plot.png'
    result['funnel_path'] = 'funnel_plot.png'
    try:
        visualizer.plot_forest(study_labels, effect_sizes, variances, result, sm_name=measure_name)
    except Exception as e:
        print(f"  Forest plot save error: {e}")
    try:
        visualizer.plot_funnel(effect_sizes, variances, result['pooled_effect'], sm_name=measure_name)
    except Exception as e:
        print(f"  Funnel plot save error: {e}")
    return result


def validate_references_simple(text: str, output_path: str = "") -> dict:
    import re as _re
    dois = _re.findall(r'10\.\d{4,9}/[-._;()/:A-Za-z0-9]+', text)
    result = {'n_references_found': len(dois), 'dois': dois, 'validated': []}
    try:
        import requests as _req
        for doi in dois[:10]:
            url = f"https://api.crossref.org/works/{doi}"
            resp = _req.get(url, headers={'User-Agent': 'DataAnalyst/1.0'}, timeout=10)
            if resp.status_code == 200:
                data = resp.json()
                title = data.get('message', {}).get('title', [''])[0]
                result['validated'].append({'doi': doi, 'status': 'valid', 'title': title})
            else:
                result['validated'].append({'doi': doi, 'status': 'not_found'})
    except ImportError:
        result['error'] = 'requests not installed'
    except Exception as e:
        result['error'] = str(e)
    if output_path:
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(result, f, indent=2)
    return result


def main():
    parser = argparse.ArgumentParser(description='Medical Research Data Analysis Pipeline')
    parser.add_argument('--data', '-d', required=True, help='Path to Excel data file (.xlsx/.csv)')
    parser.add_argument('--output', '-o', default='analysis_report.docx', help='Output Word document path')
    parser.add_argument('--brief', '-b', default='', help='Study protocol / research brief text')
    parser.add_argument('--outcome', '-dv', default='', help='Outcome/dependent variable name')
    parser.add_argument('--no-plots', action='store_true', help='Skip figure generation')
    parser.add_argument('--mice', action='store_true', help='Use MICE multiple imputation instead of single imputation')
    parser.add_argument('--psm', action='store_true', help='Run propensity score matching (requires binary outcome)')
    parser.add_argument('--journal', '-j', default='', choices=['mdpi', 'elsevier', ''],
                        help='Format output for specific journal (MDPI or Elsevier)')
    parser.add_argument('--validate-refs', action='store_true', help='Validate references in study brief')
    parser.add_argument('--palette', default='lancet', choices=['lancet', 'nejm', 'jama', 'nature', 'colorblind'],
                        help='Color palette for figures (default: lancet)')
    args = parser.parse_args()

    set_publication_style(palette=args.palette, dpi=800)

    result = run_full_pipeline(
        data_path=args.data,
        output_path=args.output,
        study_brief=args.brief,
        outcome_var=args.outcome,
        generate_plots=not args.no_plots,
        use_mice=args.mice,
        run_psm=args.psm,
        journal_format=args.journal,
        validate_refs=args.validate_refs
    )

    if args.journal:
        try:
            journal_script = os.path.join(os.path.dirname(__file__), '..', 'format_journal_cli.py')
            if os.path.exists(journal_script):
                subprocess.run([
                    sys.executable, journal_script,
                    '--input', result.get('output', args.output),
                    '--format', args.journal,
                    '--output', result.get('output', args.output).replace('.docx', f'_{args.journal}.docx')
                ], check=False)
                print(f"  Journal-formatted version saved.")
        except Exception as e:
            print(f"  Journal formatting skipped: {e}")

    return result


if __name__ == '__main__':
    main()
