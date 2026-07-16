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
from typing import Optional, Dict, List, Tuple

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from agent_core.data_cleaning import DataCleaner
from agent_core.stats_enhanced import (
    DescriptiveStatsEnhanced, PValueFormatter, GamesHowellPostHoc,
    StandardizedCoefficients, InfluenceDiagnostics
)
from agent_core.analysis_planner import StatisticalPlanner
from agent_core.report_generator import ReportGenerator
from agent_core.diagnostic_toolkit import ROC_Analysis
from agent_core.word_exporter import APAWordExporter


def read_excel(filepath: str) -> pd.DataFrame:
    ext = os.path.splitext(filepath)[1].lower()
    if ext in ('.xlsx', '.xls'):
        return pd.read_excel(filepath, engine='openpyxl')
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
            pct0 = vals[0] / ct.iloc[:, 0].sum() * 100
            pct1 = vals[1] / ct.iloc[:, 1].sum() * 100 if ct.shape[1] > 1 else 0
            if idx == 0:
                rows.append([f'{var} ({row_val})', f'{vals[0]} ({pct0:.1f}%)', f'{vals[1]} ({pct1:.1f}%)',
                             f'{chi2:.2f}', pv_str, f'V = {cramer_v:.3f}'])
            else:
                rows.append([f'{var} ({row_val})', f'{vals[0]} ({pct0:.1f}%)', f'{vals[1]} ({pct1:.1f}%)', '', '', ''])

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
    }
    try:
        from scipy.stats import chi2
        results['lr_stat'] = model.llf - model.llnull if hasattr(model, 'llnull') else None
        results['lr_pvalue'] = chi2.sf(-2 * results['lr_stat'], len(predictors)) if results['lr_stat'] else None
    except Exception:
        pass
    return results


def make_forest_plot(results: Dict, title: str = "Forest Plot") -> plt.Figure:
    coef = results['coef']
    ci = results['conf_int']
    or_vals = np.exp(coef)
    or_ci_low = np.exp(ci[0])
    or_ci_high = np.exp(ci[1])

    params = coef.index
    fig, ax = plt.subplots(figsize=(8, max(4, len(params) * 0.4)))
    y_pos = np.arange(len(params))

    ax.errorbar(or_vals.values, y_pos,
                xerr=[or_vals.values - or_ci_low.values, or_ci_high.values - or_vals.values],
                fmt='o', color='#0D9488', ecolor='gray', capsize=3, markersize=7)
    ax.axvline(x=1, color='red', linestyle='--', alpha=0.5)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(params, fontsize=9)
    ax.set_xlabel('Adjusted Odds Ratio (95% CI)', fontsize=11)
    ax.set_title(title, fontsize=13, fontweight='bold')
    ax.set_xscale('log')
    ax.grid(axis='x', alpha=0.3)
    fig.tight_layout()
    return fig


def make_boxplots(df: pd.DataFrame, outcome: str, continuous_vars: List[str],
                  max_vars: int = 9) -> List[plt.Figure]:
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
        colors = ['#E8F5E9', '#FFEBEE']
        for patch, color in zip(bp['boxes'], colors):
            patch.set_facecolor(color)
        ax.set_title(var, fontsize=10)
        ax.set_ylabel('Value', fontsize=9)
        ax.tick_params(labelsize=8)

    for j in range(i + 1, len(axes)):
        axes[j].set_visible(False)

    fig.suptitle('Distribution of Key Continuous Variables by Outcome', fontsize=13, fontweight='bold', y=1.02)
    fig.tight_layout()
    figs.append(fig)
    return figs


def make_barplots(df: pd.DataFrame, outcome: str, categorical_vars: List[str],
                  max_vars: int = 6) -> List[plt.Figure]:
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
        ct.plot(kind='bar', ax=ax, color=['#66BB6A', '#EF5350'], legend=False)
        ax.set_title(var, fontsize=10)
        ax.set_ylabel('Percentage (%)', fontsize=9)
        ax.tick_params(labelsize=8)
        ax.legend(title=outcome, fontsize=7)

    for j in range(i + 1, len(axes)):
        axes[j].set_visible(False)

    fig.suptitle('Categorical Variables Stratified by Outcome', fontsize=13, fontweight='bold', y=1.02)
    fig.tight_layout()
    figs.append(fig)
    return figs


def make_scatter_plot(df: pd.DataFrame, outcome: str, x_var: str, y_var: str) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(7, 5))
    for group in df[outcome].unique():
        subset = df[df[outcome] == group]
        ax.scatter(subset[x_var], subset[y_var], alpha=0.5, label=group, s=20)
        if len(subset) > 3:
            z = np.polyfit(subset[x_var].dropna(), subset[y_var].dropna(), 1)
            p = np.poly1d(z)
            x_line = np.linspace(subset[x_var].min(), subset[x_var].max(), 100)
            ax.plot(x_line, p(x_line), lw=1.5)
    ax.set_xlabel(x_var, fontsize=11)
    ax.set_ylabel(y_var, fontsize=11)
    ax.set_title(f'{y_var} vs {x_var} by {outcome}', fontsize=12, fontweight='bold')
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    return fig


def generate_narrative(outcome: str, table1_rows: List[List[str]],
                       logit_results: Dict, auc_val: Optional[float]) -> str:
    lines = []

    n_total_line = [r for r in table1_rows if 'Age' in r[0] or 'age' in r[0].lower()]
    lines.append(f"A total of {len(table1_rows)} candidate predictors were evaluated "
                 f"for their association with {outcome}.")

    sig_continuous = [r for r in table1_rows if len(r) > 4 and '<.001' in r[4] or (r[4].startswith('.') and float(r[4]) < 0.05)]
    if sig_continuous:
        vars_str = ', '.join(r[0] for r in sig_continuous[:5])
        lines.append(f"Bivariate analysis revealed significant associations between {outcome} and "
                     f"{vars_str} (all p < .05).")

    if logit_results and 'error' not in logit_results:
        sig_preds = []
        for var, pv in logit_results.get('pvalues', {}).items():
            if var != 'const' and pv < 0.05:
                or_val = np.exp(logit_results['coef'][var])
                ci = logit_results['conf_int']
                or_lo = np.exp(ci[0][var])
                or_hi = np.exp(ci[1][var])
                sig_preds.append(f"{var} (aOR = {or_val:.3f}, 95% CI [{or_lo:.3f}, {or_hi:.3f}])")

        lines.append(f"Multivariable logistic regression identified {len(sig_preds)} "
                     f"independent predictors of {outcome}: {'; '.join(sig_preds)}.")

        pseudo_r2 = logit_results.get('pseudo_r2', 0)
        lines.append(f"The model explained {pseudo_r2*100:.1f}% of the variance "
                     f"(McFadden's pseudo-R² = {pseudo_r2:.3f}).")

        if auc_val:
            lines.append(f"The model demonstrated {'excellent' if auc_val > 0.8 else 'good' if auc_val > 0.7 else 'moderate'} "
                         f"discrimination with an AUC of {auc_val:.3f}.")

    return '\n'.join(lines)


def run_full_pipeline(data_path: str, output_path: str, study_brief: str = "",
                      outcome_var: str = "", generate_plots: bool = True) -> Dict:
    print(f"[1/9] Reading data from {data_path}...")
    df = read_excel(data_path)
    print(f"  Loaded {df.shape[0]} rows x {df.shape[1]} columns")

    print(f"[2/9] Cleaning data...")
    cleaner = DataCleaner(df)
    cleaner.clean_pipeline()
    df_clean = cleaner.get_cleaned_df()

    outcome = auto_detect_outcome(df_clean, study_brief, outcome_var)
    print(f"  Outcome variable: {outcome}")
    if outcome not in df_clean.columns:
        outcome = df_clean.columns[-1]
        print(f"  Using last column as outcome: {outcome}")

    print(f"[3/9] Detecting variable types...")
    var_types = cleaner.detect_variable_types()
    continuous_vars = [k for k, v in var_types.items() if v in ('continuous', 'ordinal')]
    categorical_vars = [k for k, v in var_types.items() if v in ('binary', 'categorical')]
    print(f"  Continuous: {len(continuous_vars)}, Categorical: {len(categorical_vars)}")

    print(f"[4/9] Building analysis plan...")
    planner = StatisticalPlanner(df_clean, study_brief)
    plan = planner.build_plan()
    plan_type = plan[0].get('type', 'standard') if plan else 'standard'
    print(f"  Plan steps: {len(plan)}")

    print(f"[5/9] Building Table 1 (baseline characteristics)...")
    headers, table1_rows = build_table1(df_clean, outcome, var_types,
                                         continuous_vars, categorical_vars)
    print(f"  Table 1: {len(table1_rows)} rows")

    print(f"[6/9] Running multivariable regression...")
    predictors = continuous_vars + categorical_vars
    predictors = [p for p in predictors if p != outcome]
    logit_results = run_logistic_regression(df_clean, outcome, predictors)
    if 'error' in logit_results:
        print(f"  Warning: {logit_results['error']}")
        auc_val = None
    else:
        print(f"  Pseudo-R² = {logit_results['pseudo_r2']:.3f}")

    print(f"[7/9] Generating figures...")
    figs = []
    if generate_plots:
        if logit_results and 'error' not in logit_results:
            try:
                y_true = pd.Categorical(df_clean[outcome]).codes
                y_score = logit_results['model'].predict()
                roc = ROC_Analysis(y_true, y_score)
                roc_res = roc.compute()
                auc_val = roc_res.get('auc', roc_res.get('AUC'))
                fig_roc = roc.plot_roc()
                figs.append(('ROC Curve', fig_roc))
            except Exception as e:
                print(f"  ROC curve skipped: {e}")
                auc_val = None
        else:
            auc_val = None

        box_figs = make_boxplots(df_clean, outcome, continuous_vars)
        for f in box_figs:
            figs.append(('Boxplots', f))

        bar_figs = make_barplots(df_clean, outcome, categorical_vars)
        for f in bar_figs:
            figs.append(('Barplots', f))

        if logit_results and 'error' not in logit_results:
            try:
                fig_fp = make_forest_plot(logit_results)
                figs.append(('Forest Plot', fig_fp))
            except Exception as e:
                print(f"  Forest plot skipped: {e}")

        if len(continuous_vars) >= 2:
            try:
                xv = continuous_vars[1] if len(continuous_vars) > 1 else continuous_vars[0]
                yv = continuous_vars[0]
                if xv != outcome and yv != outcome:
                    fig_sc = make_scatter_plot(df_clean, outcome, xv, yv)
                    figs.append(('Scatter Plot', fig_sc))
            except Exception as e:
                print(f"  Scatter plot skipped: {e}")

    print(f"[8/9] Generating narrative...")
    narrative = generate_narrative(outcome, table1_rows, logit_results,
                                    auc_val if 'auc_val' in dir() else None)

    print(f"[9/9] Exporting Word document to {output_path}...")
    exporter = APAWordExporter(f"Analysis of {outcome}")

    exporter.add_heading("1. Background and Study Design", level=1)
    if study_brief:
        exporter.add_paragraph(study_brief[:2000])
    exporter.add_paragraph(f"This report presents a comprehensive statistical analysis of {outcome} "
                           f"among the study population. A total of {len(df_clean)} patients were analyzed.")

    exporter.add_heading("2. Data Cleaning Methodology", level=1)
    exporter.add_paragraph(cleaner.get_report()[:2000])

    exporter.add_heading("3. Table 1: Baseline Characteristics", level=1)
    exp_headers = ['Variable', f'Without {outcome}', f'With {outcome}', 'Test', 'p', 'Effect Size']
    exporter.add_apa_table(exp_headers, table1_rows,
                           caption=f"Baseline characteristics stratified by {outcome}. "
                                   f"Continuous: M ± SD or Mdn [IQR]. Effect sizes: Cohen's d or rank-biserial r.")

    exporter.add_heading("4. Bivariate Analysis Results", level=1)
    exporter.add_paragraph(narrative)

    exporter.add_heading("5. Multivariable Regression", level=1)
    if logit_results and 'error' not in logit_results:
        lr_headers = ['Predictor', 'aOR', '95% CI', 'p', '']
        lr_rows = []
        for var in logit_results['coef'].index:
            if var == 'const':
                continue
            or_val = np.exp(logit_results['coef'][var])
            ci_lo = np.exp(logit_results['conf_int'][0][var])
            ci_hi = np.exp(logit_results['conf_int'][1][var])
            pv = logit_results['pvalues'][var]
            pv_str = f'{pv:.3f}' if pv >= 0.001 else '<.001'
            sig = '***' if pv < 0.001 else '**' if pv < 0.01 else '*' if pv < 0.05 else ''
            lr_rows.append([var, f'{or_val:.3f}', f'[{ci_lo:.3f}, {ci_hi:.3f}]', pv_str, sig])

        exporter.add_apa_table(lr_headers, lr_rows,
                               caption=f"Independent predictors of {outcome} from multivariable "
                                       f"logistic regression. aOR = adjusted odds ratio; CI = confidence interval. "
                                       f"*p < .05, **p < .01, ***p < .001.")
        exporter.add_paragraph(f"Model fit: pseudo-R² = {logit_results['pseudo_r2']:.3f}, "
                               f"AIC = {logit_results['aic']:.1f}, N = {logit_results['nobs']}")

    exporter.add_heading("6. Figures", level=1)
    for label, fig in figs:
        if label == 'ROC Curve':
            exporter.add_figure(fig, caption=f"ROC curve of the logistic regression model for predicting {outcome}. "
                                             f"AUC = {auc_val:.3f}" if auc_val else "ROC curve.")
        elif label == 'Boxplots':
            exporter.add_figure(fig, caption=f"Distribution of key clinical variables by {outcome} status. "
                                             f"Boxes represent median and IQR; whiskers extend to 1.5×IQR.")
        elif label == 'Barplots':
            exporter.add_figure(fig, caption=f"Distribution of categorical variables stratified by {outcome} status. "
                                             f"Percentages computed within each group.")
        elif label == 'Forest Plot':
            exporter.add_figure(fig, caption=f"Forest plot of adjusted odds ratios. "
                                             f"Points = point estimates; error bars = 95% CI. "
                                             f"Vertical dashed line at aOR = 1.0.")
        elif label == 'Scatter Plot':
            exporter.add_figure(fig, caption=f"Relationship between key continuous variables by {outcome} status.")

    exporter.add_heading("7. Discussion and Conclusions", level=1)
    exporter.add_paragraph(narrative)
    exporter.add_paragraph("These findings should be interpreted considering the study's limitations, "
                           "including the observational design and potential residual confounding.")

    exporter.save(output_path)
    print(f"\nReport saved to: {output_path}")
    return {'status': 'success', 'output': output_path, 'rows': len(table1_rows)}


def main():
    parser = argparse.ArgumentParser(description='Medical Research Data Analysis Pipeline')
    parser.add_argument('--data', '-d', required=True, help='Path to Excel data file (.xlsx/.csv)')
    parser.add_argument('--output', '-o', default='analysis_report.docx', help='Output Word document path')
    parser.add_argument('--brief', '-b', default='', help='Study protocol / research brief text')
    parser.add_argument('--outcome', '-dv', default='', help='Outcome/dependent variable name')
    parser.add_argument('--no-plots', action='store_true', help='Skip figure generation')
    args = parser.parse_args()

    result = run_full_pipeline(
        data_path=args.data,
        output_path=args.output,
        study_brief=args.brief,
        outcome_var=args.outcome,
        generate_plots=not args.no_plots
    )
    return result


if __name__ == '__main__':
    main()
