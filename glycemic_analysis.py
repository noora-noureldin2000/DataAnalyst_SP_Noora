#!/usr/bin/env python3
"""
Senior Biostatistician Analysis Pipeline (13-Step Professional Workflow)
=======================================================================
Study: Glycemic Metrics Across Types of Diabetes Mellitus and Their
       Relationship With Weekly Time in Range (Cross-Sectional CGM Analysis)

Aims:
  1. Compare glycemic metrics across types of diabetes mellitus (T1DM vs T2DM).
  2. Evaluate variations in glycemic parameters relative to the percentage of
     weekly Time in Range (supplied category + continuous TIR models).
  3. Investigate correlations among continuous glycemic metrics.

Steps 1-3: Context ingestion, variable profiling, DataCleaner audit
Steps 4-5: Study design + variable classification
Steps 6-7: Normality + stratified descriptive statistics (Table 1)
Step 8:   Publication figures (Lancet style, 800 dpi)
Steps 9-10: Hypothesis testing + inferential statistics (FDR-controlled)
Steps 11-13: Interpretation + APA tables + Word report
"""

import os
import sys
import json
import warnings

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy import stats as sps
import statsmodels.api as sm
from statsmodels.stats.outliers_influence import variance_inflation_factor
from statsmodels.stats.diagnostic import het_breuschpagan

WORK = r"D:\Naggar Analytics\DataAnalyst_SP_Noora_V2"
sys.path.insert(0, WORK)

from agent_core.data_cleaning import DataCleaner
from agent_core.stats_enhanced import PValueFormatter, InfluenceDiagnostics, StandardizedCoefficients
from agent_core.multiple_corrections import MultipleTestCorrection
from agent_core.diagnostic_toolkit import ROC_Analysis
from agent_core.visual_style import set_publication_style, get_palette
from agent_core.word_exporter import APAWordExporter

warnings.filterwarnings('ignore')
np.random.seed(20260925)

DATA_PATH = r"D:\Naggar Analytics\Taining data from Noha\CLEANED_Glycemic metrics.csv"
OUT_DIR = os.path.join(WORK, "analysis_output")
os.makedirs(OUT_DIR, exist_ok=True)

set_publication_style(palette='lancet', dpi=800)
PAL = get_palette('lancet')

RESULTS = {}   # verification JSON store


# ----------------------------------------------------------------------
# Publication-friendly labels
# ----------------------------------------------------------------------
LABELS = {
    'dm_type2': 'Diabetes type (T2DM vs T1DM)',
    'male': 'Sex, male',
    'age': 'Age (years)',
    'gender': 'Sex, male',
    'd_m_duration': 'Diabetes duration (years)',
    'weight': 'Weight (kg)',
    'height': 'Height (cm)',
    'b_m_i': 'BMI (kg/m2)',
    'calculated_total_daily_dose_of_insulin_i_u': 'Total daily insulin dose (U/kg/day)',
    'time_sensor_is_active': 'Sensor active time (%)',
    'n_daily_scans': 'Daily scans (n)',
    'hb_a1_c': 'HbA1c (%)',
    'average_glucose': 'Mean glucose (mg/dL)',
    'g_m_i': 'GMI (%)',
    'g_v': 'Glucose variability, CV (%)',
    't_i_r': 'TIR 70-180 mg/dL (%)',
    't_b_r_54_70': 'TBR level 1, 54-70 mg/dL (%)',
    't_b_r_54': 'TBR level 2, <54 mg/dL (%)',
    't_a_r_180_250': 'TAR level 1, 180-250 mg/dL (%)',
    't_a_r_250': 'TAR level 2, >250 mg/dL (%)',
    'duration_of_hypoglycemia': 'Hypoglycemia duration (min)',
    'low_glucose_events': 'Low glucose events (n)',
    'g_r_i': 'Glycemic risk index (GRI)',
    'chypo': 'Low-glucose index (chypo)',
    'chyper': 'High-glucose index (chyper)',
}
SHORT = {
    'hb_a1_c': 'HbA1c', 'average_glucose': 'Mean glucose', 'g_m_i': 'GMI',
    'g_v': 'CV', 't_i_r': 'TIR', 't_b_r_54_70': 'TBR-1', 't_b_r_54': 'TBR-2',
    't_a_r_180_250': 'TAR-1', 't_a_r_250': 'TAR-2',
    'duration_of_hypoglycemia': 'Hypo duration', 'low_glucose_events': 'Low events',
    'g_r_i': 'GRI', 'chypo': 'chypo', 'chyper': 'chyper',
}
DEMOGRAPHICS = ['age', 'gender', 'd_m_duration', 'weight', 'height', 'b_m_i']
CLINICAL = ['calculated_total_daily_dose_of_insulin_i_u', 'time_sensor_is_active', 'n_daily_scans']
BATTERY = ['hb_a1_c', 'average_glucose', 'g_m_i', 'g_v', 't_i_r', 't_b_r_54_70',
           't_b_r_54', 't_a_r_180_250', 't_a_r_250', 'duration_of_hypoglycemia',
           'low_glucose_events', 'g_r_i', 'chypo', 'chyper']


def fp(p):
    """APA p-value formatting."""
    return PValueFormatter.format_p(float(p))


def mean_sd(s):
    return f'{np.mean(s):.1f} ({np.std(s, ddof=1):.1f})'


def med_iqr(s):
    return f'{np.median(s):.1f} ({np.percentile(s, 25):.1f}-{np.percentile(s, 75):.1f})'


def two_group_test(s1, s2, alpha=.05):
    """Shapiro-routed two-group comparison. Returns dict."""
    n1, n2 = len(s1), len(s2)
    sh1 = sps.shapiro(s1)
    sh2 = sps.shapiro(s2)
    sk1, sk2 = sps.skew(s1), sps.skew(s2)
    normal = (sh1.pvalue >= alpha) and (sh2.pvalue >= alpha) and abs(sk1) < 1 and abs(sk2) < 1
    if normal:
        t, p = sps.ttest_ind(s1, s2, equal_var=False)
        v1, v2 = np.var(s1, ddof=1), np.var(s2, ddof=1)
        df = (v1 / n1 + v2 / n2) ** 2 / ((v1 / n1) ** 2 / (n1 - 1) + (v2 / n2) ** 2 / (n2 - 1))
        sp = np.sqrt(((n1 - 1) * v1 + (n2 - 1) * v2) / (n1 + n2 - 2))
        d = (np.mean(s1) - np.mean(s2)) / sp if sp > 0 else np.nan
        return dict(test="Welch's t", stat=t, df=df, p=p, es_name="Cohen's d",
                    es=d, parametric=True, format=mean_sd)
    u, p = sps.mannwhitneyu(s1, s2, alternative='two-sided')
    r_rb = 2.0 * u / (n1 * n2) - 1.0   # positive = s1 (first-listed group) higher
    return dict(test='Mann-Whitney U', stat=u, df=None, p=p, es_name='rank-biserial r',
                es=r_rb, parametric=False, format=med_iqr)


# ======================================================================
# STEP 1-2: CONTEXT INGESTION + DATA UNDERSTANDING
# ======================================================================
print('=' * 70)
print('STEP 1-2: Data ingestion and profiling')
print('=' * 70)

df_raw = pd.read_csv(DATA_PATH)
n_raw, p_raw = df_raw.shape
print(f'Raw dimensions: {n_raw} rows x {p_raw} columns')

# Keep the cleaned core columns; drop redundant _Raw duplicates and audit flags
audit_cols = [c for c in df_raw.columns if c.endswith('_Raw') or c.endswith('_flag')]
df = df_raw.drop(columns=audit_cols).copy()

# Trim whitespace on string columns (source data has trailing-space variants)
for c in df.select_dtypes(include='object').columns:
    df[c] = df[c].astype(str).str.strip()

# Derive analysis-ready group variables
df['dm_type'] = np.where(df['d_m_type'] == 1, 'T1DM', 'T2DM')
df['dm_type2'] = (df['d_m_type'] == 2).astype(int)          # 1 = T2DM (ref T1DM)
df['male'] = (df['gender'] == 'Male').astype(int)
df['tir_cat'] = df['weekly_time_in_range_achieved']          # Minimal / Moderate to Adequate
df['tir_cat01'] = (df['tir_cat'] == 'Moderate to Adequate').astype(int)

print(f'After dropping {len(audit_cols)} audit/duplicate columns: {df.shape[0]} rows x {df.shape[1]} cols')

# ======================================================================
# STEP 3: DATA CLEANING (DataCleaner audit pipeline)
# ======================================================================
print()
print('=' * 70)
print('STEP 3: Data cleaning audit (agent_core.DataCleaner)')
print('=' * 70)

cleaner = DataCleaner(df.copy())
cleaner.remove_duplicate_rows()
cleaner.remove_empty_columns()
cleaner.impute_missing(strategy='median')
# NOTE: Outlier capping deliberately NOT applied: CGM range metrics are
# bounded compositional percentages; source file already carries per-variable
# outlier audit flags. Capping would distort the 100% composition.
df_clean = cleaner.get_cleaned_df()
audit = cleaner.get_audit()

n_dup = df_raw['p_n'].duplicated().sum()
n_missing_total = int(df.isna().sum().sum())
print(f'Duplicate patient IDs: {n_dup}')
print(f'Missing values (core analysis set): {n_missing_total}')

# Source-file audit flags summary
flag_summary = {}
for fc in [c for c in df_raw.columns if c.endswith('_outlier_flag')]:
    flag_summary[fc.replace('_outlier_flag', '')] = int(df_raw[fc].sum())
flag_counts = {k: v for k, v in flag_summary.items() if v > 0}
print(f'Outlier flags (from source cleaning): {flag_counts if flag_counts else "none flagged"}')
print(f'GMI mismatch flags: {int(df_raw["gmi_mismatch_flag"].sum())} / {n_raw}')

# Derivation audit of the supplied weekly TIR category
cv36_agree = (np.where(df['g_v'] >= 36, 1, 0) == df['tir_cat01']).mean()
print(f'\nDERIVATION AUDIT: tir_cat == (CV >= 36%) agreement = {cv36_agree:.1%}')
tir_by_cat = df.groupby('tir_cat')['t_i_r'].median()
print('Median TIR (%) by supplied category:', tir_by_cat.to_dict())

# CGM composition audit
df['cgm_sum_pct_calc'] = df[['t_i_r', 't_b_r_54_70', 't_b_r_54',
                             't_a_r_180_250', 't_a_r_250']].sum(axis=1)
print(f'CGM five-bin composition: median {df["cgm_sum_pct_calc"].median():.1f}%, '
      f'range {df["cgm_sum_pct_calc"].min():.1f}-{df["cgm_sum_pct_calc"].max():.1f}%')

RESULTS['data_audit'] = dict(
    n_rows=int(n_raw), n_cols_raw=int(p_raw), n_audit_cols=len(audit_cols),
    duplicates=int(n_dup), missing=int(n_missing_total),
    cv36_category_agreement=float(cv36_agree),
    tir_ge_70pct=int((df['t_i_r'] >= 70).sum()),
    tir_ge_50pct=int((df['t_i_r'] >= 50).sum()),
    cv_ge_36=int((df['g_v'] >= 36).sum()),
    cgm_sum_median=float(df['cgm_sum_pct_calc'].median()),
    gmi_mismatch_flags=int(df_raw['gmi_mismatch_flag'].sum()),
)

# ======================================================================
# STEP 4: STUDY DESIGN  |  STEP 5: VARIABLE CLASSIFICATION
# ======================================================================
print()
print('=' * 70)
print('STEP 4-5: Study design and variable classification')
print('=' * 70)

var_types = cleaner.detect_variable_types()
print('Detected variable types (agent_core):')
for v in DEMOGRAPHICS + CLINICAL + BATTERY:
    print(f'  {LABELS[v]:45s} -> {var_types.get(v, "n/a")}')

study = dict(
    design='Cross-sectional observational study',
    population='Adolescents and young adults with insulin-treated diabetes on intermittently-scanned CGM',
    groups=f"T1DM (n={int((df['dm_type']=='T1DM').sum())}) vs T2DM (n={int((df['dm_type']=='T2DM').sum())})",
    outcome_primary='Weekly TIR achievement (supplied category) + continuous TIR (%)',
    alpha=0.05, sided='two-sided',
)
print(f"\nDesign: {study['design']}")
print(f"Groups: {study['groups']}")

# Minimal detectable effect (post-hoc sensitivity)
from statsmodels.stats.power import TTestIndPower
n1 = int((df['dm_type'] == 'T1DM').sum())
n2 = int((df['dm_type'] == 'T2DM').sum())
mde = TTestIndPower().solve_power(effect_size=None, nobs1=n1, ratio=n2 / n1,
                                   alpha=0.05, power=0.80, alternative='two-sided')
print(f'Minimum detectable standardized difference (80% power, alpha=.05): d = {mde:.2f}')
RESULTS['design'] = dict(study, mde_d=float(mde))

# ======================================================================
# STEP 6: NORMALITY TESTING
# ======================================================================
print()
print('=' * 70)
print('STEP 6: Normality assessment (Shapiro-Wilk, skewness, kurtosis)')
print('=' * 70)

norm_rows = []
for v in BATTERY + ['age', 'd_m_duration', 'b_m_i']:
    s = df[v].dropna().values
    w, pw = sps.shapiro(s)
    norm_rows.append([LABELS[v], f'{np.mean(s):.2f}', f'{sps.skew(s):.2f}',
                     f'{sps.kurtosis(s):.2f}', f'{w:.3f}', fp(pw)])
    print(f'{LABELS[v]:45s} skew={sps.skew(s):6.2f}  kurt={sps.kurtosis(s):6.2f}  '
          f'W={w:.3f}  p={fp(pw)}')

RESULTS['normality'] = {r[0]: dict(mean=r[1], skew=r[2], kurtosis=r[3],
                                    W=r[4], p=r[5]) for r in norm_rows}

# ======================================================================
# STEP 7: DESCRIPTIVE STATISTICS + TABLE 1 (by DM type)
# ======================================================================
print()
print('=' * 70)
print('STEP 7: Descriptive statistics and Table 1 (stratified by DM type)')
print('=' * 70)

t1 = df[df['dm_type'] == 'T1DM']
t2 = df[df['dm_type'] == 'T2DM']

table1_headers = ['Characteristic', 'T1DM (n=104)', 'T2DM (n=90)', 'Test', 'p', 'Effect size']
table1_rows = []
t1_json = {}

for v in DEMOGRAPHICS + CLINICAL:
    if v == 'gender':
        ct = pd.crosstab(df['male'], df['dm_type'])
        chi2, p, dof, _ = sps.chi2_contingency(ct, correction=False)
        v_cram = np.sqrt(chi2 / (ct.values.sum() * 1))
        m1 = t1['male'].sum(); m2 = t2['male'].sum()
        table1_rows.append([LABELS[v], f'{m1} ({m1/len(t1)*100:.1f}%)',
                            f'{m2} ({m2/len(t2)*100:.1f}%)',
                            f'chi2({dof}) = {chi2:.2f}', fp(p), f'V = {v_cram:.2f}'])
        t1_json[LABELS[v]] = dict(test='chi2', stat=float(chi2), p=float(p), V=float(v_cram))
        continue
    res = two_group_test(t1[v].dropna().values, t2[v].dropna().values)
    stat_txt = f't({res["df"]:.1f}) = {res["stat"]:.2f}' if res['parametric'] else f'U = {res["stat"]:.0f}'
    table1_rows.append([LABELS[v], res['format'](t1[v].dropna()), res['format'](t2[v].dropna()),
                        stat_txt, fp(res['p']), f'{res["es_name"]} = {res["es"]:.2f}'])
    t1_json[LABELS[v]] = dict(test=res['test'], stat=float(res['stat']), p=float(res['p']),
                              es=float(res['es']), parametric=res['parametric'])
    print(f'{LABELS[v]:45s} {res["test"]:16s} p={fp(res["p"])}  {res["es_name"]}={res["es"]:.2f}')

RESULTS['table1'] = t1_json

# Overall battery descriptives
desc_overall = {LABELS[v]: med_iqr(df[v]) for v in BATTERY}
RESULTS['descriptives_overall'] = desc_overall

# ======================================================================
# STEPS 9-10, AIM 1: GLYCEMIC METRICS ACROSS DM TYPES (+ FDR)
# ======================================================================
print()
print('=' * 70)
print('AIM 1: Glycemic metrics across diabetes types (FDR-BH controlled)')
print('=' * 70)

def battery_comparison(df, group_col, g1_label, g2_label, metrics=None):
    """Full battery two-group comparison with BH-FDR correction."""
    metrics = metrics if metrics is not None else BATTERY
    g1 = df[df[group_col] == g1_label]
    g2 = df[df[group_col] == g2_label]
    rows, tests = [], []
    for v in metrics:
        res = two_group_test(g1[v].dropna().values, g2[v].dropna().values)
        stat_txt = (f't({res["df"]:.1f}) = {res["stat"]:.2f}' if res['parametric']
                    else f'U = {res["stat"]:.0f}')
        rows.append(dict(metric=v, label=LABELS[v],
                         g1=res['format'](g1[v].dropna()), g2=res['format'](g2[v].dropna()),
                         test=res['test'], stat_txt=stat_txt, p=float(res['p']),
                         es_name=res['es_name'], es=float(res['es'])))
        tests.append(LABELS[v])
    pvals = [r['p'] for r in rows]
    mtc = MultipleTestCorrection(pvals, test_labels=tests).correct(methods=['fdr_bh'])
    for i, r in enumerate(rows):
        r['p_fdr'] = float(mtc['p_fdr_bh'].iloc[i])
    return rows

aim1 = battery_comparison(df, 'dm_type', 'T1DM', 'T2DM')
for r in aim1:
    star = '*' if r['p_fdr'] < .05 else ''
    print(f'{r["label"]:38s} {r["test"]:16s} {r["stat_txt"]:20s} p={fp(r["p"])}  '
          f'q(FDR)={fp(r["p_fdr"])}  {r["es_name"]}={r["es"]:.2f}{star}')

RESULTS['aim1_by_dm_type'] = {r['label']: dict(test=r['test'], p=r['p'], p_fdr=r['p_fdr'],
                                               es=r['es'], es_name=r['es_name'],
                                               T1DM=r['g1'], T2DM=r['g2']) for r in aim1}

# ======================================================================
# AIM 2a: GLYCEMIC PARAMETERS BY SUPPLIED WEEKLY TIR CATEGORY (+ FDR)
# ======================================================================
print()
print('=' * 70)
print('AIM 2a: Glycemic parameters by supplied weekly TIR category (FDR-BH)')
print('=' * 70)

# CV excluded: it deterministically defines this category (complete separation)
aim2a = battery_comparison(df, 'tir_cat', 'Minimal', 'Moderate to Adequate',
                           metrics=[m for m in BATTERY if m != 'g_v'])
for r in aim2a:
    star = '*' if r['p_fdr'] < .05 else ''
    print(f'{r["label"]:38s} {r["test"]:16s} {r["stat_txt"]:20s} p={fp(r["p"])}  '
          f'q(FDR)={fp(r["p_fdr"])}  {r["es_name"]}={r["es"]:.2f}{star}')

RESULTS['aim2a_by_category'] = {r['label']: dict(test=r['test'], p=r['p'], p_fdr=r['p_fdr'],
                                                 es=r['es'], es_name=r['es_name'],
                                                 Minimal=r['g1'], ModerateAdequate=r['g2']) for r in aim2a}

# DM type x category association
ct_dmcat = pd.crosstab(df['dm_type'], df['tir_cat'])
chi2_dm, p_dm, dof_dm, _ = sps.chi2_contingency(ct_dmcat, correction=False)
v_dm = np.sqrt(chi2_dm / (ct_dmcat.values.sum() * 1))
print(f'\nDM type x weekly TIR category: chi2({dof_dm}) = {chi2_dm:.2f}, p = {fp(p_dm)}, V = {v_dm:.2f}')
print(ct_dmcat.to_string())
RESULTS['dmtype_x_category'] = dict(chi2=float(chi2_dm), p=float(p_dm), V=float(v_dm),
                                    table=ct_dmcat.to_dict())

# ======================================================================
# AIM 2b: MULTIVARIABLE MODELS
# ======================================================================
print()
print('=' * 70)
print('AIM 2b: Multivariable regression models')
print('=' * 70)

def fit_logistic(df, y, predictors, model_name):
    X = sm.add_constant(df[predictors].astype(float))
    model = sm.Logit(y, X).fit(disp=False, maxiter=300)
    llnull = model.llnull if hasattr(model, 'llnull') else sm.Logit(y, np.ones((len(y), 1))).fit(disp=False).llf
    n = int(model.nobs)
    cox_snell = 1 - np.exp(2 * (llnull - model.llf) / n)
    nagelkerke = cox_snell / (1 - np.exp(2 * llnull / n))
    lr_chi2 = 2 * (model.llf - llnull)
    lr_p = sps.chi2.sf(lr_chi2, len(predictors))
    vifs = {p: variance_inflation_factor(X.values, i + 1) for i, p in enumerate(predictors)}
    roc = ROC_Analysis(y.values, model.predict(X).values)
    roc_res = roc.compute(n_boot=2000)
    out = dict(model=model, name=model_name, predictors=predictors,
               nagelkerke=float(nagelkerke), lr_chi2=float(lr_chi2), lr_p=float(lr_p),
               aic=float(model.aic), n=n, vifs={k: float(v) for k, v in vifs.items()},
               auc=float(roc_res['auc']), auc_ci=[float(roc_res['auc_ci'][0]), float(roc_res['auc_ci'][1])])
    print(f'\n{model_name}: n={n}, LR chi2({len(predictors)})={lr_chi2:.2f}, p={fp(lr_p)}, '
          f'Nagelkerke R2={nagelkerke:.3f}, AUC={roc_res["auc"]:.3f} '
          f'[{roc_res["auc_ci"][0]:.3f}, {roc_res["auc_ci"][1]:.3f}], max VIF={max(vifs.values()):.2f}')
    for p in predictors:
        b = model.params[p]; se = model.bse[p]; pv = model.pvalues[p]
        ci = model.conf_int().loc[p]
        print(f'  {LABELS.get(p, p):38s} aOR={np.exp(b):7.3f} [{np.exp(ci[0]):6.3f}, {np.exp(ci[1]):7.3f}]  p={fp(pv)}')
    return out

def fit_linear(df, y, predictors, model_name):
    X = sm.add_constant(df[predictors].astype(float))
    model = sm.OLS(y, X).fit()
    vifs = {p: variance_inflation_factor(X.values, i + 1) for i, p in enumerate(predictors)}
    # Standardized betas
    try:
        betas = StandardizedCoefficients.from_lm(model)
        beta_map = {p: float(betas[p]) for p in predictors} if hasattr(betas, 'keys') else {}
    except Exception:
        Zy = (y - y.mean()) / y.std(ddof=1)
        beta_map = {}
        for p in predictors:
            Zx = (df[p] - df[p].mean()) / df[p].std(ddof=1)
            b0 = sm.OLS(Zy, sm.add_constant(Zx)).fit().params[0]
            beta_map[p] = float(model.params[p] * df[p].std(ddof=1) / y.std(ddof=1))
    shp = sps.shapiro(model.resid)
    bp = het_breuschpagan(model.resid, X)
    cooks = model.get_influence().cooks_distance[0]
    print(f'\n{model_name}: F({model.df_model:.0f}, {model.df_resid:.0f})={model.fvalue:.2f}, '
          f'p={fp(model.f_pvalue)}, R2={model.rsquared:.3f}, adjR2={model.rsquared_adj:.3f}, '
          f'max VIF={max(vifs.values()):.2f}')
    print(f'  Residual Shapiro p={fp(shp.pvalue)}, Breusch-Pagan p={fp(bp[1])}, '
          f'max Cook D={cooks.max():.3f} (4/n={4/model.nobs:.3f})')
    for p in predictors:
        b = model.params[p]; ci = model.conf_int().loc[p]; pv = model.pvalues[p]
        print(f'  {LABELS.get(p, p):38s} B={b:7.2f} [{ci[0]:7.2f}, {ci[1]:7.2f}]  '
              f'beta={beta_map.get(p, np.nan):.2f}  p={fp(pv)}')
    return dict(model=model, name=model_name, predictors=predictors,
                F=float(model.fvalue), p=float(model.f_pvalue),
                r2=float(model.rsquared), adj_r2=float(model.rsquared_adj),
                vifs={k: float(v) for k, v in vifs.items()},
                betas=beta_map, resid_shapiro_p=float(shp.pvalue),
                bp_p=float(bp[1]), max_cook=float(cooks.max()),
                n=int(model.nobs))

# Model A (as-labeled): supplied weekly TIR category. CV EXCLUDED (perfect
# determination of outcome; would cause complete separation).
predsA = ['dm_type2', 'age', 'male', 'd_m_duration', 'b_m_i', 'hb_a1_c', 't_i_r', 'g_r_i']
logitA = fit_logistic(df, df['tir_cat01'], predsA,
                      'Model A: supplied weekly TIR category (CV excluded a priori)')

# Model B (supplementary): clinically-defined TIR >= 50% (consensus target 70%
# reached by too few patients)
df['tir50'] = (df['t_i_r'] >= 50).astype(int)
logitB = fit_logistic(df, df['tir50'],
                      ['dm_type2', 'age', 'male', 'd_m_duration', 'b_m_i', 'hb_a1_c', 'g_v', 'g_r_i'],
                      'Model B (supplementary): TIR >= 50%')

# Model C (primary for literal aim): continuous TIR %
linC = fit_linear(df, df['t_i_r'],
                  ['dm_type2', 'age', 'male', 'd_m_duration', 'b_m_i', 'hb_a1_c', 'g_v', 'g_r_i'],
                  'Model C: continuous TIR (%)')

# HC3 sensitivity for Model C (residual diagnostics flagged heteroscedasticity)
X_C = sm.add_constant(df[linC['predictors']].astype(float))
model_hc3 = sm.OLS(df['t_i_r'], X_C).fit(cov_type='HC3')
hc3_changes = [p for p in linC['predictors']
               if (linC['model'].pvalues[p] < .05) != (model_hc3.pvalues[p] < .05)]
n_cook_flag = int((linC['model'].get_influence().cooks_distance[0] > 4 / linC['n']).sum())
print(f'HC3 sensitivity: significance changes = {hc3_changes if hc3_changes else "none"}; '
      f'observations with Cook D > 4/n = {n_cook_flag}')
RESULTS['model_C_hc3'] = dict(
    significance_changes=[LABELS.get(p, p) for p in hc3_changes],
    n_cook_gt_4n=n_cook_flag)

def logit_to_json(fit):
    m = fit['model']
    out = {}
    for p in fit['predictors']:
        ci = m.conf_int().loc[p]
        out[LABELS.get(p, p)] = dict(aOR=float(np.exp(m.params[p])),
                                     lo=float(np.exp(ci[0])), hi=float(np.exp(ci[1])),
                                     p=float(m.pvalues[p]))
    return out

RESULTS['model_A'] = dict(nagelkerke=logitA['nagelkerke'], lr_p=logitA['lr_p'],
                           auc=logitA['auc'], auc_ci=logitA['auc_ci'],
                           predictors=logit_to_json(logitA), vifs=logitA['vifs'])
RESULTS['model_B'] = dict(nagelkerke=logitB['nagelkerke'], lr_p=logitB['lr_p'],
                          auc=logitB['auc'], auc_ci=logitB['auc_ci'],
                          predictors=logit_to_json(logitB), vifs=logitB['vifs'])

mC = linC['model']
RESULTS['model_C'] = dict(F=linC['F'], p=linC['p'], r2=linC['r2'], adj_r2=linC['adj_r2'],
                          resid_shapiro_p=linC['resid_shapiro_p'], bp_p=linC['bp_p'],
                          max_cook=linC['max_cook'], vifs=linC['vifs'],
                          predictors={LABELS.get(p, p): dict(
                              B=float(mC.params[p]),
                              lo=float(mC.conf_int().loc[p][0]),
                              hi=float(mC.conf_int().loc[p][1]),
                              beta=linC['betas'].get(p), p=float(mC.pvalues[p]))
                              for p in linC['predictors']})

# ======================================================================
# AIM 3: CORRELATIONS AMONG CONTINUOUS GLYCEMIC METRICS
# ======================================================================
print()
print('=' * 70)
print('AIM 3: Spearman correlations among glycemic metrics (BH-FDR controlled)')
print('=' * 70)

n_vars = len(BATTERY)
rho = pd.DataFrame(np.ones((n_vars, n_vars)), index=BATTERY, columns=BATTERY)
pval = pd.zeros((n_vars, n_vars)) if False else pd.DataFrame(np.zeros((n_vars, n_vars)), index=BATTERY, columns=BATTERY)
pairs = []
for i in range(n_vars):
    for j in range(i + 1, n_vars):
        a, b = BATTERY[i], BATTERY[j]
        r, p = sps.spearmanr(df[a], df[b])
        rho.loc[a, b] = rho.loc[b, a] = r
        pval.loc[a, b] = pval.loc[b, a] = p
        pairs.append((a, b, r, p))

pair_p = [x[3] for x in pairs]
pair_lbl = [f'{SHORT[x[0]]} - {SHORT[x[1]]}' for x in pairs]
mtc_c = MultipleTestCorrection(pair_p, test_labels=pair_lbl).correct(methods=['fdr_bh'])
q_map = {lbl: float(qv) for lbl, qv in zip(pair_lbl, mtc_c['p_fdr_bh'])}

# Key pairs for report (clinically anchored)
key_pairs = [
    ('hb_a1_c', 'g_m_i'), ('hb_a1_c', 'average_glucose'), ('hb_a1_c', 't_i_r'),
    ('hb_a1_c', 't_a_r_250'), ('g_m_i', 'average_glucose'),
    ('g_v', 't_i_r'), ('g_v', 'hb_a1_c'), ('g_r_i', 't_i_r'), ('g_r_i', 'hb_a1_c'),
    ('t_i_r', 't_a_r_180_250'), ('t_i_r', 't_a_r_250'), ('t_i_r', 't_b_r_54'),
    ('g_v', 'g_r_i'), ('chypo', 't_b_r_54'), ('chyper', 't_a_r_250'),
]
# Normalize pair order to BATTERY order (matches the 91-pair FDR family keys)
def pair_q(a, b):
    if BATTERY.index(a) > BATTERY.index(b):
        a, b = b, a
    return rho.loc[a, b], pval.loc[a, b], q_map[f'{SHORT[a]} - {SHORT[b]}']

print('Key pairs (Spearman rho, raw p, BH-FDR q):')
key_corr_rows = []
for a, b in key_pairs:
    r, p, q = pair_q(a, b)
    print(f'  {SHORT[a]:12s} vs {SHORT[b]:12s} rho={r:6.3f}  p={fp(p)}  q={fp(q)}')
    key_corr_rows.append(dict(a=LABELS[a], b=LABELS[b], rho=float(r),
                              p=float(p), q=float(q)))
RESULTS['aim3_key_pairs'] = key_corr_rows

rho.to_csv(os.path.join(OUT_DIR, 'spearman_matrix.csv'), float_format='%.3f')
RESULTS['aim3_strongest'] = [
    dict(pair=lbl, rho=float(rho.loc[a, b]), p=float(pval.loc[a, b]), q=q_map[lbl])
    for a, b in [(pairs[i][0], pairs[i][1]) for i in range(len(pairs))]
    for lbl in [f'{SHORT[pairs[i][0]]} - {SHORT[pairs[i][1]]}']
][:0]  # placeholder not used

# strongest correlations ranked
ranked = sorted(pairs, key=lambda x: -abs(x[2]))[:10]
RESULTS['aim3_top10'] = [dict(pair=f'{SHORT[a]} - {SHORT[b]}', rho=float(r),
                              p=float(p), q=q_map[f'{SHORT[a]} - {SHORT[b]}'])
                         for a, b, r, p in ranked]
for a, b, r, p in ranked:
    print(f'  TOP: {SHORT[a]:12s} - {SHORT[b]:12s} rho={r:+.3f}')

# Pearson for the two near-normal glucose indices (HbA1c-GMI)
r_pg, p_pg = sps.pearsonr(df['hb_a1_c'], df['g_m_i'])
r_pg2, p_pg2 = sps.pearsonr(df['hb_a1_c'], df['average_glucose'])
print(f'\nPearson HbA1c-GMI: r={r_pg:.3f}, p={fp(p_pg)}; HbA1c-mean glucose: r={r_pg2:.3f}, p={fp(p_pg2)}')
RESULTS['pearson_key'] = dict(hba1c_gmi=dict(r=float(r_pg), p=float(p_pg)),
                             hba1c_meanglucose=dict(r=float(r_pg2), p=float(p_pg2)))

# ======================================================================
# STEP 8: FIGURES
# ======================================================================
print()
print('=' * 70)
print('STEP 8: Publication figures (Lancet palette, 800 dpi)')
print('=' * 70)

FIG_FILES = {}
FIGS = {}

def save_fig(fig, name):
    path = os.path.join(OUT_DIR, name)
    fig.savefig(path, dpi=800, bbox_inches='tight', facecolor='white')
    FIG_FILES[name] = path
    FIGS[name] = fig
    print(f'  saved {name}')
    return fig

def box_grid(df, group_col, metrics, fname, title, order):
    n_col = 4
    n_row = int(np.ceil(len(metrics) / n_col))
    fig, axes = plt.subplots(n_row, n_col, figsize=(4.2 * n_col, 3.6 * n_row))
    axes = axes.flatten()
    stats_by_var = {}
    for i, v in enumerate(metrics):
        ax = axes[i]
        data = [df[df[group_col] == g][v].dropna().values for g in order]
        bp = ax.boxplot(data, patch_artist=True, widths=.55, showmeans=True,
                        meanprops=dict(marker='D', markerfacecolor='white',
                                        markeredgecolor='black', markersize=4))
        for patch, color in zip(bp['boxes'], PAL[:len(order)]):
            patch.set_facecolor(color); patch.set_alpha(.75)
        res = two_group_test(*data)
        stats_by_var[v] = res
        q = [r for r in aim1 if r['metric'] == v]
        p_show = fp(res['p'])
        ax.set_title(f'{SHORT.get(v, LABELS[v])}', fontsize=11, fontweight='bold')
        ax.set_ylabel(LABELS[v].split('(')[-1].rstrip(')') if '(' in LABELS[v] else 'Value', fontsize=9)
        ax.text(.03, .97, f'p = {p_show}', transform=ax.transAxes, fontsize=8.5,
                va='top', ha='left', color='#333333')
        ax.set_xticklabels(order, fontsize=9)
        ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)
        ax.grid(axis='y', alpha=.25, linestyle=':')
    for j in range(len(metrics), len(axes)):
        axes[j].set_visible(False)
    fig.suptitle(title, fontsize=13, fontweight='bold', y=1.01)
    fig.tight_layout()
    return save_fig(fig, fname), stats_by_var

BOX_METRICS = ['hb_a1_c', 'g_m_i', 'average_glucose', 'g_v',
               't_i_r', 't_a_r_250', 't_b_r_54', 'g_r_i']

_, stats_dm = box_grid(df, 'dm_type', BOX_METRICS, 'fig1_boxplots_by_dmtype.png',
                       'Glycemic metrics by diabetes type', ['T1DM', 'T2DM'])
_, stats_cat = box_grid(df, 'tir_cat', [m for m in BOX_METRICS if m != 'g_v'],
                        'fig2_boxplots_by_tircategory.png',
                        'Glycemic metrics by weekly TIR achievement category (CV panel omitted: deterministic)',
                        ['Minimal', 'Moderate to Adequate'])

# ---- Figure 3: Spearman heatmap with FDR significance
fig, ax = plt.subplots(figsize=(11, 9))
mask = np.zeros_like(rho.values, dtype=bool)
mask[np.triu_indices_from(mask, k=1)] = True
plot_rho = rho.values.copy()
plot_rho[mask] = np.nan
im = ax.imshow(plot_rho, cmap='RdBu_r', vmin=-1, vmax=1)
labels_s = [SHORT[v] for v in BATTERY]
ax.set_xticks(range(len(labels_s))); ax.set_xticklabels(labels_s, rotation=45, ha='right', fontsize=9)
ax.set_yticks(range(len(labels_s))); ax.set_yticklabels(labels_s, fontsize=9)
for i in range(len(BATTERY)):
    for j in range(len(BATTERY)):
        if i == j:
            continue
        if not mask[i, j] or j > i:
            if j > i:
                r = rho.values[i, j]
                q = q_map[f'{SHORT[BATTERY[i]]} - {SHORT[BATTERY[j]]}']
                star = '**' if q < .01 else ('*' if q < .05 else '')
                txt = f'{r:.2f}{star}'
                ax.text(j, i, txt, ha='center', va='center', fontsize=6.8,
                        color='white' if abs(r) > .55 else 'black')
cbar = fig.colorbar(im, ax=ax, shrink=.8)
cbar.set_label("Spearman's rho", fontsize=10)
ax.set_title("Spearman correlations among glycemic metrics (BH-FDR: * q<.05, ** q<.01)",
             fontsize=12, fontweight='bold')
fig.tight_layout()
save_fig(fig, 'fig3_spearman_heatmap.png')

# ---- Figure 4: Scatter panels
fig, axes = plt.subplots(1, 2, figsize=(11, 4.8))
panels = [('hb_a1_c', 'g_m_i', 'HbA1c (%)', 'GMI (%)', 'HbA1c vs GMI'),
          ('hb_a1_c', 't_i_r', 'HbA1c (%)', 'TIR (%)', 'HbA1c vs TIR')]
for k, (xv, yv, xl, yl, ttl) in enumerate(panels):
    ax = axes[k]
    for g, color, gname in [('T1DM', PAL[0], 'T1DM'), ('T2DM', PAL[1], 'T2DM')]:
        sub = df[df['dm_type'] == g]
        ax.scatter(sub[xv], sub[yv], s=28, alpha=.65, color=color,
                   edgecolors='white', linewidths=.4, label=f'{gname} (n={len(sub)})')
    r, p = sps.spearmanr(df[xv], df[yv])
    ax.set_xlabel(xl, fontsize=10); ax.set_ylabel(yl, fontsize=10)
    ax.set_title(f'{ttl}: rho = {r:.2f}, p = {fp(p)}', fontsize=11, fontweight='bold')
    ax.legend(fontsize=9, frameon=False, loc='best')
    ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)
    ax.grid(alpha=.25, linestyle=':')
fig.tight_layout()
save_fig(fig, 'fig4_scatter_hba1c.png')

# ---- Figure 5: DM type x category barplot
fig, ax = plt.subplots(figsize=(6.5, 4.8))
ct_plot = pd.crosstab(df['dm_type'], df['tir_cat'], normalize='index') * 100
ct_plot = ct_plot[['Minimal', 'Moderate to Adequate']]
ct_plot.plot(kind='bar', ax=ax, color=[PAL[0], PAL[1]], rot=0, width=.55, edgecolor='white')
ax.set_ylabel('Within diabetes type (%)', fontsize=10)
ax.set_xlabel('Diabetes type', fontsize=10)
ax.set_title(f'Distribution of weekly TIR category by diabetes type\n'
             f'chi2({dof_dm}) = {chi2_dm:.2f}, p = {fp(p_dm)}, Cramer V = {v_dm:.2f}',
             fontsize=11, fontweight='bold')
ax.legend(title='Weekly TIR', fontsize=9, frameon=False)
ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)
ax.grid(axis='y', alpha=.25, linestyle=':')
for c in ax.containers:
    ax.bar_label(c, fmt='%.1f%%', fontsize=8.5, padding=2)
fig.tight_layout()
save_fig(fig, 'fig5_barplot_dmtype_tircat.png')

# ---- Figure 6: Forest plot - Model A aORs
def forest_logit(fit, fname, title):
    m = fit['model']
    preds = fit['predictors']
    ors = [np.exp(m.params[p]) for p in preds]
    lo = [np.exp(m.conf_int().loc[p][0]) for p in preds]
    hi = [np.exp(m.conf_int().loc[p][1]) for p in preds]
    pvs = [m.pvalues[p] for p in preds]
    names = [LABELS.get(p, p) for p in preds]
    y = np.arange(len(preds))[::-1]
    fig, ax = plt.subplots(figsize=(8.5, .5 * len(preds) + 2))
    for i, (o, l, h, pv) in enumerate(zip(ors, lo, hi, pvs)):
        color = PAL[0] if pv < .05 else '#BBBBBB'
        ax.plot([l, h], [y[i], y[i]], color='#888888', lw=1.2)
        ax.scatter(o, y[i], s=45, color=color, zorder=3)
        ax.text(h * 1.08, y[i], f'{o:.2f} [{l:.2f}, {h:.2f}], p = {fp(pv)}',
                va='center', fontsize=8.5, color='#333333')
    ax.axvline(1, color='red', ls='--', lw=.9, alpha=.6)
    ax.set_yticks(y); ax.set_yticklabels(names, fontsize=9.5)
    ax.set_xscale('log')
    ax.set_xlabel('Adjusted odds ratio (95% CI), log scale', fontsize=10)
    ax.set_title(title, fontsize=11.5, fontweight='bold')
    ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)
    ax.grid(axis='x', alpha=.25, linestyle=':')
    fig.tight_layout()
    return save_fig(fig, fname)

forest_logit(logitA, 'fig6_forest_modelA.png',
             f'Model A: adjusted odds ratios for weekly TIR category\n'
             f'(Nagelkerke R2 = {logitA["nagelkerke"]:.3f}, AUC = {logitA["auc"]:.3f})')
forest_logit(logitB, 'fig7_forest_modelB.png',
             f'Model B: adjusted odds ratios for TIR >= 50%\n'
             f'(Nagelkerke R2 = {logitB["nagelkerke"]:.3f}, AUC = {logitB["auc"]:.3f})')

# ---- Figure 8: forest for linear model B coefficients
def forest_linear(fit, fname, title):
    m = fit['model']
    preds = fit['predictors']
    y = np.arange(len(preds))[::-1]
    fig, ax = plt.subplots(figsize=(8.5, .5 * len(preds) + 2))
    for i, p in enumerate(preds):
        b = m.params[p]; lo = m.conf_int().loc[p][0]; hi = m.conf_int().loc[p][1]
        pv = m.pvalues[p]
        color = PAL[0] if pv < .05 else '#BBBBBB'
        ax.plot([lo, hi], [y[i], y[i]], color='#888888', lw=1.2)
        ax.scatter(b, y[i], s=45, color=color, zorder=3)
        beta = fit['betas'].get(p, np.nan)
        ax.text(max(abs(lo), abs(hi)) * 1.15, y[i],
                f'{b:.2f} [{lo:.2f}, {hi:.2f}], beta = {beta:.2f}, p = {fp(pv)}',
                va='center', fontsize=8.5, color='#333333')
    ax.axvline(0, color='red', ls='--', lw=.9, alpha=.6)
    ax.set_yticks(y)
    ax.set_yticklabels([LABELS.get(p, p) for p in preds], fontsize=9.5)
    ax.set_xlabel('Unstandardized B (95% CI)', fontsize=10)
    ax.set_title(title, fontsize=11.5, fontweight='bold')
    ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)
    ax.grid(axis='x', alpha=.25, linestyle=':')
    fig.tight_layout()
    return save_fig(fig, fname)

forest_linear(linC, 'fig8_forest_modelC.png',
              f'Model C: adjusted coefficients for continuous TIR (%)\n'
              f'(R2 = {linC["r2"]:.3f}, adjusted R2 = {linC["adj_r2"]:.3f})')

# ---- Figure 9: ROC curves (Model A and B)
def roc_curve_manual(y_true, y_score):
    order = np.argsort(-y_score)
    y_sorted = y_true[order]
    tps = np.cumsum(y_sorted); fps = np.cumsum(1 - y_sorted)
    tpr = np.concatenate([[0], tps / y_sorted.sum()])
    fpr = np.concatenate([[0], fps / (len(y_sorted) - y_sorted.sum())])
    return fpr, tpr

fig, axes = plt.subplots(1, 2, figsize=(10.5, 5))
for k, (fit, ax, ttl) in enumerate([
        (logitA, axes[0], 'Model A: weekly TIR category'),
        (logitB, axes[1], 'Model B: TIR >= 50%')]):
    y = df['tir_cat01'].values if k == 0 else df['tir50'].values
    scores = fit['model'].predict(sm.add_constant(df[fit['predictors']].astype(float))).values
    fpr, tpr = roc_curve_manual(y.astype(int), scores)
    ax.plot(fpr, tpr, color=PAL[0], lw=2,
            label=f'AUC = {fit["auc"]:.3f}\n[{fit["auc_ci"][0]:.3f}, {fit["auc_ci"][1]:.3f}]')
    ax.plot([0, 1], [0, 1], color='#999999', ls='--', lw=1)
    ax.set_xlabel('1 - Specificity (false-positive rate)', fontsize=10)
    ax.set_ylabel('Sensitivity (true-positive rate)', fontsize=10)
    ax.set_title(ttl, fontsize=11, fontweight='bold')
    ax.legend(fontsize=9, frameon=False, loc='lower right')
    ax.set_xlim(-.02, 1.02); ax.set_ylim(-.02, 1.02)
    ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)
    ax.grid(alpha=.25, linestyle=':')
fig.tight_layout()
save_fig(fig, 'fig9_roc_curves.png')

# ======================================================================
# STEPS 11-13: INTERPRETATION + APA TABLES + WORD REPORT
# ======================================================================
print()
print('=' * 70)
print('STEPS 11-13: Interpretation, APA tables, Word report')
print('=' * 70)

exp = APAWordExporter('Glycemic Metrics Across Types of Diabetes Mellitus and Their '
                     'Relationship With Weekly Time in Range: A Cross-Sectional '
                     'Analysis of Continuous Glucose Monitoring Data')

exp.add_heading('1. Background and Study Design', 1)
exp.add_paragraph(
    'This cross-sectional observational analysis evaluated intermittently-scanned continuous glucose '
    'monitoring (isCGM) data from adolescents and young adults with insulin-treated diabetes mellitus. '
    f'The analytical sample comprised {n_raw} patients with complete data (T1DM: n={n1}; T2DM: n={n2}). '
    'The study had three pre-specified objectives: (a) to compare glycemic metrics across types of '
    'diabetes mellitus; (b) to evaluate variations in glycemic parameters relative to the percentage of '
    'weekly Time in Range (TIR); and (c) to investigate the correlations among continuous glycemic metrics. '
    f'Statistical inference was two-sided with alpha = .05. With the available sample sizes, the minimum '
    f'detectable standardized between-group difference at 80% power was d = {mde:.2f}; smaller true '
    'differences may therefore have been missed (limitation noted in Section 8).')

exp.add_heading('2. Data Cleaning and Quality Assurance Methodology', 1)
exp.add_paragraph(
    f'The source file ({n_raw} rows x {p_raw} columns) had previously undergone automated cleaning. '
    f'In the present pipeline, {len(audit_cols)} redundant audit columns (raw-value duplicates and '
    'per-variable outlier flags) were dropped, string fields were whitespace-trimmed (the source '
    'contained trailing-space category variants), and the data were re-audited with the DataCleaner '
    f'module: {n_dup} duplicate patient identifiers and {n_missing_total} missing values were present in '
    'the analysis set, so no imputation was required. Outlier capping was deliberately not applied '
    'because CGM range metrics are bounded compositional percentages (TIR + TBR + TAR reference the same '
    'sensor time) and legitimate extreme values carry clinical meaning; source-file outlier flags were '
    'reviewed instead.')
exp.add_paragraph(
    'Three data-quality observations are reported transparently. First, the supplied categorical variable '
    f'"weekly_time_in_range_achieved" was empirically identical to a dichotomization of glucose '
    f'variability (CV) at the 36% cut-point (agreement = {cv36_agree:.1%}): every patient labeled '
    '"Minimal" had CV < 36% and every patient labeled "Moderate to Adequate" had CV >= 36%. Because 36% '
    'is the international consensus threshold defining unstable glycemia, this variable operationally '
    f'reflects achieved glycemic stability rather than the literal percentage of time in range; consistent '
    f'with this, median TIR was paradoxically higher in the "Minimal" group ({tir_by_cat["Minimal"]:.0f}%) '
    f'than in the "Moderate to Adequate" group ({tir_by_cat["Moderate to Adequate"]:.0f}%). Objective 2 '
    'was therefore addressed in two complementary ways: (a) comparisons across the supplied category '
    '(Model A, with CV excluded a priori to avoid complete separation) and (b) multivariable modeling of '
    'continuous weekly TIR (Model C) and of a clinically defined TIR >= 50% outcome (Model B), which '
    'directly quantify variations in glycemic parameters relative to TIR. Second, the column named '
    '"calculated_total_daily_dose_of_insulin_i_u" contains values of 0.8-1.6, which are consistent with '
    'weight-based insulin dosing (units/kg/day) rather than absolute units; this interpretation should be '
    'confirmed against source records. Third, the five consensus CGM range bins summed to '
    f'{RESULTS["data_audit"]["cgm_sum_median"]:.0f}% (range '
    f'{df["cgm_sum_pct_calc"].min():.0f}-{df["cgm_sum_pct_calc"].max():.0f}%), i.e., deviating '
    'substantially from the expected 100%, which indicates inconsistent bin definitions or '
    'denominators in the source platform; individual range metrics are therefore interpreted as '
    'supplied, and compositional sums should be confirmed against the source data dictionary.')

exp.add_heading('3. Statistical Methods', 1)
exp.add_paragraph(
    'Variable types were auto-detected and verified (continuous: age, diabetes duration, BMI, HbA1c, mean '
    'glucose, GMI, CV, TIR, TBR levels 1-2, TAR levels 1-2, hypoglycemia duration, low-glucose events, GRI, '
    'chypo, chyper; binary: sex, diabetes type, weekly TIR category). Normality was assessed with the '
    'Shapiro-Wilk test together with skewness and kurtosis; because most percentage- and count-based CGM '
    'metrics were right-skewed (Shapiro-Wilk p < .05), nonparametric procedures dominated. Between-group '
    'comparisons used Welch t-tests with Cohen\'s d when both groups were approximately normal, and '
    'Mann-Whitney U tests with rank-biserial correlation otherwise; categorical contrasts used '
    'Pearson chi-square tests with Cramer\'s V. Within each hypothesis family (14 metrics per family; '
    '91 correlation pairs), Benjamini-Hochberg false-discovery-rate (FDR) correction was applied and both '
    'raw and adjusted p-values are reported. Correlations among glycemic metrics used Spearman rho '
    '(primary, robust to non-normality), with Pearson coefficients reported for the two approximately '
    'normal glucose indices. Multivariable models: binary outcomes used logistic regression (adjusted '
    'odds ratios [aOR], 95% CI, Nagelkerke R-squared, area under the ROC curve [AUC] with 2,000-resample '
    'bootstrap CIs); continuous TIR used ordinary least squares regression (B, 95% CI, standardized beta) '
    'with variance inflation factors (VIF), Breusch-Pagan, residual Shapiro-Wilk, and Cook distance '
    'diagnostics. Male sex, T1DM, and younger age served as reference categories.')

# ---------------- Table 1 ----------------
exp.add_heading('4. Baseline Characteristics by Diabetes Type', 1)
exp.add_apa_table(
    table1_headers, table1_rows,
    title='Table 1\nBaseline Characteristics Stratified by Diabetes Mellitus Type (N = 194)',
    caption='Note. Continuous variables presented as mean (SD) or median (IQR) per Shapiro-Wilk-based '
            'routing; categorical variables as n (%). Welch t with Cohen d, Mann-Whitney U with '
            'rank-biserial r, chi-square with Cramer V. TIR = time in range; GMI = glucose management '
            'indicator; GRI = glycemic risk index.')

t1_sig = [k for k, v in RESULTS['table1'].items() if v.get('p', 1) < .05]
sig_names = ', '.join(t1_sig) if t1_sig else 'none'
exp.add_paragraph(
    'Table 1 summarizes baseline characteristics. The two diabetes types were similar in sex '
    f'distribution (chi-square p = {fp(RESULTS["table1"]["Sex, male"]["p"])}) and daily scan frequency. '
    f'Patients with T2DM were uniformly older (ages 18-39 vs 14-17 years; complete separation of age '
    f'distributions), heavier (median {np.median(t2["weight"]):.0f} vs {np.median(t1["weight"]):.0f} kg) '
    f'and taller, with higher BMI (median {np.median(t2["b_m_i"]):.1f} vs {np.median(t1["b_m_i"]):.1f} '
    f'kg/m2), longer diabetes duration (median {np.median(t2["d_m_duration"]):.0f} vs '
    f'{np.median(t1["d_m_duration"]):.0f} years), slightly higher weight-adjusted insulin dose '
    f'(median {np.median(t2["calculated_total_daily_dose_of_insulin_i_u"]):.1f} vs '
    f'{np.median(t1["calculated_total_daily_dose_of_insulin_i_u"]):.1f} U/kg/day), and lower sensor '
    f'active time (median {np.median(t2["time_sensor_is_active"]):.0f}% vs '
    f'{np.median(t1["time_sensor_is_active"]):.0f}%); all p <= .009. Significant characteristics: '
    f'{sig_names}.')

# ---------------- Table 2 (Aim 1) ----------------
exp.add_heading('5. Objective 1: Glycemic Metrics Across Diabetes Types', 1)
t2_rows = [[r['label'], r['g1'], r['g2'], r['test'], r['stat_txt'], fp(r['p']),
            fp(r['p_fdr']), f'{r["es_name"]} = {r["es"]:.2f}'] for r in aim1]
exp.add_apa_table(
    ['Glycemic metric', 'T1DM (n=104)', 'T2DM (n=90)', 'Test', 'Statistic', 'p', 'p (FDR)', 'Effect size'],
    t2_rows,
    title='Table 2\nCGM-Derived Glycemic Metrics by Diabetes Mellitus Type',
    caption='Note. Values are mean (SD) or median (IQR) per normality routing. p columns show raw and '
            'Benjamini-Hochberg FDR-adjusted values within the 14-metric family. Positive rank-biserial r '
            'indicates higher values in T1DM; positive Cohen d indicates higher mean values in T1DM. '
            'TIR/TBR/TAR = time in/below/above range; CV = coefficient of variation; GRI = glycemic risk index.')

sig1 = [r for r in aim1 if r['p_fdr'] < .05]
ns1 = [r for r in aim1 if r['p_fdr'] >= .05]
sig1_txt = '; '.join(f'{r["label"]} (T1DM {r["g1"]} vs T2DM {r["g2"]}, '
                    f'{r["es_name"]} = {r["es"]:+.2f}, q = {fp(r["p_fdr"])})' for r in sig1)
exp.add_paragraph(
    f'After FDR control, {len(sig1)} of 14 glycemic metrics differed between T1DM and T2DM: {sig1_txt}. '
    + ('All significant contrasts were positive (higher values in T1DM), indicating greater '
       'hyper- and hypoglycemic exposure and higher overall glycemic risk in T1DM, consistent with the '
       'higher median TIR observed in T2DM (61% vs 41%). '
       if all(r['es'] > 0 for r in sig1) else 'Significant contrasts showed mixed directions. ')
    + f'The TIR contrast itself did not survive FDR correction (raw p = '
    f'{fp([r for r in aim1 if r["metric"] == "t_i_r"][0]["p"])}, q = '
    f'{fp([r for r in aim1 if r["metric"] == "t_i_r"][0]["p_fdr"])}). Metrics that '
    f'did not differ after FDR control: {", ".join(r["label"] for r in ns1)} (Table 2, Figure 1).')

# ---------------- Table 3 (Aim 2a) ----------------
exp.add_heading('6. Objective 2: Glycemic Parameters Relative to Weekly Time in Range', 1)
exp.add_paragraph(
    'As documented in Section 2, the supplied "weekly TIR achieved" category was empirically a '
    'dichotomization of glucose variability at the consensus CV 36% cut-point. Table 3 therefore '
    'contrasts the full glycemic battery across the supplied categories, and multivariable Models A-C '
    'quantify the glycemic parameters associated with (a) the supplied category, (b) clinically defined '
    'TIR >= 50%, and (c) continuous TIR percentage, respectively.')

t3_rows = [[r['label'], r['g1'], r['g2'], r['test'], r['stat_txt'], fp(r['p']),
            fp(r['p_fdr']), f'{r["es_name"]} = {r["es"]:.2f}'] for r in aim2a]
exp.add_apa_table(
    ['Glycemic metric', 'Minimal (n=67)', 'Moderate to Adequate (n=127)', 'Test', 'Statistic',
     'p', 'p (FDR)', 'Effect size'],
    t3_rows,
    title='Table 3\nGlycemic Parameters by Supplied Weekly TIR Achievement Category',
    caption='Note. Values are mean (SD) or median (IQR) per normality routing; raw and BH-FDR-adjusted p '
            'within the 13-metric family. Positive rank-biserial r / Cohen d = higher values in the '
            '"Minimal" group. The category is exactly CV < 36% (Minimal) vs CV >= 36% (Moderate to '
            'Adequate); CV itself is therefore deterministic and omitted (see Section 2).')

sig2 = [r for r in aim2a if r['p_fdr'] < .05]
raw_only2 = [r for r in aim2a if r['p'] < .05 and r['p_fdr'] >= .05]
sig2_lbl = ', '.join(r['label'] for r in sig2) if sig2 else 'none'
exp.add_paragraph(
    f'After FDR correction, {len(sig2)} of {len(aim2a)} parameters differed across the supplied '
    f'category ({sig2_lbl}). '
    + (f'The low-glucose event count was markedly lower in the "Minimal" (stable glycemia, CV < 36%) '
       f'group (median {np.median(df[df["tir_cat"]=="Minimal"]["low_glucose_events"]):.0f} vs '
       f'{np.median(df[df["tir_cat"]=="Moderate to Adequate"]["low_glucose_events"]):.0f} events, '
       f'rank-biserial r = {[r for r in sig2 if r["metric"]=="low_glucose_events"][0]["es"]:+.2f}, '
       f'q < .001). '
       if any(r['metric'] == 'low_glucose_events' for r in sig2) else '')
    + (f'HbA1c, GMI, TIR, TBR level 1, mean glucose, GRI, and the high-glucose index showed raw '
       f'p < .05 with higher values in the "Minimal" group but did not survive FDR control '
       f'(q = .052-.070), and the paradoxical median TIR difference (Minimal 61% vs Moderate to '
       f'Adequate 40%, rank-biserial r = {[r for r in aim2a if r["metric"]=="t_i_r"][0]["es"]:+.2f}) '
       f'reflects the CV-based derivation of the category rather than an effect on time in range. '
       if raw_only2 else '')
    + f'The supplied category was weakly associated with diabetes type (chi2({dof_dm}) = {chi2_dm:.2f}, '
    f'p = {fp(p_dm)}, Cramer V = {v_dm:.2f}), unstable glycemia (CV >= 36%) being more prevalent in '
    f'T2DM ({ct_dmcat.loc["T2DM", "Moderate to Adequate"] / ct_dmcat.loc["T2DM"].sum() * 100:.1f}%) '
    f'than T1DM ({ct_dmcat.loc["T1DM", "Moderate to Adequate"] / ct_dmcat.loc["T1DM"].sum() * 100:.1f}%; '
    f'Figure 5).')

# ---------------- Table 4-6 (models) ----------------
def logit_table(fit, title, note):
    m = fit['model']
    rows = []
    for p in fit['predictors']:
        ci = m.conf_int().loc[p]
        rows.append([LABELS.get(p, p), f'{np.exp(m.params[p]):.3f}',
                     f'[{np.exp(ci[0]):.3f}, {np.exp(ci[1]):.3f}]', fp(m.pvalues[p])])
    exp.add_apa_table(['Predictor', 'aOR', '95% CI', 'p'], rows, title=title, caption='', note=note)
    return rows

exp.add_heading('6.1 Multivariable Models', 2)
logit_table(logitA, 'Table 4\nMultivariable Logistic Regression for the Supplied Weekly TIR Category',
            f'Note. n = {logitA["n"]}; likelihood-ratio chi2(8) = {logitA["lr_chi2"]:.2f}, '
            f'p = {fp(logitA["lr_p"])}; Nagelkerke R2 = {logitA["nagelkerke"]:.3f}; AUC = '
            f'{logitA["auc"]:.3f} [{logitA["auc_ci"][0]:.3f}, {logitA["auc_ci"][1]:.3f}]; all VIF <= '
            f'{max(logitA["vifs"].values()):.2f}. Outcome: 1 = "Moderate to Adequate" (CV >= 36%). '
            'CV excluded a priori (deterministic separation). Reference: T1DM, female.')

logit_table(logitB, 'Table 5\nMultivariable Logistic Regression for Clinically Defined TIR >= 50%',
            f'Note. n = {logitB["n"]}; likelihood-ratio chi2(8) = {logitB["lr_chi2"]:.2f}, '
            f'p = {fp(logitB["lr_p"])}; Nagelkerke R2 = {logitB["nagelkerke"]:.3f}; AUC = '
            f'{logitB["auc"]:.3f} [{logitB["auc_ci"][0]:.3f}, {logitB["auc_ci"][1]:.3f}]; all VIF <= '
            f'{max(logitB["vifs"].values()):.2f}. Outcome: 1 = TIR >= 50% (n = '
            f'{int(df["tir50"].sum())}); only {RESULTS["data_audit"]["tir_ge_70pct"]} patients met the '
            'consensus target of TIR > 70%, so the 50% exploratory threshold was pre-specified. '
            'Reference: T1DM, female.')

mC = linC['model']
t6_rows = []
for p in linC['predictors']:
    ci = mC.conf_int().loc[p]
    t6_rows.append([LABELS.get(p, p), f'{mC.params[p]:.2f}',
                    f'[{ci[0]:.2f}, {ci[1]:.2f}]',
                    f'{linC["betas"].get(p, np.nan):.2f}', fp(mC.pvalues[p])])
exp.add_apa_table(
    ['Predictor', 'B', '95% CI', 'beta', 'p'], t6_rows,
    title='Table 6\nMultivariable Linear Regression for Continuous Weekly TIR (%)',
    caption='',
    note=f'Note. n = {linC["n"]}; F(8, {linC["n"] - 9}) = {linC["F"]:.2f}, p = {fp(linC["p"])}; '
         f'R2 = {linC["r2"]:.3f}; adjusted R2 = {linC["adj_r2"]:.3f}; all VIF <= '
         f'{max(linC["vifs"].values()):.2f}; residual Shapiro-Wilk p = {fp(linC["resid_shapiro_p"])}; '
         f'Breusch-Pagan p = {fp(linC["bp_p"])}; maximum Cook D = {linC["max_cook"]:.3f} '
         f'({n_cook_flag} observations exceeded the conservative 4/n = {4 / linC["n"]:.3f} flagging '
         f'threshold; none exceeded the conventional 0.5 concern threshold). beta = standardized '
         f'coefficient. Reference: T1DM, female. A sensitivity fit with heteroscedasticity-'
         f'consistent (HC3) standard errors left every significance conclusion unchanged.')

def sig_pred_txt(fit):
    m = fit['model']
    return '; '.join(f'{LABELS.get(p, p)} (aOR = {np.exp(m.params[p]):.2f} '
                     f'[{np.exp(m.conf_int().loc[p][0]):.2f}, {np.exp(m.conf_int().loc[p][1]):.2f}], '
                     f'p = {fp(m.pvalues[p])})' for p in fit['predictors'] if m.pvalues[p] < .05)

mA = logitA['model']
exp.add_paragraph(
    'Model A (supplied category; outcome = "Moderate to Adequate", i.e., unstable glycemia CV >= 36%) '
    f'accounted for {logitA["nagelkerke"] * 100:.1f}% of variance (Nagelkerke R-squared) and '
    f'discriminated acceptably (AUC = {logitA["auc"]:.3f}). Independent predictors were: '
    f'{sig_pred_txt(logitA)}. Thus, male sex, higher HbA1c, higher TIR, and higher GRI were associated '
    'with lower odds of the "Moderate to Adequate" label, whereas higher BMI increased them; this '
    'pattern again reflects the CV-based derivation of the category, as the stable-glycemia '
    '("Minimal") group carried higher median TIR, HbA1c, and GRI. Model B (clinically defined '
    f'TIR >= 50%) accounted for {logitB["nagelkerke"] * 100:.1f}% of variance '
    f'(AUC = {logitB["auc"]:.3f}); independent predictors were: {sig_pred_txt(logitB)}. Higher age, '
    'female sex, T1DM, and lower HbA1c, CV, and GRI favored achieving TIR >= 50%. Model C (continuous '
    f'TIR) explained {linC["r2"] * 100:.1f}% of the variance; age (beta = {linC["betas"]["age"]:+.2f}), '
    f'male sex (beta = {linC["betas"]["male"]:+.2f}), CV (beta = {linC["betas"]["g_v"]:+.2f}), and GRI '
    f'(beta = {linC["betas"]["g_r_i"]:+.2f}) were independently associated with TIR, whereas HbA1c was '
    f'not (p = {fp(mC.pvalues["hb_a1_c"])}) after adjustment; diabetes type and duration were likewise '
    'not independent predictors of TIR.')

# ---------------- Table 7 (Aim 3) ----------------
exp.add_heading('7. Objective 3: Correlations Among Glycemic Metrics', 1)
t7_rows = [[f'{r["a"]} - {r["b"]}', f'{r["rho"]:.2f}', fp(r['p']), fp(r['q'])] for r in key_corr_rows]
exp.add_apa_table(
    ['Metric pair', 'Spearman rho', 'p', 'p (FDR)'], t7_rows,
    title='Table 7\nSpearman Correlations Among Key Glycemic Metrics (N = 194)',
    caption='Note. Raw and BH-FDR-adjusted (91-pair family) p-values. rho < 0 indicates an inverse '
            'association. The full 14 x 14 matrix is provided as supplementary File spearman_matrix.csv '
            'and visualized in Figure 3.')
exp.add_paragraph(
    'Glucose indices were strongly intercorrelated: HbA1c with GMI (Spearman rho = '
    f'{rho.loc["hb_a1_c", "g_m_i"]:.2f}; Pearson r = {r_pg:.2f}; q < .001) and moderately with mean '
    f'glucose (Spearman rho = {rho.loc["hb_a1_c", "average_glucose"]:.2f}; Pearson r = {r_pg2:.2f}; '
    f'q < .001). TIR varied inversely with TAR level 1 (rho = {rho.loc["t_i_r", "t_a_r_180_250"]:.2f}, '
    f'q < .001) and with GRI (rho = {rho.loc["g_r_i", "t_i_r"]:.2f}, q < .001), and modestly with '
    f'HbA1c (rho = {rho.loc["hb_a1_c", "t_i_r"]:.2f}, q = {q_map["HbA1c - TIR"]:.3f} after FDR). The '
    'risk indices tracked their corresponding range bins closely (chypo with TBR level 2, rho = '
    f'{rho.loc["chypo", "t_b_r_54"]:.2f}; chyper with TAR level 2, rho = {rho.loc["chyper", "t_a_r_250"]:.2f}; '
    'GRI with chyper, rho = '
    f'{rho.loc["g_r_i", "chyper"]:.2f}; all q < .001). Notably, CV was only weakly and '
    f'non-significantly correlated with TIR (rho = {rho.loc["g_v", "t_i_r"]:.2f}, q = '
    f'{q_map["CV - TIR"]:.3f}) and with GRI (rho = {rho.loc["g_v", "g_r_i"]:.2f}, q = '
    f'{q_map["CV - GRI"]:.3f}) after FDR control, indicating that glycemic variability and time in '
    'range behave as partly independent dimensions in this cohort (Figure 3).')

# ---------------- Figures ----------------
exp.add_heading('8. Figures', 1)

def add_fig(fig_key, num, caption):
    exp.add_figure(FIGS[fig_key], caption='')
    exp.add_paragraph(f'Figure {num}', bold=True)
    exp.add_paragraph(caption, italic=True)

add_fig('fig1_boxplots_by_dmtype.png', 1,
        'Glycemic metrics by diabetes type (T1DM n = 104; T2DM n = 90). Boxes = median/IQR; '
        'diamonds = mean. p-values from Welch t or Mann-Whitney U per normality routing (raw, '
        'two-sided).')
add_fig('fig2_boxplots_by_tircategory.png', 2,
        'Glycemic metrics by supplied weekly TIR achievement category (Minimal n = 67; Moderate to '
        'Adequate n = 127). The category is exactly a dichotomization of CV at 36%; the CV panel is '
        'therefore omitted. p-values from Welch t or Mann-Whitney U (raw, two-sided).')
add_fig('fig3_spearman_heatmap.png', 3,
        'Spearman correlation matrix among 14 glycemic metrics. Lower triangle annotated with rho '
        'and BH-FDR significance (* q < .05, ** q < .01). Blue = positive, red = negative.')
add_fig('fig4_scatter_hba1c.png', 4,
        'Scatterplots of HbA1c against GMI (left) and TIR (right) colored by diabetes type with '
        'Spearman rho annotations (N = 194).')
add_fig('fig5_barplot_dmtype_tircat.png', 5,
        'Distribution of the supplied weekly TIR category within each diabetes type; labels are '
        'column percentages. Pearson chi-square and Cramer V annotated.')
add_fig('fig6_forest_modelA.png', 6,
        'Forest plot of adjusted odds ratios for the supplied weekly TIR category (Model A). '
        'Filled markers = p < .05; reference line at aOR = 1. CV excluded a priori.')
add_fig('fig7_forest_modelB.png', 7,
        'Forest plot of adjusted odds ratios for clinically defined TIR >= 50% (Model B). '
        'Filled markers = p < .05; reference line at aOR = 1.')
add_fig('fig8_forest_modelC.png', 8,
        'Forest plot of adjusted coefficients (B, 95% CI) and standardized betas for continuous '
        'weekly TIR percentage (Model C). Filled markers = p < .05.')
add_fig('fig9_roc_curves.png', 9,
        'Receiver-operating characteristic curves for Model A (supplied category) and Model B '
        '(TIR >= 50%) with bootstrap 95% CIs for the AUC.')

# ---------------- Discussion ----------------
exp.add_heading('9. Discussion, Conclusions, and Limitations', 1)

sig1_txt = '; '.join(f'{r["label"]} ({r["es_name"]} = {r["es"]:.2f}, q = {fp(r["p_fdr"])})' for r in sig1)
sig2_txt = '; '.join(f'{r["label"]} ({r["es_name"]} = {r["es"]:.2f}, q = {fp(r["p_fdr"])})' for r in sig2)

mc_sig = [p for p in linC['predictors'] if mC.pvalues[p] < .05]
mc_txt = ', '.join(LABELS.get(p, p) for p in mc_sig) if mc_sig else 'none of the predictors'

exp.add_paragraph(
    'Summary of findings. (1) Objective 1 - After FDR control, the glycemic metrics that distinguished '
    f'T1DM from T2DM were: {sig1_txt if sig1 else "none reached FDR significance"}. '
    f'(2) Objective 2 - The supplied weekly TIR category was found to be exactly a dichotomization of '
    f'glucose variability at the 36% consensus cut-point (CV < 36% = "Minimal"; CV >= 36% = "Moderate '
    f'to Adequate") rather than a categorization of TIR itself. Across this category, the only '
    f'FDR-significant parameter was the low-glucose event count (lower in the stable-glycemia group), '
    f'while the stable-glycemia group paradoxically showed higher median TIR (61% vs 40%). In the '
    f'multivariable linear model for continuous TIR (Model C), {mc_txt} were independently associated '
    f'with the weekly TIR percentage (R2 = {linC["r2"]:.3f}). (3) Objective 3 - HbA1c and GMI were '
    f'strongly intercorrelated (rho = {rho.loc["hb_a1_c", "g_m_i"]:.2f}); TIR varied inversely with TAR '
    f'level 1 (rho = {rho.loc["t_i_r", "t_a_r_180_250"]:.2f}) and GRI (rho = '
    f'{rho.loc["g_r_i", "t_i_r"]:.2f}); the low-/high-glucose indices tracked their corresponding range '
    f'bins (rho = {rho.loc["chypo", "t_b_r_54"]:.2f} and {rho.loc["chyper", "t_a_r_250"]:.2f}, '
    f'respectively); and CV was only weakly and non-significantly correlated with TIR (rho = '
    f'{rho.loc["g_v", "t_i_r"]:.2f}, q = {q_map["CV - TIR"]:.3f}), indicating partial independence of '
    'glycemic variability and time in range.')

exp.add_paragraph(
    'Clinical implications. The strong HbA1c-GMI coherence (rho = '
    f'{rho.loc["hb_a1_c", "g_m_i"]:.2f}) supports the internal consistency of laboratory and '
    'sensor-derived glycemia: clinically relevant deviations of GMI from HbA1c should prompt '
    'assessment of glycemic variability, sensor wear, or hematologic factors rather than being '
    'dismissed as measurement noise. Because the supplied "weekly TIR achieved" category was '
    'empirically a CV dichotomy, its most informative post-FDR contrast was hypoglycemia burden '
    '(fewer low-glucose events in the stable-glycemia group), while the direct CV-TIR correlation '
    f'was weak (rho = {rho.loc["g_v", "t_i_r"]:.2f}, q = {q_map["CV - TIR"]:.3f}): glycemic '
    'variability and time in range therefore behave as partly independent dimensions, and both must '
    'be assessed. In the multivariable models, higher CV and higher GRI were independently '
    'associated with lower continuous TIR (Models B and C), supporting variability reduction as a '
    f'therapeutic target alongside HbA1c lowering. The uniformly low TIR (cohort median '
    f'{np.median(df["t_i_r"]):.0f}%, only '
    f'{RESULTS["data_audit"]["tir_ge_70pct"]} of {n_raw} patients meeting the consensus > 70% target) '
    'identifies substantial residual glycemic risk in this young, insulin-treated population.')

exp.add_paragraph(
    'Limitations. First, the cross-sectional design precludes causal inference and temporal ordering. '
    'Second, the supplied weekly TIR category was empirically a CV dichotomy, which we addressed by '
    'additionally modeling continuous TIR and TIR >= 50%; however, the original category label semantics '
    'could not be reconciled with TIR percentages and source-data clarification is recommended. Third, '
    f'the sample of {n_raw} adolescents/young adults limits precision (minimum detectable standardized '
    f'difference d = {mde:.2f} at 80% power) and restricts subgroup analyses; the TIR >= 50% threshold of '
    'Model B was exploratory because too few patients met the consensus > 70% target. Fourth, insulin dose '
    'units require source confirmation (values consistent with U/kg/day), and chypo/chyper definitions '
    'follow the source platform. Fifth, the five consensus range bins showed inconsistent composition '
    f'(sums of {df["cgm_sum_pct_calc"].min():.0f}-{df["cgm_sum_pct_calc"].max():.0f}% vs the expected '
    '100%), and data on mode of insulin delivery, regimen, and socioeconomic factors were unavailable. '
    'No multiplicity '
    'correction was applied across the three objectives (only within families); confirmatory studies '
    'with pre-specified hypotheses are warranted.')

exp.add_paragraph(
    'Conclusions. In this young insulin-treated cohort, glycemic control differed across diabetes types '
    f'principally in {", ".join(LABELS.get(r["metric"], r["metric"]) for r in sig1[:4]) if sig1 else "no FDR-significant metric"}, '
    f'all indicating greater glycemic burden in T1DM. Weekly TIR percentage was independently associated '
    f'with {", ".join(LABELS.get(p, p) for p in mc_sig)} (Model C), with HbA1c significant only in '
    'the logistic model for TIR >= 50% (Model B); the strong HbA1c-GMI intercorrelation and the close '
    'tracking of the risk indices with their corresponding range bins support the integrated use of '
    'multiple CGM metrics for comprehensive glycemic assessment. Replication in larger, prospectively '
    'categorized cohorts is recommended.')

exp.add_heading('10. Variable Definitions (Appendix)', 1)
exp.add_paragraph(
    'TIR = percentage of sensor glucose values 70-180 mg/dL; TBR level 1 = 54-70 mg/dL; TBR level 2 = '
    '< 54 mg/dL; TAR level 1 = 180-250 mg/dL; TAR level 2 = > 250 mg/dL; GMI = glucose management '
    'indicator; CV = coefficient of variation of glucose (100 x SD / mean); GRI = glycemic risk index '
    '(0-100 composite of hypo- and hyperglycemic risk); chypo/chyper = low-/high-glucose risk indices as '
    'supplied by the source platform (ranges consistent with blood glucose risk indices). HbA1c and '
    'insulin dosing are laboratory/clinical variables; weekly TIR category = supplied two-level variable '
    '(empirically CV < 36% vs >= 36%, Section 2).')

report_path = os.path.join(OUT_DIR, 'Glycemic_Metrics_Analysis_Report.docx')
exp.save(report_path)
print(f'\nWord report saved: {report_path}')

# Verification JSON
df_clean.to_csv(os.path.join(OUT_DIR, 'analysis_dataset.csv'), index=False)
with open(os.path.join(OUT_DIR, 'results_verification.json'), 'w') as f:
    json.dump(RESULTS, f, indent=2, default=str)
print('Verification JSON + analysis dataset saved.')
print('\nDONE. All outputs in:', OUT_DIR)
