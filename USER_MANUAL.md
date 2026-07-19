# Mega Medical Writer v2.1 — User Manual

## Table of Contents

1. [What This Tool Does](#1-what-this-tool-does)
2. [How to Install & Set Up](#2-how-to-install--set-up)
3. [The 13-Step Orchestration Workflow](#3-the-13-step-orchestration-workflow)
   - [Steps 1-3: Proposal Understanding & Data Cleaning](#steps-1-3-proposal-understanding--data-cleaning)
   - [Steps 4-5: Study Type & Data Types](#steps-4-5-study-type--data-types)
   - [Steps 6-7: Normality & Descriptive Statistics](#steps-6-7-normality--descriptive-statistics)
   - [Step 8: Data Visualization](#step-8-data-visualization)
   - [Steps 9-10: Hypothesis Testing & Inferential Statistics](#steps-9-10-hypothesis-testing--inferential-statistics)
   - [Steps 11-13: Interpretation & Professional Report Writing](#steps-11-13-interpretation--professional-report-writing)
4. [Trigger Words for the AI Agent](#4-trigger-words-for-the-ai-agent)
5. [Understanding the Output Document](#5-understanding-the-output-document)
6. [Troubleshooting & FAQ](#6-troubleshooting--faq)

---

## 1. What This Tool Does

This tool acts as your **Master Clinical Biostatistician & Data Analyst Orchestrator**.

It transforms raw clinical datasets into publication-ready, APA 7th edition formatted statistical reports. With the **v2.1 update**, the system strictly adheres to a zero-hallucination policy and offers a rigorous, professional 13-step statistical workflow routing for various study designs (RCT, Cohort, Case-Control, Cross-Sectional).

---

## 2. How to Install & Set Up

### What you need
- **Windows** computer
- **Python 3.10 or later**
- **The repo folder** — `D:\Naggar Analytics\DataAnalyst_SP_Noora_V2`

### Step-by-step installation
1. Open **PowerShell**.
2. Navigate to the repo: `cd "D:\Naggar Analytics\DataAnalyst_SP_Noora_V2"`
3. Install the required packages: `pip install -r requirements.txt`

---

## 3. The 13-Step Orchestration Workflow

The AI conducts data analysis through a rigid 13-step professional biostatistical pipeline ensuring absolute transparency and statistical rigor.

### Steps 1-3: Proposal Understanding & Data Cleaning
Provide the agent with your `study_brief_template.json` or write out your study background.
- The AI extracts the primary research question and clinical hypotheses.
- The agent cleans the dataset using `DataCleaner`, handling missing values and logging every dropped row.
- **Your Job:** Review the generated variables and objectives before analysis proceeds.

### Steps 4-5: Study Type & Data Types
The agent utilizes `StatisticalPlanner` to automatically determine if your study is Cross-sectional, RCT, Cohort, etc., and then mathematically profiles your variables into Continuous, Categorical, or Binary representations.

### Steps 6-7: Normality & Descriptive Statistics
Before generating p-values, the agent checks assumptions:
- **Normality:** Shapiro-Wilk tests, skewness, and kurtosis.
- **Descriptive Stats (Table 1):** Mean ± SD for parametric, Median (IQR) for non-parametric.

### Step 8: Data Visualization
The agent generates high-quality visual outputs matching the Lancet journal standards.
- Figures: Forest plots, box-jitter plots, barplots, scatter plots, and ROC curves.

### Steps 9-10: Hypothesis Testing & Inferential Statistics
The agent formally defines the null and alternative hypotheses, and runs the appropriate tests:
- **Bivariate Tests:** Paired t-tests, Wilcoxon, Friedman, multi-way ANOVA, McNemar's, etc.
- **Multivariable Models:** Logistic Regression, Linear Regression, or Cox Proportional Hazards.

### Steps 11-13: Interpretation & Professional Report Writing
The agent interprets all findings strictly against the computed tables (zero-hallucination).
- It writes a detailed narrative following APA 7th guidelines.
- It generates the final, perfectly formatted Word document including all tables, figures, and narratives.

---

## 4. Trigger Words for the AI Agent

| What you want to do | Say this |
|---|---|
| Provide study context | *"Here is my study brief JSON file."* |
| Load an Excel file | *"Attached is the raw data."* |
| Request Corrections | *"Apply Bonferroni correction to the p-values."* |
| Set outcome variable | *"The outcome is retinopathy."* |
| Approve mapping | *"Variable map approved, proceed."* |
| Format matching | *"Apply the formatting from this reference Word doc."* |

---

## 5. Understanding the Output Document

Your `.docx` file contains:
1. **Background and Study Design:** Restated objectives.
2. **Data Cleaning Methodology:** Exactly how your variables were transformed.
3. **Table 1:** Baseline Characteristics (stratified by outcome, with effect sizes).
4. **Bivariate Analysis Results:** Narrative text with strict APA 7th formatting.
5. **Multivariable Regression:** Adjusted odds ratios, standard errors, and confidence intervals.
6. **Figures:** Lancet-standard plots with automated APA captions.
7. **Discussion and Conclusions:** Objective summary.

---

## 6. Troubleshooting & FAQ

### "How do I ensure it uses my specific formatting?"
You can provide a reference document:
`python run_analysis.py --data data.xlsx --reference path/to/my_format.docx`
The system will extract fonts, heading styles, and margins.

### "Why didn't it correct for multiple comparisons?"
By design in v2.1, multiple comparisons (Bonferroni, Holm, FDR-BH) are **only applied on explicit request** to prevent over-penalizing exploratory studies.

### "The agent paused at Step 1."
This is intended! The agent may present the variable classification table and await your explicit approval before running statistical tests. Say: *"Variables look good, continue."*

---
*Mega Medical Writer v2.1 User Manual — Naggar Analytics, July 2026*
