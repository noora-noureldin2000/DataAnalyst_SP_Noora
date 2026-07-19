# Mega Medical Writer v2.0 — User Manual

## Table of Contents

1. [What This Tool Does](#1-what-this-tool-does)
2. [How to Install & Set Up](#2-how-to-install--set-up)
3. [The 6-Phase Orchestration Workflow](#3-the-6-phase-orchestration-workflow)
   - [Phase 0: Context Ingestion](#phase-0-context-ingestion)
   - [Phase 1: Data Cleaning & Core Stats](#phase-1-data-cleaning--core-stats)
   - [Phase 2: Epidemiology](#phase-2-epidemiology)
   - [Phase 3: Statistical Infrastructure](#phase-3-statistical-infrastructure)
   - [Phase 4: Visualizations & Workflows](#phase-4-visualizations--workflows)
   - [Phase 5 & 6: Integration & Verification](#phase-5--6-integration--verification)
4. [Trigger Words for the AI Agent](#4-trigger-words-for-the-ai-agent)
5. [Understanding the Output Document](#5-understanding-the-output-document)
6. [Troubleshooting & FAQ](#6-troubleshooting--faq)

---

## 1. What This Tool Does

This tool acts as your **Master Clinical Biostatistician & Data Analyst Orchestrator**.

It transforms raw clinical datasets into publication-ready, APA 7th edition formatted statistical reports. With the **v2.0 update**, the system strictly adheres to a zero-hallucination policy and offers advanced statistical routing for various study designs (RCT, Cohort, Case-Control, Cross-Sectional).

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

## 3. The 6-Phase Orchestration Workflow

The AI conducts data analysis through a rigid 6-phase pipeline ensuring absolute transparency and statistical rigor.

### Phase 0: Context Ingestion & Variable Classification
Provide the agent with your `study_brief_template.json` or write out your study background.
- The AI extracts the primary research question and clinical hypotheses.
- It auto-classifies every variable (Continuous, Categorical, Time, Status).
- **Your Job:** Review the generated Variable Role Map table and confirm it before analysis proceeds.

### Phase 1: Data Cleaning & Core Stats
The agent cleans the dataset, handling missing values and logging every dropped row in an `N-Flow Report`.
- Tests available: Paired t-tests, Wilcoxon, Friedman, multi-way ANOVA, McNemar's, Cochran Q, etc.

### Phase 2: Epidemiology
For relevant study designs, the agent calculates:
- Point/period prevalence.
- Incidence proportions/rates.
- Rate standardization.

### Phase 3: Statistical Infrastructure
Before generating p-values, the agent checks assumptions:
- **Assumptions:** Levene's, Breusch-Pagan, Shapiro-Wilk, Durbin-Watson.
- **Effect Sizes:** Cohen's d/f, Rank-Biserial r.
- **Multiple Corrections:** Bonferroni, FDR-BH, Holm (Applied **only upon explicit request**).

### Phase 4: Visualizations & Workflows
The agent runs the specific workflow for your study design (e.g., Cross-sectional, RCT, Cohort).
- Visualizations are built using the `ggplot-skills` library matching the Lancet journal standards.
- Figures: Forest plots, Kaplan-Meier curves, box-jitter plots, ROC curves.

### Phase 5 & 6: Integration & Verification
Before handing you the Word document, the AI runs the `VerificationLog`.
- It cross-checks every number in the narrative against the computed tables.
- It verifies directional claims (e.g., "Group A was higher").
- The final verification log is appended to the report as Appendix A.

---

## 4. Trigger Words for the AI Agent

| What you want to do | Say this |
|---|---|
| Provide study context | *"Here is my study brief JSON file."* |
| Load an Excel file | *"Attached is the raw data."* |
| Request Corrections | *"Apply Bonferroni correction to the p-values."* |
| Set outcome variable | *"The outcome is retinopathy."* |
| Approve mapping | *"Variable map approved, proceed with Phase 1."* |
| Format matching | *"Apply the formatting from this reference Word doc."* |

---

## 5. Understanding the Output Document

Your `.docx` file contains:
1. **Background and Study Design:** Restated objectives.
2. **N-Flow Report:** Exact counts of included/excluded subjects.
3. **Table 1:** Baseline Characteristics (stratified by outcome, with effect sizes).
4. **Primary/Secondary Outcomes:** Narrative text with strict APA 7th formatting.
5. **Figures:** Lancet-standard plots with automated APA captions.
6. **Appendix A (Verification Log):** The zero-hallucination check proving all narrative text perfectly matches the data tables.

---

## 6. Troubleshooting & FAQ

### "How do I ensure it uses my specific formatting?"
You can provide a reference document:
`python run_analysis.py --data data.xlsx --reference path/to/my_format.docx`
The system will extract fonts, heading styles, and margins.

### "Why didn't it correct for multiple comparisons?"
By design in v2.0, multiple comparisons (Bonferroni, Holm, FDR-BH) are **only applied on explicit request** to prevent over-penalizing exploratory studies.

### "The agent paused at Phase 0."
This is intended! The agent will present the variable classification table and await your explicit approval before running statistical tests. Say: *"Variables look good, continue."*

---
*Mega Medical Writer v2.0 User Manual — Naggar Analytics, July 2026*
