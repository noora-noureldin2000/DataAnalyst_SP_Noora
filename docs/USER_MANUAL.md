# Data Analyst Specialist — User Manual

## Table of Contents

1. [What This Tool Does](#1-what-this-tool-does)
2. [How to Install & Set Up](#2-how-to-install--set-up)
3. [Quick-Start (1-Minute Test)](#3-quick-start-1-minute-test)
4. [The SOP Workflow (Step by Step)](#4-the-sop-workflow-step-by-step)
   - [Phase 1: Setup & Context](#phase-1-setup--context)
   - [Phase 2: Feed Raw Data](#phase-2-feed-raw-data)
   - [Phase 3: Generate & Approve the SAP](#phase-3-generate--approve-the-sap)
   - [Phase 4: Run the Analysis](#phase-4-run-the-analysis)
   - [Phase 5: Receive the Final Report](#phase-5-receive-the-final-report)
5. [Trigger Words for the AI Agent](#5-trigger-words-for-the-ai-agent)
6. [How to Prepare Your Excel File](#6-how-to-prepare-your-excel-file)
7. [Understanding the Output Document](#7-understanding-the-output-document)
8. [Troubleshooting & FAQ](#8-troubleshooting--faq)

---

## 1. What This Tool Does

This tool is an **AI-powered medical statistician**. You give it:

| You Provide | The Tool Does |
|---|---|
| Raw Excel data (`.xlsx` or `.csv`) | Cleans the data automatically |
| Study title + objectives + aims | Understands your research context |
| | Detects variable types (continuous, binary, categorical) |
| | Builds a Statistical Analysis Plan (SAP) |
| | Runs descriptive statistics → Table 1 |
| | Runs bivariate analysis (t-tests, chi-square, etc.) |
| | Runs multivariable regression (logistic/linear) |
| | Generates 5 publication-ready figures |
| | Writes APA-formatted narrative results |
| | Exports everything as a **Word document (.docx)** |

The final output is a complete, publication-ready statistical report — similar to what you see in medical journals.

---

## 2. How to Install & Set Up

### What you need

- **Windows** computer (the tool was built on Windows)
- **Python 3.10 or later** — [Download from python.org](https://www.python.org/downloads/)
- **The repo folder** — `D:\Naggar Analytics\DataAnalyst_SP_Noora`

### Step-by-step installation

**Step 1:** Open **PowerShell** (press `Win + R`, type `powershell`, press Enter).

**Step 2:** Navigate to the repo folder:

```powershell
cd "D:\Naggar Analytics\DataAnalyst_SP_Noora"
```

**Step 3:** Install the required packages:

```powershell
pip install -r requirements.txt
```

Wait for the installation to finish. You should see "Successfully installed" messages.

**Step 4:** Test that everything works:

```powershell
python run_analysis.py --data examples\example_medical_data.csv --outcome Retinopathy --output output\test.docx
```

You should see a 9-step progress log ending with:

```
Report saved to: D:\Naggar Analytics\DataAnalyst_SP_Noora\output\test.docx
```

---

## 3. Quick Start (1-Minute Test)

If you want to see what the tool can do before using your own data, run this single command:

```powershell
python run_analysis.py --data examples\example_medical_data.csv --outcome Retinopathy --output output\quick_test.docx
```

Open the file `output\quick_test.docx` to see a complete statistical report with:

- Title page
- 7 numbered sections
- 2 APA-formatted tables
- 5 figures (ROC curve, boxplots, barplots, forest plot, scatter plot)
- Full narrative interpretation

---

## 4. The SOP Workflow (Step by Step)

This is the **recommended workflow** for getting accurate, high-quality results. Follow these phases in order.

### Phase 1: Setup & Context

**Goal:** Give the AI agent a full understanding of your research before any analysis.

**What to do:**

Send the AI agent a message with **all** of the following information:

> **Study Title:**
> *[Your study title here]*
>
> **Primary Objective:**
> *[What is the main question you want to answer?]*
>
> **Secondary Objectives:**
> *[What else do you want to explore?]*
>
> **Study Design:**
> *[Cross-sectional / Cohort / Case-Control / RCT]*
>
> **Population:**
> *[Brief description of who was studied]*
>
> **Outcome Variable:**
> *[Which variable is your main dependent variable / outcome?]*
>
> **Key Predictors:**
> *[Which variables do you expect to be important?]*
>
> **The raw Excel file:**
> *[Attach your .xlsx or .csv file]*

**What the AI does:**

- Reads your study brief
- Scans the column names in your data
- Maps your objectives to the available variables
- Asks clarifying questions if anything is unclear

**Example message:**

> *"Study Title: Risk Factors for Diabetic Retinopathy in Type 2 Diabetes Patients*
> 
> *Primary Objective: To identify independent predictors of diabetic retinopathy*
> 
> *Secondary Objectives: (1) Describe baseline characteristics by DR status, (2) Assess glycemic control patterns*
> 
> *Study Design: Cross-sectional*
> 
> *Population: 442 diabetes patients from outpatient clinic*
> 
> *Outcome: Retinopathy (Yes/No)*
> 
> *Predictors: Age, gender, BMI, HbA1c, diabetes duration, blood pressure, smoking, hypertension*
> 
> *Attached: patient_data.xlsx"*

---

### Phase 2: Feed Raw Data

**Goal:** The AI cleans your data and detects variable types.

**What the AI does automatically:**

| Step | Description |
|---|---|
| Standardize column names | `Age (years)` → `age_years`, `HbA1c (%)` → `hba1c` |
| Extract numeric values | Removes `%`, `kg`, `cm` from text fields, keeps only numbers |
| Impute missing values | Fills missing numeric values with the **median** |
| Cap outliers | Detects extreme values (1.5× IQR rule) and caps them |
| Standardize categories | `Yes/yes/Y/1` → `yes`, `No/no/N/0` → `no` |
| Remove duplicates | Deletes exact duplicate rows |
| Detect variable types | Classifies each column as continuous/binary/categorical |

**What you should do:**

- Let the AI show you the **data cleaning report** (a numbered list of all changes made)
- **Review the variable type classifications** — the AI will list which columns it identified as continuous, binary, categorical, etc.
- If a variable is misclassified, tell the AI to fix it:
  - *"BMI should be continuous, not categorical"*
  - *"Smoking is ordinal (never/former/current), not nominal"*
  - *"The outcome is 'retinopathy', the AI should use that"*

**Checklist before moving on:**

- [ ] All column names look correct
- [ ] Variable types are correctly identified
- [ ] Outcome variable is correctly identified
- [ ] Missing data handling is appropriate
- [ ] No important variables were dropped

---

### Phase 3: Generate & Approve the SAP

**Goal:** The AI creates a Statistical Analysis Plan (SAP) for you to review and approve.

**What the AI generates:**

The SAP contains:

1. **Research objectives restated** in statistical terms
2. **Variable classifications table** (which variables are predictors, which is outcome)
3. **Proposed statistical methods** for each objective:
   - Descriptive analysis → means, SDs, frequencies
   - Bivariate analysis → t-tests, Mann-Whitney, chi-square
   - Multivariable analysis → logistic regression (for binary outcomes)
   - Figures → ROC curve, boxplots, barplots, forest plot
4. **Assumption testing** — normality, homoscedasticity
5. **Software & methods** statement

**What you should do:**

- Read the SAP carefully
- Check that the proposed tests match your study design:
  - **Binary outcome** → logistic regression ✓
  - **Continuous outcome** → linear regression
  - **Time-to-event** → Cox regression
  - **Paired data** → paired t-test / Wilcoxon
- **Approve** the SAP by saying:
  - *"SAP approved, proceed with analysis"*
- **Request changes** if needed:
  - *"Add age and gender as covariates in the model"*
  - *"Use Mann-Whitney instead of t-test for skewed variables"*
  - *"I want to see subgroup analysis by gender"*

**Example SAP approval message:**

> *"The SAP looks good. I confirm the outcome is Retinopathy (binary). Please use logistic regression adjusted for age and gender. Proceed with the analysis."*

---

### Phase 4: Run the Analysis

**Goal:** The AI performs all statistical analyses and generates figures.

**What the AI does:**

| Task | Output |
|---|---|
| Table 1: Baseline Characteristics | Continuous variables: M ± SD (or Mdn [IQR]), test statistic, p-value, effect size. Categorical variables: n (%), chi-square, p-value, Cramer's V |
| Bivariate Analysis | Summary of significant predictors |
| Multivariable Regression | Adjusted odds ratios (aOR) with 95% CI, p-values, model fit statistics |
| Figure 1: ROC Curve | AUC with 95% CI |
| Figure 2: Boxplots | Key continuous variables by outcome |
| Figure 3: Barplots | Categorical variables by outcome |
| Figure 4: Forest Plot | aOR with 95% CI for all predictors |
| Figure 5: Scatter Plot | Relationship between 2 key variables |

**What you should do:**

- Wait for the AI to complete the analysis
- If the AI encounters an error (e.g., model convergence failure), ask for a fix:
  - *"Try removing variables with high multicollinearity"*
  - *"Use Firth's logistic regression for rare events"*
- If you want additional analyses:
  - *"Also run a sensitivity analysis excluding outliers"*
  - *"Add interaction terms for age × gender"*

---

### Phase 5: Receive the Final Report

**Goal:** The AI exports a complete Word document.

**What you receive:**

A `.docx` file saved to your computer with this structure:

```
1. Background and Study Design
2. Data Cleaning Methodology
3. Table 1: Baseline Characteristics
4. Bivariate Analysis Results
5. Multivariable Regression Results
6. Figures (with APA captions)
7. Discussion and Conclusions
```

**What you should do:**

- Open the `.docx` file in Microsoft Word
- Verify the tables and figures rendered correctly
- Check that your study title and objectives appear correctly
- Make any final manual edits to the narrative text
- The report is **ready for submission** to a journal, supervisor, or stakeholder

---

## 5. Trigger Words for the AI Agent

The AI agent responds to specific keywords and phrases. Here are the most important ones:

### Session Start / Loading Data

| What you want to do | Say this |
|---|---|
| Start a new analysis | *"I want to analyze my study data"* |
| | *"New analysis"* |
| | *"Start a statistical analysis"* |
| Provide study context | *"Here is my study protocol"* |
| | *"Study title and objectives"* |
| | *"Research brief"* |
| Load an Excel file | *"Here is my data file"* |
| | *"Attached is the raw data"* |
| | *"Load this Excel sheet"* |

### Data Cleaning & Preparation

| What you want to do | Say this |
|---|---|
| View cleaning report | *"Show me the data cleaning report"* |
| | *"What changes were made to my data?"* |
| | *"Cleaning summary"* |
| Fix variable type | *"BMI should be continuous"* |
| | *"Smoking is ordinal, not nominal"* |
| | *"This column is binary"* |
| Check missing data | *"How much missing data do I have?"* |
| | *"Missing values report"* |
| Set outcome variable | *"The outcome is retinopathy"* |
| | *"Use death as the dependent variable"* |
| | *"The main endpoint is..."* |

### Analysis Plan (SAP)

| What you want to do | Say this |
|---|---|
| Generate the SAP | *"Create a statistical analysis plan"* |
| | *"Build an SAP"* |
| | *"Analysis plan please"* |
| View the SAP | *"Show me the SAP"* |
| | *"What tests will you run?"* |
| Approve the SAP | *"SAP approved"* |
| | *"Proceed with analysis"* |
| | *"Looks good, continue"* |
| Request changes | *"Add age as a covariate"* |
| | *"Use non-parametric tests"* |
| | *"Adjust for multiple comparisons"* |
| | *"Include interaction terms"* |

### Running the Analysis

| What you want to do | Say this |
|---|---|
| Start the analysis | *"Run the analysis"* |
| | *"Execute the statistical tests"* |
| | *"Perform the analysis"* |
| Run specific test | *"Run logistic regression"* |
| | *"Do a chi-square test"* |
| | *"Compare groups with t-test"* |
| Add analysis | *"Also run sensitivity analysis"* |
| | *"Add a subgroup analysis by gender"* |
| | *"Check for multicollinearity"* |
| Generate figures | *"Create publication figures"* |
| | *"Generate the ROC curve"* |
| | *"Make a forest plot"* |

### Exporting Results

| What you want to do | Say this |
|---|---|
| Export Word report | *"Export the report as Word"* |
| | *"Create the final document"* |
| | *"Generate the .docx file"* |
| | *"Save as Word document"* |
| Export specific table | *"Export Table 1 as a table"* |
| | *"Save the regression results"* |
| Request APA format | *"Use APA format"* |
| | *"APA 7th edition style"* |
| | *"Format for journal submission"* |

### General Help

| What you want to do | Say this |
|---|---|
| Ask for help | *"Help"* |
| | *"What can you do?"* |
| | *"How does this work?"* |
| Reset | *"Start over"* |
| | *"Reset analysis"* |
| | *"New study"* |
| Status check | *"Where are we in the workflow?"* |
| | *"What's the status?"* |
| | *"Progress update"* |

---

## 6. How to Prepare Your Excel File

A well-prepared Excel file leads to better, faster results.

### Best practices

| Do | Don't |
|---|---|
| First row = **column headers** | Don't leave blank rows at the top |
| Use short, clear column names | Avoid spaces if possible (`Age_Years` not `Age (yrs)`) |
| One variable per column | Don't merge cells |
| One row per participant | Don't include summary rows |
| Use consistent coding | `Yes/No` everywhere, not `Y/Yes/1/True` mixed |
| Missing = **blank cell** | Don't write "N/A", "-", or "999" for missing |

### Example of a well-formatted file

| patient_id | age_years | gender | bmi | hba1c | diabetes_duration | hypertension | retinopathy |
|---|---|---|---|---|---|---|---|
| 001 | 58 | Male | 27.3 | 7.2 | 8.5 | Yes | Yes |
| 002 | 45 | Female | 31.0 | 6.8 | 3.2 | No | No |
| 003 | 62 | Male | 25.1 | 9.1 | 15.0 | Yes | Yes |

### Column name rules

The tool automatically cleans column names, but it helps to start clean:

- Use letters, numbers, and underscores only
- Avoid: spaces, %, #, @, -, parentheses
- Good: `age_years`, `hba1c`, `systolic_bp`
- Bad: `Age (years)`, `HbA1c (%)`, `Systolic-BP`

### What the tool does automatically

Even if your file isn't perfect, the tool will:

1. ✅ Convert `Age (years)` → `age_years`
2. ✅ Extract `7.2` from `7.2%`
3. ✅ Fill missing values with medians
4. ✅ Cap extreme outliers
5. ✅ Standardize `Y/Yes/1` → `yes`

---

## 7. Understanding the Output Document

Your final `.docx` file has 7 sections. Here's what each contains:

### Section 1: Background and Study Design

Your study title and objectives are repeated here for context. The AI also describes the study population (total N, inclusion criteria if provided).

### Section 2: Data Cleaning Methodology

A numbered list of every change the tool made to your data:

```
1. Standardized 31 column names to snake_case
2. Extracted numeric values from 'hba1c' (removed % signs)
3. Imputed 25 missing values in 'hba1c' with median = 7.40
4. Capped 92 outliers across 9 variables
5. Standardized categorical values (Yes/No mapping)
6. Created derived variable 'pulse_pressure' = SBP - DBP
```

### Section 3: Table 1 — Baseline Characteristics

The most important table in medical research. It shows all variables **stratified by the outcome** (e.g., patients with DR vs without DR).

**Continuous variables row example:**

| Variable | No DR (n=263) | DR (n=105) | Test | p | Effect Size |
|---|---|---|---|---|---|
| Age (years) | 53.1 ± 13.6 | 57.4 ± 14.7 | Welch's t = -2.59 | .011 | d = -0.31 |
| HbA1c (%) | 7.1 ± 1.1 | 8.0 ± 0.9 | Welch's t = -7.68 | <.001 | d = -0.91 |

**Categorical variables row example:**

| Variable | No DR (n=263) | DR (n=105) | Chi² | p | V |
|---|---|---|---|---|---|
| Hypertension (Yes) | 92 (35.0%) | 54 (51.4%) | 7.81 | .005 | .146 |

### Section 4: Bivariate Analysis Results

A narrative summary of which variables were significantly associated with the outcome in the bivariate analysis.

### Section 5: Multivariable Regression

The regression table with adjusted odds ratios:

| Predictor | aOR | 95% CI | p | |
|---|---|---|---|---|
| Diabetes Duration | 1.204 | [1.124, 1.291] | <.001 | *** |
| Hypertension | 2.822 | [1.433, 5.558] | .003 | ** |
| Nephropathy | 2.457 | [1.175, 5.135] | .017 | * |

Plus model fit statistics: pseudo-R², AIC, N.

### Section 6: Figures

**Figure 1: ROC Curve** — Shows how well the model discriminates. AUC > 0.8 = excellent.

**Figure 2: Boxplots** — Distribution of key continuous variables by outcome group.

**Figure 3: Barplots** — Percentage of categorical variables by outcome group.

**Figure 4: Forest Plot** — Visual summary of all adjusted odds ratios.

**Figure 5: Scatter Plot** — Relationship between two continuous variables, colored by outcome.

### Section 7: Discussion and Conclusions

A brief narrative summarizing:
- Key findings
- Clinical implications
- Study limitations
- Recommendations

---

## 8. Troubleshooting & FAQ

### "I got an error: ModuleNotFoundError"

You haven't installed the requirements:

```powershell
cd "D:\Naggar Analytics\DataAnalyst_SP_Noora"
pip install -r requirements.txt
```

### "The AI doesn't understand my study design"

Be explicit in your first message:

> *"This is a cross-sectional study. The outcome is retinopathy (Yes/No). I want to identify risk factors using logistic regression."*

### "The outcome variable is wrong"

Tell the AI directly:

> *"The outcome should be 'death_status', not 'retinopathy'."*

### "I want more figures"

Ask the AI:

> *"Also generate a correlation heatmap and Kaplan-Meier curves."*

### "The Word document tables look messy"

This can happen if your data has very long text values. The table cells auto-size. You can adjust column widths manually in Word.

### "Can I use this with SPSS or Stata files?"

Currently the tool supports **Excel (.xlsx)** and **CSV (.csv)** only. Export your data from SPSS/Stata to CSV first.

### "How long does the analysis take?"

For a typical dataset (200-500 patients, 10-30 variables):
- Data cleaning: < 1 second
- Table 1 + bivariate: 2-5 seconds
- Regression: 1-3 seconds
- Figures: 3-8 seconds
- Word export: 1-2 seconds
- **Total: ~10-15 seconds**

### "The regression model didn't converge"

This can happen with:
- Very small sample sizes
- Rare outcomes (< 10 events per predictor)
- Perfect separation

Tell the AI: *"Use Firth's logistic regression"* or *"Remove variables with very few events"*.

### "How do I reset and start over?"

Say: *"Start over with a new study"* or run the command again with a new Excel file.

### "Can I interrupt the analysis mid-way?"

Yes! You can stop at any time. The analysis runs in steps, but currently the pipeline runs from start to finish. If you need to customize the analysis, use the SOP workflow in [Section 4](#4-the-sop-workflow-step-by-step) to guide the AI step by step.

---

## Quick Reference Card

```
┌──────────────────────────────────────────────────────────┐
│              SOP SUMMARY                                 │
├──────────────────────────────────────────────────────────┤
│                                                          │
│  PHASE 1: SETUP                                          │
│  Say: "Study title: ... Objective: ..."                  │
│  Say: "Here is my data file [attach .xlsx]"              │
│                                                          │
│  PHASE 2: REVIEW CLEANING                                │
│  Ask: "Show cleaning report"                             │
│  Fix: "BMI is continuous, not categorical"               │
│                                                          │
│  PHASE 3: APPROVE SAP                                    │
│  Ask: "Generate SAP"                                     │
│  Say: "SAP approved, proceed"  OR  "Add age as covariate"│
│                                                          │
│  PHASE 4: RUN ANALYSIS                                   │
│  Say: "Run the analysis"                                 │
│  Ask: "Add a forest plot" (if you want extras)           │
│                                                          │
│  PHASE 5: EXPORT REPORT                                  │
│  Say: "Export as Word document"                          │
│                                                          │
└──────────────────────────────────────────────────────────┘
```

---

*Manual v1.0 — Naggar Analytics, July 2026*
