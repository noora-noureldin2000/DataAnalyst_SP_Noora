# Mega Medical Writer v2.1

A comprehensive, zero-hallucination statistical analysis toolkit for medical research data, designed for opencode AI agents. Performs end-to-end analysis from raw Excel datasets to publication-ready APA 7th edition Word documents.

## Quick Start

```bash
# Install dependencies
pip install -r requirements.txt

# Run full 13-step analysis pipeline
python run_analysis.py --data data.xlsx --output report.docx --brief "study protocol"
```

## Features (v2.1 Updates: The 13-Step Professional Workflow)

- **Steps 1-3 (Understanding & Cleaning)**: Context ingestion, rigorous variable profiling, and automated data cleaning via `DataCleaner`.
- **Steps 4-5 (Study Design & Variable Types)**: Automated study type determination and variable classification.
- **Steps 6-7 (Normality & Descriptive Statistics)**: Shapiro-Wilk/kurtosis checks generating a beautifully stratified baseline Table 1.
- **Step 8 (Data Visualization)**: Publication-ready ROC curves, Forest plots, and Boxplots matching Lancet-standard guidelines.
- **Steps 9-10 (Hypothesis Testing & Inferential Statistics)**: Bivariate and multivariable linear/logistic/Cox regression suites strictly routing tests based on variable types.
- **Steps 11-13 (Interpretation, Tables, Report Writing)**: Zero-hallucination APA 7th edition Word document generation featuring strictly mapped p-values, aORs, and effect sizes.

## Project Structure

```
DataAnalyst_SP_Noora_V2/
├── SKILL.md                    # opencode AI agent instructions
├── run_analysis.py             # Main orchestrator CLI
├── requirements.txt            # Dependencies
├── .agents/skills/             # Customization root containing ggplot-skills
├── agent_core/                 # Core analysis modules
│   ├── study_context.py        # [Phase 0] Context ingestion & parsing
│   ├── numerical_tests.py      # [Phase 1] Core stats
│   ├── categorical_tests.py    # [Phase 1] Contingency & categorical stats
│   ├── epidemiology.py         # [Phase 2] Incidence & prevalence
│   ├── assumption_checker.py   # [Phase 3] Assumption test battery
│   ├── effect_sizes.py         # [Phase 3] Robust effect sizes
│   ├── multiple_corrections.py # [Phase 3] FDR, Bonferroni, Holm
│   ├── visualizations.py       # [Phase 4] Publication-quality figures
│   ├── study_design.py         # [Phase 4] Cross-sectional, RCT, Cohort workflows
│   ├── verification.py         # [Phase 6] Zero-hallucination log
│   ├── analysis_planner.py     # Statistical analysis plan generator
│   ├── report_generator.py     # APA report generation
│   ├── word_exporter.py        # Word document export
│   └── __init__.py             # v2.0 Exports
├── templates/                  # JSON Context briefs and APA writing templates
├── scripts/                    # Helper scripts
├── output/                     # Generated reports
└── examples/                   # Example datasets
```

## Usage

### CLI Pipeline (fully automated)

```bash
python run_analysis.py -d diabetes_data.xlsx -o report.docx -b templates/study_brief_template.json
python run_analysis.py --data data.xlsx --outcome retinopathy --output report.docx
```

### Custom Analysis (Python)

```python
from agent_core.study_context import StudyContextParser, VariableRoleClassifier
from agent_core.numerical_tests import MultiGroupTests

# Phase 0
context = StudyContextParser("templates/study_brief_template.json").parse()
classifier = VariableRoleClassifier(df, context)
roles = classifier.classify()

# Phase 1
results = MultiGroupTests.one_way_anova(data, groups)
```

## Dependencies

- numpy, pandas, scipy, statsmodels
- matplotlib, seaborn
- python-docx, openpyxl

## License

Proprietary — Naggar Analytics
