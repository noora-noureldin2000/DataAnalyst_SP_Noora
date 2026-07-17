# Mega Medical Writer v2.0

A comprehensive, zero-hallucination statistical analysis toolkit for medical research data, designed for opencode AI agents. Performs end-to-end analysis from raw Excel datasets to publication-ready APA 7th edition Word documents.

## Quick Start

```bash
# Install dependencies
pip install -r requirements.txt

# Run full 6-phase analysis pipeline
python run_analysis.py --data data.xlsx --output report.docx --brief "study protocol"
```

## Features (v2.0 Updates)

- **Phase 0 (Context Ingestion & Variables)**: Auto-maps variables to formal statistical roles. Parses study briefs and handles psychometric scoring (PHQ-9, GAD-7, SF-36).
- **Phase 1 (Core Statistical Tests)**: Extensive coverage including Paired t-tests, Wilcoxon, Friedman, multi-way ANOVA, McNemar's, Cochran Q, and robust 2x2 contingency tables.
- **Phase 2 (Epidemiology)**: Point/period prevalence, incidence proportions/rates, and rate standardizations.
- **Phase 3 (Statistical Infrastructure)**: Strict assumption checking (Levene, Breusch-Pagan, Durbin-Watson) and extensive effect sizes (Cohen's d/f, Rank-Biserial r). On-request multiple comparisons (Bonferroni, FDR-BH, Holm).
- **Phase 4 (Workflows & Visualizations)**: Cross-sectional, Case-Control, Cohort, and RCT routing. Lancet-standard ggplot2-style figures including forest plots, box-jitter plots, KM curves, and prevalence charts.
- **Phase 5 & 6 (Verification Log)**: Strict zero-hallucination guardrails via `verification.py`, cross-checking all narrative claims and p-values against computed data tables before Word document generation.

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
