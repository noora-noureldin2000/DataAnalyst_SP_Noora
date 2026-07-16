# Data Analyst Specialist for Medical Research

A comprehensive statistical analysis toolkit for medical research data, designed for opencode AI agents. Performs end-to-end analysis from raw Excel data to publication-ready Word documents.

## Quick Start

```bash
# Install dependencies
pip install -r requirements.txt

# Run full analysis pipeline
python run_analysis.py --data data.xlsx --output report.docx --brief "study protocol"
```

## Features

- **Automated data cleaning**: Column standardization, missing value imputation, outlier capping
- **Variable type detection**: Continuous, binary, categorical, ordinal auto-classification
- **Statistical analysis plan**: Auto-generated SAP based on data profile and study brief
- **Descriptive statistics**: Table 1 stratified by outcome with appropriate tests
- **Bivariate analysis**: t-tests, Mann-Whitney, chi-square, Fisher's exact with effect sizes
- **Multivariable modeling**: Logistic, linear, and Cox regression
- **Publication-quality figures**: ROC curves, boxplots, barplots, forest plots, scatter plots
- **APA-formatted reports**: Complete with tables, figures, captions, and narrative
- **Word document export**: Properly formatted .docx with professional layout

## Project Structure

```
DataAnalyst_SP_Noora/
├── SKILL.md                    # opencode AI agent instructions
├── run_analysis.py             # Main orchestrator CLI
├── requirements.txt            # Dependencies
├── agent_core/                 # Core analysis modules
│   ├── data_cleaning.py        # Automated data cleaning
│   ├── stats_enhanced.py       # 14+ statistical test classes
│   ├── analysis_planner.py     # Statistical analysis plan generator
│   ├── report_generator.py     # APA report generation
│   ├── diagnostic_toolkit.py   # ROC, Bland-Altman, ICC, meta-analysis
│   ├── bayesian_analysis.py    # Bayes factors, Bayesian estimation
│   ├── biostats.py             # Survival analysis, clinical regression
│   ├── survival_enhanced.py    # Competing risks, time-varying Cox
│   ├── causal_inference.py     # DAGs, confounding, mediation
│   ├── meta_analysis.py        # Meta pooling, forest/funnel plots
│   ├── sample_size_calculator.py # Power analysis, sample size
│   └── word_exporter.py        # Word document export
├── templates/                  # APA writing templates
├── scripts/                    # Helper scripts
├── output/                     # Generated reports
└── examples/                   # Example datasets
```

## Usage

### CLI Pipeline (fully automated)

```bash
python run_analysis.py -d diabetes_data.xlsx -o report.docx -b "Study objectives..."
python run_analysis.py --data data.xlsx --outcome retinopathy --output report.docx
python run_analysis.py --data data.xlsx --no-plots --output report.docx
```

### Custom Analysis (Python)

```python
from agent_core.data_cleaning import DataCleaner
from agent_core.stats_enhanced import DescriptiveStatsEnhanced
from agent_core.word_exporter import APAWordExporter

df = pd.read_excel('data.xlsx')
cleaner = DataCleaner(df)
cleaner.clean_pipeline()
df_clean = cleaner.get_cleaned_df()

# Run analyses, generate figures, export
exporter = APAWordExporter("Analysis Report")
exporter.add_heading("Results", level=1)
exporter.add_apa_table(headers, rows, caption="Table 1.")
exporter.add_figure(fig, caption="Figure 1. Description")
exporter.save("report.docx")
```

## Dependencies

- numpy, pandas, scipy, statsmodels
- matplotlib, seaborn
- python-docx, openpyxl

## License

Proprietary — Naggar Analytics
