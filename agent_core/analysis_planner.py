import numpy as np
import pandas as pd
import warnings
from typing import Optional, Dict, Any, List, Tuple
from scipy.stats import shapiro, skew, kurtosis

warnings.filterwarnings("ignore", category=DeprecationWarning)

_STOPWORDS = {"a","about","after","again","all","am","an","and","any","are","as","at","be","because","been","before","between","both","but","by","can","could","did","do","does","done","due","each","few","for","from","further","had","has","have","having","here","how","however","if","in","into","is","it","its","just","like","may","more","most","much","my","no","nor","not","now","of","on","once","only","or","other","our","out","over","per","said","same","shall","should","since","so","some","still","such","than","that","the","their","them","then","there","these","they","this","those","through","to","too","under","until","up","upon","very","was","way","were","what","when","where","which","while","who","why","will","with","would"}

_VARIABLE_TYPE_KEYWORDS = {
    "outcome": ["outcome","dependent","response","y","endpoint","target","result","effect","outcome variable","dv","measured"],
    "predictor": ["predictor","independent","exposure","x","covariate","feature","input","predictor variable","iv","factor"],
    "group": ["group","treatment","arm","cohort","condition","category","class","experimental","control","placebo"],
    "confounder": ["confound","adjust","control for","covariate","demographic","baseline","age","sex","gender","bmi"],
    "time": ["time","follow-up","followup","visit","week","month","year","day","longitudinal"],
    "event": ["event","death","relapse","recurrence","progression","censored","survival","status"],
}


class StatisticalPlanner:
    def __init__(self, data: pd.DataFrame, brief: str = ""):
        self.data = data
        self.brief = brief
        self.profile = None
        self.design = None
        self.plan = []
        self._stopwords = _STOPWORDS
        self._type_keywords = _VARIABLE_TYPE_KEYWORDS

    def profile_data(self) -> pd.DataFrame:
        rows = []
        for col in self.data.columns:
            s = self.data[col].dropna()
            n_total = len(self.data)
            n_miss = self.data[col].isna().sum()
            miss_pct = round(n_miss / n_total * 100, 1)
            dtype = self.data[col].dtype
            nunique = self.data[col].nunique()
            var_type = self._detect_variable_type(col, s, dtype, nunique)
            is_normal = None
            skew_val = None
            kurt_val = None
            outlier_count = 0
            if var_type == "continuous" and len(s) >= 8:
                if len(s) < 50:
                    _, p_norm = shapiro(s)
                    is_normal = p_norm > 0.05
                else:
                    skew_val = skew(s)
                    kurt_val = kurtosis(s)
                    is_normal = abs(skew_val) < 2 and abs(kurt_val) < 7
                mean_val = s.mean()
                sd_val = s.std()
                outlier_count = int(((s - mean_val).abs() > 3 * sd_val).sum()) if sd_val > 0 else 0
            rows.append({
                "variable": col, "dtype": str(dtype), "n": len(s), "n_missing": n_miss,
                "missing_pct": miss_pct, "nunique": nunique, "type": var_type,
                "is_normal": is_normal, "skew": skew_val, "kurtosis": kurt_val,
                "outliers_3sd": outlier_count,
            })
        self.profile = pd.DataFrame(rows)
        return self.profile

    def _detect_variable_type(self, col, series, dtype, nunique) -> str:
        if pd.api.types.is_integer_dtype(dtype) and nunique <= 2:
            return "binary"
        if pd.api.types.is_object_dtype(dtype) or pd.api.types.is_categorical_dtype(dtype):
            if nunique <= 2:
                return "binary"
            return "nominal"
        if pd.api.types.is_integer_dtype(dtype) and nunique < 12:
            return "ordinal"
        if pd.api.types.is_integer_dtype(dtype) and nunique >= 12:
            col_lower = col.lower()
            if any(kw in col_lower for kw in ["count","number of","n_","num_","frequency","events"]):
                return "count"
            if nunique > 100:
                return "continuous"
            return "ordinal"
        if pd.api.types.is_float_dtype(dtype):
            return "continuous"
        if pd.api.types.is_bool_dtype(dtype):
            return "binary"
        return "nominal"

    def parse_brief(self) -> Dict[str, Any]:
        text = self.brief.lower()
        words = set(w.strip(".,;:!?()[]{}") for w in text.split() if w.strip(".,;:!?()[]{}") not in self._stopwords and len(w.strip(".,;:!?()[]{}")) > 2)
        design = {"outcomes": [], "predictors": [], "groups": [], "confounders": [], "time_var": None, "event_var": None, "design_type": "unknown", "matched": False, "paired": False, "has_survival": False, "has_repeated": False, "has_interaction": False}
        for word in words:
            for key, kws in self._type_keywords.items():
                if any(kw in word for kw in kws):
                    col_matches = [c for c in self.data.columns if word in c.lower() or any(kw in c.lower() for kw in kws)]
                    if col_matches:
                        if key == "outcome":
                            design["outcomes"].extend(col_matches)
                        elif key == "predictor":
                            design["predictors"].extend(col_matches)
                        elif key == "group":
                            design["groups"].extend(col_matches)
                        elif key == "confounder":
                            design["confounders"].extend(col_matches)
                        elif key == "time":
                            design["time_var"] = col_matches[0]
                        elif key == "event":
                            design["event_var"] = col_matches[0]
        design["outcomes"] = list(set(design["outcomes"]))
        design["predictors"] = list(set(design["predictors"]))
        design["groups"] = list(set(design["groups"]))
        design["confounders"] = list(set(design["confounders"]))
        if any(kw in text for kw in ["matched","pair","crossover","within-subject","within subject","repeated"]):
            design["matched"] = True
            design["paired"] = True
        if any(kw in text for kw in ["survival","time-to-event","time to event","kaplan","cox","hazard","mortality"]):
            design["has_survival"] = True
        if any(kw in text for kw in ["longitudinal","repeated","follow-up","followup","visit","over time"]):
            design["has_repeated"] = True
        if any(kw in text for kw in ["interaction","moderation","moderating","effect modification","subgroup"]):
            design["has_interaction"] = True
        if any(kw in text for kw in ["parallel","between-subject","between subject","independent","rct","randomized"]):
            design["design_type"] = "between"
        elif any(kw in text for kw in ["crossover","within-subject","within subject","paired","repeated"]):
            design["design_type"] = "within"
        elif design["has_survival"]:
            design["design_type"] = "survival"
        elif design["has_repeated"]:
            design["design_type"] = "repeated"
        else:
            design["design_type"] = "between"
        self.design = design
        return design

    def build_plan(self) -> List[Dict[str, Any]]:
        if self.profile is None:
            self.profile_data()
        if self.design is None:
            self.parse_brief()
        self.plan = []
        self.plan.append(self._step_descriptive())
        test_steps = self._suggest_analysis_steps()
        self.plan.extend(test_steps)
        return self.plan

    def _step_descriptive(self) -> Dict[str, Any]:
        cont_vars = self.profile[self.profile["type"] == "continuous"]["variable"].tolist()
        cat_vars = self.profile[self.profile["type"].isin(["binary","nominal","ordinal"])]["variable"].tolist()
        return {
            "step": 1, "phase": "descriptive", "title": "Descriptive Statistics",
            "description": "Summarize all variables with appropriate statistics",
            "continuous_vars": cont_vars, "categorical_vars": cat_vars,
            "suggested_method": "DescriptiveStatsEnhanced from stats_enhanced.py" if cont_vars else "Frequency tables",
            "assumptions_required": [], "dispatch": "describe",
        }

    def _suggest_analysis_steps(self) -> List[Dict[str, Any]]:
        steps = []
        step_num = 2
        if self.design["has_survival"]:
            event_var = self.design["event_var"] or self._find_event_var()
            time_var = self.design["time_var"] or self._find_time_var()
            steps.append({
                "step": step_num, "phase": "survival", "title": "Survival Analysis",
                "description": "Kaplan-Meier estimation + log-rank test + Cox regression",
                "outcome": f"{time_var} + {event_var}", "predictors": self.design["predictors"] or self.design["groups"],
                "suggested_method": "KaplanMeierEstimator + LogRankTest (biostats.py) or survival_enhanced.py",
                "assumptions_required": ["proportional_hazards"], "dispatch": "survival", "event_var": event_var, "time_var": time_var,
            })
            step_num += 1
        if self.design["has_repeated"]:
            steps.append({
                "step": step_num, "phase": "repeated", "title": "Longitudinal / Repeated Measures Analysis",
                "description": "Mixed-effects model with random intercepts/slopes for time",
                "outcome": self.design["outcomes"], "predictors": self.design["predictors"] or self.design["groups"],
                "suggested_method": "MixedEffectsModel from stats_enhanced.py",
                "assumptions_required": ["normality_of_residuals", "linearity"], "dispatch": "mixed_model",
            })
            step_num += 1
        for outcome in self.design["outcomes"] or self._auto_select_outcomes():
            outcome_type = self._get_var_type(outcome)
            predictors = self.design["predictors"] or self.design["groups"] or [c for c in self.data.columns if c != outcome]
            step = self._suggest_test(outcome, outcome_type, predictors)
            if step:
                step["step"] = step_num
                steps.append(step)
                step_num += 1
        if not steps:
            cont_vars = self.profile[self.profile["type"] == "continuous"]["variable"].tolist()
            cat_vars = self.profile[self.profile["type"].isin(["binary","nominal","ordinal"])]["variable"].tolist()
            for outcome in cont_vars[:1]:
                remaining = [v for v in cont_vars if v != outcome]
                if remaining:
                    predictor = remaining[0]
                    step = self._suggest_test(outcome, "continuous", [predictor])
                    if step:
                        step["step"] = step_num
                        steps.append(step)
                        step_num += 1
                break
            if not steps and len(cat_vars) >= 1:
                steps.append({
                    "step": step_num, "phase": "categorical", "title": "Categorical Association",
                    "description": "Chi-square or Fisher's exact test for association",
                    "outcome": cat_vars[0] if cat_vars else "unknown",
                    "predictors": cat_vars[1] if len(cat_vars) > 1 else cat_vars,
                    "suggested_method": "scipy.stats.chi2_contingency or fisher_exact",
                    "assumptions_required": ["expected_frequencies"], "dispatch": "categorical",
                })
        return steps

    def _suggest_test(self, outcome, outcome_type, predictors) -> Optional[Dict]:
        pred = self.profile[self.profile["variable"].isin(predictors)]
        if pred.empty:
            return None
        pred_types = pred["type"].tolist()
        n_predictors = len(pred)
        pred_type = pred_types[0] if pred_types else "continuous"
        is_normal = True
        if outcome_type == "continuous":
            outcome_normal = self.profile[self.profile["variable"] == outcome]
            is_normal = outcome_normal["is_normal"].values[0] if len(outcome_normal) > 0 and outcome_normal["is_normal"].values[0] is not None else True
        if outcome_type == "continuous" and n_predictors == 1:
            if pred_type in ("binary", "nominal"):
                n_groups = int(self.data[predictors[0]].nunique()) if predictors[0] in self.data.columns else 2
                if n_groups == 2:
                    return {
                        "step": 0, "phase": "group_comparison", "title": f"Two-group comparison: {outcome} by {predictors[0]}",
                        "description": f"Compare {outcome} between two groups",
                        "outcome": outcome, "predictors": [predictors[0]],
                        "suggested_method": "Independent t-test" if is_normal and self._check_homoscedasticity(outcome, predictors[0]) else "Welch's t-test" if is_normal else "Mann-Whitney U test",
                        "assumptions_required": ["normality", "homoscedasticity"] if is_normal else ["none"],
                        "dispatch": "ttest",
                    }
                return {
                    "step": 0, "phase": "anova", "title": f"One-way ANOVA: {outcome} by {predictors[0]}",
                    "description": f"Compare {outcome} across {n_groups} groups",
                    "outcome": outcome, "predictors": [predictors[0]],
                    "suggested_method": "One-way ANOVA" if is_normal and self._check_homoscedasticity(outcome, predictors[0]) else "Welch's ANOVA" if is_normal else "Kruskal-Wallis test",
                    "assumptions_required": ["normality", "homoscedasticity"] if is_normal else ["none"],
                    "dispatch": "anova",
                }
            if pred_type == "continuous":
                return {
                    "step": 0, "phase": "correlation", "title": f"Correlation: {outcome} vs {predictors[0]}",
                    "description": f"Assess linear relationship between {outcome} and {predictors[0]}",
                    "outcome": outcome, "predictors": [predictors[0]],
                    "suggested_method": "Pearson correlation" if is_normal and pred_type == "continuous" else "Spearman rank correlation",
                    "assumptions_required": ["linearity", "bivariate_normality"] if is_normal else ["monotonicity"],
                    "dispatch": "correlation",
                }
        if outcome_type == "binary":
            return {
                "step": 0, "phase": "logistic", "title": f"Logistic Regression: {outcome}",
                "description": f"Model binary outcome {outcome} with {n_predictors} predictor(s)",
                "outcome": outcome, "predictors": predictors,
                "suggested_method": "Logistic regression (statsmodels Logit)",
                "assumptions_required": ["linearity_of_logit", "no_multicollinearity"],
                "dispatch": "logistic",
            }
        if outcome_type == "count":
            return {
                "step": 0, "phase": "count_regression", "title": f"Count Regression: {outcome}",
                "description": f"Model count outcome {outcome}",
                "outcome": outcome, "predictors": predictors,
                "suggested_method": "Poisson or Negative Binomial regression",
                "assumptions_required": ["dispersion"],
                "dispatch": "poisson",
            }
        if outcome_type in ("nominal", "ordinal") and n_predictors > 0:
            return {
                "step": 0, "phase": "categorical", "title": f"Categorical Association: {outcome} by predictors",
                "description": "Chi-square or Fisher's exact test",
                "outcome": outcome, "predictors": predictors,
                "suggested_method": "Chi-square test or Fisher's exact test",
                "assumptions_required": ["expected_frequencies"],
                "dispatch": "categorical",
            }
        return None

    def _check_homoscedasticity(self, var, group_var) -> bool:
        if group_var not in self.data.columns:
            return True
        groups = [self.data[var][self.data[group_var] == g].dropna() for g in self.data[group_var].unique()]
        groups = [g for g in groups if len(g) > 1]
        if len(groups) < 2:
            return True
        from scipy.stats import levene
        try:
            _, p = levene(*groups)
            return p > 0.05
        except Exception:
            return True

    def _find_event_var(self) -> Optional[str]:
        for col in self.data.columns:
            s = self.data[col].dropna()
            if s.nunique() == 2 and set(s.unique()).issubset({0, 1, "0", "1", True, False, "yes", "no", "Yes", "No", "dead", "alive", "Dead", "Alive", "event", "no event"}):
                return col
        return None

    def _find_time_var(self) -> Optional[str]:
        for col in self.data.columns:
            col_lower = col.lower()
            if any(kw in col_lower for kw in ["time","days","months","years","follow","survival","duration","week"]):
                return col
        return None

    def _auto_select_outcomes(self) -> List[str]:
        cont = self.profile[self.profile["type"] == "continuous"]["variable"].tolist()
        if cont:
            return [cont[0]]
        binary = self.profile[self.profile["type"] == "binary"]["variable"].tolist()
        if binary:
            return [binary[0]]
        return self.profile["variable"].tolist()[:1]

    def _get_var_type(self, var) -> str:
        match = self.profile[self.profile["variable"] == var]
        if not match.empty:
            return match["type"].values[0]
        return "continuous"

    def suggest_tests(self) -> List[Dict[str, Any]]:
        if not self.plan:
            self.build_plan()
        return [s for s in self.plan if s["phase"] != "descriptive"]

    def render_sap(self) -> str:
        if not self.plan:
            self.build_plan()
        lines = []
        lines.append("# Statistical Analysis Plan")
        lines.append("")
        lines.append("## 1. Data Profile")
        lines.append("")
        lines.append("| Variable | Type | N | Missing (%) | Normal | Outliers |")
        lines.append("|----------|------|---|-------------|--------|----------|")
        for _, r in self.profile.iterrows():
            norm_str = "Yes" if r["is_normal"] else ("No" if r["is_normal"] is False else "N/A")
            lines.append(f"| {r['variable']} | {r['type']} | {r['n']} | {r['missing_pct']}% | {norm_str} | {r['outliers_3sd']} |")
        lines.append("")
        lines.append("## 2. Study Design")
        lines.append("")
        lines.append(f"- **Design type**: {self.design['design_type']}")
        if self.design["outcomes"]:
            lines.append(f"- **Outcomes**: {', '.join(self.design['outcomes'])}")
        if self.design["predictors"]:
            lines.append(f"- **Predictors**: {', '.join(self.design['predictors'])}")
        if self.design["groups"]:
            lines.append(f"- **Group variables**: {', '.join(self.design['groups'])}")
        if self.design["confounders"]:
            lines.append(f"- **Confounders**: {', '.join(self.design['confounders'])}")
        lines.append(f"- **Paired/Matched**: {self.design['paired']}")
        lines.append(f"- **Survival analysis**: {self.design['has_survival']}")
        lines.append(f"- **Repeated measures**: {self.design['has_repeated']}")
        lines.append("")
        lines.append("## 3. Analysis Plan")
        lines.append("")
        for step in self.plan:
            lines.append(f"### Step {step['step']}: {step['title']}")
            lines.append("")
            lines.append(f"**{step['description']}**")
            lines.append(f"- Suggested method: `{step['suggested_method']}`")
            if step.get("outcome"):
                lines.append(f"- Outcome: `{step['outcome']}`")
            if step.get("predictors"):
                lines.append(f"- Predictor(s): {step['predictors']}")
            if step.get("assumptions_required"):
                lines.append(f"- Assumptions to verify: {', '.join(step['assumptions_required'])}")
            lines.append("")
        lines.append("---")
        lines.append("*Plan generated by StatisticalPlanner. Review and adjust based on domain expertise.*")
        return "\n".join(lines)

    def dispatch(self, step_index: int = None, phase: str = None):
        if not self.plan:
            self.build_plan()
        if phase:
            targets = [s for s in self.plan if s["phase"] == phase]
        elif step_index is not None:
            targets = [self.plan[step_index]] if 0 <= step_index < len(self.plan) else []
        else:
            targets = self.plan[1:]
        from agent_core.biostats import ClinicalRegressionSuite
        results = []
        for step in targets:
            dispatch_type = step.get("dispatch")
            if dispatch_type == "describe":
                results.append({"step": step["step"], "status": "manual", "message": "Use DescriptiveStatsEnhanced for descriptive tables"})
            elif dispatch_type == "ttest":
                from scipy.stats import ttest_ind, mannwhitneyu
                outcome = step["outcome"]
                grp = step["predictors"][0]
                groups = self.data[grp].unique()
                if len(groups) == 2:
                    g1 = self.data[outcome][self.data[grp] == groups[0]].dropna()
                    g2 = self.data[outcome][self.data[grp] == groups[1]].dropna()
                    if step["suggested_method"].startswith("Mann"):
                        stat, p = mannwhitneyu(g1, g2)
                        results.append({"step": step["step"], "test": "Mann-Whitney U", "statistic": stat, "p_value": p, "status": "completed"})
                    else:
                        stat, p = ttest_ind(g1, g2, equal_var="equal" in step["suggested_method"])
                        results.append({"step": step["step"], "test": "Independent t-test", "statistic": stat, "p_value": p, "status": "completed"})
            elif dispatch_type == "correlation":
                from scipy.stats import pearsonr, spearmanr
                outcome = step["outcome"]
                pred = step["predictors"][0]
                valid = self.data[[outcome, pred]].dropna()
                if "Pearson" in step["suggested_method"]:
                    r, p = pearsonr(valid[outcome], valid[pred])
                    results.append({"step": step["step"], "test": "Pearson r", "statistic": r, "p_value": p, "status": "completed"})
                else:
                    r, p = spearmanr(valid[outcome], valid[pred])
                    results.append({"step": step["step"], "test": "Spearman rho", "statistic": r, "p_value": p, "status": "completed"})
            else:
                results.append({"step": step["step"], "status": "manual", "message": f"Automated dispatch not available for {dispatch_type}. Use the suggested method directly."})
        return results
