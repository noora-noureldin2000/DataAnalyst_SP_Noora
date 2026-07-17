class StudyDesignDetector:
    def detect(self, df, brief_text):
        return {'design': 'cross-sectional', 'confidence': 0.8, 'evidence': []}

class CrossSectionalWorkflow:
    def run(self, df, outcome, predictors, alpha=0.05):
        return {}

class CaseControlWorkflow:
    def run(self, df, case_col, exposure_cols, matched_col=None, alpha=0.05):
        return {}

class CohortWorkflow:
    def run(self, df, time_col, event_col, exposure_col, covariates, alpha=0.05):
        return {}

class RCTWorkflow:
    def run(self, df, arm_col, outcome_col, baseline_col, covariates, alpha=0.05, analysis='itt'):
        return {}

    def consort_table(self, df, arm_col, screened_n, enrolled_n):
        pass
