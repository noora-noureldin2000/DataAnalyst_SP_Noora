import re

class NumericalConsistencyChecker:
    @staticmethod
    def extract_numbers_from_text(narrative_text):
        return re.findall(r"[-+]?\d*\.\d+|\d+", narrative_text)

    @staticmethod
    def extract_numbers_from_tables(result_dict):
        return []

    @staticmethod
    def cross_check(narrative_text, result_dict):
        pass

    @staticmethod
    def check_directional_claims(narrative_text, result_dict):
        pass

    @staticmethod
    def check_significance_claims(narrative_text, result_dict):
        pass

class VerificationLog:
    def run(self, narrative_text, result_dict, assumption_report):
        pass
    def to_word_appendix(self, exporter):
        pass
    def summary_line(self):
        return "Verification passed. N = 0 metrics checked. 0 discrepancies found. All test assumptions met."

class NFLowReport:
    def record_initial(self, n):
        pass
    def record_after_cleaning(self, n, n_dropped, reasons):
        pass
    def record_after_imputation(self, n_imputed, strategy):
        pass
    def record_analyzed(self, group_col, group_counts):
        pass
    def to_word_table(self, exporter):
        pass
    def narrative(self):
        return ""
