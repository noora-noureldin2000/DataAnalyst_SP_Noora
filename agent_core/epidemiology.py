import numpy as np
import pandas as pd
from scipy import stats
import statsmodels.api as sm
from statsmodels.stats.proportion import proportion_confint

class PrevalenceCalculator:
    @staticmethod
    def point_prevalence(cases, population, per=1000):
        prev = cases / population if population > 0 else 0
        ci_low, ci_high = proportion_confint(cases, population, method='wilson')
        return {
            "prevalence": prev * per,
            "ci_lower": ci_low * per,
            "ci_upper": ci_high * per,
            "per": per
        }

    @staticmethod
    def period_prevalence(new_cases, existing_cases, population, per=1000):
        total_cases = new_cases + existing_cases
        return PrevalenceCalculator.point_prevalence(total_cases, population, per)

    @staticmethod
    def crude_vs_adjusted(strata_data, standard_population):
        pass

class IncidenceCalculator:
    @staticmethod
    def incidence_proportion(new_cases, population_at_risk, per=1000):
        inc = new_cases / population_at_risk if population_at_risk > 0 else 0
        ci_low, ci_high = proportion_confint(new_cases, population_at_risk, method='beta')
        return {
            "incidence": inc * per,
            "ci_lower": ci_low * per,
            "ci_upper": ci_high * per,
            "per": per
        }

    @staticmethod
    def incidence_rate(events, person_time, per=1000):
        rate = events / person_time if person_time > 0 else 0
        ci_low = stats.chi2.ppf(0.025, 2 * events) / (2 * person_time) if events > 0 else 0
        ci_high = stats.chi2.ppf(0.975, 2 * (events + 1)) / (2 * person_time)
        return {
            "rate": rate * per,
            "ci_lower": ci_low * per,
            "ci_upper": ci_high * per,
            "per": per
        }

    @staticmethod
    def incidence_rate_ratio(rate1, pt1, rate2, pt2):
        irr = rate1 / rate2 if rate2 > 0 else np.nan
        return {"IRR": irr, "ci_lower": np.nan, "ci_upper": np.nan}

class RateStandardizer:
    pass

class EpiVisualizer:
    pass
