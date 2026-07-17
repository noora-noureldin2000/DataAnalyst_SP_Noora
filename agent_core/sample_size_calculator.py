import math
from typing import Dict, Any, Optional

def probit(p: float) -> float:
    """Approximation of the inverse cumulative distribution function of the standard normal distribution (Probit).
    Hastings' approximation (error < 0.00045).
    """
    if p <= 0 or p >= 1:
        raise ValueError("Probability must be between 0 and 1 exclusive.")
    
    if p < 0.5:
        t = math.sqrt(-2.0 * math.log(p))
        return -(t - (2.515517 + 0.802853 * t + 0.010328 * t**2) / 
                  (1.0 + 1.432788 * t + 0.189269 * t**2 + 0.001308 * t**3))
    else:
        t = math.sqrt(-2.0 * math.log(1.0 - p))
        return (t - (2.515517 + 0.802853 * t + 0.010328 * t**2) / 
                (1.0 + 1.432788 * t + 0.189269 * t**2 + 0.001308 * t**3))

def get_z_score(confidence_level: float) -> float:
    """Get the Z-score for a given two-sided confidence level (e.g. 0.95 -> 1.96)."""
    alpha = 1.0 - confidence_level
    p_lookup = 1.0 - (alpha / 2.0)
    # Common exact lookup values for precision
    exact_vals = {
        0.90: 1.64485,
        0.95: 1.95996,
        0.98: 2.32635,
        0.99: 2.57583,
        0.999: 3.29053
    }
    # Check if we have an exact match in common ones
    for key, val in exact_vals.items():
        if abs(confidence_level - key) < 1e-4:
            return val
    return probit(p_lookup)

def get_z_beta(power: float) -> float:
    """Get the Z-score corresponding to the statistical power (1 - beta) (e.g., 0.80 power -> 0.8416)."""
    # Common exact lookup values for power
    exact_vals = {
        0.80: 0.84162,
        0.85: 1.03643,
        0.90: 1.28155,
        0.95: 1.64485,
        0.99: 2.32635
    }
    for key, val in exact_vals.items():
        if abs(power - key) < 1e-4:
            return val
    return probit(power)

class SampleSizeCalculator:
    """Core engine for advanced sample size calculations and academic rationales."""
    
    # 1. Cochran's Formula
    def cochran(self, p: float = 0.5, e: float = 0.05, confidence: float = 0.95) -> Dict[str, Any]:
        """Cochran's Formula for large/infinite populations."""
        if not (0 < p < 1):
            raise ValueError("Proportion (p) must be between 0 and 1.")
        if e <= 0:
            raise ValueError("Margin of error (e) must be positive.")
        
        z = get_z_score(confidence)
        n0 = (z**2 * p * (1 - p)) / (e**2)
        n_final = math.ceil(n0)
        
        rationale = (
            f"**Method**: Cochran's Formula (for large/infinite populations).\n"
            f"**Equation**: $n_0 = \\frac{{Z^2 \\cdot p \\cdot (1 - p)}}{{e^2}}$\n\n"
            f"**Calculation Steps**:\n"
            f"1. Confidence level is set to {confidence * 100}%, giving a two-sided critical Z-score of $Z = {z:.4f}$.\n"
            f"2. Expected proportion $p$ is set to {p} (or {p*100}%), representing estimated attribute presence (0.50 yields maximum variance).\n"
            f"3. Margin of error $e$ is set to {e} (or {e*100}%).\n"
            f"4. Substituting these values into the formula: \n"
            f"   $$n_0 = \\frac{{{z:.4f}^2 \\cdot {p} \\cdot (1 - {p})}}{{{e}^2}} = \\frac{{{z**2:.4f} \\cdot {p * (1 - p):.4f}}}{{{e**2:.6f}}} = {n0:.4f}$$\n"
            f"5. Rounding up to the nearest integer yields a final sample size of **{n_final}** participants.\n\n"
            f"**Clinical Context**: Cochran's formula is ideal for large-scale epidemiological surveys where the source "
            f"population size is effectively infinite or extremely large (>10,000) and the primary objective is estimating "
            f"a clinical proportion with specified precision."
        )
        
        return {
            "method": "Cochran's Formula",
            "sample_size": n_final,
            "formula": "n_0 = (Z^2 * p * (1 - p)) / e^2",
            "parameters": {"p": p, "e": e, "confidence": confidence, "z_score": round(z, 4)},
            "rationale": rationale
        }
    
    # 2. Yamane's (or Slovin's) Formula
    def yamane(self, N: int, e: float = 0.05) -> Dict[str, Any]:
        """Yamane's Formula for finite populations when proportion is unknown."""
        if N <= 0:
            raise ValueError("Population size (N) must be greater than zero.")
        if e <= 0:
            raise ValueError("Margin of error (e) must be positive.")
        
        n = N / (1 + N * e**2)
        n_final = math.ceil(n)
        
        rationale = (
            f"**Method**: Yamane's Formula (or Slovin's Formula) for finite populations.\n"
            f"**Equation**: $n = \\frac{{N}}{{1 + N \\cdot e^2}}$\n\n"
            f"**Calculation Steps**:\n"
            f"1. Target population size $N$ is known and equals {N}.\n"
            f"2. Permissible margin of error $e$ is set to {e} (or {e*100}%).\n"
            f"3. Substituting these values into the formula:\n"
            f"   $$n = \\frac{{{N}}}{{1 + {N} \\cdot {e}^2}} = \\frac{{{N}}}{{1 + {N} \\cdot {e**2:.6f}}} = \\frac{{{N}}}{{{1 + N * e**2:.4f}}} = {n:.4f}$$\n"
            f"4. Rounding up to the nearest integer yields a final sample size of **{n_final}** participants.\n\n"
            f"**Clinical Context**: Yamane's formula is a simplified, non-probabilistic approach often used in survey research "
            f"targeting defined local groups (such as staff in a single hospital network) when no prior research exists "
            f"to estimate standard deviation or proportion variance, making power calculations impossible."
        )
        
        return {
            "method": "Yamane's Formula",
            "sample_size": n_final,
            "formula": "n = N / (1 + N * e^2)",
            "parameters": {"N": N, "e": e},
            "rationale": rationale
        }

    # 3. Statistical Power Analysis
    def power_analysis(self, test_type: str, alpha: float = 0.05, power: float = 0.8, **kwargs) -> Dict[str, Any]:
        """Statistical Power Analysis for Hypothesis Testing."""
        if not (0 < alpha < 1):
            raise ValueError("Alpha must be between 0 and 1.")
        if not (0 < power < 1):
            raise ValueError("Power must be between 0 and 1.")
        
        z_alpha = get_z_score(1.0 - alpha)  # standard two-sided critical value
        z_beta = get_z_beta(power)
        
        if test_type == "two_sample_t_test":
            # Comparison of two independent means (continuous)
            sd = kwargs.get("sd")
            mean_diff = kwargs.get("mean_diff")
            cohens_d = kwargs.get("cohens_d")
            
            if cohens_d is None:
                if sd is None or mean_diff is None or sd <= 0 or mean_diff == 0:
                    raise ValueError("For two-sample t-test, provide positive 'sd' and non-zero 'mean_diff', or direct 'cohens_d'.")
                cohens_d = abs(mean_diff) / sd
            else:
                cohens_d = abs(cohens_d)
                
            if cohens_d == 0:
                raise ValueError("Effect size (Cohen's d) cannot be zero.")
                
            n_per_group = (2 * (z_alpha + z_beta)**2) / (cohens_d**2)
            n_final = math.ceil(n_per_group)
            total_n = n_final * 2
            
            formula_str = "n_per_group = 2 * (Z_alpha + Z_beta)^2 / d^2"
            rationale = (
                f"**Method**: Statistical Power Analysis (Two-Sample Independent t-test, Continuous Outcome).\n"
                f"**Equation**: $n_{{\\text{{group}}}} = \\frac{{2 \\cdot (Z_{{\\alpha/2}} + Z_{{\\beta}})^2}}{{d^2}}$\n\n"
                f"**Calculation Steps**:\n"
                f"1. Significance level $\\alpha$ is set to {alpha} (two-sided), yielding $Z_{{\\alpha/2}} = {z_alpha:.4f}$.\n"
                f"2. Statistical power is set to {power * 100}% (type II error $\\beta = {1-power:.2f}$), yielding $Z_{{\\beta}} = {z_beta:.4f}$.\n"
                f"3. Effect size (Cohen's $d$) calculated or provided is $d = {cohens_d:.4f}$"
            )
            if "sd" in kwargs and "mean_diff" in kwargs:
                rationale += f" (based on expected mean difference $\\Delta = {kwargs['mean_diff']}$ and pooled standard deviation $\\sigma = {sd}$)."
            else:
                rationale += "."
            rationale += (
                f"\n4. Substituting these values into the two-sample comparison formula:\n"
                f"   $$n_{{\\text{{group}}}} = \\frac{{2 \\cdot ({z_alpha:.4f} + {z_beta:.4f})^2}}{{{cohens_d:.4f}^2}} = \\frac{{2 \\cdot ({z_alpha + z_beta:.4f})^2}}{{{cohens_d**2:.6f}}} = \\frac{{2 \\cdot {(z_alpha + z_beta)**2:.4f}}}{{{cohens_d**2:.6f}}} = {n_per_group:.4f}$$\n"
                f"5. Rounding up to the nearest integer yields **{n_final}** participants per group, for a total study sample size of **{total_n}** (2 groups).\n\n"
                f"**Clinical Context**: This calculation determines the sample size required to detect a clinically meaningful "
                f"difference in a continuous biomarker or outcome (e.g., blood pressure, HbA1c reduction) between an active intervention "
                f"group and a control group, minimizing both false-positive (Type I) and false-negative (Type II) rates."
            )
            return {
                "method": "Statistical Power Analysis (Two-Sample Independent t-test)",
                "sample_size": total_n,
                "sample_size_per_group": n_final,
                "formula": formula_str,
                "parameters": {"alpha": alpha, "power": power, "cohens_d": round(cohens_d, 4), "z_alpha": round(z_alpha, 4), "z_beta": round(z_beta, 4)},
                "rationale": rationale
            }
            
        elif test_type == "one_sample_t_test":
            # One sample mean comparison
            sd = kwargs.get("sd")
            mean_diff = kwargs.get("mean_diff")
            cohens_d = kwargs.get("cohens_d")
            
            if cohens_d is None:
                if sd is None or mean_diff is None or sd <= 0 or mean_diff == 0:
                    raise ValueError("For one-sample t-test, provide positive 'sd' and non-zero 'mean_diff', or direct 'cohens_d'.")
                cohens_d = abs(mean_diff) / sd
            else:
                cohens_d = abs(cohens_d)
                
            n = ((z_alpha + z_beta)**2) / (cohens_d**2)
            n_final = math.ceil(n)
            
            formula_str = "n = (Z_alpha + Z_beta)^2 / d^2"
            rationale = (
                f"**Method**: Statistical Power Analysis (One-Sample / Paired t-test).\n"
                f"**Equation**: $n = \\frac{{(Z_{{\\alpha/2}} + Z_{{\\beta}})^2}}{{d^2}}$\n\n"
                f"**Calculation Steps**:\n"
                f"1. Significance level $\\alpha$ is set to {alpha} (two-sided), yielding $Z_{{\\alpha/2}} = {z_alpha:.4f}$.\n"
                f"2. Statistical power is set to {power * 100}%, yielding $Z_{{\\beta}} = {z_beta:.4f}$.\n"
                f"3. Effect size (Cohen's $d$ or paired difference ratio) is $d = {cohens_d:.4f}$.\n"
                f"4. Substituting into the formula:\n"
                f"   $$n = \\frac{{({z_alpha:.4f} + {z_beta:.4f})^2}}{{{cohens_d:.4f}^2}} = {n:.4f}$$\n"
                f"5. Rounding up to the nearest integer yields a final sample size of **{n_final}** participants.\n\n"
                f"**Clinical Context**: Paired/one-sample testing is used for pre-test vs. post-test designs in a single cohort, "
                f"or when comparing clinical results from a single treated cohort directly against a validated historical standard mean."
            )
            return {
                "method": "Statistical Power Analysis (One-Sample/Paired t-test)",
                "sample_size": n_final,
                "formula": formula_str,
                "parameters": {"alpha": alpha, "power": power, "cohens_d": round(cohens_d, 4)},
                "rationale": rationale
            }

        elif test_type == "two_sample_proportion":
            # Comparison of two independent proportions (binary)
            p1 = kwargs.get("p1")
            p2 = kwargs.get("p2")
            if p1 is None or p2 is None or not (0 < p1 < 1) or not (0 < p2 < 1):
                raise ValueError("For two-sample proportion test, provide 'p1' and 'p2' between 0 and 1.")
            
            p_bar = (p1 + p2) / 2
            numerator = (z_alpha * math.sqrt(2 * p_bar * (1 - p_bar)) + z_beta * math.sqrt(p1 * (1 - p1) + p2 * (1 - p2)))**2
            denominator = (p1 - p2)**2
            
            if denominator == 0:
                raise ValueError("Proportions p1 and p2 must be different.")
                
            n_per_group = numerator / denominator
            n_final = math.ceil(n_per_group)
            total_n = n_final * 2
            
            formula_str = "n_per_group = (Z_alpha*sqrt(2*p_bar*(1-p_bar)) + Z_beta*sqrt(p1(1-p1) + p2(1-p2)))^2 / (p1 - p2)^2"
            rationale = (
                f"**Method**: Statistical Power Analysis (Two-Sample Proportions / Comparison of Two Proportions).\n"
                f"**Equation**: $n_{{\\text{{group}}}} = \\frac{{\\left(Z_{{\\alpha/2}}\\sqrt{{2\\bar{{p}}(1-\\bar{{p}})}} + Z_{{\\beta}}\\sqrt{{p_1(1-p_1) + p_2(1-p_2)}}\\right)^2}}{{(p_1 - p_2)^2}}$\n\n"
                f"**Calculation Steps**:\n"
                f"1. Significance level $\\alpha$ is set to {alpha} (two-sided), yielding $Z_{{\\alpha/2}} = {z_alpha:.4f}$.\n"
                f"2. Statistical power is set to {power * 100}%, yielding $Z_{{\\beta}} = {z_beta:.4f}$.\n"
                f"3. Anticipated proportions are $p_1 = {p1}$ (group 1) and $p_2 = {p2}$ (group 2). The pooled proportion $\\bar{{p}} = {p_bar:.4f}$.\n"
                f"4. Substituting into the formula:\n"
                f"   - Standard error component: $Z_{{\\alpha/2}}\\sqrt{{2\\bar{{p}}(1-\\bar{{p}})}} = {z_alpha:.4f} \\times \\sqrt{{2 \\times {p_bar} \\times (1 - {p_bar})}} = {z_alpha * math.sqrt(2 * p_bar * (1 - p_bar)):.4f}$\n"
                f"   - Power component: $Z_{{\\beta}}\\sqrt{{p_1(1-p_1) + p_2(1-p_2)}} = {z_beta:.4f} \\times \\sqrt{{{p1}(1-{p1}) + {p2}(1-{p2})}} = {z_beta * math.sqrt(p1*(1-p1) + p2*(1-p2)):.4f}$\n"
                f"   - Numerator: $({z_alpha * math.sqrt(2 * p_bar * (1 - p_bar)):.4f} + {z_beta * math.sqrt(p1*(1-p1) + p2*(1-p2)):.4f})^2 = {numerator:.4f}$\n"
                f"   - Denominator: $({p1} - {p2})^2 = {denominator:.6f}$\n"
                f"   - $n_{{\\text{{group}}}} = {numerator:.4f} / {denominator:.6f} = {n_per_group:.4f}$\n"
                f"5. Rounding up to the nearest integer yields **{n_final}** participants per group, for a total sample size of **{total_n}** (2 groups).\n\n"
                f"**Clinical Context**: Standard sample size formula for comparing event rates or treatment responses between two groups in randomized clinical trials (RCTs) or comparative cohort studies (e.g., comparing cure rates, mortality rates, or side-effect incidences)."
            )
            return {
                "method": "Statistical Power Analysis (Two-Sample Proportions)",
                "sample_size": total_n,
                "sample_size_per_group": n_final,
                "formula": formula_str,
                "parameters": {"alpha": alpha, "power": power, "p1": p1, "p2": p2},
                "rationale": rationale
            }
            
        elif test_type == "one_sample_proportion":
            # Comparison of single proportion to historical control
            p1 = kwargs.get("p1")
            p0 = kwargs.get("p0")
            if p1 is None or p0 is None or not (0 < p1 < 1) or not (0 < p0 < 1):
                raise ValueError("For one-sample proportion test, provide 'p1' (expected) and 'p0' (null) between 0 and 1.")
            
            numerator = (z_alpha * math.sqrt(p0 * (1 - p0)) + z_beta * math.sqrt(p1 * (1 - p1)))**2
            denominator = (p1 - p0)**2
            
            if denominator == 0:
                raise ValueError("Proportions p1 and p0 must be different.")
                
            n = numerator / denominator
            n_final = math.ceil(n)
            
            formula_str = "n = (Z_alpha*sqrt(p0*(1-p0)) + Z_beta*sqrt(p1*(1-p1)))^2 / (p1 - p0)^2"
            rationale = (
                f"**Method**: Statistical Power Analysis (One-Sample Proportion vs. Historical Standard).\n"
                f"**Equation**: $n = \\frac{{\\left(Z_{{\\alpha/2}}\\sqrt{{p_0(1-p_0)}} + Z_{{\\beta}}\\sqrt{{p_1(1-p_1)}}\\right)^2}}{{(p_1 - p_0)^2}}$\n\n"
                f"**Calculation Steps**:\n"
                f"1. Significance level $\\alpha$ is set to {alpha} (two-sided), yielding $Z_{{\\alpha/2}} = {z_alpha:.4f}$.\n"
                f"2. Statistical power is set to {power * 100}%, yielding $Z_{{\\beta}} = {z_beta:.4f}$.\n"
                f"3. Proportions are $p_0 = {p0}$ (historical control) and $p_1 = {p1}$ (expected/clinical outcome).\n"
                f"4. Substituting into the formula:\n"
                f"   - Numerator: $({z_alpha:.4f}\\sqrt{{{p0}(1-{p0})}} + {z_beta:.4f}\\sqrt{{{p1}(1-{p1})}})^2 = {numerator:.4f}$\n"
                f"   - Denominator: $({p1} - {p0})^2 = {denominator:.6f}$\n"
                f"   - $n = {numerator:.4f} / {denominator:.6f} = {n:.4f}$\n"
                f"5. Rounding up to the nearest integer yields a final sample size of **{n_final}** participants.\n\n"
                f"**Clinical Context**: Applied in phase II clinical trials where a new treatment is evaluated to see if its event rate "
                f"(e.g., response rate, survival rate) is superior to a well-known historical standard, without requiring a concurrent control group."
            )
            return {
                "method": "Statistical Power Analysis (One-Sample Proportion)",
                "sample_size": n_final,
                "formula": formula_str,
                "parameters": {"alpha": alpha, "power": power, "p1": p1, "p0": p0},
                "rationale": rationale
            }
        else:
            raise ValueError("Unknown test_type for power analysis. Use: 'two_sample_t_test', 'one_sample_t_test', 'two_sample_proportion', or 'one_sample_proportion'.")

    # 4. Krejcie–Morgan Table / Formula
    def krejcie_morgan(self, N: int, P: float = 0.5, e: float = 0.05, confidence: float = 0.95) -> Dict[str, Any]:
        """Krejcie–Morgan Formula for sample size determination from a finite population."""
        if N <= 0:
            raise ValueError("Population size (N) must be greater than zero.")
        if not (0 < P < 1):
            raise ValueError("Proportion (P) must be between 0 and 1.")
        if e <= 0:
            raise ValueError("Margin of error (e) must be positive.")
            
        z = get_z_score(confidence)
        chi2 = z**2  # Chi-squared value for 1 degree of freedom (standard chi^2_1 = Z^2)
        
        numerator = chi2 * N * P * (1 - P)
        denominator = (e**2 * (N - 1)) + (chi2 * P * (1 - P))
        
        n = numerator / denominator
        n_final = math.ceil(n)
        
        rationale = (
            f"**Method**: Krejcie–Morgan Formula (Finite Population Proportion).\n"
            f"**Equation**: $n = \\frac{{\\chi^2 \\cdot N \\cdot P \\cdot (1 - P)}}{{e^2 \\cdot (N - 1) + \\chi^2 \\cdot P \\cdot (1 - P)}}$\n\n"
            f"**Calculation Steps**:\n"
            f"1. Target population size $N = {N}$.\n"
            f"2. Chi-squared value (for 1 degree of freedom at {confidence*100}% confidence level, equivalent to $Z^2$) is $\\chi^2 = {chi2:.4f}$ (derived from $Z = {z:.4f}$).\n"
            f"3. Expected population proportion $P$ is set to {P} (or {P*100}%), representing maximum variance assumptions (0.50).\n"
            f"4. Desired accuracy margin $e$ is set to {e} (or {e*100}%).\n"
            f"5. Substituting into the equation:\n"
            f"   - Numerator: ${chi2:.4f} \\times {N} \\times {P} \\times (1 - {P}) = {numerator:.4f}$\n"
            f"   - Denominator: $({e}^2 \\times ({N} - 1)) + ({chi2:.4f} \\times {P} \\times (1 - {P})) = {e**2 * (N - 1):.6f} + {chi2 * P * (1 - P):.4f} = {denominator:.4f}$\n"
            f"   - $n = {numerator:.4f} / {denominator:.4f} = {n:.4f}$\n"
            f"6. Rounding up to the nearest integer yields a final sample size of **{n_final}** participants.\n\n"
            f"**Clinical Context**: The Krejcie-Morgan formula (published in 1970) is widely cited in research methodologies to "
            f"establish representative sample sizes for survey populations of known, finite dimensions (e.g. patients diagnosed with a "
            f"rare syndrome in a national registry, or doctors registered in a specific clinical society)."
        )
        return {
            "method": "Krejcie–Morgan Formula",
            "sample_size": n_final,
            "formula": "n = (X^2 * N * P * (1 - P)) / (e^2 * (N - 1) + X^2 * P * (1 - P))",
            "parameters": {"N": N, "P": P, "e": e, "confidence": confidence, "chi_squared": round(chi2, 4)},
            "rationale": rationale
        }

    # 5. Confidence Interval Method
    def confidence_interval(self, estimate_type: str, margin_of_error: float, alpha: float = 0.05, **kwargs) -> Dict[str, Any]:
        """Confidence Interval (Precision-based) Sample Size Estimation."""
        if margin_of_error <= 0:
            raise ValueError("Margin of error must be positive.")
        if not (0 < alpha < 1):
            raise ValueError("Alpha must be between 0 and 1.")
            
        z = get_z_score(1.0 - alpha)
        
        if estimate_type == "mean":
            sd = kwargs.get("sd")
            if sd is None or sd <= 0:
                raise ValueError("For mean estimation, provide a positive standard deviation 'sd'.")
            
            n = (z * sd / margin_of_error)**2
            n_final = math.ceil(n)
            
            rationale = (
                f"**Method**: Confidence Interval Method (Estimation of a Population Mean).\n"
                f"**Equation**: $n = \\left(\\frac{{Z \\cdot \\sigma}}{{E}}\\right)^2$\n\n"
                f"**Calculation Steps**:\n"
                f"1. Confidence level is set to {(1.0-alpha)*100}%, yielding $Z = {z:.4f}$.\n"
                f"2. Expected standard deviation $\\sigma$ is {sd}.\n"
                f"3. Acceptable margin of error (precision half-width) $E$ is {margin_of_error}.\n"
                f"4. Substituting into the formula:\n"
                f"   $$n = \\left(\\frac{{{z:.4f} \\times {sd}}}{{{margin_of_error}}}\\right)^2 = \\left(\\frac{{{z * sd:.4f}}}{{{margin_of_error}}}\\right)^2 = {n:.4f}$$\n"
                f"5. Rounding up yields a final sample size of **{n_final}** participants.\n\n"
                f"**Clinical Context**: This method is used when the researcher aims to estimate the true mean value of a "
                f"continuous physiological variable (e.g. mean cholesterol, BMI, blood pressure) in a population with a specified degree "
                f"of precision, rather than conducting hypothesis tests comparing groups."
            )
            return {
                "method": "Confidence Interval Method (Mean Estimation)",
                "sample_size": n_final,
                "formula": "n = (Z * sd / E)^2",
                "parameters": {"alpha": alpha, "margin_of_error": margin_of_error, "sd": sd},
                "rationale": rationale
            }
            
        elif estimate_type == "proportion":
            p = kwargs.get("p", 0.5)
            N = kwargs.get("N")  # Optional finite population size
            
            if not (0 < p < 1):
                raise ValueError("Expected proportion 'p' must be between 0 and 1.")
                
            if N is not None:
                if N <= 0:
                    raise ValueError("Population size 'N' must be greater than zero.")
                # Finite population proportion CI formula
                numerator = N * z**2 * p * (1 - p)
                denominator = (margin_of_error**2 * (N - 1)) + (z**2 * p * (1 - p))
                n = numerator / denominator
                n_final = math.ceil(n)
                
                equation_str = "$n = \\frac{{N \\cdot Z^2 \\cdot p(1-p)}}{{E^2(N-1) + Z^2 \\cdot p(1-p)}}$"
                formula_txt = "n = (N * Z^2 * p * (1-p)) / (E^2 * (N-1) + Z^2 * p * (1-p))"
                rationale = (
                    f"**Method**: Confidence Interval Method (Proportion Estimation for Finite Population).\n"
                    f"**Equation**: {equation_str}\n\n"
                    f"**Calculation Steps**:\n"
                    f"1. Target population $N = {N}$.\n"
                    f"2. Confidence level is set to {(1.0-alpha)*100}%, giving a critical Z-score of $Z = {z:.4f}$.\n"
                    f"3. Expected proportion $p$ is {p}.\n"
                    f"4. Margin of error $E$ is {margin_of_error}.\n"
                    f"5. Substituting into the equation:\n"
                    f"   - Numerator: ${N} \\times {z:.4f}^2 \\times {p} \\times (1 - {p}) = {numerator:.4f}$\n"
                    f"   - Denominator: $({margin_of_error}^2 \\times ({N} - 1)) + ({z:.4f}^2 \\times {p} \\times (1 - {p})) = {denominator:.4f}$\n"
                    f"   - $n = {numerator:.4f} / {denominator:.4f} = {n:.4f}$\n"
                    f"6. Rounding up yields a final sample size of **{n_final}** participants.\n\n"
                    f"**Clinical Context**: Precision-based proportion estimation with finite correction is used when sampling from a "
                    f"well-defined hospital patient population or institutional list to estimate a prevalence or diagnostic rate."
                )
            else:
                # Infinite proportion CI (same as Cochran's)
                res = self.cochran(p=p, e=margin_of_error, confidence=1.0-alpha)
                n_final = res["sample_size"]
                rationale = res["rationale"].replace("Cochran's Formula", "Confidence Interval Method (Proportion Estimation, Large Population)")
                formula_txt = res["formula"]
                
            return {
                "method": "Confidence Interval Method (Proportion Estimation)",
                "sample_size": n_final,
                "formula": formula_txt,
                "parameters": {"alpha": alpha, "margin_of_error": margin_of_error, "p": p, "N": N},
                "rationale": rationale
            }
        else:
            raise ValueError("Unknown estimate_type. Use 'mean' or 'proportion'.")

    # 6. Rules of Thumb
    def rules_of_thumb(self, analysis_type: str, **kwargs) -> Dict[str, Any]:
        """Rules of Thumb for multivariate models and complex designs."""
        rationale = ""
        n_final = 0
        formula_txt = ""
        params = {}
        
        if analysis_type == "linear_regression":
            k = kwargs.get("num_predictors")
            if k is None or k <= 0:
                raise ValueError("Provide a positive number of predictors 'num_predictors' for linear regression.")
            
            rule = kwargs.get("rule", "green_both")
            params["num_predictors"] = k
            params["rule"] = rule
            
            n_overall = 50 + 8 * k  # Green's overall R^2 test
            n_indiv = 104 + k      # Green's individual predictor test
            n_ratio = 15 * k       # Common ratio (15 subjects per predictor)
            
            if rule == "green_overall":
                n_final = max(50, n_overall)
                formula_txt = "N >= 50 + 8k"
                rationale = (
                    f"**Method**: Rules of Thumb (Multiple Linear Regression - Green's Rule for Overall Fit $R^2$).\n"
                    f"**Equation**: $N \\ge 50 + 8k$ (where $k$ is the number of predictors).\n\n"
                    f"**Calculation Steps**:\n"
                    f"1. Number of predictive variables $k = {k}$.\n"
                    f"2. Substituting into Green's overall formula:\n"
                    f"   $$N = 50 + 8 \\times {k} = 50 + {8*k} = {n_final}$$\n"
                    f"3. Sample size required is **{n_final}** participants.\n\n"
                    f"**Clinical Context**: Green (1991) developed this heuristic rule to ensure that regression analyses have "
                    f"adequate statistical power to test the overall variance explained ($R^2$) by the set of independent predictors."
                )
            elif rule == "green_individual":
                n_final = max(104, n_indiv)
                formula_txt = "N >= 104 + k"
                rationale = (
                    f"**Method**: Rules of Thumb (Multiple Linear Regression - Green's Rule for Individual Predictors).\n"
                    f"**Equation**: $N \\ge 104 + k$ (where $k$ is the number of predictors).\n\n"
                    f"**Calculation Steps**:\n"
                    f"1. Number of predictive variables $k = {k}$.\n"
                    f"2. Substituting into Green's individual coefficients formula:\n"
                    f"   $$N = 104 + {k} = {n_final}$$\n"
                    f"3. Sample size required is **{n_final}** participants.\n\n"
                    f"**Clinical Context**: This rule ensures adequate statistical power to test the specific partial regression "
                    f"coefficients (individual beta-weights) of each predictor in the model."
                )
            elif rule == "ratio":
                n_final = n_ratio
                formula_txt = "N = 15 * k"
                rationale = (
                    f"**Method**: Rules of Thumb (Multiple Linear Regression - Subject-to-Variable Ratio).\n"
                    f"**Equation**: $N = 15 \\cdot k$ (where a standard ratio of 15 subjects per predictor is applied).\n\n"
                    f"**Calculation Steps**:\n"
                    f"1. Number of predictive variables $k = {k}$.\n"
                    f"2. Using the ratio of 15 subjects per predictor: $N = 15 \\times {k} = {n_final}$.\n"
                    f"3. Sample size required is **{n_final}** participants.\n\n"
                    f"**Clinical Context**: General guidelines in statistical textbooks suggest ratios between 10:1 and 20:1. A ratio "
                    f"of 15:1 is a balanced heuristic commonly accepted for exploratory clinical modeling."
                )
            else:  # green_both (recommended)
                n_final = max(n_overall, n_indiv)
                formula_txt = "N >= max(50 + 8k, 104 + k)"
                rationale = (
                    f"**Method**: Rules of Thumb (Multiple Linear Regression - Green's Comprehensive Rule).\n"
                    f"**Equation**: $N \\ge \\max(50 + 8k, 104 + k)$ (to satisfy power for both overall fit and individual predictors).\n\n"
                    f"**Calculation Steps**:\n"
                    f"1. Number of predictive variables $k = {k}$.\n"
                    f"2. For overall fit test: $N = 50 + 8 \\times {k} = {n_overall}$.\n"
                    f"3. For individual predictor coefficients test: $N = 104 + {k} = {n_indiv}$.\n"
                    f"4. Taking the maximum of these two: $\\max({n_overall}, {n_indiv}) = {n_final}$.\n"
                    f"5. Comprehensive sample size required is **{n_final}** participants.\n\n"
                    f"**Clinical Context**: This is the most conservative and rigorous of Green's heuristic rules, ensuring the study "
                    f"is sufficiently powered both to determine if the clinical model works overall and to isolate which specific "
                    f"risk factors/predictors are statistically significant."
                )
                
        elif analysis_type == "logistic_regression":
            k = kwargs.get("num_predictors")
            prevalence = kwargs.get("prevalence")
            epv = kwargs.get("epv", 10)
            
            if k is None or k <= 0:
                raise ValueError("Provide a positive number of predictors 'num_predictors' for logistic regression.")
            if prevalence is None or not (0 < prevalence < 1):
                raise ValueError("Provide outcome prevalence 'prevalence' between 0 and 1.")
            if epv <= 0:
                raise ValueError("Events Per Variable (epv) must be positive (commonly 10 or 15).")
                
            n = (epv * k) / prevalence
            n_final = math.ceil(n)
            formula_txt = "N = (EPV * k) / prevalence"
            params["num_predictors"] = k
            params["prevalence"] = prevalence
            params["epv"] = epv
            rationale = (
                f"**Method**: Rules of Thumb (Logistic Regression - Events Per Variable (EPV) Rule).\n"
                f"**Equation**: $N = \\frac{{\\text{{EPV}} \\cdot k}}{{P_{{\\text{{prevalence}}}}}}$ (where $k$ is the number of predictors "
                f"and $P_{{\\text{{prevalence}}}}$ is the expected prevalence of the outcome event).\n\n"
                f"**Calculation Steps**:\n"
                f"1. Number of predictor variables $k = {k}$.\n"
                f"2. Outcome prevalence/rate $P_{{\\text{{prevalence}}}}$ is set to {prevalence} (or {prevalence*100}%).\n"
                f"3. The desired Events Per Variable threshold is set to {epv} (minimum standard is 10; 15 is preferred for robustness).\n"
                f"4. Substituting into the formula:\n"
                f"   $$N = \\frac{{{epv} \\times {k}}}{{{prevalence}}} = \\frac{{{epv * k}}}{{{prevalence}}} = {n:.4f}$$\n"
                f"5. Rounding up to the nearest integer yields a final sample size of **{n_final}** participants (expected to yield "
                f"approximately {epv * k} positive outcome events in the sample).\n\n"
                f"**Clinical Context**: In logistic regression modeling (e.g., predicting 30-day readmission, mortality, or disease presence), "
                f"statistical stability is limited by the number of events in the smaller outcome class. The EPV rule is standard "
                f"epidemiological practice (Peduzzi et al., 1996) to prevent model overfitting and unreliable confidence intervals."
            )
            
        elif analysis_type == "factor_analysis":
            num_vars = kwargs.get("num_variables")
            if num_vars is None or num_vars <= 0:
                raise ValueError("Provide a positive number of variables 'num_variables' for factor analysis.")
            
            rule = kwargs.get("rule", "comrey")
            params["num_variables"] = num_vars
            params["rule"] = rule
            
            if rule == "ratio":
                ratio = kwargs.get("ratio_multiplier", 10)  # default 10:1 subject-to-variable ratio
                n_final = num_vars * ratio
                formula_txt = f"N = {ratio} * num_variables"
                rationale = (
                    f"**Method**: Rules of Thumb (Factor Analysis / PCA - Subject-to-Variable Ratio).\n"
                    f"**Equation**: $N = {ratio} \\cdot V$ (where a ratio of {ratio} subjects per scale item/variable is applied).\n\n"
                    f"**Calculation Steps**:\n"
                    f"1. Number of test items/variables $V = {num_vars}$.\n"
                    f"2. Using the ratio of {ratio}:1, sample size $N = {ratio} \\times {num_vars} = {n_final}$.\n"
                    f"3. Sample size required is **{n_final}** participants.\n\n"
                    f"**Clinical Context**: For validating clinical scales, surveys, or psychometric questionnaires, a minimum "
                    f"subject-to-variable ratio (often 10:1 or 20:1) is recommended to ensure the sample is large enough to construct "
                    f"stable factor matrices."
                )
            else:  # comrey & lee (default)
                comrey_rating = kwargs.get("rating", "good")
                ratings = {
                    "poor": 100,
                    "fair": 200,
                    "good": 300,
                    "very_good": 500,
                    "excellent": 1000
                }
                n_final = ratings.get(comrey_rating.lower(), 300)
                formula_txt = f"N = Comrey & Lee rating ({comrey_rating})"
                rationale = (
                    f"**Method**: Rules of Thumb (Factor Analysis - Comrey & Lee Flat Sample Size Guideline).\n"
                    f"**Standard Scale**: 100 = Poor, 200 = Fair, 300 = Good, 500 = Very Good, 1000 = Excellent.\n\n"
                    f"**Calculation Steps**:\n"
                    f"1. Desired factor analysis rating selected is '{comrey_rating}'.\n"
                    f"2. Based on the Comrey & Lee (1992) classification, the corresponding sample size threshold is **{n_final}**.\n\n"
                    f"**Clinical Context**: While subject-to-variable ratio is useful, absolute sample size is critical in factor analysis "
                    f"to reduce correlation coefficient sampling error. A sample size of 300 is widely considered 'good' and adequate "
                    f"for clinical questionnaire validation regardless of the number of items, provided communalities are high."
                )
                
        elif analysis_type == "sem":
            # Structural Equation Modeling
            num_params = kwargs.get("num_parameters")
            ratio = kwargs.get("ratio_multiplier", 10)  # default 10:1 ratio
            
            params["num_parameters"] = num_params
            params["ratio_multiplier"] = ratio
            
            if num_params is not None and num_params > 0:
                n_calc = num_params * ratio
                n_final = max(200, n_calc)  # 200 is widely cited as the absolute minimum for SEM
                formula_txt = f"N = max(200, {ratio} * num_parameters)"
                rationale = (
                    f"**Method**: Rules of Thumb (Structural Equation Modeling - Bentler & Chou Parameter Ratio).\n"
                    f"**Equation**: $N = \\max(200, {ratio} \\cdot q)$ (where $q$ is the number of estimated parameters, and an "
                    f"absolute minimum baseline of 200 is enforced).\n\n"
                    f"**Calculation Steps**:\n"
                    f"1. Number of estimated model parameters $q = {num_params}$.\n"
                    f"2. Applying the parameter-to-subject ratio ({ratio}:1): $N_{{\\text{{calc}}}} = {ratio} \\times {num_params} = {n_calc}$.\n"
                    f"3. Applying the widely accepted SEM absolute minimum size barrier of 200:\n"
                    f"   $$N = \\max(200, {n_calc}) = {n_final}$$\n"
                    f"4. Sample size required is **{n_final}** participants.\n\n"
                    f"**Clinical Context**: Structural Equation Modeling (SEM) involves estimating numerous structural and measurement "
                    f"coefficients concurrently. Maximum Likelihood estimation (standard in SEM) relies on large-sample asymptotic properties. "
                    f"Thus, samples smaller than 200 risk convergence failures, model estimation errors, or inflated fit statistics (Kline, 2015)."
                )
            else:
                n_final = 200
                formula_txt = "N = 200 (absolute minimum)"
                rationale = (
                    f"**Method**: Rules of Thumb (Structural Equation Modeling - Baseline Minimum).\n"
                    f"**Standard Baseline**: $N = 200$ cases is the widely accepted standard baseline for stable SEM estimation.\n\n"
                    f"**Calculation Steps**:\n"
                    f"1. Estimated parameter count is unspecified.\n"
                    f"2. Enforcing the general rule-of-thumb minimum requirement for SEM yields a sample size of **200** participants.\n\n"
                    f"**Clinical Context**: In clinical research using path diagrams or latent variables, a sample size of at least 200 is "
                    f"considered a standard prerequisite for publishing in peer-reviewed journals to avoid convergence issues."
                )
        else:
            raise ValueError("Unknown analysis_type for rules of thumb. Use 'linear_regression', 'logistic_regression', 'factor_analysis', or 'sem'.")
            
        return {
            "method": f"Rule of Thumb ({analysis_type.replace('_', ' ').title()})",
            "sample_size": n_final,
            "formula": formula_txt,
            "parameters": params,
            "rationale": rationale
        }

    # 7. Pilot Study Method
    def pilot_study(self, pilot_type: str, **kwargs) -> Dict[str, Any]:
        """Determine sample size for a pilot study, or use pilot data to determine main study size."""
        n_final = 0
        formula_txt = ""
        params = {"pilot_type": pilot_type}
        
        if pilot_type == "flat":
            rule = kwargs.get("rule", "julious")
            params["rule"] = rule
            if rule == "julious":
                n_final = 12  # Julious rule: 12 per group
                formula_txt = "n = 12 per group"
                rationale = (
                    f"**Method**: Pilot Study Method (Julious Rule of Thumb for Pilot Trials).\n"
                    f"**Rule**: $n = 12$ participants per treatment arm.\n\n"
                    f"**Rationale & Justification**:\n"
                    f"1. Julious (2005) demonstrated that a sample size of 12 per group provides a feasible and statistically defensible "
                    f"baseline to estimate the population variance (standard deviation) and feasibility parameters for a future trial.\n"
                    f"2. For a standard 2-arm trial, this results in a total pilot sample size of **{n_final * 2}** participants (12 in treatment, 12 in control).\n\n"
                    f"**Clinical Context**: Pilot studies are conducted to evaluate feasibility, recruitment rates, and protocol adherence. "
                    f"A flat sample of 12 per group is standard in clinical trial design to obtain a reliable standard deviation estimate to "
                    f"power the subsequent phase III main trial."
                )
            elif rule == "lancaster":
                n_final = 30  # Lancaster rule: 30 overall
                formula_txt = "N = 30 overall"
                rationale = (
                    f"**Method**: Pilot Study Method (Lancaster flat size rule).\n"
                    f"**Rule**: $N = 30$ participants total for the entire pilot trial.\n\n"
                    f"**Rationale & Justification**:\n"
                    f"1. Lancaster et al. (2004) recommended a total sample size of 30 overall for pilot studies to assess feasibility, "
                    f"as this sample size is large enough to identify major implementation bottlenecks without unnecessarily wasting patient resources.\n"
                    f"2. Applying this rule yields a total pilot study sample of **30** participants.\n\n"
                    f"**Clinical Context**: Flat rules of thumb ensure study progress can occur without complex power mathematics when "
                    f"moving into completely novel clinical interventions."
                )
            else:
                n_final = 20  # General default
                formula_txt = "N = 20-40 range (using 20)"
                rationale = (
                    f"**Method**: Pilot Study Method (General range rule).\n"
                    f"**Rule**: Minimum sample size $N = 20$ participants overall.\n\n"
                    f"**Rationale**: Typical general clinical guidelines suggest a pilot size of 20 to 40 participants overall to test survey "
                    f"reliability or feasibility."
                )
                
        elif pilot_type == "percentage":
            main_n = kwargs.get("main_sample_size")
            pct = kwargs.get("percent", 0.10)
            if main_n is None or main_n <= 0:
                raise ValueError("For percentage-based pilot calculation, provide a positive 'main_sample_size'.")
            if not (0 < pct < 1):
                raise ValueError("Percent fraction 'percent' must be between 0 and 1 (e.g. 0.10 for 10%).")
                
            n = main_n * pct
            n_final = math.ceil(n)
            formula_txt = f"N_pilot = main_sample_size * percent"
            params["main_sample_size"] = main_n
            params["percent"] = pct
            rationale = (
                f"**Method**: Pilot Study Method (Percentage of Projected Main Trial Sample Size).\n"
                f"**Equation**: $N_{{\\text{{pilot}}}} = N_{{\\text{{main}}}} \\cdot P_{{\\text{{percent}}}}$\n\n"
                f"**Calculation Steps**:\n"
                f"1. Projected sample size for the planned main phase III trial $N_{{\\text{{main}}}} = {main_n}$ participants.\n"
                f"2. Target percentage proportion for the pilot stage is set to {pct*100}% ($P_{{\\text{{percent}}}} = {pct}$).\n"
                f"3. Substituting into the equation:\n"
                f"   $$N_{{\\text{{pilot}}}} = {main_n} \\times {pct} = {n:.4f}$$\n"
                f"4. Rounding up to the nearest integer yields a pilot study sample size of **{n_final}** participants.\n\n"
                f"**Clinical Context**: Designing a pilot study as 10% to 20% of the planned main trial is a common guideline in clinical trial "
                f"design (Browne, 1995). It guarantees the pilot study is proportionally scaled to the main trial size, ensuring adequate "
                f"numbers to test the feasibility of study logs, drop-outs, and randomization."
            )
        else:
            raise ValueError("Unknown pilot_type. Use 'flat' or 'percentage'.")
            
        return {
            "method": "Pilot Study Sample Size",
            "sample_size": n_final,
            "formula": formula_txt,
            "parameters": params,
            "rationale": rationale
        }

    # 8. Finite Population Correction (FPC)
    def finite_correction(self, n0: int, N: int) -> Dict[str, Any]:
        """Apply Finite Population Correction (FPC) to an infinite-population sample size."""
        if n0 <= 0:
            raise ValueError("Initial sample size (n0) must be greater than zero.")
        if N <= 0:
            raise ValueError("Population size (N) must be greater than zero.")
            
        if n0 >= N:
            # If initial calculated sample is greater than or equal to population, study whole population
            n_final = N
            fpc_applied = True
        else:
            n = n0 / (1 + (n0 - 1) / N)
            n_final = math.ceil(n)
            fpc_applied = True
            
        sampling_fraction = n0 / N
        formula_txt = "n = n0 / (1 + (n0 - 1) / N)"
        
        rationale = (
            f"**Method**: Finite Population Correction (FPC) Adjustment.\n"
            f"**Equation**: $n = \\frac{{n_0}}{{1 + \\frac{{n_0 - 1}}{{N}}}}$ (where $n_0$ is the initial sample size and $N$ is the population size).\n\n"
            f"**Calculation Steps**:\n"
            f"1. Initial sample size calculated for an infinite population $n_0 = {n0}$.\n"
            f"2. Finite population size $N = {N}$.\n"
            f"3. The sampling fraction is $n_0 / N = {n0} / {N} = {sampling_fraction:.4f}$ (or {sampling_fraction*100:.2f}%).\n"
        )
        if sampling_fraction <= 0.05:
            rationale += f"   *Note*: The sampling fraction is <= 5% (under the standard 0.05 threshold). FPC adjustment is technically optional but applied here for mathematical precision.\n"
        else:
            rationale += f"   *Note*: The sampling fraction is > 5% ({sampling_fraction*100:.2f}%). FPC adjustment is highly recommended here to avoid over-sampling.\n"
            
        if n0 >= N:
            rationale += (
                f"4. Because the calculated sample size $n_0$ ({n0}) equals or exceeds the total available population $N$ ({N}), "
                f"the study must recruit the entire population. Hence, adjusted sample size is **{n_final}**.\n\n"
                f"**Clinical Context**: Censuses are conducted instead of sampling when the disease group or patient roster is extremely small."
            )
        else:
            rationale += (
                f"4. Substituting into the FPC formula:\n"
                f"   $$n = \\frac{{{n0}}}{{1 + \\frac{{{n0} - 1}}{{{N}}}}} = \\frac{{{n0}}}{{1 + \\frac{{{n0 - 1}}}{{{N}}}}} = \\frac{{{n0}}}{{1 + { (n0 - 1) / N:.6f}}} = \\frac{{{n0}}}{{{1 + (n0 - 1) / N:.6f}}} = {n:.4f}$$\n"
                f"5. Rounding up to the nearest integer yields a corrected sample size of **{n_final}** participants.\n\n"
                f"**Clinical Context**: FPC reduces the required sample size when sampling without replacement from a small, closed population. "
                f"Since we absorb a significant portion of the population, each consecutive sample provides more relative information, "
                f"allowing us to reduce the required size without losing statistical precision."
            )
            
        return {
            "method": "Finite Population Correction (FPC)",
            "sample_size": n_final,
            "formula": formula_txt,
            "parameters": {"initial_n0": n0, "population_N": N, "sampling_fraction": round(sampling_fraction, 4)},
            "rationale": rationale
        }

    # 9. Resource-Based (or Resource Equation) Method
    def resource_based(self, num_groups: int, num_blocks: int = 0) -> Dict[str, Any]:
        """Resource Equation Method for animal/lab studies (based on ANOVA degrees of freedom)."""
        if num_groups <= 1:
            raise ValueError("Number of treatment groups must be greater than 1.")
        if num_blocks < 0:
            raise ValueError("Number of blocks must be non-negative.")
            
        # Error df: E = N - B - T
        # Target: 10 <= E <= 20
        # T (treatment df) = num_groups - 1
        # B (block df) = num_blocks - 1 if num_blocks > 0 else 0
        t_df = num_groups - 1
        b_df = num_blocks - 1 if num_blocks > 0 else 0
        
        # Min N: 10 = N_min - B - T => N_min = 10 + B + T
        n_min = 10 + b_df + t_df
        # Max N: 20 = N_max - B - T => N_max = 20 + B + T
        n_max = 20 + b_df + t_df
        
        # Group sizes
        n_per_group_min = math.ceil(n_min / num_groups)
        n_per_group_max = math.ceil(n_max / num_groups)
        
        # Realized N after rounding group sizes
        n_realized_min = n_per_group_min * num_groups
        n_realized_max = n_per_group_max * num_groups
        
        # Realized E
        e_realized_min = n_realized_min - b_df - t_df
        e_realized_max = n_realized_max - b_df - t_df
        
        formula_txt = "E = N - B - T  where 10 <= E <= 20"
        
        rationale = (
            f"**Method**: Resource Equation Method (for animal or laboratory studies).\n"
            f"**Equation**: $E = N - B - T$ (where $E$ is error degrees of freedom, $N$ is total sample size, "
            f"$B$ is block degrees of freedom, and $T$ is treatment degrees of freedom).\n"
            f"**Decision Rule**: Statistical validity requires $10 \\le E \\le 20$. If $E < 10$, the study is underpowered; "
            f"if $E > 20$, resources and animal subjects are wasted.\n\n"
            f"**Calculation Steps**:\n"
            f"1. Treatment groups = {num_groups}. Treatment degrees of freedom $T = \\text{{groups}} - 1 = {t_df}$.\n"
        )
        if num_blocks > 0:
            rationale += f"2. Randomized blocks = {num_blocks}. Block degrees of freedom $B = \\text{{blocks}} - 1 = {b_df}$.\n"
        else:
            rationale += f"2. No blocking factor is specified ($B = 0$).\n"
            
        rationale += (
            f"3. Solving for the target range $10 \\le E \\le 20$:\n"
            f"   - Minimum total subjects: $N_{{\\text{{min}}}} = 10 + B + T = 10 + {b_df} + {t_df} = {n_min}$.\n"
            f"   - Maximum total subjects: $N_{{\\text{{max}}}} = 20 + B + T = 20 + {b_df} + {t_df} = {n_max}$.\n"
            f"4. Adjusting for equal group allocation (rounding up group sizes):\n"
            f"   - Minimum group size: $\\lceil {n_min} / {num_groups} \\rceil = {n_per_group_min}$ animals per group.\n"
            f"     Total Minimum Sample Size: **{n_realized_min}** (yielding realized $E = {e_realized_min}$).\n"
            f"   - Maximum group size: $\\lceil {n_max} / {num_groups} \\rceil = {n_per_group_max}$ animals per group.\n"
            f"     Total Maximum Sample Size: **{n_realized_max}** (yielding realized $E = {e_realized_max}$).\n\n"
            f"**Recommended Study Allocation**: Recruit **{n_per_group_min} to {n_per_group_max}** subjects per group (Total $N$ = **{n_realized_min} to {n_realized_max}**).\n\n"
            f"**Clinical/Ethics Context**: This method is standard in preclinical research involving small animal models (Mead, 1988) where "
            f"standard deviations and clinical effect sizes are unknown, making standard power calculation impossible. It is widely "
            f"accepted by Institutional Animal Care and Use Committees (IACUC) for ethical validation of animal use numbers."
        )
        
        return {
            "method": "Resource Equation Method",
            "sample_size_min": n_realized_min,
            "sample_size_max": n_realized_max,
            "sample_size_per_group_range": f"{n_per_group_min}-{n_per_group_max}",
            "formula": formula_txt,
            "parameters": {"num_groups": num_groups, "num_blocks": num_blocks, "treatment_df": t_df, "block_df": b_df},
            "rationale": rationale
        }
