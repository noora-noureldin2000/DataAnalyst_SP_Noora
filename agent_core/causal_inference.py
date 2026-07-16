import numpy as np
import pandas as pd
import scipy.stats as stats
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.patches import FancyBboxPatch
import statsmodels.api as sm
import statsmodels.formula.api as smf
from statsmodels.stats.outliers_influence import variance_inflation_factor
from scipy.special import expit as plogis
from collections import deque
import warnings
import networkx as nx


class CausalDAG:
    """A simple DAG structure for causal inference reasoning.

    Stores nodes, directed edges, and optional coordinates for layout.
    Provides graph traversal methods and backdoor/frontdoor path finding.
    """

    def __init__(self):
        self.nodes = set()
        self.edges = {}  # parent -> children
        self.coords = {}  # node -> (x, y)
        self._parents_cache = {}
        self._children_cache = {}

    def add_node(self, name: str, x: float = 0, y: float = 0):
        """Add a node with optional coordinates for layout."""
        self.nodes.add(name)
        self.coords[name] = (x, y)
        if name not in self.edges:
            self.edges[name] = []
        if name not in self._parents_cache:
            self._parents_cache[name] = []
        if name not in self._children_cache:
            self._children_cache[name] = []

    def add_edge(self, cause: str, effect: str):
        """Add directed edge from cause to effect."""
        if cause not in self.nodes:
            self.add_node(cause)
        if effect not in self.nodes:
            self.add_node(effect)
        if cause not in self.edges:
            self.edges[cause] = []
        self.edges[cause].append(effect)
        if effect not in self._parents_cache:
            self._parents_cache[effect] = []
        self._parents_cache[effect].append(cause)
        if cause not in self._children_cache:
            self._children_cache[cause] = []
        self._children_cache[cause].append(effect)

    def get_parents(self, node: str) -> list[str]:
        """Return list of direct parents (causes) of node."""
        return list(self._parents_cache.get(node, []))

    def get_children(self, node: str) -> list[str]:
        """Return list of direct children (effects) of node."""
        return list(self._children_cache.get(node, []))

    def get_descendants(self, node: str) -> set[str]:
        """Return set of all descendants of node (children, grandchildren, etc)."""
        descendants = set()
        stack = list(self.get_children(node))
        visited = {node}
        while stack:
            current = stack.pop()
            if current in visited:
                continue
            visited.add(current)
            descendants.add(current)
            stack.extend(self.get_children(current))
        return descendants

    def get_ancestors(self, node: str) -> set[str]:
        """Return set of all ancestors of node (parents, grandparents, etc)."""
        ancestors = set()
        stack = list(self.get_parents(node))
        visited = {node}
        while stack:
            current = stack.pop()
            if current in visited:
                continue
            visited.add(current)
            ancestors.add(current)
            stack.extend(self.get_parents(current))
        return ancestors

    def is_collider(self, node: str) -> bool:
        """A node is a collider if it has at least 2 parents."""
        return len(self.get_parents(node)) >= 2

    def find_backdoor_paths(self, exposure: str, outcome: str) -> list[list[str]]:
        """Use BFS to find all paths from exposure to outcome going backwards (into parents).
        
        Backdoor paths are non-causal associations between exposure and outcome
        through common causes (confounders).
        """
        paths = []
        queue = deque()
        queue.append([exposure])
        while queue:
            path = queue.popleft()
            current = path[-1]
            if current == outcome:
                if len(path) > 2:
                    paths.append(path)
                continue
            if current in self._parents_cache:
                for parent in self._parents_cache[current]:
                    if parent not in path:
                        queue.append(path + [parent])
            if current in self._children_cache:
                for child in self._children_cache[current]:
                    if child not in path and child != outcome:
                        queue.append(path + [child])
        return paths

    def find_frontdoor_paths(self, exposure: str, outcome: str) -> list[list[str]]:
        """Use BFS to find all directed paths from exposure to outcome."""
        paths = []
        queue = deque()
        queue.append([exposure])
        while queue:
            path = queue.popleft()
            current = path[-1]
            if current == outcome:
                if len(path) > 2:
                    paths.append(path)
                continue
            if current in self._children_cache:
                for child in self._children_cache[current]:
                    if child not in path:
                        queue.append(path + [child])
        return paths

    def minimal_adjustment_set(self, exposure: str, outcome: str) -> set[str]:
        """Find minimal set of variables to adjust for using backdoor criterion.

        A variable should be adjusted for if it's a parent of both exposure
        and outcome (confounder). Exclude colliders and descendants of exposure.
        """
        descendants_of_exposure = self.get_descendants(exposure)
        ancestors_of_outcome = self.get_ancestors(outcome)
        common_ancestors = ancestors_of_outcome.intersection(
            set(self._parents_cache.get(exposure, []))
        )
        adjustment = set()
        for node in self.nodes:
            if node == exposure or node == outcome:
                continue
            if node in descendants_of_exposure:
                continue
            if self.is_collider(node):
                continue
            parents_node = set(self.get_parents(node))
            parents_exp = set(self.get_parents(exposure))
            parents_out = set(self.get_parents(outcome))
            if node in parents_exp and node in parents_out:
                adjustment.add(node)
            if parents_node.intersection(common_ancestors):
                for backdoor_path in self.find_backdoor_paths(exposure, outcome):
                    if node in backdoor_path:
                        if not self.is_collider(node):
                            adjustment.add(node)
        return adjustment


class ConfoundingDetector:
    """Detect and quantify confounding bias in observational data."""

    def detect_confounding(self, data: pd.DataFrame, treatment: str, outcome: str,
                           potential_confounders: list[str]) -> dict:
        """Detect and quantify confounding from a list of potential confounders.

        For each potential confounder C, checks if C is associated with both
        treatment and outcome. Fits unadjusted and adjusted models and computes
        the bias introduced by confounding.
        """
        results = {
            "suspected_confounders": [],
            "confounder_associations": {},
            "models": {},
            "bias_per_confounder": {},
            "overall_bias": None,
            "percent_bias": None,
            "recommendation": ""
        }
        suspected = []
        bias_per_conf = {}

        for conf in potential_confounders:
            assoc_with_treatment = False
            assoc_with_outcome = False
            assoc_details = {"treatment_association": None, "outcome_association": None}

            if data[conf].dtype in (np.int64, np.float64):
                treatment_values = data[treatment].values
                unique_t = np.unique(treatment_values)
                if len(unique_t) == 2:
                    group0 = data.loc[data[treatment] == unique_t[0], conf].dropna()
                    group1 = data.loc[data[treatment] == unique_t[1], conf].dropna()
                    if len(group0) > 1 and len(group1) > 1:
                        t_stat, p_val = stats.ttest_ind(group0, group1, equal_var=False)
                        assoc_with_treatment = p_val < 0.05
                        assoc_details["treatment_association"] = {
                            "test": "Welch t-test", "stat": t_stat, "p": p_val
                        }
                else:
                    corr, p_val = stats.pearsonr(
                        data[conf].fillna(data[conf].mean()),
                        data[treatment].fillna(data[treatment].mean())
                    )
                    assoc_with_treatment = p_val < 0.05
                    assoc_details["treatment_association"] = {
                        "test": "Pearson correlation", "r": corr, "p": p_val
                    }
            else:
                ct = pd.crosstab(data[conf], data[treatment])
                if ct.size > 0:
                    chi2, p_val, _, _ = stats.chi2_contingency(ct)
                    assoc_with_treatment = p_val < 0.05
                    assoc_details["treatment_association"] = {
                        "test": "Chi-square", "chi2": chi2, "p": p_val
                    }

            try:
                X_out = sm.add_constant(data[[conf]].fillna(data[conf].mean()))
                y_out = data[outcome].fillna(data[outcome].mean())
                model_out = sm.OLS(y_out, X_out).fit()
                assoc_with_outcome = model_out.pvalues[conf] < 0.05
                assoc_details["outcome_association"] = {
                    "coef": model_out.params[conf],
                    "p": model_out.pvalues[conf]
                }
            except Exception:
                assoc_details["outcome_association"] = None

            results["confounder_associations"][conf] = assoc_details

            if assoc_with_treatment and assoc_with_outcome:
                suspected.append(conf)
                try:
                    X_unadj = sm.add_constant(data[[treatment]].fillna(data[treatment].mean()))
                    y = data[outcome].fillna(data[outcome].mean())
                    model_unadj = sm.OLS(y, X_unadj).fit()

                    X_adj = sm.add_constant(
                        data[[treatment] + suspected].fillna(data[[treatment] + suspected].mean())
                    )
                    model_adj = sm.OLS(y, X_adj).fit()

                    coef_unadj = model_unadj.params[treatment]
                    coef_adj = model_adj.params[treatment]
                    bias_per_conf[conf] = {
                        "unadjusted_coef": coef_unadj,
                        "adjusted_coef": coef_adj,
                        "bias": coef_unadj - coef_adj,
                        "pct_change": abs((coef_unadj - coef_adj) / coef_adj) * 100 if coef_adj != 0 else np.nan
                    }
                except Exception:
                    bias_per_conf[conf] = None

        results["suspected_confounders"] = suspected

        if suspected:
            try:
                X_unadj = sm.add_constant(data[[treatment]].fillna(data[treatment].mean()))
                y = data[outcome].fillna(data[outcome].mean())
                model_unadj = sm.OLS(y, X_unadj).fit()

                X_adj = sm.add_constant(
                    data[[treatment] + suspected].fillna(data[[treatment] + suspected].mean())
                )
                model_adj = sm.OLS(y, X_adj).fit()

                coef_unadj = model_unadj.params[treatment]
                coef_adj = model_adj.params[treatment]

                results["models"]["unadjusted"] = {
                    "coef": model_unadj.params.to_dict(),
                    "pvalues": model_unadj.pvalues.to_dict(),
                    "ci": model_unadj.conf_int().to_dict()
                }
                results["models"]["adjusted"] = {
                    "coef": model_adj.params.to_dict(),
                    "pvalues": model_adj.pvalues.to_dict(),
                    "ci": model_adj.conf_int().to_dict()
                }
                results["overall_bias"] = coef_unadj - coef_adj
                results["percent_bias"] = abs(results["overall_bias"] / coef_adj) * 100 if coef_adj != 0 else np.nan

                if abs(results["percent_bias"]) > 10:
                    results["recommendation"] = (
                        f"Substantial confounding detected ({results['percent_bias']:.1f}% bias). "
                        f"Adjust for {', '.join(suspected)} in the final model."
                    )
                else:
                    results["recommendation"] = (
                        f"Minimal confounding bias ({results['percent_bias']:.1f}%). "
                        "Unadjusted estimate may be acceptable."
                    )
            except Exception:
                results["recommendation"] = "Could not fit adjustment models."

        results["bias_per_confounder"] = bias_per_conf
        return results

    def simulate_confounding(self, n: int = 200, treatment_effect: float = 5.0,
                             confounder_effect: float = 20.0) -> tuple[pd.DataFrame, dict]:
        """Simulate data with a confounder structure.

        Structure:
            Z ~ N(5, 1)
            W ~ Bernoulli(plogis(Z - 5))
            Y ~ N(2 + treatment_effect*W + confounder_effect*Z, 3)
        """
        np.random.seed(42)
        Z = np.random.normal(5, 1, n)
        prob_w = plogis(Z - 5)
        W = np.random.binomial(1, prob_w)
        y_mean = 2 + treatment_effect * W + confounder_effect * Z
        Y = np.random.normal(y_mean, 3, n)

        data = pd.DataFrame({
            "W": W,
            "Z": Z,
            "Y": Y
        })
        ground_truth = {
            "treatment_effect": treatment_effect,
            "confounder_effect": confounder_effect,
            "structure": "Z -> W, Z -> Y, W -> Y"
        }
        return data, ground_truth


class ColliderBiasDetector:
    """Detect and quantify collider/selection bias."""

    def detect_collider(self, data: pd.DataFrame, exposure: str, outcome: str,
                        conditioning_var: str) -> dict:
        """Detect if conditioning on a variable induces collider bias.

        Fits models with and without the conditioning variable and compares
        the exposure coefficient to detect collider-induced confounding.
        """
        result = {
            "unadjusted_coef": None,
            "adjusted_coef": None,
            "difference": None,
            "warning": "",
            "collider_status": "unknown"
        }
        try:
            y = data[outcome].fillna(data[outcome].mean())
            X_unadj = sm.add_constant(data[[exposure]].fillna(data[exposure].mean()))
            model_unadj = sm.OLS(y, X_unadj).fit()

            X_adj = sm.add_constant(
                data[[exposure, conditioning_var]].fillna(data[[exposure, conditioning_var]].mean())
            )
            model_adj = sm.OLS(y, X_adj).fit()

            result["unadjusted_coef"] = model_unadj.params[exposure]
            result["adjusted_coef"] = model_adj.params[exposure]
            result["difference"] = result["unadjusted_coef"] - result["adjusted_coef"]

            assoc_exposure = False
            assoc_outcome = False
            try:
                if data[conditioning_var].dtype in (np.int64, np.float64):
                    corr_e, p_e = stats.pearsonr(
                        data[conditioning_var].fillna(data[conditioning_var].mean()),
                        data[exposure].fillna(data[exposure].mean())
                    )
                    assoc_exposure = p_e < 0.05
                    corr_o, p_o = stats.pearsonr(
                        data[conditioning_var].fillna(data[conditioning_var].mean()),
                        data[outcome].fillna(data[outcome].mean())
                    )
                    assoc_outcome = p_o < 0.05
            except Exception:
                pass

            result["assoc_with_exposure"] = assoc_exposure
            result["assoc_with_outcome"] = assoc_outcome

            pct_change = abs(result["difference"] / result["unadjusted_coef"]) * 100 if result["unadjusted_coef"] != 0 else 0

            if assoc_exposure and assoc_outcome and pct_change > 10:
                result["collider_status"] = "suspected_collider"
                result["warning"] = (
                    f"Collider bias suspected: conditioning on '{conditioning_var}' "
                    f"changes the exposure coefficient by {pct_change:.1f}%. "
                    f"The variable is associated with both exposure and outcome."
                )
            elif pct_change > 10:
                result["collider_status"] = "possible_collider"
                result["warning"] = (
                    f"Coefficient changes by {pct_change:.1f}% when conditioning on "
                    f"'{conditioning_var}'. Check DAG structure for collider bias."
                )
            else:
                result["collider_status"] = "unlikely"
                result["warning"] = (
                    f"Collider bias unlikely: coefficient change of {pct_change:.1f}% "
                    f"when conditioning on '{conditioning_var}' is minimal."
                )
        except Exception as e:
            result["warning"] = f"Error in analysis: {str(e)}"

        return result

    def simulate_collider(self, n: int = 500, treatment_effect: float = 5.0) -> tuple[pd.DataFrame, dict]:
        """Simulate college admissions-like collider bias.

        Structure:
            intellectual_ability ~ N(0,1)
            athletic_ability ~ N(0,1)
            admission = (intellectual > 1) | (athletic > 1.5) [collider]

        In the full data, intellectual and athletic ability are uncorrelated,
        but in the selected sample (admitted students), they become negatively correlated.
        """
        np.random.seed(42)
        intellectual = np.random.normal(0, 1, n)
        athletic = np.random.normal(0, 1, n)
        admission = (intellectual > 1.0) | (athletic > 1.5)

        data = pd.DataFrame({
            "intellectual": intellectual,
            "athletic": athletic,
            "admission": admission.astype(int),
            "outcome": np.random.normal(50 + treatment_effect * intellectual, 5, n)
        })
        ground_truth = {
            "treatment_effect": treatment_effect,
            "structure": "intellectual -> admission, athletic -> admission (collider)",
            "full_correlation": np.corrcoef(intellectual, athletic)[0, 1],
            "selected_correlation": np.corrcoef(intellectual[admission], athletic[admission])[0, 1]
        }
        return data, ground_truth


class MediationAnalysis:
    """Decompose total effects into direct and indirect effects."""

    def analyze(self, data: pd.DataFrame, treatment: str, mediator: str,
                outcome: str) -> dict:
        """Perform mediation analysis using the Baron & Kenny approach.

        Fits three models and computes direct, indirect, and total effects
        with Sobel test for significance of the indirect effect.
        """
        result = {
            "total_effect": None,
            "direct_effect": None,
            "indirect_effect": None,
            "proportion_mediated": None,
            "sobel_z": None,
            "sobel_p": None,
            "models": {},
            "interpretation": ""
        }
        try:
            y = data[outcome].fillna(data[outcome].mean())
            t = data[treatment].fillna(data[treatment].mean())
            m = data[mediator].fillna(data[mediator].mean())

            X_total = sm.add_constant(t)
            model_total = sm.OLS(y, X_total).fit()
            total_effect = model_total.params[treatment]

            X_mediator = sm.add_constant(t)
            model_mediator = sm.OLS(m, X_mediator).fit()
            a_coef = model_mediator.params[treatment]
            a_se = model_mediator.bse[treatment]

            X_direct = sm.add_constant(pd.DataFrame({treatment: t, mediator: m}))
            model_direct = sm.OLS(y, X_direct).fit()
            direct_effect = model_direct.params[treatment]
            b_coef = model_direct.params[mediator]
            b_se = model_direct.bse[mediator]

            indirect_ab = a_coef * b_coef
            indirect_diff = total_effect - direct_effect
            indirect_effect = (indirect_ab + indirect_diff) / 2

            sobel_z = (a_coef * b_coef) / np.sqrt(b_coef**2 * a_se**2 + a_coef**2 * b_se**2 + 1e-10)
            sobel_p = 2 * (1 - stats.norm.cdf(abs(sobel_z)))

            proportion_mediated = (indirect_effect / total_effect) * 100 if total_effect != 0 else np.nan

            result["total_effect"] = total_effect
            result["direct_effect"] = direct_effect
            result["indirect_effect"] = indirect_effect
            result["proportion_mediated"] = proportion_mediated
            result["sobel_z"] = sobel_z
            result["sobel_p"] = sobel_p

            result["models"]["total_effect"] = {
                "coef": model_total.params.to_dict(),
                "pvalues": model_total.pvalues.to_dict(),
                "ci": model_total.conf_int().to_dict()
            }
            result["models"]["mediator"] = {
                "coef": model_mediator.params.to_dict(),
                "pvalues": model_mediator.pvalues.to_dict(),
                "ci": model_mediator.conf_int().to_dict()
            }
            result["models"]["direct_effect"] = {
                "coef": model_direct.params.to_dict(),
                "pvalues": model_direct.pvalues.to_dict(),
                "ci": model_direct.conf_int().to_dict()
            }

            if sobel_p < 0.05:
                sig_str = "significant"
            else:
                sig_str = "not significant"
            result["interpretation"] = (
                f"Total effect = {total_effect:.3f}, Direct effect = {direct_effect:.3f}, "
                f"Indirect effect = {indirect_effect:.3f}. "
                f"Proportion mediated = {proportion_mediated:.1f}%. "
                f"Sobel test: z = {sobel_z:.3f}, p = {sobel_p:.4f} ({sig_str})."
            )
        except Exception as e:
            result["interpretation"] = f"Error in mediation analysis: {str(e)}"

        return result

    def simulate_mediation(self, n: int = 200, direct_effect: float = 5.0,
                           mediator_effect: float = 2.0,
                           mediator_outcome_effect: float = 3.0) -> tuple[pd.DataFrame, dict]:
        """Simulate data with mediation structure.

        Structure:
            W ~ Bernoulli(0.5)
            M ~ N(direct_effect*W, 1)
            Y ~ N(2 + treatment_effect*W + mediator_outcome_effect*M, 3)
        where treatment_effect = direct_effect and
        a*b = mediator_effect * mediator_outcome_effect is the indirect effect.
        """
        np.random.seed(42)
        W = np.random.binomial(1, 0.5, n)
        M = np.random.normal(mediator_effect * W, 1, n)
        y_mean = 2 + direct_effect * W + mediator_outcome_effect * M
        Y = np.random.normal(y_mean, 3, n)

        data = pd.DataFrame({
            "W": W,
            "M": M,
            "Y": Y
        })
        indirect_effect = mediator_effect * mediator_outcome_effect
        total_effect = direct_effect + indirect_effect
        ground_truth = {
            "direct_effect": direct_effect,
            "mediator_effect_a": mediator_effect,
            "mediator_effect_b": mediator_outcome_effect,
            "indirect_effect": indirect_effect,
            "total_effect": total_effect,
            "structure": "W -> M -> Y, W -> Y"
        }
        return data, ground_truth


class SimpsonParadoxDetector:
    """Detect Simpson's Paradox: when a trend appears in several groups but
    disappears or reverses when groups are combined."""

    def detect(self, data: pd.DataFrame, x_col: str, y_col: str,
               group_col: str) -> dict:
        """Detect Simpson's Paradox in grouped data.

        Fits overall regression and within-group regressions, then compares
        slopes to detect paradox.
        """
        result = {
            "overall_slope": None,
            "overall_p": None,
            "within_slopes": {},
            "average_within_slope": None,
            "paradox_detected": False,
            "severity": None,
            "explanation": ""
        }
        try:
            X_overall = sm.add_constant(data[[x_col]].fillna(data[x_col].mean()))
            y = data[y_col].fillna(data[y_col].mean())
            model_overall = sm.OLS(y, X_overall).fit()
            overall_slope = model_overall.params[x_col]
            overall_p = model_overall.pvalues[x_col]
            result["overall_slope"] = overall_slope
            result["overall_p"] = overall_p

            within_slopes = {}
            groups = data[group_col].unique()
            for group in groups:
                subset = data[data[group_col] == group]
                if len(subset) < 3:
                    continue
                try:
                    X_sub = sm.add_constant(subset[[x_col]].fillna(subset[x_col].mean()))
                    y_sub = subset[y_col].fillna(subset[y_col].mean())
                    model_sub = sm.OLS(y_sub, X_sub).fit()
                    within_slopes[str(group)] = {
                        "slope": model_sub.params[x_col],
                        "p": model_sub.pvalues[x_col],
                        "intercept": model_sub.params.iloc[0],
                        "n": len(subset)
                    }
                except Exception:
                    continue

            result["within_slopes"] = within_slopes

            if within_slopes:
                avg_within_slope = np.mean([v["slope"] for v in within_slopes.values()])
                result["average_within_slope"] = avg_within_slope
                result["severity"] = overall_slope - avg_within_slope

                sign_overall = np.sign(overall_slope)
                signs_within = [np.sign(v["slope"]) for v in within_slopes.values()]
                all_opposite = all(s != sign_overall for s in signs_within if s != 0)

                if all_opposite and len(within_slopes) >= 2:
                    result["paradox_detected"] = True
                    result["explanation"] = (
                        f"Simpson's Paradox detected! Overall slope = {overall_slope:.3f} (p={overall_p:.4f}), "
                        f"but within-group slopes average = {avg_within_slope:.3f}. "
                        f"The direction of association reverses when data are aggregated."
                    )
                elif abs(result["severity"]) > 0.5:
                    result["paradox_detected"] = True
                    result["explanation"] = (
                        f"Potential Simpson's Paradox: overall slope ({overall_slope:.3f}) differs "
                        f"substantially from average within-group slope ({avg_within_slope:.3f}). "
                        f"Severity = {result['severity']:.3f}."
                    )
                else:
                    result["explanation"] = (
                        f"No Simpson's Paradox. Overall slope ({overall_slope:.3f}) is similar to "
                        f"average within-group slope ({avg_within_slope:.3f})."
                    )
            else:
                result["explanation"] = "Could not compute within-group slopes (insufficient data)."
        except Exception as e:
            result["explanation"] = f"Error in detection: {str(e)}"

        return result

    def simulate(self, n: int = 400, n_groups: int = 5) -> tuple[pd.DataFrame, dict]:
        """Simulate data exhibiting Simpson's Paradox.

        Creates groups with different means. Within each group, y has a negative
        slope with x, but overall the slope is positive due to group mean differences.
        """
        np.random.seed(42)
        n_per_group = n // n_groups
        all_data = []
        group_centers_x = np.linspace(0, 20, n_groups)
        group_centers_y = np.linspace(5, 25, n_groups)

        for i in range(n_groups):
            x = np.random.normal(group_centers_x[i], 1.5, n_per_group)
            y = group_centers_y[i] + (-1.0) * (x - group_centers_x[i]) + np.random.normal(0, 2, n_per_group)
            grp = pd.DataFrame({"x": x, "y": y, "group": f"Group {i+1}"})
            all_data.append(grp)

        data = pd.concat(all_data, ignore_index=True)
        ground_truth = {
            "within_slope": -1.0,
            "overall_slope": None,
            "n_groups": n_groups,
            "n_per_group": n_per_group,
            "description": "Within each group, y decreases with x (slope=-1). "
                           "Overall, y increases with x due to group mean differences."
        }
        return data, ground_truth


class SimpsonsParadoxVisualizer:
    """Visualize Simpson's Paradox and other causal inference concepts."""

    def plot_simpsons_paradox(self, data: pd.DataFrame, x_col: str, y_col: str,
                              group_col: str, output_path: str = None) -> plt.Figure:
        """Two-panel figure showing Simpson's Paradox.

        Left: Overall regression with all points in one color.
        Right: Grouped regression with colored groups and separate lines.
        """
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

        colors = plt.cm.Set1(np.linspace(0, 1, data[group_col].nunique()))
        color_map = {g: colors[i] for i, g in enumerate(data[group_col].unique())}

        ax1.scatter(data[x_col], data[y_col], alpha=0.4, color="gray", s=30)
        X_overall = sm.add_constant(data[x_col])
        model = sm.OLS(data[y_col], X_overall).fit()
        x_range = np.linspace(data[x_col].min(), data[x_col].max(), 100)
        y_pred = model.params.iloc[0] + model.params.iloc[1] * x_range
        ax1.plot(x_range, y_pred, "r-", linewidth=2.5,
                 label=f"Overall: slope={model.params.iloc[1]:.2f}")
        ax1.set_xlabel(x_col)
        ax1.set_ylabel(y_col)
        ax1.set_title("Overall Regression")
        ax1.legend()
        ax1.grid(True, alpha=0.3)

        for group in data[group_col].unique():
            subset = data[data[group_col] == group]
            ax2.scatter(subset[x_col], subset[y_col], alpha=0.5,
                        color=color_map[group], s=30, label=str(group))
            if len(subset) >= 3:
                X_sub = sm.add_constant(subset[x_col])
                try:
                    model_sub = sm.OLS(subset[y_col], X_sub).fit()
                    y_pred_sub = model_sub.params.iloc[0] + model_sub.params.iloc[1] * x_range
                    ax2.plot(x_range, y_pred_sub, "--", linewidth=1.5, color=color_map[group])
                except Exception:
                    pass

        ax2.set_xlabel(x_col)
        ax2.set_ylabel(y_col)
        ax2.set_title("Grouped Regression")
        ax2.legend(loc="best", fontsize=8, ncol=2)
        ax2.grid(True, alpha=0.3)

        plt.tight_layout()
        if output_path:
            fig.savefig(output_path, dpi=150, bbox_inches="tight")
        return fig

    def plot_causal_dag(self, dag: CausalDAG, title: str = "Causal DAG",
                        output_path: str = None) -> plt.Figure:
        """Render the DAG using networkx and matplotlib.

        Nodes as circles with labels, directed edges as arrows.
        Color coding: exposure=blue, outcome=red, confounder=orange,
        collider=purple, mediator=green.
        """
        G = nx.DiGraph()
        for node in dag.nodes:
            G.add_node(node)
        for parent in dag.edges:
            for child in dag.edges[parent]:
                G.add_edge(parent, child)

        fig, ax = plt.subplots(1, 1, figsize=(10, 8))

        pos = {}
        for node in dag.nodes:
            if node in dag.coords:
                pos[node] = dag.coords[node]

        if not pos or all(v == (0, 0) for v in pos.values()):
            pos = nx.spring_layout(G, seed=42, k=2.0, iterations=50)

        node_colors = []
        node_sizes = []
        for node in G.nodes():
            if dag.is_collider(node) and len(dag.get_parents(node)) >= 2:
                node_colors.append("purple")
            elif node == dag.coords.get(node) and False:
                node_colors.append("blue")
            else:
                node_colors.append("lightblue")
            node_sizes.append(2000)

        nx.draw_networkx_nodes(G, pos, ax=ax, node_size=node_sizes,
                               node_color=node_colors, edgecolors="black",
                               linewidths=2, alpha=0.9)
        nx.draw_networkx_labels(G, pos, ax=ax, font_size=11, font_weight="bold")
        nx.draw_networkx_edges(G, pos, ax=ax, arrows=True, arrowsize=20,
                               arrowstyle="-|>", width=2, alpha=0.7,
                               connectionstyle="arc3,rad=0.1")

        legend_handles = [
            plt.Line2D([0], [0], marker="o", color="w", markerfacecolor="lightblue",
                       markersize=10, label="Node"),
            plt.Line2D([0], [0], marker="o", color="w", markerfacecolor="purple",
                       markersize=10, label="Collider"),
            plt.Line2D([0], [0], color="black", linewidth=2, label="Directed edge"),
        ]
        ax.legend(handles=legend_handles, loc="upper right", fontsize=9)
        ax.set_title(title, fontsize=14, fontweight="bold")
        ax.axis("off")

        if output_path:
            fig.savefig(output_path, dpi=150, bbox_inches="tight")
        return fig

    def plot_collider_bias(self, data: pd.DataFrame, x_col: str, y_col: str,
                           selection_col: str, output_path: str = None) -> plt.Figure:
        """Two-panel figure showing collider bias.

        Left: All data with overall regression line.
        Right: Selected subset (where selection_col is True) with regression line,
        showing the induced correlation.
        """
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

        x_all = data[x_col].fillna(data[x_col].mean())
        y_all = data[y_col].fillna(data[y_col].mean())
        X_all = sm.add_constant(x_all)
        model_all = sm.OLS(y_all, X_all).fit()

        ax1.scatter(x_all, y_all, alpha=0.3, color="gray", s=25)
        x_range = np.linspace(x_all.min(), x_all.max(), 100)
        y_pred_all = model_all.params.iloc[0] + model_all.params.iloc[1] * x_range
        ax1.plot(x_range, y_pred_all, "b-", linewidth=2.5,
                 label=f"All data: slope={model_all.params.iloc[1]:.3f}")
        ax1.set_xlabel(x_col)
        ax1.set_ylabel(y_col)
        ax1.set_title("Full Data (No Selection)")
        ax1.legend()
        ax1.grid(True, alpha=0.3)

        sel_mask = data[selection_col].astype(bool)
        x_sel = data.loc[sel_mask, x_col].fillna(data[x_col].mean())
        y_sel = data.loc[sel_mask, y_col].fillna(data[y_col].mean())

        if len(x_sel) >= 3:
            X_sel = sm.add_constant(x_sel)
            model_sel = sm.OLS(y_sel, X_sel).fit()
            ax2.scatter(x_sel, y_sel, alpha=0.4, color="red", s=25, label="Selected")
            ax2.scatter(x_all[~sel_mask], y_all[~sel_mask], alpha=0.15,
                        color="gray", s=15, label="Not selected")
            y_pred_sel = model_sel.params.iloc[0] + model_sel.params.iloc[1] * x_range
            ax2.plot(x_range, y_pred_sel, "r-", linewidth=2.5,
                     label=f"Selected: slope={model_sel.params.iloc[1]:.3f}")
            ax2.plot(x_range, y_pred_all, "b--", linewidth=1.5, alpha=0.5,
                     label="Original slope (ref)")
        else:
            ax2.scatter(x_all, y_all, alpha=0.3, color="gray", s=25)

        ax2.set_xlabel(x_col)
        ax2.set_ylabel(y_col)
        ax2.set_title("Selected Subset (Collider Bias)")
        ax2.legend(loc="best", fontsize=8)
        ax2.grid(True, alpha=0.3)

        plt.tight_layout()
        if output_path:
            fig.savefig(output_path, dpi=150, bbox_inches="tight")
        return fig


class DAGDataSimulator:
    """Simulate data from known DAG structures for testing methods."""

    def simulate_confounding_dag(self, n: int = 200, treatment_effect: float = 5) -> pd.DataFrame:
        """Simulate confounding DAG structure: Z -> W, Z -> Y, W -> Y.

        Z is a confounder affecting both treatment (W) and outcome (Y).
        """
        np.random.seed(42)
        Z = np.random.normal(5, 1, n)
        prob_w = plogis(Z - 5)
        W = np.random.binomial(1, prob_w)
        y_mean = 2 + treatment_effect * W + 20 * Z
        Y = np.random.normal(y_mean, 3, n)
        return pd.DataFrame({"W": W, "Z": Z, "Y": Y})

    def simulate_collider_dag(self, n: int = 200, treatment_effect: float = 5) -> pd.DataFrame:
        """Simulate collider DAG structure: W -> Z, Y -> Z (Z is collider).

        W (exposure) and Y (outcome) both cause Z. Conditioning on Z induces
        a non-causal association between W and Y.
        """
        np.random.seed(42)
        W = np.random.normal(0, 1, n)
        Y = np.random.normal(2 + treatment_effect * W, 2, n)
        Z = W + Y + np.random.normal(0, 1, n)
        return pd.DataFrame({"W": W, "Z": Z, "Y": Y})

    def simulate_mediator_dag(self, n: int = 200, direct_effect: float = 5,
                              indirect_effect: float = 6) -> pd.DataFrame:
        """Simulate mediator DAG structure: W -> M -> Y and W -> Y.

        M mediates the relationship between W (exposure) and Y (outcome).
        indirect_effect represents the product a*b (mediator path).
        """
        np.random.seed(42)
        W = np.random.binomial(1, 0.5, n)
        a_path = indirect_effect / 2
        b_path = 2
        M = np.random.normal(a_path * W, 1, n)
        y_mean = 2 + direct_effect * W + b_path * M
        Y = np.random.normal(y_mean, 3, n)
        ground_truth = {
            "direct_effect": direct_effect,
            "indirect_effect": a_path * b_path,
            "a_path": a_path,
            "b_path": b_path
        }
        return pd.DataFrame({"W": W, "M": M, "Y": Y, **ground_truth})
