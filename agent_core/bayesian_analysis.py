"""
Bayesian analysis tools for the Data Analyst Specialist.
Uses analytical formulas from scipy.stats — no PyMC or Stan.
Provides Bayes Factors, conjugate estimation, plotting, model comparison, and APA reporting.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import integrate
from scipy.special import gammaln, beta as beta_func
from scipy.stats import (
    t as t_dist,
    nct as noncentral_t,
    norm,
    beta as beta_dist,
    cauchy,
)
from scipy.optimize import minimize_scalar
from typing import Optional, Union, Dict, Any, List, Tuple
import warnings


def _interpret_bf(log_bf: float) -> str:
    if abs(log_bf) < 1:
        return "Barely worth mentioning"
    elif abs(log_bf) < 3:
        return "Positive"
    elif abs(log_bf) < 5:
        return "Strong"
    return "Very strong"


def _jzs_bf(t_stat: float, n_eff: float, df: float, r: float = 0.707) -> float:
    r"""JZS Bayes Factor via the :math:`\theta`-transformation.

    Uses :math:`\delta = r \tan(\theta)` to map the Cauchy prior on effect size
    :math:`\delta \sim \text{Cauchy}(0, r)` to a uniform prior on
    :math:`\theta`, yielding a numerically stable 1-d integral.

    .. math::

        \text{BF}_{10} = \frac{\int_{-\pi/2}^{\pi/2}
        f(t \mid \text{df}, \sqrt{n_{\text{eff}}}\,r\tan(\theta)) \,
        \frac{1}{\pi} \, d\theta}{f(t \mid \text{df}, 0)}

    Parameters
    ----------
    t_stat : float
        Observed t-statistic.
    n_eff : float
        Effective sample size (n for one-sample/paired;
        n1*n2/(n1+n2) for two-sample).
    df : float
        Degrees of freedom.
    r : float, optional
        Cauchy prior scale (default 0.707).

    Returns
    -------
    float
        BF10 (evidence for the alternative).
    """
    def integrand(theta):
        delta = r * np.tan(theta)
        ncp = np.sqrt(n_eff) * delta
        return noncentral_t.pdf(t_stat, df, ncp) / np.pi

    numerator, _ = integrate.quad(integrand, -np.pi / 2, np.pi / 2,
                                  limit=200)
    denominator = t_dist.pdf(t_stat, df)
    if denominator <= 0:
        return np.inf
    return numerator / denominator


# ---------------------------------------------------------------------------
# BayesFactor
# ---------------------------------------------------------------------------
class BayesFactor:
    """Bayes Factor computation for common null-hypothesis significance tests.

    All methods return a dictionary with keys:
        bf10, bf01, log_bf, interpretation
    """

    def __init__(self):
        pass

    def ttest(self, x, y=None, paired=False, prior_cauchy_r=0.707):
        """JZS Bayes factor for a one-sample, paired, or two-sample t-test.

        Parameters
        ----------
        x : array-like
            Data for one-sample, or first group.
        y : array-like or None
            Second group (ignored for one-sample).
        paired : bool
            If True, treat as paired (compute differences).
        prior_cauchy_r : float
            Cauchy prior scale on effect size (default 0.707).

        Returns
        -------
        dict
            bf10, bf01, log_bf, interpretation, t_stat, df, n_eff
        """
        x = np.asarray(x, dtype=float)
        if y is not None:
            y = np.asarray(y, dtype=float)

        if y is None or paired:
            if paired and y is not None:
                data = x - y
            else:
                data = x
            n = len(data)
            df = n - 1
            t_stat = np.mean(data) / (np.std(data, ddof=1) / np.sqrt(n))
            n_eff = n
        else:
            n1, n2 = len(x), len(y)
            df = n1 + n2 - 2
            sp = np.sqrt(((n1 - 1) * np.var(x, ddof=1) +
                          (n2 - 1) * np.var(y, ddof=1)) / df)
            t_stat = (np.mean(x) - np.mean(y)) / (sp * np.sqrt(1 / n1 + 1 / n2))
            n_eff = n1 * n2 / (n1 + n2)

        bf10 = _jzs_bf(t_stat, n_eff, df, prior_cauchy_r)
        log_bf = np.log(bf10)
        return {
            "bf10": bf10,
            "bf01": 1.0 / bf10,
            "log_bf": log_bf,
            "interpretation": _interpret_bf(log_bf),
            "t_stat": t_stat,
            "df": df,
            "n_eff": n_eff,
            "prior_cauchy_r": prior_cauchy_r,
        }

    def anova_bf(self, data, groups):
        r"""Approximate BF10 for a one-way ANOVA using the BIC approximation.

        .. math::

            \log(\text{BF}_{10}) \approx \frac{\text{BIC}_{H_0} -
            \text{BIC}_{H_1}}{2}

        Parameters
        ----------
        data : array-like
            Response values.
        groups : array-like
            Group labels (same length as data).

        Returns
        -------
        dict
            bf10, bf01, log_bf, interpretation, bic_h0, bic_h1, rss_h0,
            rss_h1, k (number of groups), n
        """
        data = np.asarray(data, dtype=float)
        groups = np.asarray(groups)
        n = len(data)
        unique_groups = np.unique(groups)
        k = len(unique_groups)

        grand_mean = np.mean(data)
        rss_h0 = np.sum((data - grand_mean) ** 2)

        rss_h1 = 0.0
        for g in unique_groups:
            mask = groups == g
            g_mean = np.mean(data[mask])
            rss_h1 += np.sum((data[mask] - g_mean) ** 2)

        bic_h0 = n * np.log(rss_h0 / n) + 1.0 * np.log(n)
        bic_h1 = n * np.log(rss_h1 / n) + k * np.log(n)
        log_bf = (bic_h0 - bic_h1) / 2.0
        bf10 = np.exp(log_bf)
        return {
            "bf10": bf10,
            "bf01": 1.0 / bf10,
            "log_bf": log_bf,
            "interpretation": _interpret_bf(log_bf),
            "bic_h0": bic_h0,
            "bic_h1": bic_h1,
            "rss_h0": rss_h0,
            "rss_h1": rss_h1,
            "k": k,
            "n": n,
        }

    def correlation(self, x, y, prior_beta=1.0):
        r"""BF10 for Pearson correlation (Wetzels & Wagenmakers, 2012).

        Uses Fisher z-transformation and numerical integration over a
        Beta(prior_beta, prior_beta) prior on :math:`\rho` stretched to
        :math:`[-1, 1]`.

        Parameters
        ----------
        x : array-like
            First variable.
        y : array-like
            Second variable.
        prior_beta : float
            Beta shape parameter for prior on :math:`\rho`
            (default 1.0 = uniform).

        Returns
        -------
        dict
            bf10, bf01, log_bf, interpretation, r (observed), n
        """
        x = np.asarray(x, dtype=float)
        y = np.asarray(y, dtype=float)
        n = len(x)
        r_stat = np.corrcoef(x, y)[0, 1]

        z = np.arctanh(r_stat)
        z_se = 1.0 / np.sqrt(n - 3)

        def integrand(rho):
            if abs(rho) >= 1.0:
                return 0.0
            log_lik = norm.logpdf(z, loc=np.arctanh(rho), scale=z_se)
            log_prior = ((prior_beta - 1) * np.log(1 - rho ** 2)
                         - (2 * prior_beta - 1) * np.log(2)
                         - np.log(beta_func(prior_beta, prior_beta)))
            if np.isinf(log_prior):
                return 0.0
            return np.exp(log_lik + log_prior)

        numerator, _ = integrate.quad(integrand, -1, 1, limit=200,
                                      epsabs=1e-12, epsrel=1e-8)
        # Use log-space to avoid underflow/overflow
        log_numer = np.log(max(numerator, 1e-300))
        log_denom = norm.logpdf(z, loc=0.0, scale=z_se)
        log_bf10 = log_numer - log_denom
        log_bf10 = max(min(log_bf10, 100), -100)
        bf10 = np.exp(log_bf10)

        log_bf = np.log(bf10)
        return {
            "bf10": bf10,
            "bf01": 1.0 / bf10,
            "log_bf": log_bf,
            "interpretation": _interpret_bf(log_bf),
            "r": r_stat,
            "n": n,
            "prior_beta": prior_beta,
        }

    def contingency(self, table, prior_concentration=1.0):
        r"""Bayes factor for contingency tables (Gunel & Dickey, 1974).

        Computes :math:`\text{BF}_{10}` for association vs independence
        using the joint multinomial sampling plan with a Dirichlet prior.

        Parameters
        ----------
        table : array-like (R x C)
            Contingency table of counts.
        prior_concentration : float
            Dirichlet concentration parameter (default 1.0).

        Returns
        -------
        dict
            bf10, bf01, log_bf, interpretation, R, C, N
        """
        table = np.asarray(table, dtype=float)
        R, C = table.shape
        N = table.sum()
        row_sums = table.sum(axis=1)
        col_sums = table.sum(axis=0)
        alpha = prior_concentration

        # Under H1 (saturated): Dir(alpha) prior on all R*C cells
        log_p_H1 = (np.sum(gammaln(table + alpha))
                    + gammaln(R * C * alpha)
                    - gammaln(N + R * C * alpha)
                    - R * C * gammaln(alpha))

        # Under H0 (independence): row ~ Dir(alpha), col ~ Dir(alpha)
        log_p_H0 = (np.sum(gammaln(row_sums + alpha))
                    + gammaln(R * alpha)
                    - gammaln(N + R * alpha)
                    - R * gammaln(alpha)
                    + np.sum(gammaln(col_sums + alpha))
                    + gammaln(C * alpha)
                    - gammaln(N + C * alpha)
                    - C * gammaln(alpha))

        log_bf10 = log_p_H1 - log_p_H0
        bf10 = np.exp(log_bf10)
        return {
            "bf10": bf10,
            "bf01": 1.0 / bf10,
            "log_bf": log_bf10,
            "interpretation": _interpret_bf(log_bf10),
            "R": R,
            "C": C,
            "N": N,
            "prior_concentration": alpha,
        }


# ---------------------------------------------------------------------------
# BayesianEstimation
# ---------------------------------------------------------------------------
class BayesianEstimation:
    """Conjugate Bayesian estimation for common parameters.

    All methods return posterior summaries with credible intervals.
    """

    def __init__(self):
        pass

    def mean_ci(self, x, prior_mean=0, prior_sd=1, credible_mass=0.95):
        r"""Bayesian credible interval for the mean (normal-normal conjugate).

        :math:`\mu \mid \text{data} \sim \mathcal{N}(\mu_n, \sigma_n^2)`

        Parameters
        ----------
        x : array-like
            Data.
        prior_mean : float
            Prior mean of :math:`\mu` (default 0).
        prior_sd : float
            Prior standard deviation of :math:`\mu` (default 1).
        credible_mass : float
            Desired credible level (default 0.95).

        Returns
        -------
        dict
            posterior_mean, posterior_sd, credible_interval, prior_mean,
            prior_sd, n
        """
        x = np.asarray(x, dtype=float)
        n = len(x)
        x_bar = np.mean(x)
        s2 = np.var(x, ddof=1)

        prior_prec = 1.0 / (prior_sd ** 2)
        data_prec = n / s2
        post_prec = prior_prec + data_prec
        post_mean = (prior_mean * prior_prec + x_bar * data_prec) / post_prec
        post_sd = np.sqrt(1.0 / post_prec)

        alpha = 1.0 - credible_mass
        ci = norm.ppf([alpha / 2, 1.0 - alpha / 2], loc=post_mean, scale=post_sd)
        return {
            "posterior_mean": post_mean,
            "posterior_sd": post_sd,
            "credible_interval": tuple(ci),
            "credible_mass": credible_mass,
            "prior_mean": prior_mean,
            "prior_sd": prior_sd,
            "n": n,
        }

    def proportion(self, k, n, prior_alpha=1, prior_beta=1, credible_mass=0.95):
        r"""Bayesian credible interval for a proportion (beta-binomial conjugate).

        :math:`\theta \mid k, n \sim \text{Beta}(\alpha_0 + k, \beta_0 + n - k)`

        The credible interval is a true HPD interval (shortest interval).

        Parameters
        ----------
        k : int
            Number of successes.
        n : int
            Number of trials.
        prior_alpha : float
            Beta prior shape1 (default 1).
        prior_beta : float
            Beta prior shape2 (default 1).
        credible_mass : float
            Desired credible level (default 0.95).

        Returns
        -------
        dict
            posterior_alpha, posterior_beta, posterior_mean, posterior_sd,
            credible_interval, prior_alpha, prior_beta
        """
        post_alpha = prior_alpha + k
        post_beta = prior_beta + n - k

        post_mean = post_alpha / (post_alpha + post_beta)
        post_var = (post_alpha * post_beta
                    / ((post_alpha + post_beta) ** 2
                       * (post_alpha + post_beta + 1)))
        post_sd = np.sqrt(post_var)

        ci = _hpd_beta(post_alpha, post_beta, credible_mass)
        return {
            "posterior_alpha": post_alpha,
            "posterior_beta": post_beta,
            "posterior_mean": post_mean,
            "posterior_sd": post_sd,
            "credible_interval": ci,
            "credible_mass": credible_mass,
            "prior_alpha": prior_alpha,
            "prior_beta": prior_beta,
            "n": n,
            "k": k,
        }

    def correlation(self, r, n, prior_kappa=1, credible_mass=0.95):
        r"""Bayesian credible interval for a Pearson correlation.

        Uses a Beta(prior_kappa, prior_kappa) prior on
        :math:`(\rho + 1)/2` (uniform on :math:`\rho` when kappa=1).

        The posterior is approximated as:

        .. math::

            \frac{\rho + 1}{2} \mid r, n \sim
            \text{Beta}(\kappa + \frac{n(r+1)}{2},
            \kappa + \frac{n(1-r)}{2})

        Parameters
        ----------
        r : float
            Observed Pearson correlation.
        n : int
            Sample size.
        prior_kappa : float
            Beta shape for prior on :math:`(\rho+1)/2` (default 1).
        credible_mass : float
            Desired credible level (default 0.95).

        Returns
        -------
        dict
            posterior_mode, credible_interval, r, n, prior_kappa
        """
        a = prior_kappa + n * (r + 1) / 2.0
        b = prior_kappa + n * (1 - r) / 2.0

        # Mode of the transformed Beta
        if a > 1 and b > 1:
            mode_transformed = (a - 1) / (a + b - 2)
        else:
            mode_transformed = a / (a + b)
        posterior_mode = 2.0 * mode_transformed - 1.0

        alpha_p = 1.0 - credible_mass
        lower_t = beta_dist.ppf(alpha_p / 2, a, b)
        upper_t = beta_dist.ppf(1.0 - alpha_p / 2, a, b)
        ci = (2.0 * lower_t - 1.0, 2.0 * upper_t - 1.0)

        return {
            "posterior_mode": posterior_mode,
            "credible_interval": ci,
            "credible_mass": credible_mass,
            "r": r,
            "n": n,
            "prior_kappa": prior_kappa,
        }

    def effect_size(self, t, df, prior_cauchy_r=0.707, credible_mass=0.95):
        r"""Bayesian estimation of Cohen's :math:`\delta` (shifted-t posterior).

        Assumes a one-sample design (:math:`n = df + 1`). The posterior is
        computed via numerical integration over a Cauchy(0, r) prior.

        Parameters
        ----------
        t : float
            Observed t-statistic.
        df : float
            Degrees of freedom.
        prior_cauchy_r : float
            Cauchy prior scale (default 0.707).
        credible_mass : float
            Desired credible level (default 0.95).

        Returns
        -------
        dict
            posterior_mean, posterior_sd, credible_interval, credible_mass,
            prior_cauchy_r, t, df
        """
        n = df + 1.0
        n_eff = n
        delta_max = 5.0
        n_points = 20001
        delta = np.linspace(-delta_max, delta_max, n_points)

        ncp = np.sqrt(n_eff) * delta
        likelihood = noncentral_t.pdf(t, df, ncp)
        prior = cauchy.pdf(delta, 0, prior_cauchy_r)
        posterior = likelihood * prior
        post_norm = np.trapezoid(posterior, delta)
        if post_norm <= 0:
            posterior = np.ones_like(delta) / (2 * delta_max)
            post_norm = np.trapezoid(posterior, delta)
        posterior /= post_norm

        post_mean = np.trapezoid(delta * posterior, delta)
        post_var = np.trapezoid((delta - post_mean) ** 2 * posterior, delta)
        post_sd = np.sqrt(post_var)

        # CDF via cumulative trapezoidal integration
        cdf = np.zeros_like(delta)
        cdf[1:] = np.cumsum(0.5 * (posterior[1:] + posterior[:-1])
                            * np.diff(delta))
        cdf /= cdf[-1]

        alpha_p = 1.0 - credible_mass
        lower = np.interp(alpha_p / 2, cdf, delta)
        upper = np.interp(1.0 - alpha_p / 2, cdf, delta)

        return {
            "posterior_mean": post_mean,
            "posterior_sd": post_sd,
            "credible_interval": (float(lower), float(upper)),
            "credible_mass": credible_mass,
            "prior_cauchy_r": prior_cauchy_r,
            "t": t,
            "df": df,
        }


# ---------------------------------------------------------------------------
# Helper: HPD interval for Beta distribution
# ---------------------------------------------------------------------------
def _hpd_beta(a, b, mass=0.95):
    """Highest Posterior Density interval for a Beta(a, b) distribution."""
    if a == b:
        lower = beta_dist.ppf((1 - mass) / 2, a, b)
        upper = beta_dist.ppf((1 + mass) / 2, a, b)
        return (float(lower), float(upper))

    def interval_width(p):
        return beta_dist.ppf(p + mass, a, b) - beta_dist.ppf(p, a, b)

    res = minimize_scalar(interval_width, bounds=(0, 1 - mass), method="bounded")
    p_opt = res.x
    lower = beta_dist.ppf(p_opt, a, b)
    upper = beta_dist.ppf(p_opt + mass, a, b)
    return (float(lower), float(upper))


# ---------------------------------------------------------------------------
# BayesianPlot
# ---------------------------------------------------------------------------
class BayesianPlot:
    """Visualisation tools for Bayesian analysis."""

    def __init__(self):
        pass

    def posterior_plot(self, posterior_samples, prior_samples=None,
                       figsize=(10, 6)):
        """Plot the posterior distribution, optionally overlaying the prior.

        Parameters
        ----------
        posterior_samples : array-like
            Posterior samples.
        prior_samples : array-like or None
            Prior samples for overlay.
        figsize : tuple
            Figure dimensions.

        Returns
        -------
        matplotlib.figure.Figure
        """
        fig, ax = plt.subplots(figsize=figsize)
        ax.hist(posterior_samples, bins="auto", density=True,
                alpha=0.6, color="steelblue", label="Posterior")
        if prior_samples is not None:
            ax.hist(prior_samples, bins="auto", density=True,
                    alpha=0.4, color="grey", label="Prior")
        ax.set_xlabel("Parameter value")
        ax.set_ylabel("Density")
        ax.set_title("Prior and Posterior Distributions")
        ax.legend()
        fig.tight_layout()
        return fig

    def sequential_analysis(self, data, prior_mean=0, prior_sd=1,
                            figsize=(10, 6)):
        """Plot how the posterior mean evolves as sample size increases.

        Uses the normal-normal conjugate model with the empirical variance.

        Parameters
        ----------
        data : array-like
            Full data vector.
        prior_mean : float
            Prior mean (default 0).
        prior_sd : float
            Prior standard deviation (default 1).
        figsize : tuple
            Figure dimensions.

        Returns
        -------
        matplotlib.figure.Figure
        """
        data = np.asarray(data, dtype=float)
        n_total = len(data)
        s2 = np.var(data, ddof=1)
        prior_prec = 1.0 / (prior_sd ** 2)

        posterior_means = np.zeros(n_total)
        posterior_sds = np.zeros(n_total)
        for i in range(1, n_total + 1):
            x_bar_k = np.mean(data[:i])
            data_prec_k = i / s2
            post_prec_k = prior_prec + data_prec_k
            posterior_means[i - 1] = (
                prior_mean * prior_prec + x_bar_k * data_prec_k
            ) / post_prec_k
            posterior_sds[i - 1] = np.sqrt(1.0 / post_prec_k)

        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=figsize, sharex=True)
        ax1.plot(range(1, n_total + 1), posterior_means,
                 color="steelblue", lw=2)
        ax1.fill_between(range(1, n_total + 1),
                          posterior_means - 1.96 * posterior_sds,
                          posterior_means + 1.96 * posterior_sds,
                          alpha=0.2, color="steelblue")
        ax1.axhline(prior_mean, color="grey", ls="--", label="Prior mean")
        ax1.set_ylabel("Posterior mean")
        ax1.legend()
        ax1.set_title("Sequential Bayesian Updating")

        ax2.plot(range(1, n_total + 1), posterior_sds,
                 color="coral", lw=2)
        ax2.set_xlabel("Sample size")
        ax2.set_ylabel("Posterior SD")
        fig.tight_layout()
        return fig


# ---------------------------------------------------------------------------
# BayesianModelComparison
# ---------------------------------------------------------------------------
class BayesianModelComparison:
    """Tools for comparing multiple models via Bayes factors."""

    def __init__(self):
        pass

    def compare_bf(self, bf_values, model_names=None):
        """Convert Bayes factors to posterior probabilities.

        Assumes equal prior odds across models.

        Parameters
        ----------
        bf_values : array-like of float
            Bayes factors BF10 for each model compared to a baseline,
            OR a square matrix where bf_values[i][j] = BF_ij.
        model_names : list of str or None
            Labels for each model.

        Returns
        -------
        pandas.DataFrame
            Columns: model, BF, log_BF, posterior_probability
        """
        bf_values = np.asarray(bf_values, dtype=float)

        if bf_values.ndim == 1:
            # Treat as BF for each model vs a common baseline
            k = len(bf_values)
            if model_names is None:
                model_names = [f"M{i}" for i in range(k)]
            log_bf = np.log(bf_values)
            # Equal prior odds: posterior odds = BF
            log_post_odds = log_bf
            max_log = np.max(log_post_odds)
            log_denom = max_log + np.log(np.sum(np.exp(log_post_odds - max_log)))
            post_probs = np.exp(log_post_odds - log_denom)
            rows = []
            for i in range(k):
                rows.append({
                    "model": model_names[i],
                    "BF": bf_values[i],
                    "log_BF": log_bf[i],
                    "posterior_probability": post_probs[i],
                })
            return pd.DataFrame(rows)

        if bf_values.ndim == 2:
            k = bf_values.shape[0]
            if model_names is None:
                model_names = [f"M{i}" for i in range(k)]
            # Convert pairwise BF to BF against M0 (first model)
            bf_vs_first = bf_values[:, 0]
            return self.compare_bf(bf_vs_first, model_names)

        raise ValueError("bf_values must be 1-d or 2-d array")

    def robustness_check(self, bf_function, r_values=None, figsize=(10, 6)):
        """Visualise how the Bayes factor changes across Cauchy prior widths.

        Parameters
        ----------
        bf_function : callable
            A function ``f(r)`` that returns a dictionary with key ``"bf10"``
            (e.g. a lambda wrapping a :class:`BayesFactor` method).
        r_values : list of float or None
            Cauchy scale values to evaluate (default
            ``[0.2, 0.5, 0.707, 1.0, 1.5]``).
        figsize : tuple
            Figure dimensions.

        Returns
        -------
        matplotlib.figure.Figure
        """
        if r_values is None:
            r_values = [0.2, 0.5, 0.707, 1.0, 1.5]
        bfs = []
        for r in r_values:
            result = bf_function(r)
            bfs.append(result["bf10"])
        bfs = np.asarray(bfs)

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)
        ax1.plot(r_values, bfs, "o-", color="steelblue", lw=2)
        ax1.axhline(1, color="grey", ls="--", label="BF = 1 (no evidence)")
        ax1.set_xlabel("Cauchy prior width $r$")
        ax1.set_ylabel("BF$_{10}$")
        ax1.set_title("Bayes Factor vs Prior Width")
        ax1.legend()

        ax2.plot(r_values, np.log(bfs), "o-", color="coral", lw=2)
        ax2.axhline(0, color="grey", ls="--", label="log(BF) = 0")
        ax2.set_xlabel("Cauchy prior width $r$")
        ax2.set_ylabel("log(BF$_{10}$)")
        ax2.set_title("Log Bayes Factor vs Prior Width")
        ax2.legend()

        fig.tight_layout()
        return fig


# ---------------------------------------------------------------------------
# BayesianAPA
# ---------------------------------------------------------------------------
class BayesianAPA:
    """APA-formatted narrative reporting of Bayesian results."""

    @staticmethod
    def write_apa_ttest(bf_result, method_name="Bayesian t-test"):
        """APA narrative for a Bayesian t-test result.

        Parameters
        ----------
        bf_result : dict
            Output from :meth:`BayesFactor.ttest`.
        method_name : str
            Descriptive method name.

        Returns
        -------
        str
        """
        bf10 = bf_result["bf10"]
        bf01 = bf_result["bf01"]
        interp = bf_result["interpretation"]
        t_val = bf_result.get("t_stat", "?")
        df_val = bf_result.get("df", "?")
        r = bf_result.get("prior_cauchy_r", 0.707)

        text = (
            f"A {method_name} was conducted (t({df_val}) = {t_val:.3f}, "
            f"BF10 = {bf10:.3f}, "
            f"BF01 = {bf01:.3f}, "
            f"Cauchy({r}) prior). "
            f"The Bayes factor provides {interp.lower()} evidence "
            f"{'for' if bf10 >= 1 else 'against'} the alternative hypothesis."
        )
        return text

    @staticmethod
    def write_apa_correlation(bf_result):
        """APA narrative for a Bayesian correlation result.

        Parameters
        ----------
        bf_result : dict
            Output from :meth:`BayesFactor.correlation`.

        Returns
        -------
        str
        """
        bf10 = bf_result["bf10"]
        bf01 = bf_result["bf01"]
        interp = bf_result["interpretation"]
        r_val = bf_result.get("r", "?")
        n_val = bf_result.get("n", "?")
        beta_val = bf_result.get("prior_beta", 1.0)

        text = (
            f"A Bayesian Pearson correlation was computed "
            f"(r({n_val - 2}) = {r_val:.3f}, "
            f"BF10 = {bf10:.3f}, "
            f"BF01 = {bf01:.3f}, "
            f"Beta({beta_val}, {beta_val}) prior). "
            f"The Bayes factor provides {interp.lower()} evidence "
            f"{'for' if bf10 >= 1 else 'against'} the presence of a correlation."
        )
        return text

    @staticmethod
    def write_apa_estimation(est_result, var_name="parameter"):
        """APA narrative for a Bayesian estimation result.

        Parameters
        ----------
        est_result : dict
            Output from :class:`BayesianEstimation` methods.
        var_name : str
            Human-readable name of the estimated parameter.

        Returns
        -------
        str
        """
        post_mean = est_result.get("posterior_mean", est_result.get("posterior_mode", "?"))
        ci = est_result.get("credible_interval")
        cm = est_result.get("credible_mass", 0.95)

        if isinstance(post_mean, float):
            mean_str = f"{post_mean:.3f}"
        else:
            mean_str = str(post_mean)

        if ci is not None:
            ci_str = f"[{ci[0]:.3f}, {ci[1]:.3f}]"
        else:
            ci_str = "[?]"

        text = (
            f"The posterior mean for {var_name} was {mean_str} "
            f"({cm * 100:.0f}% credible interval {ci_str})."
        )
        return text
