import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.stats import binom, norm
from typing import Any


def mpn_maximum_likelihood_estimation(experiment: Any, inoculum_amounts: list[float] | np.ndarray, positives: list[int] | np.ndarray, replicates: int) -> float:
    """
    Estimate Most-probable number (MPN) using Maximum Likelihood Estimation (MLE).

    Parameters:
        experiment (Any): Experiment identifier for error report purposes.
        inoculum_amounts (list[float] | np.ndarray): Log10 of sample inoculum (g | ml) (e.g., [-1.0, -2.0, -3.0, ...]).
        positives (list[int] | np.ndarray): Number of positive tubes at each dilution.
        replicates (int): Number of replicates per dilution.

    Returns:
        mpn (float): MPN estimate
    """

    def neg_log_likelihood(log_mpn_guess: float) -> float:
        """
        Negative log-likelihood function using log MPN guess.
        """
        # Convert log_mpn_guess to linear for probability calculation
        mpn_guess = 10 ** log_mpn_guess

        # Probability p for success (positive response) for each inoculum amount (or dilution) given an mpn_guess (geometric mean of all inoculum_amounts(log10) levels)
        probabilities = 1 - np.exp(-mpn_guess * (10 ** np.array(inoculum_amounts, dtype=float)))
        
        # Probability mass function for binomial distribution
        # Computes probability of exactly k successes (positives) in n independent trials (replicates) with success probability of p (probabilities)
        likelihoods = binom.pmf(positives, replicates, probabilities)

        # Return the sum of negative log-likelihood with minimum likelihood of 1e-10 to avoid log(0)
        # This function will be minimized to estimate the MPN
        return -np.sum(np.log(np.maximum(likelihoods, 1e-10)))
    
    def neg_log_likelihood_bayesian_prior(log_mpn: float) -> float:
        """
        Set a prior distribution on the MPN estimate to reduce overfitting to "zero positives"
        """
        prior = norm.logpdf(log_mpn, loc=0, scale=1)
        return neg_log_likelihood(log_mpn) - prior

    # Initial guess calculated as geometric mean of observed log10 inoculum amounts
    log_mpn_guess = np.log10(10 ** np.mean(inoculum_amounts))

    # Optimize the negative log-likelihood
    result = minimize(neg_log_likelihood, log_mpn_guess, method='Nelder-Mead')

    if result.success:
        # Check if "zero positives" were overfitted and if True use bayesian prior
        # Threshold of calculated MPN < 1 was chosen arbitrarily
        if 10 ** result.x[0] < 1:
            result = minimize(neg_log_likelihood_bayesian_prior, log_mpn_guess, method='Nelder-Mead')
        mpn = 10 ** result.x[0]

        # Return MPN in linear scale
        return mpn
    
    else:
        raise ValueError(f"Experiment {experiment}: MPN estimation failed to converge")
    

def calculate_mpn_adjustment(mpn: float, sigma: float, skewness: float, z_alpha_over_2: float) -> tuple[float]:
    """
    Calculate the confidence interval given the MPN value.

    Parameters:
        mpn (float): The MPN estimate
        sigma (float): The standard deviation
        skewness (float): The skewness of the distribution
        z_alpha_over_2 (float): The critical value for the confidence interval (e.g., 1.96 for 95% CI)

    Returns:
        confidence_interval (tuple[float]): Confidence interval limits (CI_high, CI_low) in linear scale
    """

    # Convert MPN to log10
    log_mpn = np.log10(mpn)

    # Calculate confidence interval limits
    adjustment = z_alpha_over_2 * sigma + (((z_alpha_over_2 ** 2) - 1) / 6) * skewness * sigma
    log_adjusted_mpn_low = log_mpn - adjustment
    log_adjusted_mpn_high = log_mpn + adjustment

    # Convert to linear scale
    adjusted_mpn_low = 10 ** log_adjusted_mpn_low
    adjusted_mpn_high = 10 ** log_adjusted_mpn_high

    return (adjusted_mpn_low, adjusted_mpn_high)


def calculate_mpn(data: pd.DataFrame, replicates: int, column_names: list[str], sigma: float=0.2, skewness: float=0.1, z_alpha_over_2: float=1.96, transform_inoculum_log10: bool=True) -> pd.DataFrame:
    """
    Calculate Most-probable number (MPN) for a DataFrame using Maximum Likelihood Estimation (MLE),
    and compute confidence intervals.

    Parameters:
        data (pd.DataFrame): Input data with three columns for (1) experiment identifier, (2) inoculum used in g or ml, and (3) the number of positive outcomes.
        column_names (list[str]): List containing the three column names for (1) experiment identifier, (2) inoculum used in g or ml, and (3) the number of positive outcomes.
        replicates (int): Number of indipendent trials per dilution level.
        sigma (float, optional): Standard deviation for confidence interval calculation.
        skewness (float, optional): Skewness for confidence interval calculation.
        z_alpha_over_2 (float, optional): The critical value for the confidence interval (default 1.96 for 95% CI).
        transform_inoculum_log10 (bool, optional): Boolean if inoculum is not yet log10 transfomed (True if inoculum = [0.1, 0.01, 0.001, etc ...]).

    Returns:
        result (pandas.DataFrame): Contains experiment identifier, MPN, and confidence intervals.
    """

    print("\nInitiate mpn calculation.\n")

    results = []
    if transform_inoculum_log10:
        data[column_names[1]] = np.log10(data[column_names[1]])
        print("Transform inoculum amount to log10.\n")

    for experiment, group in data.groupby(column_names[0]):
        inoculum_amounts = group[column_names[1]].tolist()  # Convert to list
        positives = group[column_names[2]].tolist()  # Convert to list
        replicates = replicates
        print(f"Calculate mpn for {experiment}.")

        try:
            mpn = mpn_maximum_likelihood_estimation(experiment, inoculum_amounts=inoculum_amounts, positives=positives, replicates=replicates)
            ci_limits = calculate_mpn_adjustment(mpn, sigma, skewness, z_alpha_over_2)
            results.append({
                column_names[0]: experiment,
                "MPN": mpn,
                "CI_low": ci_limits[0],
                "CI_high": ci_limits[1]
            })
        except ValueError as e:
            print(f"Error for experiment {experiment}: {e}")

    print("\nMPN calculation done.\n")
    return pd.DataFrame(results)
