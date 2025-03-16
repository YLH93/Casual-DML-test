# Double Machine Learning Example: Estimating the Causal Effect of Drug Dosage
This repository contains a self-contained Python example demonstrating how to use Double Machine Learning (DML) to estimate the causal effect of a continuous treatment—in this case, drug dosage—on an outcome, such as patient recovery time. The example uses synthetic data to illustrate the core concepts behind DML including nuisance function estimation, orthogonalization, and cross-fitting.

# Overview
Double Machine Learning is a powerful method that combines flexible machine learning techniques with econometric theory to estimate causal effects in the presence of high-dimensional confounding. In this example, we simulate data for:

* Covariates (X): Patient characteristics (e.g., age, baseline inflammation)
* Treatment (D): Drug dosage (continuous variable)
* Outcome (Y): Recovery time (continuous variable)

The true data-generating process follows a partially linear model:
$Y = g(X) + \theta D + \epsilon$
with a known causal effect $\theta$. DML works by estimating the nuisance functions $m(X) = E(D|X)$ and $g(X) = E(Y|X)$ using machine learning models, computing residuals, and then regressing the outcome residual on the treatment residual to recover an unbiased estimate of $\theta$

# References
Chernozhukov, V., Chetverikov, D., Demirer, M., Duflo, E., Hansen, C., Newey, W., & Robins, J. (2018). Double/Debiased Machine Learning for Treatment and Structural Parameters. The Econometrics Journal, 21(1), C1–C68. 

Schwab, P., et al. (2019). Learning Counterfactual Representations for Estimating Individual Dose-Response Curves.
  
