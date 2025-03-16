# Double Machine Learning Example: Estimating the Causal Effect of Drug Dosage
This repository contains a self-contained Python example demonstrating how to use Double Machine Learning (DML) to estimate the causal effect of a continuous treatment—in this case, drug dosage—on an outcome, such as patient recovery time. The example uses synthetic data to illustrate the core concepts behind DML including nuisance function estimation, orthogonalization, and cross-fitting.

# Overview
Double Machine Learning is a powerful method that combines flexible machine learning techniques with econometric theory to estimate causal effects in the presence of high-dimensional confounding. In this example, we simulate data for:

* Covariates (X): Patient characteristics (e.g., age, baseline inflammation)
* Treatment (D): Drug dosage (continuous variable)
* Outcome (Y): Recovery time (continuous variable)

The true data-generating process follows a partially linear model:
$$ Y = g(X) + \theta D + \epsilon $$
  
