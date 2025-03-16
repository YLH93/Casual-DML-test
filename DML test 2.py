import numpy as np
import statsmodels.api as sm
from sklearn.model_selection import KFold
from sklearn.ensemble import GradientBoostingRegressor

# Set seed for reproducibility
np.random.seed(42)

# ------------------------------
# 1. Simulate Synthetic Data
# ------------------------------

n = 1000  # number of samples

# Covariates:
# - Age: average 70 years with some variability.
# - Comorbidity score: count of comorbidities (Poisson distributed).
# - Sex: binary indicator (0 = female, 1 = male)
age = np.random.normal(loc=70, scale=5, size=n)
comorbidity = np.random.poisson(lam=2, size=n)
sex = np.random.binomial(1, 0.5, size=n)
X = np.column_stack((age, comorbidity, sex))

# Treatment (D): Sedative dose (continuous)
# Assume doctors prescribe lower doses for older patients and higher doses for patients with more comorbidities.
m_true = 0.3 * age - 1.0 * comorbidity + 0.5 * sex
D = m_true + np.random.normal(scale=2, size=n)  # add noise

# Outcome (Y): Risk score for respiratory depression (continuous)
# Baseline risk depends on age and comorbidity.
# The causal effect of D on Y is modified by comorbidity.
theta_true = 0.1      # baseline causal effect of dose
gamma = 0.2           # effect modification factor with comorbidity
baseline = 50 + 0.2 * age + 1.0 * comorbidity  # baseline risk
# Outcome model: Y = baseline + (theta_true + gamma * comorbidity) * D + noise
Y = baseline + (theta_true + gamma * comorbidity) * D + np.random.normal(scale=3, size=n)

# ------------------------------------
# 2. Implement Double Machine Learning (DML)
# ------------------------------------
# We use 2-fold cross-fitting to estimate the nuisance functions and then obtain the debiased causal effect.
kf = KFold(n_splits=2, shuffle=True, random_state=42)
theta_estimates = []  # store the estimated treatment effects from each fold

for train_index, test_index in kf.split(X):
    # Split data into training (for nuisance estimation) and validation (for residual estimation)
    X_train, X_val = X[train_index], X[test_index]
    D_train, D_val = D[train_index], D[test_index]
    Y_train, Y_val = Y[train_index], Y[test_index]
    
    # --- Estimate the treatment model: predict D from X ---
    model_D = GradientBoostingRegressor(random_state=42)
    model_D.fit(X_train, D_train)
    m_hat_val = model_D.predict(X_val)
    
    # --- Estimate the outcome model: predict Y from X ---
    model_Y = GradientBoostingRegressor(random_state=42)
    model_Y.fit(X_train, Y_train)
    g_hat_val = model_Y.predict(X_val)
    
    # --- Compute residuals ---
    # Treatment residual: difference between actual and predicted dose.
    D_res = D_val - m_hat_val
    # Outcome residual: difference between actual and predicted outcome.
    Y_res = Y_val - g_hat_val
    
    # --- Regress outcome residual on treatment residual ---
    # Use OLS (without an intercept) to obtain the treatment effect estimate.
    ols_model = sm.OLS(Y_res, D_res).fit()
    theta_estimates.append(ols_model.params[0])

# Average the estimates from each fold to obtain the final causal effect estimate.
theta_hat = np.mean(theta_estimates)

print("Estimated overall treatment effect (theta):", theta_hat)
print("True baseline treatment effect (theta):", theta_true)

# ------------------------------------
# (Optional) Subgroup Analysis
# ------------------------------------
# For instance, analyze the effect in patients with high comorbidity.
high_comorbidity_idx = np.where(comorbidity >= np.percentile(comorbidity, 75))[0]
X_high = X[high_comorbidity_idx]
D_high = D[high_comorbidity_idx]
Y_high = Y[high_comorbidity_idx]

kf_high = KFold(n_splits=2, shuffle=True, random_state=42)
theta_estimates_high = []

for train_idx, val_idx in kf_high.split(X_high):
    X_train_high, X_val_high = X_high[train_idx], X_high[val_idx]
    D_train_high, D_val_high = D_high[train_idx], D_high[val_idx]
    Y_train_high, Y_val_high = Y_high[train_idx], Y_high[val_idx]
    
    model_D_high = GradientBoostingRegressor(random_state=42)
    model_D_high.fit(X_train_high, D_train_high)
    m_hat_val_high = model_D_high.predict(X_val_high)
    
    model_Y_high = GradientBoostingRegressor(random_state=42)
    model_Y_high.fit(X_train_high, Y_train_high)
    g_hat_val_high = model_Y_high.predict(X_val_high)
    
    D_res_high = D_val_high - m_hat_val_high
    Y_res_high = Y_val_high - g_hat_val_high
    
    ols_model_high = sm.OLS(Y_res_high, D_res_high).fit()
    theta_estimates_high.append(ols_model_high.params[0])

theta_hat_high = np.mean(theta_estimates_high)
print("Estimated treatment effect for high comorbidity patients:", theta_hat_high)
