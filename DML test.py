import numpy as np
import statsmodels.api as sm
from sklearn.model_selection import KFold
from sklearn.ensemble import GradientBoostingRegressor

# Seed for reproducibility
np.random.seed(42)

# --------------------------
# 1. Generate synthetic data
# --------------------------
n = 1000  # total number of samples

# Generate two covariates: age and baseline inflammation
age = np.random.normal(loc=50, scale=10, size=n)
inflammation = np.random.normal(loc=5, scale=2, size=n)
X = np.column_stack((age, inflammation))

# Define true models:
# Treatment model: dose D is generated as a function of covariates + noise.
m_true = 0.1 * age + 0.5 * inflammation
D = m_true + np.random.normal(scale=1, size=n)

# Outcome model: recovery time Y depends on covariates, the dose, and noise.
# True causal effect (theta_true): each unit increase in dose reduces recovery time by 0.15 days.
theta_true = -0.15
g_true = 2 * age + 1.5 * inflammation
Y = g_true + theta_true * D + np.random.normal(scale=1, size=n)

# ------------------------------------
# 2. Double Machine Learning with 2-fold cross-fitting
# ------------------------------------
kf = KFold(n_splits=2, shuffle=True, random_state=42)
theta_estimates = []  # to store treatment effect estimates from each fold

for train_index, test_index in kf.split(X):
    # Split data into training and validation (for nuisance estimation)
    X_train, X_val = X[train_index], X[test_index]
    D_train, D_val = D[train_index], D[test_index]
    Y_train, Y_val = Y[train_index], Y[test_index]
    
    # --- Estimate the treatment model: Predict D from X ---
    model_D = GradientBoostingRegressor(random_state=42)
    model_D.fit(X_train, D_train)
    m_hat_val = model_D.predict(X_val)
    
    # --- Estimate the outcome model: Predict Y from X ---
    model_Y = GradientBoostingRegressor(random_state=42)
    model_Y.fit(X_train, Y_train)
    g_hat_val = model_Y.predict(X_val)
    
    # --- Compute residuals ---
    # Residual for treatment: deviation of actual dose from predicted dose.
    D_res = D_val - m_hat_val
    # Residual for outcome: deviation of actual outcome from predicted outcome.
    Y_res = Y_val - g_hat_val
    
    # --- Regress outcome residual on treatment residual ---
    # Note: We run an OLS regression without an intercept.
    # This regression yields the estimate for the causal effect (theta).
    ols_model = sm.OLS(Y_res, D_res).fit()
    theta_estimates.append(ols_model.params[0])

# Average the two fold estimates to obtain the final causal effect estimate.
theta_hat = np.mean(theta_estimates)
print("Estimated causal effect (theta):", theta_hat)
print("True causal effect (theta):", theta_true)
