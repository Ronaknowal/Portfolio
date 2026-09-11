def cox_score_hessian(beta, T, E, X):
    """
    Score (gradient) and Hessian of log partial likelihood.
    At each event time i, the risk set R(t_i) contributes a weighted
    mean covariate x_bar and a variance term to the Hessian.
    """
    n, d = X.shape
    eta = X @ beta                            # linear predictor
    score = np.zeros(d)
    hessian = np.zeros((d, d))
    for i in range(n):
        if E[i] == 0:
            continue
        risk_set = T >= T[i]
        w = np.exp(eta[risk_set])
        w_sum = np.sum(w)
        X_risk = X[risk_set]
        x_bar = (w @ X_risk) / w_sum         # risk-set weighted covariate mean
        score += X[i] - x_bar
        X_c = X_risk - x_bar
        hessian += -(w[:, None] * X_c).T @ X_c / w_sum
    return score, hessian

def fit_cox_nr(T, E, X, max_iter=20, tol=1e-6):
    """Newton-Raphson: beta_{k+1} = beta_k - H^{-1} score"""
    d = X.shape[1]
    beta = np.zeros(d)
    for it in range(max_iter):
        score, hessian = cox_score_hessian(beta, T, E, X)
        delta = np.linalg.solve(-hessian, score)
        beta = beta + delta
        if np.max(np.abs(delta)) < tol:
            print(f"Converged at iteration {it + 1}")
            break
    return beta

X_mat = x.reshape(-1, 1).astype(float)
beta_hat = fit_cox_nr(T, E, X_mat)
# Converged at iteration 4

print(f"Cox beta (treatment): {beta_hat[0]:.4f}")
# Cox beta (treatment): -0.8609
print(f"Hazard ratio exp(beta): {np.exp(beta_hat[0]):.4f}")
# Hazard ratio exp(beta): 0.4228
print(f"True beta=-0.7  ->  true HR={np.exp(-0.7):.4f}")
# True beta=-0.7  ->  true HR=0.4966