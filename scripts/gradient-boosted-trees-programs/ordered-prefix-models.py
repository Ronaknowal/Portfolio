from statistics import mean

# Algorithmic miniature: each base learner is a constant, not a CatBoost tree.
# Prefix model M[k] only trains on the first k rows, using earlier-prefix residuals.
targets = [1.0, 3.0, 2.0, 6.0]
models = [0.0] * (len(targets) + 1)
rate = 0.5
for round_index in range(1, 3):
    residual = [target - models[index] for index, target in enumerate(targets)]
    previous = models.copy()
    for prefix in range(1, len(models)):
        models[prefix] = previous[prefix] + rate * mean(residual[:prefix])
    print(f"round={round_index}; earlier-prefix residuals={residual}")
    print("prefix models:", models)
print(f"new-row prediction uses full-prefix model={models[-1]:.6f}")
print("An earlier-prefix residual need not be zero; it excludes the row's own training target.")
