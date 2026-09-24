from fractions import Fraction


def pinball(target, prediction, quantile):
    residual = target - prediction
    return quantile * residual if residual >= 0 else (quantile - 1) * residual


# Constructed spare-part demand within one leaf, not operational evidence.
demand = [0, 1, 1, 2, 8]
quantile = Fraction(4, 5)
scores = [(candidate, sum(pinball(value, candidate, quantile) for value in demand))
          for candidate in range(9)]
best = min(loss for _, loss in scores)
print("integer minimizers:", [candidate for candidate, loss in scores if loss == best])
print(f"minimum total pinball loss={best}")
print(f"mean demand={sum(demand)/len(demand):.1f}; mean prediction pinball loss={float(sum(pinball(value,Fraction(sum(demand),len(demand)),quantile) for value in demand)):.3f}")
print("At tau=.8, underprediction costs four times overprediction per unit.")
print("The optimum is an interval here; a conditional quantile is not a calibrated prediction interval by itself.")
