from itertools import combinations
from fractions import Fraction

gradient = [8, -6, 3, -2, 1, -1]
kept, remaining, draw = [0, 1], [2, 3, 4, 5], 2
probability = Fraction(draw, len(remaining))
weight = 1 / probability
totals = [sum(gradient[i] for i in kept) + weight * sum(gradient[i] for i in sample)
          for sample in combinations(remaining, draw)]
expected = sum(totals) / len(totals)
expected_square = sum(value ** 2 for value in totals) / len(totals)
print(f"inclusion probability={probability}; small-row weight={weight}")
print("all estimated sums:", [int(value) for value in totals])
print(f"full sum={sum(gradient)}; expected estimate={expected}")
print(f"square of full sum={sum(gradient)**2}; expected squared estimate={expected_square}")
print(f"variance of estimate={expected_square-expected**2}")
