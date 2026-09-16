"""Optional documented library route; written, not executed in phase one.

Install a compatible pgmpy in your own environment, then run this file.
The five-node teaching network is unrelated to the larger named ALARM dataset.
"""
from pgmpy.models import DiscreteBayesianNetwork
from pgmpy.factors.discrete import TabularCPD
from pgmpy.inference import VariableElimination

model = DiscreteBayesianNetwork(
    [("B", "A"), ("E", "A"), ("A", "J"), ("A", "M")]
)
# State order is 0, 1. For A, columns are (B,E) = 00, 01, 10, 11.
model.add_cpds(
    TabularCPD("B", 2, [[.999], [.001]]),
    TabularCPD("E", 2, [[.998], [.002]]),
    TabularCPD("A", 2, [[.999, .71, .06, .05], [.001, .29, .94, .95]],
               evidence=["B", "E"], evidence_card=[2, 2]),
    TabularCPD("J", 2, [[.95, .1], [.05, .9]], evidence=["A"], evidence_card=[2]),
    TabularCPD("M", 2, [[.99, .3], [.01, .7]], evidence=["A"], evidence_card=[2]),
)
model.check_model()
posterior = VariableElimination(model).query(
    variables=["B"], evidence={"J": 1, "M": 1}, show_progress=False
)
print(posterior.values)
# Mathematically expected: approximately [0.71582816, 0.28417184].
