"""BernoulliRBM fit/transform plus a matched, finite-state probability oracle.

Authoring target: NumPy 2.3.5, SciPy 1.18.1, scikit-learn 1.9.1.
Run beside rbm-study.py. This short constructed fit is not a new digit benchmark.
"""
from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path
import numpy as np
from scipy.special import expit
from sklearn.neural_network import BernoulliRBM


def load_mechanisms():
    # The earlier program's hyphenated download name is not a Python identifier.
    spec = spec_from_file_location("rbm_mechanisms", Path(__file__).with_name("rbm-study.py"))
    module = module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def main():
    scratch = load_mechanisms()
    # Counts, not class labels: these four binary patterns form a tiny training set.
    states = scratch.bits(2)
    fitting = np.repeat(states, [1, 2, 2, 5], axis=0)
    model = BernoulliRBM(n_components=1, learning_rate=.05, batch_size=5,
                         n_iter=20, random_state=19).fit(fitting)
    hidden = model.transform(states)
    weights = model.components_.T.copy()       # library H x D -> lesson D x H
    visible_bias = model.intercept_visible_.copy()
    hidden_bias = model.intercept_hidden_.copy()
    np.testing.assert_allclose(hidden, expit(states @ weights + hidden_bias), atol=1e-14)
    free_energy = scratch.free_energy(states, weights, visible_bias, hidden_bias)
    log_z = scratch.hidden_distribution(weights, visible_bias, hidden_bias)[3]
    visible_probability = np.exp(-free_energy-log_z)
    np.testing.assert_allclose(visible_probability.sum(), 1., atol=1e-14)

    # A deterministic transition matrix sums over every hidden/visible draw.
    h = scratch.bits(1)
    visible_given_h = expit(h @ weights.T + visible_bias)
    probability_v_given_h = np.prod(
        visible_given_h[:, None, :]**states[None, :, :]
        * (1-visible_given_h[:, None, :])**(1-states[None, :, :]), axis=2)
    transition = np.c_[1-hidden[:, 0], hidden[:, 0]] @ probability_v_given_h
    np.testing.assert_allclose(visible_probability @ transition, visible_probability,
                               atol=1e-14)
    sampled_next = model.gibbs(states)          # actual binary draws, not hidden means
    assert set(np.unique(sampled_next)).issubset({0, 1})
    pseudo_likelihood = model.score_samples(states)
    print("Hidden probabilities:", hidden[:, 0])
    print("Exact probabilities of 00,01,10,11:", visible_probability)
    print("One sampled visible transition:", sampled_next.astype(int))
    print("Random-bit pseudo-likelihood estimate:", pseudo_likelihood)
    print("Exact log probability:", -free_energy-log_z)


if __name__ == "__main__":
    main()
