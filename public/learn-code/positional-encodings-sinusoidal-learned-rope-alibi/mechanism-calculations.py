"""Independent NumPy geometry/cache fixtures; no training or network access."""
import json
import math
from pathlib import Path
import numpy as np

PACKET = Path(__file__).resolve().parent


def sinusoidal(positions, width, base=10000.0):
    angles = np.asarray(positions)[..., None] * base ** (-np.arange(0, width, 2) / width)
    return np.stack((np.sin(angles), np.cos(angles)), -1).reshape(*angles.shape[:-1], width)


def rotate(values, positions, base=10000.0):
    values = np.asarray(values, dtype=float)
    width = values.shape[-1]
    angles = np.asarray(positions)[..., None] * base ** (-np.arange(0, width, 2) / width)
    even, odd = values[..., ::2], values[..., 1::2]
    return np.stack((even * np.cos(angles) - odd * np.sin(angles),
                     even * np.sin(angles) + odd * np.cos(angles)), -1).reshape(values.shape)


def softmax(logits):
    weights = np.exp(logits - np.max(logits, axis=-1, keepdims=True))
    return weights / weights.sum(axis=-1, keepdims=True)


def attention(query, key, value, query_positions, key_positions, mode="rope", base=10000.0):
    if mode == "rope":
        query, key = rotate(query, query_positions, base), rotate(key, key_positions, base)
    logits = query @ key.T / np.sqrt(query.shape[-1])
    if mode == "alibi":
        logits -= .5 * (np.asarray(query_positions)[:, None] - np.asarray(key_positions)[None, :])
    legal = np.asarray(key_positions)[None, :] <= np.asarray(query_positions)[:, None]
    weights = softmax(np.where(legal, logits, -np.inf))
    return weights @ value, weights


def t5_bucket(offsets, buckets=32, maximum=128, bidirectional=True):
    offsets = np.asarray(offsets)
    sign = np.zeros(offsets.shape, dtype=int)
    if bidirectional:
        buckets //= 2
        sign = (offsets > 0).astype(int) * buckets
        distance = np.abs(offsets)
    else:
        distance = np.maximum(-offsets, 0)
    exact = buckets // 2
    large = exact + np.floor(np.log(np.maximum(distance, exact) / exact)
                             / math.log(maximum / exact) * (buckets - exact)).astype(int)
    return sign + np.where(distance < exact, distance, np.minimum(large, buckets - 1))


def main():
    query = np.array([[1, 0, .5, -1], [.5, 1, -1, .25], [2, -.5, .25, 1]])
    key = np.array([[.5, 1, 1, 0], [1, -.5, .5, 1], [-.5, .75, 1, -1]])
    value = np.array([[2, 0], [0, 3], [1, -1]])
    positions = np.array([7, 8, 9])
    caches = {}
    for mode in ("rope", "alibi"):
        full, weights = attention(query, key, value, positions, positions, mode)
        cached, _ = attention(query[-1:], key, value, positions[-1:], positions, mode)
        shifted, _ = attention(query, key, value, positions + 100, positions + 100, mode)
        caches[mode] = {"full": full.tolist(), "weights": weights.tolist(),
                        "cached_last": cached.tolist(), "shift_error": float(np.max(np.abs(full - shifted))),
                        "cached_error": float(np.max(np.abs(full[-1:] - cached)))}
    # Wrong angle for a new query with otherwise legal cached entries: do not
    # conflate a rotation bug with the separate legality-mask bug.
    rotated_key = rotate(key, positions)
    wrong_query = rotate(query[-1:], [0])
    wrong_weights = softmax(wrong_query @ rotated_key.T / 2)
    caches["wrong_query_offset"] = {"weights": wrong_weights.tolist(), "output": (wrong_weights @ value).tolist()}
    changed_base = 100.0
    mixed_weights = softmax(rotate(query[-1:], [9], changed_base) @ rotated_key.T / 2)
    fresh, fresh_weights = attention(query[-1:], key, value, [9], positions, base=changed_base)
    caches["mixed_frequency"] = {"stale_output": (mixed_weights @ value).tolist(),
                                 "fresh_output": fresh.tolist(), "fresh_weights": fresh_weights.tolist()}
    rephased = rotate(rotated_key, -positions)
    assert np.allclose(rephased, key)
    q, k = np.array([.8, -.5, .3, 1.2]), np.array([1, .25, -.5, .75])
    q3, k7 = rotate(q, 3), rotate(k, 7)
    geometry = {"query": q.tolist(), "key": k.tolist(), "query3": q3.tolist(), "key7": k7.tolist(),
                "dot": float(q3 @ k7), "relative_dot": float(q @ rotate(k, 4)),
                "shifted_dot": float(rotate(q, 103) @ rotate(k, 107)),
                "norm_query": float(np.linalg.norm(q)), "norm_rotated_query": float(np.linalg.norm(q3)),
                "query3_key3_dot": float(q3 @ rotate(k, 3)),
                "query3_key8_dot": float(q3 @ rotate(k, 8))}
    constant = np.tile(q, (4, 1))
    changing = constant.copy()
    changing[1] *= 2
    for name, values in (("constant_content", constant), ("changing_content", changing)):
        rotated = rotate(values, np.arange(4))
        geometry[name] = (rotated @ rotated.T).tolist()
    geometry["single_pair_nonmonotonic"] = {str(delta): math.cos(delta) for delta in (0, 1, 2, 3, 4, 5, 6)}
    raw_logits = np.array([2., 0., 0., 0.])
    bias = -.5 * np.array([3., 2., 1., 0.])
    alibi = {"raw_logits": raw_logits.tolist(), "bias": bias.tolist(),
             "without_bias": softmax(raw_logits).tolist(), "with_bias": softmax(raw_logits + bias).tolist(),
             "equal_content": softmax(bias).tolist(),
             "odds_distance10": math.exp(-5), "half_distance": math.log(2) / .5,
             "row_constant_error": float(np.max(np.abs(softmax(raw_logits + bias)
                                                        - softmax(raw_logits + .5 * np.arange(4)))))}
    width, base, context, scale = 64, 10000., 4096, 8
    pair = np.arange(width // 2)
    frequencies = base ** (-2 * pair / width)
    changed = (base * scale ** (width / (width - 2))) ** (-2 * pair / width)
    rotations = context * frequencies / (2 * np.pi)
    ramp = np.clip((rotations - 1) / 31, 0, 1)
    yarn = (1 - ramp) * frequencies / scale + ramp * frequencies
    frequencies_record = {"width": width, "base": base, "context": context, "scale": scale,
                          "frequency": frequencies.tolist(), "wavelength": (2 * np.pi / frequencies).tolist(),
                          "pi_frequency": (frequencies / scale).tolist(), "ntk_frequency": changed.tolist(),
                          "yarn_paper_ramp_frequency": yarn.tolist(), "rotations_in_training": rotations.tolist(),
                          "yarn_qk_multiplier": 1 + .1 * math.log(scale),
                          "yarn_logit_multiplier": (1 + .1 * math.log(scale)) ** 2,
                          "base500000_wavelength": (2 * np.pi / (500000 ** (-2 * pair / width))).tolist()}
    results = {"sinusoidal_width8_positions0to3": sinusoidal(np.arange(4), 8).tolist(),
               "sinusoidal_width32_positions0to15": sinusoidal(np.arange(16), 32).tolist(),
               "geometry": geometry, "alibi": alibi, "cache": caches,
               "cache_inputs": {"query": query.tolist(), "key": key.tolist(), "value": value.tolist(),
                                 "positions": positions.tolist()}, "frequencies": frequencies_record,
               "t5": {"offsets": [-129, -128, -16, -8, -7, -1, 0, 1, 7, 8, 16, 128, 129],
                       "buckets": t5_bucket([-129, -128, -16, -8, -7, -1, 0, 1, 7, 8, 16, 128, 129]).tolist()},
               "practice": {"distance8_odds_m025": math.exp(-2), "d8_last_wavelength": 2 * math.pi * 1000,
                             "pi_255": 255 / 4, "xpos_ratio512_firstpair": 2 / 7}}
    (PACKET / "mechanism-fixtures.json").write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(json.dumps({"geometry": geometry, "alibi": alibi, "cache": caches,
                      "longest_wavelengths": [frequencies_record["wavelength"][-1],
                                              frequencies_record["base500000_wavelength"][-1]],
                      "t5": results["t5"]}, indent=2))


if __name__ == "__main__":
    main()
