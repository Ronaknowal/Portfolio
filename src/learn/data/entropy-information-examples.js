export const entropyInformationExamples = {
  "coin": {
    "title": "A biased coin in bits",
    "question": "Should exchanging heads and tails change this answer?",
    "code": "import math\n\np_heads = 0.9\nentropy_bits = -(\n    p_heads * math.log2(p_heads)\n    + (1 - p_heads) * math.log2(1 - p_heads)\n)\nprint(round(entropy_bits, 3))",
    "expected": "0.469"
  },
  "bernoulli": {
    "title": "The same Bernoulli mismatch in nats",
    "question": "Which number changes if Q is replaced by P?",
    "code": "import math\n\n# True Bernoulli probability P(x=1)=0.7; model believes Q(x=1)=0.9.\np, q = 0.7, 0.9\nentropy = -(p * math.log(p) + (1 - p) * math.log(1 - p))\ncross_entropy = -(p * math.log(q) + (1 - p) * math.log(1 - q))\nkl = cross_entropy - entropy\n\nprint(round(entropy, 3), round(cross_entropy, 3), round(kl, 3))",
    "expected": "0.611 0.765 0.154"
  },
  "prefix": {
    "title": "Encode and decode without separators",
    "question": "Why does reaching a complete codeword allow the decoder to reset?",
    "code": "CODE = {\"A\": \"0\", \"B\": \"10\", \"C\": \"110\", \"D\": \"111\"}\n\n\ndef decode(bits, code):\n    words = list(code.values())\n    if len(set(words)) != len(words) or any(\n        not word or set(word) - {\"0\", \"1\"} for word in words\n    ):\n        raise ValueError(\"Codewords must be distinct nonempty binary strings.\")\n    if any(a != b and b.startswith(a) for a in words for b in words):\n        raise ValueError(\"Codebook is not prefix-free.\")\n    if set(bits) - {\"0\", \"1\"}:\n        raise ValueError(\"The stream must contain only bits.\")\n    reverse = {word: symbol for symbol, word in code.items()}\n    output, prefix = [], \"\"\n    for bit in bits:\n        prefix += bit\n        if prefix in reverse:\n            output.append(reverse[prefix])\n            prefix = \"\"\n        elif not any(word.startswith(prefix) for word in words):\n            raise ValueError(\"This bit sequence has no matching word.\")\n    if prefix:\n        raise ValueError(\"The stream ends inside a codeword.\")\n    return \"\".join(output)\n\n\nmessage = \"AAAABBCD\"\nbits = \"\".join(CODE[symbol] for symbol in message)\nprint(bits)\nprint(decode(bits, CODE))\nprint(f\"{len(bits)} bits; {len(bits) / len(message):.2f} bits/symbol\")\ntry:\n    decode(\"11\", CODE)\nexcept ValueError as error:\n    print(error)",
    "expected": "00001010110111\nAAAABBCD\n14 bits; 1.75 bits/symbol\nThe stream ends inside a codeword."
  },
  "blocks": {
    "title": "Block coding spreads the rounding overhead",
    "question": "A binary source still needs one-bit single-symbol words. Can grouped words approach less than one bit per draw?",
    "code": "from fractions import Fraction\nfrom itertools import product\nfrom math import log2\n\np = Fraction(9, 10)\nh = -float(p) * log2(float(p)) - float(1 - p) * log2(float(1 - p))\nfor n in (1, 2, 4, 8):\n    expected_length, kraft = Fraction(0), Fraction(0)\n    for sequence in product((0, 1), repeat=n):\n        probability = p ** sum(sequence) * (1 - p) ** (n - sum(sequence))\n        length = 0\n        while Fraction(1, 2**length) > probability:\n            length += 1\n        expected_length += probability * length\n        kraft += Fraction(1, 2**length)\n    rate = float(expected_length) / n\n    assert kraft <= 1\n    assert h <= rate < h + 1 / n\n    print(f\"n={n}: rate={rate:.6f}, H={h:.6f}, upper={h+1/n:.6f}\")",
    "expected": "n=1: rate=1.300000, H=0.468996, upper=1.468996\nn=2: rate=0.800000, H=0.468996, upper=0.968996\nn=4: rate=0.550925, H=0.468996, upper=0.718996\nn=8: rate=0.550054, H=0.468996, upper=0.593996"
  },
  "support": {
    "title": "Compute finite distribution costs without hiding zero support",
    "question": "Which zero rows contribute zero, and which produce an infinite result?",
    "code": "import math\n\n\ndef information(p, q, base=2):\n    if len(p) != len(q) or not p:\n        raise ValueError(\"Use the same nonempty labeled alphabet.\")\n    if not math.isfinite(base) or base <= 1:\n        raise ValueError(\"Log base must be finite and greater than one.\")\n    for distribution in (p, q):\n        if any(not math.isfinite(x) or x < 0 for x in distribution):\n            raise ValueError(\"Probabilities must be finite and nonnegative.\")\n        if not math.isclose(sum(distribution), 1, rel_tol=0, abs_tol=1e-12):\n            raise ValueError(\"Each distribution must sum to one.\")\n    entropy = cross_entropy = kl = 0.0\n    for mass, model in zip(p, q):\n        if mass == 0:\n            continue\n        entropy -= mass * math.log(mass, base)\n        if model == 0:\n            cross_entropy = kl = math.inf\n        else:\n            cross_entropy -= mass * math.log(model, base)\n            kl += mass * (math.log(mass, base) - math.log(model, base))\n    return entropy, cross_entropy, kl\n\n\nfor p, q in [\n    ([0.5, 0.25, 0.125, 0.125], [0.125, 0.125, 0.25, 0.5]),\n    ([1, 0], [0.5, 0.5]),\n    ([0.5, 0.5], [1, 0]),\n    ([1, 0], [1, 0]),\n]:\n    print(tuple(round(value, 6) for value in information(p, q)))",
    "expected": "(1.75, 2.625, 0.875)\n(0.0, 1.0, 1.0)\n(1.0, inf, inf)\n(0.0, 0.0, 0.0)"
  },
  "conditional": {
    "title": "Averaging prediction cost over the actual joint distribution",
    "question": "The marginal label distribution stays fair. Which term reveals whether the context helps?",
    "code": "import math\n\nnoise, trust = 0.1, 0.8\njoint = {(x, y): 0.5 * (1 - noise if x == y else noise) for x in (0, 1) for y in (0, 1)}\nconditional_entropy = conditional_loss = 0.0\nfor (x, y), mass in joint.items():\n    true_conditional = 1 - noise if x == y else noise\n    model_conditional = trust if x == y else 1 - trust\n    if mass:\n        conditional_entropy -= mass * math.log2(true_conditional)\n        conditional_loss -= mass * math.log2(model_conditional)\nprint(\"P(Y=1):\", sum(mass for (x, y), mass in joint.items() if y == 1))\nprint(f\"H(Y|X)={conditional_entropy:.6f} bits\")\nprint(f\"Model loss={conditional_loss:.6f} bits\")\nprint(f\"Excess={conditional_loss-conditional_entropy:.6f} bits\")",
    "expected": "P(Y=1): 0.5\nH(Y|X)=0.468996 bits\nModel loss=0.521928 bits\nExcess=0.052933 bits"
  },
  "logits": {
    "title": "Keep log probabilities on the stable path",
    "question": "With scores 1000,0,−1000, what is the loss when the third class occurs?",
    "code": "import math\n\n\ndef log_probabilities(logits):\n    if not logits or any(not math.isfinite(x) for x in logits):\n        raise ValueError(\"Use a nonempty list of finite logits.\")\n    maximum = max(logits)\n    shifted = [x - maximum for x in logits]\n    log_normalizer = math.log(sum(math.exp(x) for x in shifted))\n    return [x - log_normalizer for x in shifted]\n\n\nfor scores in ([2, 0, -2], [1002, 1000, 998], [1000, 0, -1000]):\n    logp = log_probabilities(scores)\n    print(\"log probabilities:\", [round(x, 6) for x in logp])\n    print(\"last-class loss:\", round(-logp[-1], 6))",
    "expected": "log probabilities: [-0.142932, -2.142932, -4.142932]\nlast-class loss: 4.142932\nlog probabilities: [-0.142932, -2.142932, -4.142932]\nlast-class loss: 4.142932\nlog probabilities: [0.0, -1000.0, -2000.0]\nlast-class loss: 2000.0"
  },
  "torch": {
    "title": "Use the framework contract: logits in, class indices or valid targets",
    "question": "Does a common offset change either example’s loss?",
    "code": "import torch\nimport torch.nn.functional as F\n\nscores = torch.tensor([[2.0, 0.0, -2.0], [1000.0, 0.0, -1000.0]], dtype=torch.float64)\nlabels = torch.tensor([0, 2], dtype=torch.long)\nlosses = F.cross_entropy(scores, labels, reduction=\"none\")\nassert torch.allclose(losses, F.cross_entropy(scores + 1000, labels, reduction=\"none\"))\nsoft_targets = torch.tensor([[0.7, 0.2, 0.1], [0.0, 0.0, 1.0]], dtype=torch.float64)\nassert bool(torch.all(soft_targets >= 0))\nassert torch.allclose(soft_targets.sum(dim=1), torch.ones(2, dtype=torch.float64))\nsoft_losses = F.cross_entropy(scores, soft_targets, reduction=\"none\")\nprint(\"Class-index losses:\", [round(x, 6) for x in losses.tolist()])\nprint(\"Soft-target losses:\", [round(x, 6) for x in soft_losses.tolist()])\nprint(\"Mean class-index loss:\", round(losses.mean().item(), 6))",
    "expected": "Class-index losses: [0.142932, 2000.0]\nSoft-target losses: [0.942932, 2000.0]\nMean class-index loss: 1000.071466"
  },
  "evaluation": {
    "title": "Score every observation on the same ten-case test set",
    "question": "Both models have eight successes. Which two observations decide the log-loss comparison?",
    "code": "import math\n\ncorrect_label_probabilities = {\n    \"A\": [0.6] * 8 + [0.4] * 2,\n    \"B\": [0.99] * 8 + [0.01] * 2,\n}\nfor name, probabilities in correct_label_probabilities.items():\n    losses = [-math.log(p) for p in probabilities]\n    accuracy = sum(p > 0.5 for p in probabilities) / len(probabilities)\n    mean_loss = sum(losses) / len(losses)\n    geometric_probability = math.exp(-mean_loss)\n    print(f\"{name}: accuracy={accuracy:.0%}, loss={mean_loss:.6f} nats\")\n    print(f\"geometric correct-label probability={geometric_probability:.6f}\")\n    print(f\"perplexity={math.exp(mean_loss):.6f}\")",
    "expected": "A: accuracy=80%, loss=0.591919 nats\ngeometric correct-label probability=0.553265\nperplexity=1.807453\nB: accuracy=80%, loss=0.929074 nats\ngeometric correct-label probability=0.394919\nperplexity=2.532164"
  },
  "continuous": {
    "title": "Change units while preserving probability and KL",
    "question": "Which quantities change when metres become centimetres?",
    "code": "import math\n\nwidth, bins = 0.25, 4\nfor scale in (1, 100):\n    coordinate_width = scale * width\n    h = math.log2(coordinate_width)\n    delta = coordinate_width / bins\n    print(\n        f\"scale={scale}: h={h:.6f}, bin width={delta:g}, H(bin)={h-math.log2(delta):.6f}\"\n    )\n\n\ndef normal_kl(mean_p, sd_p, mean_q, sd_q):\n    if (\n        not all(math.isfinite(x) for x in (mean_p, sd_p, mean_q, sd_q))\n        or min(sd_p, sd_q) <= 0\n    ):\n        raise ValueError(\n            \"Use finite means and strictly positive finite standard deviations.\"\n        )\n    return (\n        math.log(sd_q / sd_p) + (sd_p**2 + (mean_p - mean_q) ** 2) / (2 * sd_q**2) - 0.5\n    )\n\n\nfor scale in (1, 100):\n    print(\n        f\"Normal KL after scaling by {scale}: {normal_kl(0,scale,scale,2*scale):.6f} nats\"\n    )",
    "expected": "scale=1: h=-2.000000, bin width=0.0625, H(bin)=2.000000\nscale=100: h=4.643856, bin width=6.25, H(bin)=2.000000\nNormal KL after scaling by 1: 0.443147 nats\nNormal KL after scaling by 100: 0.443147 nats"
  },
  "maximum": {
    "title": "Certify a maximum by the gap, not just a plotted peak",
    "question": "All three candidate distributions have mean10/7. Which maximizes entropy, and why is the gap guaranteed nonnegative?",
    "code": "import math\n\np_star = [1 / 7, 2 / 7, 4 / 7]\nmean = 10 / 7\n\n\ndef entropy(p):\n    return -sum(x * math.log2(x) for x in p if x)\n\n\nfor t in (3 / 7, 4 / 7, 5 / 7):\n    q = [1 - mean + t, mean - 2 * t, t]\n    # Exact boundaries may accumulate tiny floating subtraction noise.\n    q = [0.0 if abs(x) < 1e-14 else x for x in q]\n    assert min(q) >= 0 and math.isclose(sum(q), 1)\n    assert math.isclose(q[1] + 2 * q[2], mean)\n    kl = sum(x * math.log2(x / y) for x, y in zip(q, p_star) if x)\n    gap = entropy(p_star) - entropy(q)\n    assert math.isclose(kl, gap, abs_tol=1e-12)\n    print(f\"q={[round(x,6) for x in q]}, H={entropy(q):.6f}, KL gap={kl:.6f}\")\nbase_tilt = [0.25, 0.5, 0.25]\nuniform = [1 / 3] * 3\nprint(f\"Nonuniform base at eta=0: H={entropy(base_tilt):.6f}\")\nprint(f\"Same mean, uniform: H={entropy(uniform):.6f}\")",
    "expected": "q=[0.0, 0.571429, 0.428571], H=0.985228, KL gap=0.393555\nq=[0.142857, 0.285714, 0.571429], H=1.378783, KL gap=0.000000\nq=[0.285714, 0.0, 0.714286], H=0.863121, KL gap=0.515663\nNonuniform base at eta=0: H=1.500000\nSame mean, uniform: H=1.584963"
  }
};
