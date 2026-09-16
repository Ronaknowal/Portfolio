// Displayed programs for the hidden Markov models lesson.
//
// Extracted verbatim from the frozen packet files and the manuscript's own fenced
// block, each pinned by SHA-256, and executed by scripts/verify-hmm-examples.py.
// The complete program's excerpts are exact source segments taken through Python's
// AST, with the line numbers they occupy in the served file. The optional hmmlearn
// program was executed in an isolated environment whose resolved versions are
// recorded beside its output. Do not edit by hand.
export const hmmExamples = {
  "experiments": {
    "title": "The complete offline program: inference, learning and the real tagging comparison",
    "question": "Which rows does one changed report move, and which does it leave exactly alone?",
    "file": "hmm-experiments.py",
    "setup": "python hmm-experiments.py",
    "language": "python",
    "download": "/learn-assets/hmm/hmm-experiments.py",
    "dataFile": "/learn-assets/hmm/ewt-sequences.json",
    "lineCount": 331,
    "sha256": "25e7e4016c5420b7aae33a21e869275564d6951bc22865fd173f566c1d7b2f77",
    "expected": "Recorded exact mechanisms, four EM fits, and four supervised tagging configurations.",
    "executed": true,
    "excerpts": [
      {
        "key": "inference",
        "title": "The trellis itself: one sum, one backward sum, one maximum",
        "guidance": "Follow one forward sum, one backward sum, the posterior normalisation, and then the separate maximum-and-backpointer recurrence that does not share them.",
        "functions": [
          "log_values",
          "infer"
        ],
        "code": "def log_values(values):\n    with np.errstate(divide=\"ignore\"):\n        return np.log(values)\n\n\ndef infer(start, transition, emission, observations):\n    \"\"\"Return joint evidence, filtered/smoothed beliefs and exact best-path decoding.\n\n    Rows of transition/emission sum to one. Observation -1 is missing at a\n    retained time step: sum over its categories, giving emission likelihood one.\n    \"\"\"\n    observations = np.asarray(observations, dtype=int)\n    if len(observations) == 0:\n        raise ValueError(\"This teaching function requires at least one time step.\")\n    states = len(start)\n    log_transition = log_values(transition)\n    local = np.array([\n        np.zeros(states) if value == -1 else log_values(emission[:, value])\n        for value in observations\n    ])\n    forward = np.empty_like(local)\n    forward[0] = log_values(start) + local[0]\n    for time in range(1, len(observations)):\n        forward[time] = np.logaddexp.reduce(\n            forward[time - 1, :, None] + log_transition, axis=0\n        ) + local[time]\n    log_evidence = np.logaddexp.reduce(forward[-1])\n    if not np.isfinite(log_evidence):\n        raise ValueError(\"Impossible observation sequence under this model.\")\n    backward = np.zeros_like(local)\n    for time in range(len(observations) - 2, -1, -1):\n        backward[time] = np.logaddexp.reduce(\n            log_transition + local[time + 1] + backward[time + 1], axis=1\n        )\n    smoothed = np.exp(forward + backward - log_evidence)\n    filtered = np.exp(\n        forward - np.logaddexp.reduce(forward, axis=1, keepdims=True)\n    )\n    pair = np.empty((len(observations) - 1, states, states))\n    for time in range(len(observations) - 1):\n        pair[time] = np.exp(\n            forward[time, :, None] + log_transition\n            + local[time + 1] + backward[time + 1] - log_evidence\n        )\n    best = np.empty_like(local)\n    predecessor = np.zeros(local.shape, dtype=int)\n    best[0] = forward[0]\n    for time in range(1, len(observations)):\n        candidates = best[time - 1, :, None] + log_transition\n        predecessor[time] = np.argmax(candidates, axis=0)\n        best[time] = candidates[predecessor[time], np.arange(states)] + local[time]\n    path = np.zeros(len(observations), dtype=int)\n    path[-1] = np.argmax(best[-1])\n    for time in range(len(observations) - 2, -1, -1):\n        path[time] = predecessor[time + 1, path[time + 1]]\n    return {\n        \"log_evidence\": log_evidence, \"forward_log\": forward,\n        \"backward_log\": backward, \"filtered\": filtered, \"smoothed\": smoothed,\n        \"pair\": pair, \"viterbi_log\": best, \"predecessor\": predecessor,\n        \"path\": path, \"path_joint\": np.exp(best[-1, path[-1]]),\n        \"path_posterior\": np.exp(best[-1, path[-1]] - log_evidence),\n        \"marginal_modes\": np.argmax(smoothed, axis=1),\n    }",
        "lines": [
          9,
          71
        ],
        "sha256": "40e5586711b0fbeb9cd07621e46c6cc9a08f63fc0ed83e70d9068a5ce8c643ad"
      },
      {
        "key": "scaling",
        "title": "Filtering that keeps its scale factors instead of its logarithms",
        "guidance": "Each row is normalised and its factor kept, so the evidence is the product of the factors and its logarithm is their sum. This is the exact alternative to log space.",
        "functions": [
          "forward_scaled"
        ],
        "code": "def forward_scaled(start, transition, emission, observations):\n    belief = start.copy()\n    factors = []\n    for time, value in enumerate(observations):\n        prediction = belief if time == 0 else belief @ transition\n        mass = prediction if value == -1 else prediction * emission[:, value]\n        factor = mass.sum()\n        if factor == 0:\n            raise ValueError(\"Impossible observation sequence under this model.\")\n        belief = mass / factor\n        factors.append(factor)\n    return {\"filtered_last\": belief, \"factors\": factors,\n            \"log_evidence\": np.log(factors).sum()}",
        "lines": [
          74,
          86
        ],
        "sha256": "a7009192ae09ab57586daca6f6b43c8abeab1e7dfb34dec56d82186023ad0031"
      },
      {
        "key": "learning",
        "title": "Fractional events, one recording at a time",
        "guidance": "Add fractional events within each recording and reset the initial event at every boundary. An unvisited state's row is unidentified, so it is retained rather than divided by zero.",
        "functions": [
          "expected_counts",
          "normalize_counts",
          "em_step"
        ],
        "code": "def expected_counts(model, sequences):\n    start, transition, emission = model\n    initial = np.zeros_like(start)\n    edge = np.zeros_like(transition)\n    symbol = np.zeros_like(emission)\n    log_likelihood = 0.0\n    for observations in sequences:\n        result = infer(*model, observations)\n        initial += result[\"smoothed\"][0]\n        edge += result[\"pair\"].sum(axis=0)\n        for time, value in enumerate(observations):\n            if value != -1:\n                symbol[:, value] += result[\"smoothed\"][time]\n        log_likelihood += result[\"log_evidence\"]\n    return initial, edge, symbol, log_likelihood\n\n\ndef normalize_counts(counts, previous):\n    \"\"\"An unvisited row is unidentified; retain it instead of inventing counts.\"\"\"\n    total = counts.sum(axis=-1, keepdims=True)\n    return np.divide(counts, total, out=previous.copy(), where=total > 0)\n\n\ndef em_step(model, sequences):\n    initial, edge, symbol, _ = expected_counts(model, sequences)\n    return (\n        initial / initial.sum(),\n        normalize_counts(edge, model[1]),\n        normalize_counts(symbol, model[2]),\n    )",
        "lines": [
          89,
          118
        ],
        "sha256": "2322440a089e27ad202595b068769088a0ba582b0557ce04c95f449caad20f72"
      },
      {
        "key": "tagging",
        "title": "The supervised fit on real sentences, and two decision rules from it",
        "guidance": "Fit category counts on the training sentences, then compare the declared decoders on the development sentences. The same fitted counts serve both rules.",
        "functions": [
          "real_tagging"
        ],
        "code": "def real_tagging(directory):\n    records = json.loads((directory / \"ewt-sequences.json\").read_text(encoding=\"utf-8\"))\n    # The retained extract is a flat list; split membership is preserved from EWT.\n    train = [row for row in records if row[\"split\"] == \"train\"]\n    development = [row for row in records if row[\"split\"] == \"dev\"]\n    labels = [\"NOUN\", \"VERB\", \"OTHER\"]\n    def coarse(upos):\n        return 0 if upos in {\"NOUN\", \"PROPN\"} else 1 if upos in {\"VERB\", \"AUX\"} else 2\n    frequency = Counter(word.lower() for row in train for word in row[\"tokens\"])\n    vocabulary = [\"<UNKNOWN>\"] + sorted(word for word, count in frequency.items() if count >= 2)\n    index = {word: number for number, word in enumerate(vocabulary)}\n    def encode(words):\n        return [index.get(word.lower(), 0) for word in words]\n    configurations = []\n    for smoothing in [.1, 1.]:\n        initial = np.full(3, smoothing)\n        transition = np.full((3, 3), smoothing)\n        emission = np.full((3, len(vocabulary)), smoothing)\n        occupancy = np.full(3, smoothing)\n        for row in train:\n            tags = [coarse(value) for value in row[\"upos\"]]\n            initial[tags[0]] += 1\n            for previous, following in zip(tags, tags[1:]):\n                transition[previous, following] += 1\n            for tag, symbol in zip(tags, encode(row[\"tokens\"])):\n                emission[tag, symbol] += 1\n                occupancy[tag] += 1\n        model = (initial / initial.sum(), transition / transition.sum(axis=1, keepdims=True),\n                 emission / emission.sum(axis=1, keepdims=True))\n        prior = occupancy / occupancy.sum()\n        for method in [\"lexical\", \"hmm\"]:\n            rows = []\n            for row in development:\n                symbols = encode(row[\"tokens\"])\n                truth = np.array([coarse(value) for value in row[\"upos\"]])\n                if method == \"hmm\":\n                    result = infer(*model, symbols)\n                    prediction = result[\"path\"]\n                    beliefs = result[\"smoothed\"]\n                else:\n                    mass = model[2][:, symbols].T * prior\n                    beliefs = mass / mass.sum(axis=1, keepdims=True)\n                    prediction = beliefs.argmax(axis=1)\n                rows.append({\"id\": row[\"id\"], \"tokens\": row[\"tokens\"], \"symbols\": symbols,\n                             \"truth\": truth, \"predicted\": prediction, \"beliefs\": beliefs,\n                             \"correct\": int((truth == prediction).sum())})\n            configurations.append({\n                \"method\": method, \"smoothing\": smoothing, \"model\": {\n                    \"start\": model[0], \"transition\": model[1], \"emission\": model[2],\n                    \"occupancy\": prior},\n                \"correct\": sum(row[\"correct\"] for row in rows),\n                \"tokens\": sum(len(row[\"tokens\"]) for row in rows),\n                \"sentences_correct\": sum(row[\"correct\"] == len(row[\"tokens\"]) for row in rows),\n                \"rows\": rows,\n            })\n    # Predeclared ties favor lexical, then smaller smoothing.\n    chosen = max(range(len(configurations)), key=lambda i: (\n        configurations[i][\"correct\"], configurations[i][\"method\"] == \"lexical\",\n        -configurations[i][\"smoothing\"]))\n    majority = Counter(coarse(tag) for row in train for tag in row[\"upos\"]).most_common(1)[0][0]\n    return {\"labels\": labels, \"vocabulary\": vocabulary, \"train_ids\": [row[\"id\"] for row in train],\n            \"development_ids\": [row[\"id\"] for row in development],\n            \"train_tokens\": sum(len(row[\"tokens\"]) for row in train),\n            \"unknown_development_tokens\": sum(symbol == 0 for row in development for symbol in encode(row[\"tokens\"])),\n            \"majority_label\": majority,\n            \"majority_development_correct\": sum(coarse(tag) == majority for row in development for tag in row[\"upos\"]),\n            \"configurations\": configurations, \"selected_configuration_index\": chosen,\n            \"reserved_test_scored\": False}",
        "lines": [
          233,
          300
        ],
        "sha256": "5b1d8be6a74413acef3996678c458fbd222892489ba2fd7389427200d6a15029"
      }
    ],
    "writes": "calculated-inputs.json"
  },
  "hmmlearn": {
    "title": "The same four reports through hmmlearn 0.3.3",
    "question": "What exactly does each returned number mean, and which one is not a log probability?",
    "file": "hmmlearn-examples.py",
    "setup": "python -m venv hmm-optional\nhmm-optional/Scripts/python -m pip install \"hmmlearn==0.3.3\"\nhmm-optional/Scripts/python hmmlearn-examples.py",
    "language": "python",
    "download": "/learn-assets/hmm/hmmlearn-examples.py",
    "code": "\"\"\"Optional hmmlearn 0.3.3 examples; dependency unavailable during authoring.\n\nRun after installing compatible numpy and hmmlearn in a separate environment.\nNo stdout or native results from this file are claimed until it is executed.\n\"\"\"\nimport numpy as np\nfrom hmmlearn.hmm import CategoricalHMM, GaussianHMM\n\nactivities = np.array([[0], [1], [0], [2]], dtype=int)\nfixed = CategoricalHMM(n_components=2, n_features=3, init_params=\"\", params=\"\")\nfixed.startprob_ = np.array([0.6, 0.4])\nfixed.transmat_ = np.array([[0.7, 0.3], [0.4, 0.6]])\nfixed.emissionprob_ = np.array([[0.1, 0.4, 0.5], [0.6, 0.3, 0.1]])\nprint(\"Observation log probability:\", fixed.score(activities))\njoint_log_score, path = fixed.decode(activities, algorithm=\"viterbi\")\nprint(\"Best joint log score and path:\", joint_log_score, path)\nprint(\"Smoothed marginals:\", fixed.predict_proba(activities))\nexpected_correct_count, pointwise = fixed.decode(activities, algorithm=\"map\")\nprint(\"Pointwise expected correct count and modes:\", expected_correct_count, pointwise)\nassert np.isclose(fixed.score(activities), np.log(0.00933936))\nassert np.array_equal(path, [1, 1, 1, 0])\n\n# Store independent recordings together, but retain their boundaries.\nrecordings = [np.array([[0], [1]]), np.array([[0], [2]])]\njoined = np.concatenate(recordings)\nlengths = [len(recording) for recording in recordings]\nlearned = CategoricalHMM(n_components=2, n_features=3, n_iter=30,\n                         tol=1e-6, random_state=7)\nlearned.fit(joined, lengths)\nprint(\"Fitted independent-sequence score:\", learned.score(joined, lengths))\nprint(\"Training history (check budget and objective):\", list(learned.monitor_.history))\n\n# A constructed scalar sensor: continuous emissions, two independent recordings.\nrng = np.random.default_rng(8)\nsensor_recordings = []\nfor _ in range(2):\n    hidden = 0\n    readings = []\n    for time in range(60):\n        if time and rng.random() > 0.9:\n            hidden = 1 - hidden\n        readings.append([rng.normal(-1.5 if hidden == 0 else 1.5, 0.4)])\n    sensor_recordings.append(np.asarray(readings))\nsensor = np.concatenate(sensor_recordings)\nsensor_lengths = [len(recording) for recording in sensor_recordings]\ngaussian = GaussianHMM(n_components=2, covariance_type=\"diag\", n_iter=50,\n                       tol=1e-5, random_state=9)\ngaussian.fit(sensor, sensor_lengths)\nprint(\"Sensor log density:\", gaussian.score(sensor, sensor_lengths))\nprint(\"Means (state indices have no fixed meaning):\", gaussian.means_.ravel())\nprint(\"Variances:\", gaussian.covars_)\nprint(\"First recording smoothed probabilities:\",\n      gaussian.predict_proba(sensor_recordings[0])[:5])",
    "sha256": "991ccca13385eb983d1364fb736bc4a5ac2640a05ef375bab5dae257fb8f033e",
    "expected": "Observation log probability: -4.673517551573099\nBest joint log score and path: -5.773003943698154 [1 1 1 0]\nSmoothed marginals: [[0.19494269 0.80505731]\n [0.43382844 0.56617156]\n [0.23631598 0.76368402]\n [0.8051087  0.1948913 ]]\nPointwise expected correct count and modes: 2.940021586061572 [1 1 1 0]\nFitted independent-sequence score: -1.3862943611198906\nTraining history (check budget and objective): [-4.1113936445247035, -2.406285658870865, -1.6977983129293435, -1.4124336007892535, -1.386466292948082, -1.3862943685103468, -1.3862943611198906]\nSensor log density: -103.6618400298901\nMeans (state indices have no fixed meaning): [ 1.52034801 -1.47503011]\nVariances: [[[0.17103118]]\n\n [[0.19913031]]]\nFirst recording smoothed probabilities: [[6.12029779e-77 1.00000000e+00]\n [9.99999208e-01 7.91567342e-07]\n [9.99996523e-01 3.47712396e-06]\n [9.99999999e-01 7.38995814e-10]\n [1.00000000e+00 7.26447366e-15]]",
    "warning": "Fitting a model with 7 free scalar parameters with only 4 data points will result in a degenerate solution.",
    "executed": true,
    "environment": {
      "python": "3.12.14",
      "hmmlearn": "0.3.3",
      "numpy": "2.5.3",
      "scipy": "1.18.1",
      "scikit-learn": "1.9.1"
    },
    "environmentNote": "Run in an isolated environment, not in this project's shared lesson runtime: resolving hmmlearn moves NumPy, and other completed lessons' recorded outputs depend on the exact versions there."
  }
};
