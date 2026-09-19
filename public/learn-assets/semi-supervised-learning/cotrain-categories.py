"""Complete categorical co-training mechanism; authored fixture, not a benchmark."""
from collections import defaultdict


def learn_rules(categories, labels):
    observed = defaultdict(set)
    for category, label in zip(categories, labels):
        if label is not None:
            observed[category].add(label)
    return {category: next(iter(values)) for category, values in observed.items()
            if len(values) == 1}


def cotrain(rows, initial, max_rounds=8):
    views = [[row[view] for row in rows] for view in [0, 1]]
    labels = [initial.copy(), initial.copy()]
    history = []
    for iteration in range(1, max_rounds + 1):
        rules = [learn_rules(views[v], labels[v]) for v in [0, 1]]
        predicted = [[rules[v].get(category) for category in views[v]] for v in [0, 1]]
        offers = []
        conflicts = []
        for row in range(len(rows)):
            left, right = predicted[0][row], predicted[1][row]
            if left is not None and right is not None and left != right:
                conflicts.append(row)
                continue
            for donor in [0, 1]:
                recipient = 1 - donor
                value = predicted[donor][row]
                if value is not None and labels[recipient][row] is None:
                    offers.append({"row": row, "donor": donor, "recipient": recipient, "label": value})
        # Both learners predict before either receives a label.
        for offer in offers:
            labels[offer["recipient"]][offer["row"]] = offer["label"]
        history.append({"round": iteration, "rules_before": rules, "offers": offers,
                        "conflicts": conflicts, "labels_after": [list(v) for v in labels]})
        if not offers:
            break
    return {"history": history, "labels": labels,
            "rules": [learn_rules(views[v], labels[v]) for v in [0, 1]]}


ROWS = [("red", "round"), ("blue", "square"), ("red", "triangle"),
        ("green", "triangle"), ("orange", "square"), ("orange", "hexagon"),
        ("red", "square")]
INITIAL = [0, 1, None, None, None, None, None]


if __name__ == "__main__":
    result = cotrain(ROWS, INITIAL)
    for step in result["history"]:
        print("round", step["round"], "offers",
              [(v["row"], v["donor"] + 1, v["recipient"] + 1, v["label"]) for v in step["offers"]],
              "conflicts", step["conflicts"])
    print("rules", result["rules"])
    duplicate = cotrain([(left, left) for left, _ in ROWS], INITIAL)
    print("duplicate-view unresolved", [i for i, y in enumerate(duplicate["labels"][0]) if y is None])
