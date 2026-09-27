"""Fetch one pinned UniMorph file, audit groups, and retain a licensed small extract."""
from pathlib import Path
from collections import defaultdict, Counter
import csv
import hashlib
import io
import json
import re
import urllib.request

ROOT = Path(__file__).resolve().parent
COMMIT = "66e0e9e8e2dcd196da081a25a48e5c1fe3d8b49b"
URL = f"https://raw.githubusercontent.com/unimorph/eng/{COMMIT}/eng"
EXPECTED = "20a191cefdc7cad6fa74b00f49d6f658684f17b14541aae372e5a3d5a8c15c67"
TAGS = {"V;PST": "past", "V;V.PTCP;PRS": "participle", "V;PRS;3;SG": "third_person"}


def group_shared_forms(records):
    """Keep selected spellings linked by a common target in one conservative group."""
    parent = {row["lemma"]: row["lemma"] for row in records}
    def find(word):
        while parent[word] != word:
            parent[word] = parent[parent[word]]
            word = parent[word]
        return word
    seen = {}
    for row in records:
        if row["form"] in seen:
            parent[find(row["lemma"])] = find(seen[row["form"]])
        seen[row["form"]] = row["lemma"]
    train_groups = {find(row["lemma"]) for row in records if row["partition"] == "train"}
    changed = []
    for row in records:
        revised = "train" if find(row["lemma"]) in train_groups else "development"
        if revised != row["partition"]:
            changed.append({"lemma": row["lemma"], "feature": row["feature"], "form": row["form"]})
        row["partition"] = revised
    return changed


def main():
    with urllib.request.urlopen(URL, timeout=60) as response:
        payload = response.read()
    assert hashlib.sha256(payload).hexdigest() == EXPECTED
    groups = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
    rows, malformed, duplicates = 0, 0, 0
    all_triples = set()
    for line_number, line in enumerate(payload.decode("utf-8").splitlines(), 1):
        if not line.strip():
            continue
        rows += 1
        fields = line.split("\t")
        if len(fields) != 3:
            malformed += 1
            continue
        lemma, form, tag = fields
        triple = tuple(fields)
        duplicates += triple in all_triples
        all_triples.add(triple)
        if tag in TAGS and re.fullmatch("[a-z]{3,8}", lemma) and re.fullmatch("[a-z]{3,12}", form):
            groups[lemma][tag][form].append(line_number)
    ambiguous = sum(any(len(forms) > 1 for forms in group.values()) for group in groups.values())
    eligible = [lemma for lemma, group in groups.items()
                if set(group) == set(TAGS) and all(len(forms) == 1 for forms in group.values())]
    # Hash order is reproducible, independent of outputs and alphabetical/source order.
    eligible.sort(key=lambda word: hashlib.sha256(("seq2seq-inflection-v1:"+word).encode()).hexdigest())
    selected = eligible[:600]
    records = []
    for index, lemma in enumerate(selected):
        partition = "train" if index < 450 else "development"
        for tag, label in TAGS.items():
            form, line_numbers = next(iter(groups[lemma][tag].items()))
            records.append({"partition": partition, "lemma": lemma, "feature": label,
                            "unimorph_features": tag, "form": form,
                            "source_rows": ";".join(map(str, line_numbers))})
    moved_records = group_shared_forms(records)
    output = io.StringIO(newline="")
    writer = csv.DictWriter(output, fieldnames=list(records[0]), lineterminator="\n")
    writer.writeheader()
    writer.writerows(records)
    extracted = output.getvalue().encode()
    (ROOT/"english-inflections.csv").write_bytes(extracted)
    partitions = {name: [row for row in records if row["partition"] == name] for name in ("train", "development")}
    shared_lemmas = set(row["lemma"] for row in partitions["train"]) & set(row["lemma"] for row in partitions["development"])
    assert not shared_lemmas and len(records) == 1800
    report = {"url": URL, "commit": COMMIT, "source_bytes": len(payload), "source_sha256": EXPECTED,
              "nonempty_rows": rows, "malformed_rows": malformed, "duplicate_exact_triple_rows": duplicates,
              "filtered_lemma_groups": len(groups), "ambiguous_filtered_lemma_groups": ambiguous,
              "eligible_complete_unambiguous_lemmas": len(eligible), "selected_lemmas": len(selected),
              "extract_bytes": len(extracted), "extract_sha256": hashlib.sha256(extracted).hexdigest(),
              "partitions": {name: {"rows": len(rows), "lemmas": len(set(row["lemma"] for row in rows)),
                  "lemma_lengths": dict(sorted(Counter(len(row["lemma"]) for row in rows).items()))} for name, rows in partitions.items()},
              "cross_partition_lemma_overlap": len(shared_lemmas),
              "shared_form_group_moved_records": moved_records,
              "cross_partition_target_overlap": len(set(row["form"] for row in partitions["train"]) & set(row["form"] for row in partitions["development"])),
              "license": "CC BY-SA 3.0", "attribution": "UniMorph English repository contributors; source named Wikipedia in the pinned repository README.",
              "readme_url": f"https://github.com/unimorph/eng/blob/{COMMIT}/README.md"}
    (ROOT/"data-extraction.json").write_text(json.dumps(report, indent=2)+"\n", encoding="utf-8")
    print(json.dumps(report, indent=2))
    print("First examples:", records[:9])


if __name__ == "__main__":
    main()
