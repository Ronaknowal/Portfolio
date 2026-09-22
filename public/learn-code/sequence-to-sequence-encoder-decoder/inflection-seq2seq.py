"""Complete CPU character encoder-decoder on a fixed, lemma-disjoint real extract."""
from pathlib import Path
import csv
import json
import platform
import string
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torch.nn.utils.rnn import pack_padded_sequence

ROOT = Path(__file__).resolve().parent
TOKENS = ["<pad>", "<bos>", "<eos>", "<past>", "<participle>", "<third_person>"] + list(string.ascii_lowercase)
INDEX = {token: index for index, token in enumerate(TOKENS)}
PAD, BOS, EOS = 0, 1, 2
ALLOWED = [EOS] + list(range(6, len(TOKENS)))
MAX_OUTPUT = 16  # Generated tokens, including EOS; BOS is not counted.


def load_records():
    with (ROOT/"english-inflections.csv").open(encoding="utf-8", newline="") as stream:
        records = list(csv.DictReader(stream))
    train = [row for row in records if row["partition"] == "train"]
    development = [row for row in records if row["partition"] == "development"]
    assert len(train) == 1353 and len(development) == 447
    assert not set(row["lemma"] for row in train) & set(row["lemma"] for row in development)
    assert not set(row["form"] for row in train) & set(row["form"] for row in development)
    return train, development


def source_ids(lemma, feature):
    if not lemma or any(letter not in string.ascii_lowercase for letter in lemma):
        raise ValueError("Use a nonempty lower-case a-z lemma.")
    return [INDEX[f"<{feature}>"]] + [INDEX[letter] for letter in lemma] + [EOS]


def source_batch(records):
    sources = [source_ids(row["lemma"], row["feature"]) for row in records]
    source = torch.full((len(records), max(map(len, sources))), PAD, dtype=torch.long)
    for index, ids in enumerate(sources):
        source[index, :len(ids)] = torch.tensor(ids)
    return source, torch.tensor(list(map(len, sources)), dtype=torch.long)


def batch(records):
    source, lengths = source_batch(records)
    targets = [[INDEX[letter] for letter in row["form"]] + [EOS] for row in records]
    target = torch.full((len(records), max(map(len, targets))), PAD, dtype=torch.long)
    for index, y in enumerate(targets):
        target[index, :len(y)] = torch.tensor(y)
    decoder_input = torch.cat([torch.full((len(records), 1), BOS), target[:, :-1]], dim=1)
    return source, lengths, decoder_input, target


class Inflector(nn.Module):
    def __init__(self):
        super().__init__()
        self.embedding = nn.Embedding(len(TOKENS), 24, padding_idx=PAD)
        self.encoder = nn.GRU(24, 64, batch_first=True)
        self.decoder = nn.GRU(24, 64, batch_first=True)
        self.readout = nn.Linear(64, len(TOKENS))
        invalid = torch.ones(len(TOKENS), dtype=torch.bool)
        invalid[ALLOWED] = False
        self.register_buffer("invalid_output", invalid)

    def encode(self, source, lengths):
        packed = pack_padded_sequence(self.embedding(source), lengths, batch_first=True, enforce_sorted=False)
        _, final = self.encoder(packed)
        return final

    def output_logits(self, hidden):
        return self.readout(hidden).masked_fill(self.invalid_output, -torch.inf)

    def forward(self, source, lengths, decoder_input):
        state = self.encode(source, lengths)
        hidden, _ = self.decoder(self.embedding(decoder_input), state)
        return self.output_logits(hidden)

    def step(self, previous_token, state):
        hidden, state = self.decoder(self.embedding(previous_token[:, None]), state)
        return self.output_logits(hidden[:, 0]), state


def edit_distance(left, right):
    previous = list(range(len(right)+1))
    for row, a in enumerate(left, 1):
        current = [row]
        for column, b in enumerate(right, 1):
            current.append(min(current[-1]+1, previous[column]+1, previous[column-1]+(a != b)))
        previous = current
    return previous[-1]


def suffix_baseline(lemma, feature):
    consonant_y = lemma.endswith("y") and lemma[-2] not in "aeiou"
    if feature == "past":
        return lemma[:-1]+"ied" if consonant_y else lemma+"d" if lemma.endswith("e") else lemma+"ed"
    if feature == "participle":
        if lemma.endswith("ie"):
            return lemma[:-2]+"ying"
        return lemma[:-1]+"ing" if lemma.endswith("e") and not lemma.endswith("ee") else lemma+"ing"
    if consonant_y:
        return lemma[:-1]+"ies"
    return lemma+"es" if lemma.endswith(("s", "x", "z", "ch", "sh")) else lemma+"s"


@torch.no_grad()
def greedy(model, records, max_output=MAX_OUTPUT):
    model.eval()
    source, lengths = source_batch(records)
    state = model.encode(source, lengths)
    previous = torch.full((len(records),), BOS, dtype=torch.long)
    finished = torch.zeros(len(records), dtype=torch.bool)
    outputs, traces = [[] for _ in records], [[] for _ in records]
    for _ in range(max_output):
        logits, state = model.step(previous, state)
        log_probabilities = logits.log_softmax(-1)
        next_token = log_probabilities.argmax(-1)
        for index in range(len(records)):
            if not finished[index]:
                token = int(next_token[index])
                outputs[index].append(token)
                traces[index].append(float(log_probabilities[index, token]))
        finished |= next_token == EOS
        previous = torch.where(finished, EOS, next_token)
        if bool(finished.all()):
            break
    return [{"prediction": "".join(TOKENS[token] for token in output if token != EOS),
             "tokens": output, "ended_with_eos": bool(output and output[-1] == EOS),
             "log_probability": sum(logs), "token_log_probabilities": logs}
            for output, logs in zip(outputs, traces)]


@torch.no_grad()
def beam(model, record, width=3, alpha=0., max_output=MAX_OUTPUT):
    model.eval()
    source, lengths = source_batch([record])
    # Candidate owns tokens, recurrent state after consuming its previous input, and raw logP.
    candidates = [([], model.encode(source, lengths), 0.)]
    def score(item):
        ids, _, log_probability = item
        return log_probability / (((5+len(ids))/6)**alpha)
    for _ in range(max_output):
        expanded = []
        for ids, state, log_probability in candidates:
            if ids and ids[-1] == EOS:
                expanded.append((ids, state, log_probability))
                continue
            previous = torch.tensor([ids[-1] if ids else BOS])
            logits, new_state = model.step(previous, state)
            values = logits.log_softmax(-1)[0]
            for token in ALLOWED:
                expanded.append((ids+[token], new_state, log_probability+float(values[token])))
        candidates = sorted(expanded, key=lambda item: (-score(item), item[0]))[:width]
        if all(ids and ids[-1] == EOS for ids, _, _ in candidates):
            break
    ids, _, log_probability = candidates[0]
    return {"prediction": "".join(TOKENS[token] for token in ids if token != EOS), "tokens": ids,
            "ended_with_eos": ids[-1] == EOS, "log_probability": log_probability,
            "ranking_score": score(candidates[0]), "beam_width": width, "alpha": alpha}


def summarize(records, predictions):
    errors = [edit_distance(row["form"], item["prediction"]) for row, item in zip(records, predictions)]
    return {"count": len(records), "exact": sum(error == 0 and item.get("ended_with_eos", True) for error, item in zip(errors, predictions)),
            "character_edits": sum(errors), "reference_characters": sum(len(row["form"]) for row in records),
            "character_error_rate": sum(errors)/sum(len(row["form"]) for row in records),
            "no_eos": sum(not item.get("ended_with_eos", True) for item in predictions)}


@torch.no_grad()
def assess(model, records):
    source, lengths, decoder_input, target = batch(records)
    logits = model(source, lengths, decoder_input)
    loss = F.cross_entropy(logits.flatten(0, 1), target.flatten(), ignore_index=PAD)
    predictions = greedy(model, records)
    return {"teacher_forced_nll": float(loss), **summarize(records, predictions)}, predictions


def main():
    torch.set_num_threads(1)
    train, development = load_records()
    report = {"versions": {"python": platform.python_version(), "numpy": np.__version__, "torch": torch.__version__},
              "tokens": TOKENS, "allowed_output_ids": ALLOWED, "protocol": {"seeds": [1, 2, 3], "updates": 1200,
              "batch_size": 64, "adam_lr": .003, "gradient_clip": 1., "max_output": MAX_OUTPUT}, "baselines": {}, "runs": []}
    for partition, records in (("train", train), ("development", development)):
        report["baselines"][partition] = {}
        for name, predictor in (("copy", lambda lemma, feature: lemma), ("predeclared_suffix_rules", suffix_baseline)):
            predictions = [{"prediction": predictor(row["lemma"], row["feature"])} for row in records]
            report["baselines"][partition][name] = summarize(records, predictions)
    for seed in report["protocol"]["seeds"]:
        torch.manual_seed(seed)
        model = Inflector()
        optimizer = torch.optim.Adam(model.parameters(), lr=.003)
        generator = torch.Generator().manual_seed(100+seed)
        checkpoints, clipped = [], 0
        for update in range(1201):
            if update in (0, 100, 400, 800, 1200):
                metrics, _ = assess(model, development)
                checkpoints.append({"update": update, **metrics})
                print(json.dumps({"seed": seed, **checkpoints[-1]}), flush=True)
            if update == 1200:
                break
            model.train()
            chosen = torch.randint(len(train), (64,), generator=generator).tolist()
            source, lengths, decoder_input, target = batch([train[index] for index in chosen])
            optimizer.zero_grad(set_to_none=True)
            logits = model(source, lengths, decoder_input)
            loss = F.cross_entropy(logits.flatten(0, 1), target.flatten(), ignore_index=PAD)
            loss.backward()
            norm = nn.utils.clip_grad_norm_(model.parameters(), 1.)
            clipped += float(norm) > 1
            optimizer.step()
        training, _ = assess(model, train)
        metrics, predictions = assess(model, development)
        details = [{"lemma": row["lemma"], "feature": row["feature"], "reference": row["form"], **item} for row, item in zip(development, predictions)]
        run = {"seed": seed, "parameters": sum(p.numel() for p in model.parameters()), "clipped_updates": clipped,
               "checkpoints": checkpoints, "train": training, "development": metrics,
               "development_predictions": details,
               "weights": {key: value.tolist() for key, value in model.state_dict().items()}}
        if seed == 1:
            beam_predictions = [beam(model, row, width=3) for row in development]
            run["beam3_development"] = summarize(development, beam_predictions)
            run["beam3_predictions"] = beam_predictions
            for row, expected in zip(development[:12], predictions[:12]):
                actual = beam(model, row, width=1)
                assert actual["tokens"] == expected["tokens"]
        report["runs"].append(run)
        (ROOT/"calculated-inputs.json").write_text(json.dumps(report, indent=2)+"\n", encoding="utf-8")
        print(json.dumps({"seed": seed, "parameters": run["parameters"], "train": training, "development": metrics,
                          "beam3": run.get("beam3_development")}), flush=True)


if __name__ == "__main__":
    main()
