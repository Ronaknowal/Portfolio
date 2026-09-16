"""CPU attentive inflection: exact shared data, two explicit decoder orders."""
from pathlib import Path
import csv
import json
import platform
import string
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

ROOT = Path(__file__).resolve().parent
TOKENS = ["<pad>", "<bos>", "<eos>", "<past>", "<participle>", "<third_person>"] + list(string.ascii_lowercase)
INDEX = {token: index for index, token in enumerate(TOKENS)}
PAD, BOS, EOS = 0, 1, 2
ALLOWED = [EOS] + list(range(6, len(TOKENS)))
MAX_OUTPUT = 16

def load_records():
    with (ROOT/"english-inflections.csv").open(encoding="utf-8", newline="") as stream:
        rows = list(csv.DictReader(stream))
    train = [row for row in rows if row["partition"] == "train"]
    development = [row for row in rows if row["partition"] == "development"]
    assert (len(train), len(development)) == (1353, 447)
    for field in ("lemma", "form"):
        assert not {row[field] for row in train} & {row[field] for row in development}
    return train, development

def source_batch(rows):
    sources = []
    for row in rows:
        lemma, feature = row["lemma"], row["feature"]
        if not 1 <= len(lemma) <= 32 or any(ch not in string.ascii_lowercase for ch in lemma):
            raise ValueError("Use 1 to 32 lowercase a-z characters.")
        if feature not in ("past", "participle", "third_person"):
            raise ValueError("Unknown inflection request.")
        sources.append([INDEX[f"<{feature}>"]] + [INDEX[ch] for ch in lemma] + [EOS])
    source = torch.full((len(rows), max(map(len, sources))), PAD, dtype=torch.long)
    for index, ids in enumerate(sources):
        source[index, :len(ids)] = torch.tensor(ids)
    return source, torch.tensor(list(map(len, sources)), dtype=torch.long)

def batch(rows):
    source, lengths = source_batch(rows)
    targets = [[INDEX[ch] for ch in row["form"]] + [EOS] for row in rows]
    target = torch.full((len(rows), max(map(len, targets))), PAD, dtype=torch.long)
    for index, ids in enumerate(targets):
        target[index, :len(ids)] = torch.tensor(ids)
    previous = torch.cat([torch.full((len(rows), 1), BOS), target[:, :-1]], dim=1)
    return source, lengths, previous, target

class AttentiveInflector(nn.Module):
    def __init__(self, kind="additive", input_feeding=False):
        super().__init__()
        if kind not in ("additive", "general"):
            raise ValueError("Choose additive or general.")
        if kind == "additive" and input_feeding:
            raise ValueError("Input feeding here names only the Luong extension.")
        self.kind, self.input_feeding = kind, input_feeding
        self.embedding = nn.Embedding(len(TOKENS), 24, padding_idx=PAD)
        self.encoder = nn.GRU(24, 64, batch_first=True)
        extra = 64 if kind == "additive" or input_feeding else 0
        self.decoder = nn.GRUCell(24+extra, 64)
        self.key_projection = nn.Linear(64, 32 if kind == "additive" else 64, bias=False)
        if kind == "additive":
            self.query_projection = nn.Linear(64, 32, bias=False)
            self.score_projection = nn.Linear(32, 1, bias=False)
        self.combine = nn.Linear(128, 64)
        self.readout = nn.Linear(64, len(TOKENS))
        invalid = torch.ones(len(TOKENS), dtype=torch.bool)
        invalid[ALLOWED] = False
        self.register_buffer("invalid_output", invalid)

    def encode(self, source, lengths):
        packed = pack_padded_sequence(self.embedding(source), lengths, batch_first=True, enforce_sorted=False)
        output, final = self.encoder(packed)
        memory, _ = pad_packed_sequence(output, batch_first=True, total_length=source.shape[1])
        valid = torch.arange(source.shape[1])[None, :] < lengths[:, None]
        # Memory, projected keys and mask are source-only; cache once for decoding.
        cache = (memory, self.key_projection(memory), valid)
        return cache, final[0], torch.zeros_like(final[0])

    def attend(self, query, cache):
        memory, keys, valid = cache
        if not bool(valid.any(-1).all()):
            raise ValueError("Every source needs at least one valid position.")
        if self.kind == "additive":
            scores = self.score_projection(torch.tanh(keys+self.query_projection(query)[:, None, :])).squeeze(-1)
        else:
            scores = torch.bmm(keys, query[:, :, None]).squeeze(-1)
        scores = scores.masked_fill(~valid, -torch.inf)
        weights = scores.softmax(-1)
        context = torch.bmm(weights[:, None, :], memory).squeeze(1)
        return context, weights, scores

    def step(self, previous, state, fed, cache):
        embedding = self.embedding(previous)
        if self.kind == "additive":
            context, weights, scores = self.attend(state, cache)
            state = self.decoder(torch.cat([embedding, context], dim=-1), state)
        else:
            decoder_input = torch.cat([embedding, fed], dim=-1) if self.input_feeding else embedding
            state = self.decoder(decoder_input, state)
            context, weights, scores = self.attend(state, cache)
        attentional = torch.tanh(self.combine(torch.cat([state, context], dim=-1)))
        logits = self.readout(attentional).masked_fill(self.invalid_output, -torch.inf)
        return logits, state, attentional, weights, context, scores

    def forward(self, source, lengths, previous):
        cache, state, fed = self.encode(source, lengths)
        logits, weights = [], []
        for token in previous.unbind(1):
            output, state, fed, attention, _, _ = self.step(token, state, fed, cache)
            logits.append(output)
            weights.append(attention)
        return torch.stack(logits, 1), torch.stack(weights, 1)

def edit_distance(left, right):
    previous = list(range(len(right)+1))
    for row, a in enumerate(left, 1):
        current = [row]
        for column, b in enumerate(right, 1):
            current.append(min(current[-1]+1, previous[column]+1, previous[column-1]+(a != b)))
        previous = current
    return previous[-1]

@torch.no_grad()
def greedy(model, rows, max_output=MAX_OUTPUT):
    model.eval()
    source, lengths = source_batch(rows)
    cache, state, fed = model.encode(source, lengths)
    previous = torch.full((len(rows),), BOS, dtype=torch.long)
    finished = torch.zeros(len(rows), dtype=torch.bool)
    outputs, logs = [[] for _ in rows], [[] for _ in rows]
    for _ in range(max_output):
        logits, state, fed, _, _, _ = model.step(previous, state, fed, cache)
        log_probabilities = logits.log_softmax(-1)
        chosen = log_probabilities.argmax(-1)
        for index in range(len(rows)):
            if not finished[index]:
                token = int(chosen[index])
                outputs[index].append(token)
                logs[index].append(float(log_probabilities[index, token]))
        finished |= chosen == EOS
        previous = torch.where(finished, EOS, chosen)
        if bool(finished.all()):
            break
    return [{"prediction": "".join(TOKENS[token] for token in output if token != EOS),
             "tokens": output, "ended_with_eos": bool(output and output[-1] == EOS),
             "log_probability": sum(values)} for output, values in zip(outputs, logs)]

def summarize(rows, predictions):
    errors = [edit_distance(row["form"], item["prediction"]) for row, item in zip(rows, predictions)]
    return {"count": len(rows), "exact": sum(error == 0 and item["ended_with_eos"] for error, item in zip(errors, predictions)),
            "character_edits": sum(errors), "reference_characters": sum(len(row["form"]) for row in rows),
            "character_error_rate": sum(errors)/sum(len(row["form"]) for row in rows),
            "no_eos": sum(not item["ended_with_eos"] for item in predictions)}

@torch.no_grad()
def assess(model, rows):
    model.eval()
    source, lengths, previous, target = batch(rows)
    logits, _ = model(source, lengths, previous)
    loss = F.cross_entropy(logits.flatten(0, 1), target.flatten(), ignore_index=PAD)
    predictions = greedy(model, rows)
    return {"teacher_forced_nll": float(loss), **summarize(rows, predictions)}, predictions

def main():
    torch.set_num_threads(1)
    train, development = load_records()
    report = {"versions": {"python": platform.python_version(), "torch": torch.__version__, "numpy": np.__version__},
              "protocol": {"kinds": ["additive", "general"], "seeds": [1, 2, 3], "input_feeding": False,
              "updates": 1200, "batch_size": 64, "adam_lr": .003, "gradient_clip": 1., "max_output": MAX_OUTPUT},
              "tokens": TOKENS, "runs": []}
    for kind in report["protocol"]["kinds"]:
        for seed in report["protocol"]["seeds"]:
            torch.manual_seed(seed)
            model = AttentiveInflector(kind)
            optimizer = torch.optim.Adam(model.parameters(), lr=.003)
            generator = torch.Generator().manual_seed(100+seed)
            checkpoints, clipped = [], 0
            for update in range(1201):
                if update in (0, 100, 400, 800, 1200):
                    metrics, _ = assess(model, development)
                    checkpoints.append({"update": update, **metrics})
                    print(json.dumps({"kind": kind, "seed": seed, **checkpoints[-1]}), flush=True)
                if update == 1200:
                    break
                model.train()
                chosen = torch.randint(len(train), (64,), generator=generator).tolist()
                source, lengths, previous, target = batch([train[index] for index in chosen])
                optimizer.zero_grad(set_to_none=True)
                logits, _ = model(source, lengths, previous)
                loss = F.cross_entropy(logits.flatten(0, 1), target.flatten(), ignore_index=PAD)
                loss.backward()
                norm = nn.utils.clip_grad_norm_(model.parameters(), 1.)
                clipped += float(norm) > 1
                optimizer.step()
            training, _ = assess(model, train)
            metrics, predictions = assess(model, development)
            report["runs"].append({"kind": kind, "seed": seed, "parameters": sum(p.numel() for p in model.parameters()),
                "clipped_updates": clipped, "checkpoints": checkpoints, "train": training, "development": metrics,
                "development_predictions": [{"lemma": row["lemma"], "feature": row["feature"], "reference": row["form"], **item}
                                            for row, item in zip(development, predictions)],
                "weights": {key: value.tolist() for key, value in model.state_dict().items()}})
            (ROOT/"calculated-inputs.json").write_text(json.dumps(report, indent=2)+"\n", encoding="utf-8")
            print(json.dumps({"kind": kind, "seed": seed, "train": training, "development": metrics}), flush=True)

if __name__ == "__main__":
    main()

