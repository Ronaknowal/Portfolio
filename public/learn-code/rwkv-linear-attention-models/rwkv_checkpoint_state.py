"""Optional local-checkpoint state continuation using the official rwkv API.

No download, fine-tuning or benchmark. Supply a trusted checkpoint and its tokenizer.
Contract inspected 22 September 2026; this optional package program is unexecuted.
"""
import argparse
import copy
import importlib.metadata
import os
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--tokenizer", required=True, help="local tokenizer JSON, or rwkv_vocab_v20230424 for a matching World model")
    parser.add_argument("--generation", choices=("4", "7"), required=True)
    parser.add_argument("--text", default="A memory carries information forward.")
    args = parser.parse_args()
    if not args.checkpoint.is_file():
        raise ValueError("Supply the local checkpoint .pth file")
    if args.tokenizer != "rwkv_vocab_v20230424" and not Path(args.tokenizer).is_file():
        raise ValueError("Supply the matching local tokenizer JSON")
    os.environ["RWKV_V7_ON"] = "1" if args.generation == "7" else "0"
    os.environ["RWKV_CUDA_ON"] = "0"
    os.environ["RWKV_JIT_ON"] = "0"
    import torch
    from rwkv.model import RWKV
    from rwkv.utils import PIPELINE
    model = RWKV(model=args.checkpoint.as_posix(), strategy="cpu fp32")
    tokenizer = PIPELINE(model, args.tokenizer)
    tokens = tokenizer.encode(args.text)
    if not 2 <= len(tokens) <= 128:
        raise ValueError("Use a short prompt producing 2 to 128 tokens for this continuation probe")
    split = len(tokens)//2
    with torch.no_grad():
        full, _ = model.forward(tokens, None)
        _, prefix = model.forward(tokens[:split], None)
        # The API may mutate state: branch from an independent copy.
        chunked, state = model.forward(tokens[split:], copy.deepcopy(prefix))
        tokenwise, token_state = None, copy.deepcopy(prefix)
        for token in tokens[split:]:
            tokenwise, token_state = model.forward([token], token_state)
    torch.testing.assert_close(full, chunked, atol=3e-3, rtol=3e-3)
    torch.testing.assert_close(full, tokenwise, atol=3e-3, rtol=3e-3)
    print("rwkv package:", importlib.metadata.version("rwkv"))
    print("torch:", torch.__version__, "tokens:", len(tokens), "state tensors:", len(state))
    print("chunk maximum logit error:", float((full-chunked).abs().max()))
    print("tokenwise maximum logit error:", float((full-tokenwise).abs().max()))
    print("state bytes:", sum(t.numel()*t.element_size() for t in state))


if __name__ == "__main__":
    main()
