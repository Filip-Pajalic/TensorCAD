"""Hold a packed row to what its documents compute alone.

The row is packed the way Hugging Face's DataCollatorWithFlattening packs one
(transformers/data/data_collator.py): whole examples laid end to end, their
input_ids concatenated and their position_ids each counted from zero. The
documents are where position_ids is zero, which is how FlashAttention's
variable-length kernel is given them.

Then the generated model, keeping documents apart, is run three ways: on the
packed row with those positions, on the packed row with positions counted
straight through, and on each document by itself. The runtime's own
positions_of is held against the collator's position_ids on the way.

Arguments: <model.py> <T>.
"""

import importlib.util
import inspect
import json
import sys

import torch

from tensorcad_runtime.packing import positions_of

spec = importlib.util.spec_from_file_location("gen", sys.argv[1])
gen = importlib.util.module_from_spec(spec)
sys.modules["gen"] = gen
spec.loader.exec_module(gen)
T = int(sys.argv[2])


def flatten(features):
    """DataCollatorWithFlattening, the parts that make a row: input_ids end to
    end, and position_ids counted from zero in each example."""
    ret = {"input_ids": [], "position_ids": []}
    for feature in features:
        ret["input_ids"] += feature["input_ids"]
        ret["position_ids"] += list(range(len(feature["input_ids"])))
    return {k: torch.tensor([v]) for k, v in ret.items()}


torch.manual_seed(0)
cls = next(
    c
    for c in vars(gen).values()
    if isinstance(c, type) and issubclass(c, torch.nn.Module) and c.__module__ == "gen" and "init_weights" in vars(c)
)
model = cls().init_weights().eval()
vocab = model.embed.num_embeddings
names = list(inspect.signature(model.forward).parameters)

lengths = [37, 80, 11, 64]
lengths.append(T - sum(lengths))
features = [{"input_ids": torch.randint(0, vocab, (n,)).tolist()} for n in lengths]
row = flatten(features)
tokens, positions = row["input_ids"], row["position_ids"]
documents = torch.cumsum(positions == 0, dim=-1) - 1


def run(tokens, documents, positions):
    given = {"tokens": tokens, "docs": documents, "positions": positions}
    with torch.no_grad():
        return model(*[given[n] for n in names])


out = {"matches_collator": bool(torch.equal(positions_of(documents), positions))}
restarted = run(tokens, documents, positions)
continued = run(tokens, documents, torch.arange(T).unsqueeze(0))
worst_restarted = worst_continued = 0.0
start = 0
for n, feature in zip(lengths, features):
    alone = run(torch.tensor([feature["input_ids"]]), torch.zeros(1, n, dtype=torch.long), torch.arange(n).unsqueeze(0))
    worst_restarted = max(worst_restarted, float((restarted[0, start : start + n] - alone[0]).abs().max()))
    worst_continued = max(worst_continued, float((continued[0, start : start + n] - alone[0]).abs().max()))
    start += n
out["restarted_vs_alone"] = worst_restarted
out["continued_vs_alone"] = worst_continued
out["scale"] = float(restarted.abs().max())
print(json.dumps(out))
