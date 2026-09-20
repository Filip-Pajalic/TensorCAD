"""Count a model's parameters from its safetensors headers, without the weights.

Every safetensors file begins with an 8-byte little-endian header length and
then a JSON header naming each tensor and its shape. A range request for the
first few hundred kilobytes is enough to read it, which is how a 55 GB model's
exact parameter count costs a few seconds.
"""

import json
import struct
import sys
import urllib.request

REPO = sys.argv[1]
PREFIX = sys.argv[2] if len(sys.argv) > 2 else ""


def get(url: str, start: int | None = None, end: int | None = None) -> bytes:
    req = urllib.request.Request(url)
    if start is not None:
        req.add_header("Range", "bytes=%d-%d" % (start, end))
    with urllib.request.urlopen(req, timeout=60) as r:
        return r.read()


base = "https://huggingface.co/%s/resolve/main/" % REPO
index = json.loads(get(base + "model.safetensors.index.json"))
shards = sorted(set(index["weight_map"].values()))

total = 0
matched = 0
by_group: dict[str, int] = {}
for shard in shards:
    url = base + shard
    size = struct.unpack("<Q", get(url, 0, 7))[0]
    header = json.loads(get(url, 8, 8 + size - 1))
    for name, meta in header.items():
        if name == "__metadata__":
            continue
        n = 1
        for d in meta["shape"]:
            n *= d
        total += n
        if PREFIX and not name.startswith(PREFIX):
            continue
        matched += n
        # Group by the part of the name that is not a layer index.
        parts = [p for p in name.split(".") if not p.isdigit()]
        by_group[".".join(parts[:4])] = by_group.get(".".join(parts[:4]), 0) + n

print("shards:", len(shards))
print("total:  %,d" % total if False else "total:  {:,}".format(total))
if PREFIX:
    print("%s*: {:,}".format(matched) % PREFIX)
    for k, v in sorted(by_group.items(), key=lambda kv: -kv[1])[:12]:
        print("   {:>16,}  {}".format(v, k))
