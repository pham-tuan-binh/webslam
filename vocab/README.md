# vocab

The DBoW-style vocabulary artifact used by L4 place recognition.

spec.md §7: *"DBoW2 is small. The vocabulary is the artifact; the code is a tree
search over binary descriptors. Reimplementable in days, and the trained
vocabulary file is reusable as data."*

## What is here

| File | What |
|---|---|
| `wslam-vocab-v1.bin` | Trained vocabulary. **git-lfs**, not a plain blob. |
| `wslam-vocab-v1.json` | Training provenance: corpus, seed, branching, depth. |

## Why we train our own

Not a licence question — a compatibility one. Our binary descriptor is not
bit-compatible with ORB's, so an ORB vocabulary would cluster our descriptors
into meaningless words. The tree is only useful if it was built from the same
descriptor definition that will query it.

## Retraining

```sh
# Dump descriptors from a representative corpus, then:
cargo xtask train-vocab <descriptors.bin> --branching 10 --depth 5 --seed 20260801
```

The seed is recorded in the provenance JSON and the training is deterministic,
so a vocabulary can be reproduced exactly from its metadata. That matters
because place-recognition recall shifts when the vocabulary shifts, and an
unreproducible vocabulary makes a recall regression impossible to bisect.

**Retraining changes recall.** Re-run the false-positive measurement
(spec.md §6 L4) before shipping a new vocabulary — it is a release gate, and a
vocabulary swap is exactly the kind of change that quietly moves it.

## Corpus

The corpus should span the scenes we expect, not every scene on earth. A
vocabulary trained on drone footage will underperform indoors. spec.md §5 makes
the same point about learned models: *"compression comes from narrowing the
domain."*
