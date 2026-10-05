# models/griffin: Griffin with STEER

STEER is implemented inside the neighbor sampler of this model. This directory
is an inference-only copy of Griffin, the graph-neural-network relational
foundation model (Wang et al., "Griffin: Towards a Graph-Centric Relational
Database Foundation Model", ICML 2025), with STEER's traversal filter added.
Training and preprocessing code were removed. Attribution and license details
are in the top-level [NOTICE](../../NOTICE).

You normally don't call anything here directly: `steer run --model griffin ...`
from the repository root builds a sampler config and calls `steer_eval.py`.

Griffin represents each row as a single graph node, so the column filter does
not apply to it; only the traversal filter does.

## Contents

- `relfm/griffin/hmodel.py`: the model architecture.
- `relfm/griffin/hdataset.py`, `hdataset_prob.py`, `hloaderwrapper.py`: the
  graph, the neighbor sampler and its batching.
- `relfm/griffin/hFloatEmb.py` and the `float*.pt` files: the float encoder
  and decoder.
- `evaluate.py`, `steer_eval.py`: load a checkpoint, evaluate one task's test
  split, and write the metric and the sampled node and edge counts as JSON.
  Neighbor sampling is seeded (seed 42) so a run is repeatable.

## What STEER changes

All of it is in the sampler, `relfm/griffin/hdataset.py` and `hdataset_prob.py`:

- The sampler reads a per-task JSON config: which filters are on, the tier of
  each edge, and a keep probability per tier. The path is passed explicitly
  through `subgraphargs` from `evaluate.py`.
- In `getedge`, after the fanout cap (fanout 20, 2 hops), each remaining
  candidate edge is kept with the probability of its tier, using a coin flip.
  This is Algorithm 1 / `SteerFilter` in the paper.

## Environment

```bash
cd models/griffin
pixi install     # pure Python, nothing to compile
```

Checkpoints (one per task) and preprocessed data are fetched by `steer setup` (see the top-level README).
