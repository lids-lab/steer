# models/rt: Relational Transformer (RT-12) with STEER

STEER is implemented inside the sampler of this model. This directory is an
inference-only copy of the [Relational Transformer](https://github.com/stanford-star/relational-transformer)
(MIT, see [LICENSE](LICENSE)) with STEER's traversal and column filters added.
Training, preprocessing and embedding-generation code were removed.

You normally don't call anything here directly: `steer run --model rt ...`
from the repository root builds a sampler config and calls `steer_eval.py`.

## Contents

- `model.py`, `data.py`: the architecture and the sampler-backed dataset.
- `evaluate.py`, `steer_eval.py`: load a checkpoint, evaluate one task's test
  split, and write the metric and average token count as JSON.
- `rustler/`: the Rust sampler (`src/fly.rs`) that builds each context.

## What STEER changes

All of it is in the sampler, `rustler/src/fly.rs`, and in `data.py`, which passes
the config path down to it:

- The sampler reads a per-task JSON config: which filters are on, the tier of
  each edge and column, and a keep probability per tier. The path is passed
  explicitly from Python to Rust.
- `p2f_edge_probabilistic_at_push_keep_probability`,
  `f2p_edge_probabilistic_at_push_keep_probability` and
  `column_filter_keep_probability` map an edge's or column's tier to its keep
  probability.
- Where the sampler adds a neighbor to the frontier (or writes a column), a coin
  flip against that probability decides whether it is kept. This is
  Algorithm 1 / `SteerFilter` in the paper.

## Environment

```bash
cd models/rt
pixi install
pixi run build-sampler     # compiles the Rust sampler (needs a few minutes)
```

Checkpoints and preprocessed data are fetched by `steer setup` (see the top-level README).
