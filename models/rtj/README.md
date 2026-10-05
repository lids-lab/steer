# models/rtj: RT-J with STEER

STEER is implemented inside the sampler of this model. This directory is an
inference-only copy of RT-J from the
[Relational Transformer repository](https://github.com/stanford-star/relational-transformer)
(MIT, see [LICENSE](LICENSE)) with STEER's traversal and column filters added.
Training, preprocessing and embedding-generation code were removed.

You normally don't call anything here directly: `steer run --model rtj ...`
from the repository root builds a sampler config and calls `steer_eval.py`.

RT-J is zero-shot: one classification checkpoint and one regression checkpoint
cover every task.

## Contents

- `rt/`: the model package (`model.py`, `data.py`, `evaluator.py`,
  `eval_utils.py`, `checkpoints.py`, and `pre.py`, which resolves the
  preprocessed-data and checkpoint locations).
- `steer_eval.py`: evaluates one task's test split and writes the metric and
  average token count as JSON. Context sizes are fixed (outer buffer 8192,
  local context 1024).
- `rustler/`: the Rust sampler (`src/fly.rs`).

## What STEER changes

All of it is in the sampler, `rustler/src/fly.rs`:

- The sampler reads a per-task JSON config (`load_task_config`,
  `resolve_task_config`): which filters are on, the tier of each edge and
  column, and a keep probability per tier. The config path is passed explicitly
  from Python to Rust.
- `p2f_edge_probabilistic_at_push_keep_probability`,
  `f2p_edge_probabilistic_at_push_keep_probability` and
  `column_filter_keep_probability` map an edge's or column's tier to its keep
  probability.
- Where the sampler adds a neighbor to the frontier (or writes a column), a coin
  flip against that probability decides whether it is kept. This is
  Algorithm 1 / `SteerFilter` in the paper.

## Environment

```bash
cd models/rtj
pixi install
pixi run build-sampler     # compiles the Rust sampler
```

Checkpoints and preprocessed data are fetched by `steer setup` (see the top-level README).
