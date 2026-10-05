# models/rtj: RT-J with STEER

STEER is implemented inside the sampler of this model. This directory is an
inference-only copy of RT-J from the
[Relational Transformer repository](https://github.com/stanford-star/relational-transformer)
(MIT, see [LICENSE](LICENSE)) with STEER's traversal and column filters added.
It keeps the sampler, the model's forward pass and the evaluation loop;
training, preprocessing, ablation, visualization and benchmarking code were removed.

You normally don't call anything here directly: `steer run --model rtj ...`
from the repository root builds a sampler config and calls `steer_eval.py`.

RT-J is zero-shot: one classification checkpoint and one regression checkpoint
cover every task.

## Contents

- `rt/`: the model package: `model.py` (forward pass), `data.py` (wraps the
  sampler), `evaluator.py` (runs the model over a task's items),
  `eval_utils.py` (metric and evaluator construction) and `checkpoints.py`
  (loads a local checkpoint directory).
- `steer_eval.py`: evaluates one task's test split and writes the metric and
  average token count as JSON. The outer context buffer is fixed at 8192 cells;
  the local (BFS) context defaults to 1024 (`--local-ctx-size`).
- `rustler/`: the Rust sampler (`src/fly.rs`). Each context is the target cell
  followed by one BFS around the target row, capped at the local context size;
  the buffer is not topped up with other rows.

## What STEER changes

All of it is in the sampler, `rustler/src/fly.rs`:

- The sampler reads a per-task JSON config (`load_task_config`,
  `resolve_task_config`): which filters are on, the tier of each edge and
  column, and a keep probability per tier. The config path is passed explicitly
  from Python to Rust, and `keep_probabilities` is required in it.
- `p2f_edge_probabilistic_at_push_keep_probability`,
  `f2p_edge_probabilistic_at_push_keep_probability` and
  `column_filter_keep_probability` map an edge's or column's tier to its keep
  probability.
- Where the sampler adds a neighbor to the frontier (or writes a column), a coin
  flip against that probability decides whether it is kept (the `// STEER:`
  comments in `bfs_collect_nodes` and `seq_build` mark these three places).
  This is Algorithm 1 / `SteerFilter` in the paper.

## Environment

```bash
cd models/rtj
pixi install
pixi run build-sampler     # compiles the Rust sampler
```

Checkpoints and preprocessed data are fetched by `steer setup` (see the top-level README).
