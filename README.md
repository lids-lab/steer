# STEER

Code for **STEER: Reducing Inference Cost in Relational Foundation Models through
Semantically Informed Sampling** (Abdalla Mohamed and Ashraf Aboulnaga, arxiv 2026).

Relational foundation models (RFMs) answer a prediction by sampling a
neighborhood of rows around the target row and reading it as context. STEER
shrinks that context by having an LLM rank the foreign-key edges of the schema
into four relevance tiers for the task, and then following each edge during
sampling with a probability that depends on its tier. This repository contains
STEER's sampler changes for three RFMs (RT, RT-J and Griffin), the shipped
rankings for the 24 RelBench tasks used in the paper, and the command-line
tools to run them.

The three models are the hosts STEER is implemented in; they are not part of
STEER itself. Each lives in `models/<name>/` as an inference-only copy that keeps
the sampler, the STEER changes, and what is needed for a forward pass. See each
directory's README and [NOTICE](NOTICE) for their origin and licenses.

## Requirements

Linux, an NVIDIA GPU, and [pixi](https://pixi.sh). Everything else is installed by pixi.

## Install

```bash
git clone https://github.com/lids-lab/steer && cd steer
pixi install                                  # the STEER command-line tools

# one environment per model you want to use
cd models/rt   && pixi install && pixi run build-sampler && cd ../..   # RT (compiles a Rust sampler)
cd models/rtj  && pixi install && pixi run build-sampler && cd ../..   # RT-J (compiles a Rust sampler)
cd models/griffin && pixi install && cd ../..                          # Griffin (pure Python)
```

## Download checkpoints and data

Checkpoints and preprocessed data are hosted on Hugging Face under
[`abdallam1007`](https://huggingface.co/abdallam1007), except RT-J's checkpoints and its data for
six of the RelBench databases, which come from the RT-J authors' own repositories
(pinned to the revisions our results use). Download what one task needs, or everything for a model:

```bash
pixi run steer setup rt --task rel_f1_driver_dnf      # small: rel-f1 is about 15 MB
pixi run steer setup rt --all                         # everything for RT (about 216 GB)
```

Files shared between tasks (a dataset, an RT database checkpoint, RT-J's two
checkpoints) are downloaded once. `rel_f1_driver_dnf` is a good first task.

## Run STEER

```bash
pixi run steer run --model rt --task rel_f1_driver_dnf
```

By default this applies STEER's traversal filter in both directions
(parent to child and child to parent), with the universal probability profile
and the Claude-sourced ranking shipped for the task. Add `--compare` to also run
the unmodified baseline sampler and print both results (the metric and `Tokens`).

| Option | Meaning |
|---|---|
| `--model {rt,rtj,griffin}` | Which model to run. |
| `--task TASK` | One of the 24 RelBench tasks (see `steer run --help`). |
| `--column-filter` | Also apply the column filter. RT and RT-J only. |
| `--config PATH` | A tiered config (default: `configs/tasks/claude/<task>.json`). |
| `--profile PATH` | A probability profile (default: `configs/profiles/universal.json`). |
| `--compare` | Also run the baseline sampler. |

Each run uses the paper's settings: 1280 evaluation items, a 1024-token context
for RT and RT-J, and fanout 20 with 2 hops for Griffin.

`Tokens` is the average number of tokens in the sampled context. Griffin does not
tokenize its input, so for Griffin it is estimated from the number of sampled
nodes (times the average number of columns per table in the database).

### Other configs and profiles

- **Rankings.** `configs/tasks/claude/` and `configs/tasks/gpt/` hold the
  rankings used in the paper, one file per task, from Claude and GPT. To use the
  GPT ones: `--config configs/tasks/gpt/<task>.json`.
- **Probability profile.** `configs/profiles/universal.json` maps Very Strong,
  Strong, Moderate and Weak to 1.0, 0.6, 0.2 and 0.0, for every task and model.
- **Tuned profiles.** `configs/profiles/tuned/{claude,gpt}/{rt,rtj,griffin}/<task>.json`
  are per-task profiles tuned for STEER's traversal filter, e.g.
  `--profile configs/profiles/tuned/claude/rt/<task>.json`. Use a tuned profile
  with the ranking it was tuned against (the `claude` ones with the Claude
  config). They were evaluated without the column filter, so their
  column-filter section is just the universal one.

## Generate your own ranking (optional)

The only step that calls an LLM API. It writes a config you can pass to
`steer run --config`. You need `ANTHROPIC_API_KEY` or `OPENAI_API_KEY` set, and
RelBench access for the task's database.

```bash
pixi run steer-generate-config --task rel_f1_driver_dnf --provider anthropic
pixi run steer-generate-config --task rel_f1_driver_dnf --provider openai --column-filter
pixi run steer run --model rt --task rel_f1_driver_dnf --config generated/anthropic/rel_f1_driver_dnf.json
```

Two LLM calls are made (one per traversal direction), and a third for the column
filter with `--column-filter`. A config generated without it has no
column-filter section, and `steer run --column-filter` will tell you to
regenerate it. `--model` picks a different model within the provider, and
`--reasoning-effort` defaults to `high`, as in the paper (`none` omits it, for
models without reasoning). LLM output is not reproducible bit for bit, which is
why the configs used in the paper are shipped.

## Citation

```bibtex
@article{mohamed2026steer,
  title   = {STEER: Reducing Inference Cost in Relational Foundation Models through Semantically Informed Sampling},
  author  = {Mohamed, Abdalla and Aboulnaga, Ashraf},
  journal = {arXiv preprint arXiv:2610.00907},
  year    = {2026}
}
```

## License

STEER's own code is released under the Apache License 2.0 ([LICENSE](LICENSE)).
The model code in `models/` and the hosted checkpoints and data keep their
original licenses; see [NOTICE](NOTICE) and the per-directory LICENSE files.
