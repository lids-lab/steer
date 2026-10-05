"""Standalone inference for Griffin: a slim rewrite of pretrain.py's
--mode test path. Unlike pretrain.py, this never constructs the training
dataset, the validation datasets, or the optimizer/LR-scheduler -- pretrain.py
builds all of those unconditionally even when only testing.
"""

from __future__ import annotations

import os
import os.path as osp
import re
import sys
import tempfile

import accelerate
import numpy as np
import torch
from accelerate import Accelerator
from accelerate.utils import ProjectConfiguration
from torch.utils.data import DataLoader
from tqdm import tqdm

from relfm.griffin.hFloatEmb import SimpleRepeater, getfloatdec
from relfm.griffin.hdataset import Graph, Task
from relfm.griffin.hloaderwrapper import LoaderWrapperTask
from relfm.griffin.hmodel import GriffinMod
from relfm.griffin.metric import compute_metric

# Fixed hyperparameters matching the one LODO recipe this project trained
# with (griffin.sh's reference config) -- not configurable per STEER run,
# same as RT always being RT-12.
HIDDIM = 728
NUM_MP = 4
FANOUT = 20
HOP = 2
FEWSHOTFANOUT = 0
BATCH_SIZE = 32
USE_REV = True
USE_GATE = False
ISOLATE_TASK_TABLES = True
ADD_TASK_TABLE = True


def compute_output(model, dec, data):
    label, y, mapping = data[-3:]
    data = data[:-3]
    if y is None:
        output = dec(model(*data)[mapping])
    else:
        output = model(*data)[mapping] @ y.T
    return output, label


_SAMPLE_SIZE_RE = re.compile(rb"SAMPLE_SIZE task=\S+ nodes=(\d+) edges=(\d+)")


class _CaptureFd:
    """Duplicates OS file descriptor 1 (stdout) to a temp file for the
    duration of the block, in addition to the real stream, so
    Graph.subgraph()'s own 'SAMPLE_SIZE task=... nodes=... edges=...' print
    (already-correct node/edge counts, computed before the DataLoader
    reshapes the batch for the model) can be recovered afterward -- at the
    fd level because subgraph() runs inside DataLoader worker *processes*,
    which inherit OS file descriptors, not a Python-level sys.stdout
    reassignment made in the parent."""

    def __enter__(self):
        self._tmp = tempfile.TemporaryFile(mode="w+b")
        self._saved_fd = os.dup(1)
        sys.stdout.flush()
        os.dup2(self._tmp.fileno(), 1)
        return self

    def __exit__(self, *exc):
        sys.stdout.flush()
        os.dup2(self._saved_fd, 1)
        os.close(self._saved_fd)
        self._tmp.seek(0)
        self.captured = self._tmp.read()
        self._tmp.close()

    def sample_size(self):
        nodes = edges = 0
        n = 0
        for match in _SAMPLE_SIZE_RE.finditer(self.captured):
            nodes += int(match.group(1))
            edges += int(match.group(2))
            n += 1
        if n == 0:
            return {"avg_num_nodes": None, "avg_num_edges": None}
        return {"avg_num_nodes": nodes / n, "avg_num_edges": edges / n}


def eval_task(model, dec, dataset, max_eval_steps, accelerator, metric):
    model.eval()
    dataset.rebuild_indice(accelerator)
    batchsize = dataset.batch_size
    # num_workers=0 (single-process loading): with ~7-8 concurrent array-job
    # processes on one node, worker multiprocessing hit a shared-memory-file
    # race in /dev/shm regardless of worker count (num_workers=8, then 3,
    # both eventually hung mid-eval with "could not unlink the shared memory
    # file" -- a genuine PyTorch DataLoader multiprocessing bug under this
    # much node-wide concurrency, not something to tune around). Eliminating
    # worker processes entirely removes the whole bug class.
    loader = DataLoader(
        dataset, shuffle=False, batch_size=1, collate_fn=lambda xlist: xlist[0],
        num_workers=0,
    )
    loader = accelerator.prepare(loader)
    outputs, labels = [], []
    capture = _CaptureFd()
    with torch.inference_mode(), capture:
        for batch_idx, data in enumerate(tqdm(loader, desc="Evaluating", disable=not accelerator.is_main_process)):
            output, label = compute_output(model, dec, data)
            if output.shape[0] < batchsize:
                padnum = batchsize - output.shape[0]
                if torch.is_floating_point(label):
                    label = torch.concat((label, torch.empty_like(label[[0]].expand(padnum)).fill_(torch.nan)), dim=0)
                else:
                    label = torch.concat((label, torch.empty_like(label[[0]].expand(padnum)).fill_(-1)), dim=0)
                output = torch.concat((output, torch.empty_like(output[[0]].expand(padnum, -1)).fill_(torch.nan)), dim=0)
            output, label = output.unsqueeze(0), label.unsqueeze(0)
            output, label = accelerator.gather_for_metrics((output, label))
            output, label = output.flatten(0, 1), label.flatten(0, 1)
            if accelerator.is_main_process:
                mask = torch.isnan(label).logical_not_() if torch.is_floating_point(label) else label >= 0
                output, label = output[mask], label[mask]
                outputs.append(output.cpu())
                labels.append(label.cpu())

            if max_eval_steps > -1 and batch_idx >= max_eval_steps:
                break

    if not accelerator.is_main_process:
        return None, None
    labels = torch.concat(labels, dim=0)
    outputs = torch.concat(outputs, dim=0)
    metric_value = compute_metric(outputs, labels, metric)
    if metric == "mae":
        # compute_metric negates mae to keep Griffin's internal "higher score
        # is better" convention; flip back to a positive, lower-is-better
        # value to match RT/rt-j's mae convention.
        metric_value = -metric_value
    sample_size = capture.sample_size()
    return metric_value, sample_size


def evaluate_checkpoint(data_path, checkpoint_path, dataset, task_table, max_eval_steps, sampler_config_path=None, seed=42, split="test", reg_metric=None):
    # Griffin's fanout-based neighbor sampling (hdataset.py) draws from the
    # global torch/numpy RNG with no local generator -- pretrain.py's own
    # __main__ seeds this (default 42) before eval for reproducibility;
    # this standalone entry point must do the same.
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)

    tbconfig = ProjectConfiguration(project_dir="/tmp/steer_griffin_logs", logging_dir="/tmp/steer_griffin_logs")
    accelerator = Accelerator(log_with=None, project_config=tbconfig)

    model = GriffinMod(hiddim=HIDDIM, num_mp=NUM_MP, use_rev=USE_REV, use_gate=USE_GATE)
    accelerate.load_checkpoint_in_model(model, checkpoint_path)
    dec = getfloatdec(HIDDIM)
    floatembmodel = SimpleRepeater(HIDDIM)

    graph = Graph(data_path, HIDDIM)
    task = Task(data_path, HIDDIM)

    # Task-key naming inside a dataset's own metatask.yaml/task/ dir is
    # inconsistent across Griffin's preprocessed datasets: rel-f1 prefixes
    # with the dataset name ("rel-f1-driver-dnf"), every other dataset uses
    # the bare task_table ("user-churn") -- a quirk of how each was
    # preprocessed, not something to normalize here. Resolve whichever key
    # actually exists rather than assuming one convention.
    candidates = (f"{dataset}-{task_table}", task_table)
    taskname = next((c for c in candidates if c in task.metatask), None)
    if taskname is None:
        raise KeyError(f"Task table '{task_table}' not found for dataset '{dataset}' (tried {candidates}); known: {sorted(task.metatask)}")

    metric = task.metatask[taskname]["metric"]
    # reg_metric lets the caller override which regression metric gets used
    # (defaults to whatever's baked into this dataset's metatask.yaml, "r2"
    # for every regression task in this project); never touches classification's "auc".
    if reg_metric is not None and metric in ("r2", "mae"):
        metric = reg_metric

    # RunSpec/steer_eval spell it "val" (matching RT/rt-j); Griffin's own
    # LoaderWrapperTask spells the same split "valid".
    griffin_split = "valid" if split == "val" else split

    test_dataset = LoaderWrapperTask(
        graph, batch_size=BATCH_SIZE,
        subgraphargs={
            "floatemb": floatembmodel, "fanout": FANOUT, "hop": HOP,
            "isolate_task_tables": ISOLATE_TASK_TABLES, "add_task_table": ADD_TASK_TABLE,
            "sampler_config_path": sampler_config_path,
        },
        shuffle=False, task=task, tasknames=[taskname], split=griffin_split,
        fewshotfanout=FEWSHOTFANOUT,
    )

    model, dec = accelerator.prepare(model, dec)
    metric_value, sample_size = eval_task(model, dec, test_dataset, max_eval_steps, accelerator, metric)
    accelerator.end_training()

    return {"metric_name": metric, "metric_value": metric_value, "sample_size": sample_size}
