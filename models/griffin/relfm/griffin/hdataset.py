import os.path as osp
from typing import Iterable, Union, Literal

import yaml
import torch
import torch.nn.functional as F
import datasets as hds

from relfm.griffin.hdataset_prob import (
    load_tier_config,
    DEFAULT_COLUMN_FILTER_KEEP as COLUMN_KEEP,
)

INF = 100000000
MAXINT64 = 1<<62
TIMESTAMPADJNAME = "___TIMESTAMP"

def normalize_emb(embeddings):
    return F.normalize(embeddings, p=2, dim=-1)

class Node:
    meta: dict
    feat: hds.Dataset
    textemb: hds.Dataset
    adj: Union[hds.Dataset, None]

    def __init__(self, meta, feat, textemb, adj, feat_dim=512) -> None:
        self.meta = meta
        self.feat = feat
        self.textemb = textemb
        self.adj = adj
        self.feat_dim = feat_dim
        assert len(self.feat) == self.meta["num"]
        if self.adj is not None:
            assert (
                len(self.adj) == self.meta["num"]
            ), f"adj has {len(self.adj)} while there are {self.meta['num']} nodes"

    def __len__(self):
        return self.meta["num"]

    @property
    def is_target(self):
        return self.meta["is_target"]

    @property
    def featlist(self):
        return self.meta["feat"]
    

    def getfeat(self, idx: Union[int, Iterable[int]], floatemb, column_tier: dict = None, column_filter_enabled: bool = True, column_filter_keep: dict = None) -> torch.Tensor:
        data: dict = self.feat[idx]
        def unique_query_textemb(idx, input_dim=None):
            unique_idx, inv = torch.unique(idx, return_inverse=True)
            text_embeddings = self.textemb[unique_idx]["emb"][inv]
            if input_dim is not None:
                assert text_embeddings.shape[1] >= input_dim, f"input_dim {input_dim} is larger than text embedding dimension {text_embeddings.shape[1]}"
                text_embeddings = text_embeddings[:, :input_dim]
            return normalize_emb(text_embeddings)

        def unique_float_emb(val, input_dim=None):
            unique_val, inv = torch.unique(val, return_inverse=True)
            float_embeddings = floatemb(unique_val)[inv]
            if input_dim is not None:
                assert float_embeddings.shape[1] >= input_dim, f"input_dim {input_dim} is larger than float embedding dimension {float_embeddings.shape[1]}"
                float_embeddings = float_embeddings[:, :input_dim]
            return normalize_emb(float_embeddings)
        
        data = torch.stack(
            [
                (
                    unique_query_textemb(data[_], input_dim=self.feat_dim)
                    if "Griffin_text_" in _
                    else unique_float_emb(data[_])#floatemb(data[_])
                )
                for _ in self.featlist
            ],
            dim=1,
        )  # (B, num_col, dim)

        # Opt-in column_filter tiered sampling (see hdataset_prob.py);
        # column_tier is {colname: tier}, scoped to this node type by the
        # caller. None (the default) leaves data untouched. Matching
        # fly.rs: the column_filter toggle gates independently of whether
        # column_tier is populated.
        if column_tier is not None and column_filter_enabled:
            keep_map = column_filter_keep if column_filter_keep is not None else COLUMN_KEEP
            keep = torch.ones(data.shape[1])
            for i, colname in enumerate(self.featlist):
                tier = column_tier.get(colname, "weak")  # unassigned columns default to "weak", matching fly.rs
                keep[i] = keep_map[tier]
            if not (keep >= 1.0).all():
                coin = torch.rand(data.shape[0], data.shape[1])
                keep_mask = (coin < keep.unsqueeze(0)).unsqueeze(-1)  # (B, num_col, 1)
                data = data * keep_mask

        return data

    def getedge(self, idx: torch.LongTensor, fanout: int = INF, timestamp: list[int] = None, edge_tier: dict = None, edge_toggles: dict = None, p2f_edge_keep: dict = None, f2p_edge_keep: dict = None):
        if self.adj is None or self.is_target:
            return {}
        num_q = len(idx)
        # uni_idx, inv = torch.unique(idx, return_inverse=True)
        subadj: dict = self.adj[idx]# [uni_idx]
        subadj.pop("number")
        
        hastimestamp = timestamp is not None
        if hastimestamp:
            assert len(timestamp) == num_q
        ret = {}
        for key in subadj:
            if key.endswith(TIMESTAMPADJNAME):
                continue
            adj = subadj[key]
            if len(adj) == 0:
                continue
            if not isinstance(adj, torch.Tensor):
                assert isinstance(adj, list)
                adj = torch.nn.utils.rnn.pad_sequence(adj, batch_first=True, padding_value=-1)
            # adj = adj[inv]
            if len(adj.flatten()) == 0:
                continue
            assert adj.ndim == 2, f"{adj.shape}"
            assert adj.shape[0] == num_q
            rootnode = torch.arange(num_q).reshape(-1, 1).repeat(1, adj.shape[1])
            if hastimestamp:# and key + TIMESTAMPADJNAME in subadj:
                adjtimestamp = subadj[key + TIMESTAMPADJNAME]
                if not isinstance(adjtimestamp, torch.Tensor):
                    adjtimestamp = torch.nn.utils.rnn.pad_sequence(adjtimestamp, batch_first=True, padding_value=-1)
                # adjtimestamp = adjtimestamp[inv]
                assert adjtimestamp.ndim == 2
                assert adjtimestamp.shape[0] == num_q
                adj.masked_fill_(adjtimestamp>=timestamp.reshape(-1, 1), -1)

            # Fanout cap and tiered gate -- order matters, see below.
            def _apply_fanout():
                nonlocal adj, rootnode
                if adj.shape[1] > fanout:
                    rank_val = torch.rand_like(adj, dtype=torch.float)
                    rank_val.masked_fill_(adj<0, -1000.)
                    topk_ind = torch.topk(rank_val, fanout, dim=-1)[1]
                    adj = torch.gather(adj, 1, topk_ind)
                    rootnode = rootnode[:, :fanout]

            def _apply_gate():
                # Opt-in tiered probabilistic sampling (column_filter /
                # p2f_edge_probabilistic_at_push / f2p_edge_probabilistic_at_push,
                # mirroring RT's sampler -- see hdataset_prob.py).
                # edge_tier/edge_toggles are None unless a sampler config path is
                # given and a config exists for the running task, so this is a
                # no-op by default. Task-table edges are always exempt. Matching
                # fly.rs: each direction's toggle gates independently of whether
                # the tier dict is populated -- an empty/absent-entry tier dict
                # with the toggle ON still means "unassigned defaults to weak",
                # NOT "mechanism off".
                nonlocal adj
                if edge_tier is not None and "__TASK__" not in key:
                    is_f2p_role = key.startswith("head of ")  # this node's own FK -> parent
                    toggle_key = "f2p_edge_probabilistic_at_push" if is_f2p_role else "p2f_edge_probabilistic_at_push"
                    if edge_toggles is not None and edge_toggles.get(toggle_key, True):
                        keep_map = f2p_edge_keep if is_f2p_role else p2f_edge_keep
                        tier = edge_tier.get(key, "weak")  # unassigned edges default to "weak", matching fly.rs
                        keep_prob = keep_map[tier]
                        if keep_prob <= 0.0:
                            adj.masked_fill_(adj >= 0, -1)
                        elif keep_prob < 1.0:
                            coin = torch.rand_like(adj, dtype=torch.float)
                            adj.masked_fill_((coin >= keep_prob) & (adj >= 0), -1)

            # Gating AFTER the fanout cap is correct: RT's own
            # sampler applies its width limit (max_bfs_width) before rolling
            # the tier coin (fly.rs), and that order matters here too --
            # gate-then-select would make keep_prob a near no-op whenever the
            # candidate pool comfortably exceeds fanout (a random top-`fanout`
            # pick from a randomly-gated pool is distributionally close to a
            # random top-`fanout` pick from the whole pool, since the final
            # selection ignores which tier a survivor came from). Gating the
            # already-capped set instead means keep_prob directly controls how
            # many of the `fanout` candidates actually survive.
            _apply_fanout()
            _apply_gate()

            mask = adj >= 0
            rootnode, adj = rootnode[mask], adj[mask]
            if len(rootnode) == 0:
                continue
            ret[key] = torch.stack(
                (rootnode, adj), dim=0
            )
        return ret


def edgename2tail(edgename: str):
    if edgename.startswith("head of "):
        return edgename.split(":")[-1]
    elif edgename.startswith("tail of "):
        return edgename.split(":")[0].removeprefix("tail of ")
    else:
        print("cannot parse edgename", edgename)
        raise NotImplementedError


def edgename2head(edgename: str):
    if edgename.startswith("head of "):
        return edgename.split(":")[0].removeprefix("head of ")
    elif edgename.startswith("tail of "):
        return edgename.split(":")[-1]
    else:
        print("cannot parse edgename", edgename)
        raise NotImplementedError


class Graph:

    def __init__(self, path, feat_dim=512) -> None:
        with open(osp.join(path, "metanode.yaml")) as f:
            metanode = yaml.safe_load(f)
        with open(osp.join(path, "metaadj.yaml")) as f:
            metaadj = yaml.safe_load(f)
        for nodetype in metanode:
            metanode[nodetype].update(metaadj[nodetype])
        self.feat_dim = feat_dim
        self.metanode = metanode
        self.edgenameemb = torch.load(
            osp.join(path, "edgenameemb.pt"), map_location="cpu", weights_only=True
        )
        self.edgenameemb = {name: normalize_emb(emb[:feat_dim]) for name, emb in self.edgenameemb.items()}
        self.featnameemb = torch.load(
            osp.join(path, "featnameemb.pt"), map_location="cpu", weights_only=True
        )
        self.featnameemb = {name: normalize_emb(emb[:feat_dim]) for name, emb in self.featnameemb.items()}
        self.nodes = {
            nodetype: Node(
                self.metanode[nodetype],
                hds.load_from_disk(osp.join(path, "node", nodetype, "feat")).with_format("torch"),
                (
                    hds.load_from_disk(
                        osp.join(path, "node", nodetype, "textemb")
                    ).with_format("torch")
                    if osp.exists(osp.join(path, "node", nodetype, "textemb"))
                    else None
                ),
                (
                    None
                    if len(
                        self.metanode[nodetype]["in"] + self.metanode[nodetype]["out"]
                    )
                    == 0
                    else hds.load_from_disk(
                        osp.join(path, "edge", nodetype, "adj")
                    ).with_format("torch")
                ),
                feat_dim=self.feat_dim,
            )
            for nodetype in self.metanode
        }

    def subgraph(
        self,
        root_nodetype: str,
        root_nodeidx: Iterable[int],
        hop: int,
        floatemb,
        fanout: int = INF,
        timestamp: Union[list[int], None] = None,
        taskname: Union[str, None] = None,
        isolate_task_tables: bool = True,
        add_task_table: bool = True,
        sampler_config_path: Union[str, None] = None,
    ):
        hastimestamp: bool = timestamp is not None
        adj = {}
        node = {}
        root = {root_nodetype: root_nodeidx}
        roottimestamp = {root_nodetype: timestamp}

        # Opt-in tiered probabilistic sampling: no-op unless
        # sampler_config_path is given and a config exists there (see
        # hdataset_prob.py).
        tier_config = load_tier_config(taskname, sampler_config_path)
        edge_tier = tier_config["edge_tier"] if tier_config else None
        edge_toggles = tier_config["toggles"] if tier_config else None
        column_tier_all = tier_config["column_tier"] if tier_config else None
        column_filter_enabled = tier_config["toggles"]["column_filter"] if tier_config else True
        column_filter_keep = tier_config["column_filter_keep"] if tier_config else None
        p2f_edge_keep = tier_config["p2f_edge_keep"] if tier_config else None
        f2p_edge_keep = tier_config["f2p_edge_keep"] if tier_config else None

        def dictgetlen(d: dict[str, torch.Tensor], key: str, dim: int = 0):
            if key not in d:
                return 0
            return d[key].shape[dim]

        def dictupdate(
            d: dict[str, torch.Tensor],
            key: str,
            value: torch.Tensor,
            concatdim: int = 0,
        ):
            if key not in d:
                d[key] = value
            else:
                d[key] = torch.concat((d[key], value), dim=concatdim)
            return d

        for h in range(hop):
            nroot, nroottimestamp = {}, {}
            for nodetype in root:
                found_src_num = dictgetlen(node, nodetype)
                # print(list(self.nodes.keys()), list(root.keys()), list(roottimestamp.keys()))
                ttadj = self.nodes[nodetype].getedge(
                    root[nodetype], fanout, roottimestamp[nodetype], edge_tier=edge_tier, edge_toggles=edge_toggles,
                    p2f_edge_keep=p2f_edge_keep, f2p_edge_keep=f2p_edge_keep,
                )
                for edgetype in ttadj:
                    srcidx, taridx = ttadj[edgetype][0], ttadj[edgetype][1]

                    tartype = edgename2tail(edgetype)
                    found_tar_num = (
                        dictgetlen(node, tartype)
                        + dictgetlen(root, tartype)
                        + dictgetlen(nroot, tartype)
                    )

                    if hastimestamp:
                        tartimestamp = roottimestamp[nodetype][srcidx]
                        nroottimestamp = dictupdate(
                            nroottimestamp, tartype, tartimestamp
                        )
                    else:
                        nroottimestamp[tartype] = None
                    nroot = dictupdate(nroot, tartype, taridx)

                    srcidx = srcidx + found_src_num
                    taridx = (
                        torch.arange(taridx.shape[0], device=taridx.device)
                        + found_tar_num
                    )

                    adj = dictupdate(
                        adj, edgetype, torch.stack((srcidx, taridx), dim=0), concatdim=1
                    )

            for nodetype in root:
                node = dictupdate(node, nodetype, root[nodetype])
            del root
            del roottimestamp
            root = nroot
            roottimestamp = nroottimestamp

        for nodetype in root:
            node = dictupdate(node, nodetype, root[nodetype])

        mapping = torch.arange(len(root_nodeidx), device=root_nodeidx.device)
        # not change, latter code depends on mapping == arange

        for nodetype in node:
            assert len(node[nodetype]), f"{list(node.keys())} {root_nodeidx.shape} {list(adj.keys())}"
            #unique_idx, inv = torch.unique(node[nodetype], return_inverse=True)
            #node[nodetype] = self.nodes[nodetype].getfeat(unique_idx, floatemb)[inv]
            column_tier = None
            if column_tier_all is not None and not nodetype.endswith("__TASK__"):
                column_tier = {
                    colname: tier
                    for (nt, colname), tier in column_tier_all.items()
                    if nt == nodetype
                }
            node[nodetype] = self.nodes[nodetype].getfeat(
                node[nodetype], floatemb, column_tier=column_tier, column_filter_enabled=column_filter_enabled,
                column_filter_keep=column_filter_keep,
            )

        edgenameemb = {edgetype: self.edgenameemb[edgetype] for edgetype in adj}
        nodenameemb = {
            nodetype: torch.stack(
                [self.featnameemb[_] for _ in self.nodes[nodetype].featlist], dim=0
            )
            for nodetype in node
        }
        # print("loader: ", len(node), len(adj), list(nodenameemb.keys()), list(edgenameemb.keys()))

        nodetypes = list(node.keys())
        for nodetype in nodetypes:
            if nodetype.endswith("__TASK__"):
                if add_task_table and nodetype.startswith(taskname):
                    continue
                elif add_task_table and not isolate_task_tables:
                    continue
                # remove table
                node.pop(nodetype)
                nodenameemb.pop(nodetype)

        edgetypes = list(adj.keys())
        for edgetype in edgetypes:
            tailtype = edgename2tail(edgetype)
            if tailtype.endswith("__TASK__"):
                if add_task_table and tailtype.startswith(taskname):
                    continue
                elif add_task_table and not isolate_task_tables:
                    continue
                # remove edgetype
                adj.pop(edgetype)
                edgenameemb.pop(edgetype)
            headtype = edgename2head(edgetype)
            if headtype.endswith("__TASK__"):
                if add_task_table and headtype.startswith(taskname):
                    continue
                elif add_task_table and not isolate_task_tables:
                    continue
                # remove edgetype
                adj.pop(edgetype)
                edgenameemb.pop(edgetype)

        # Per-example sample-size instrumentation (node/edge count of the
        # finalized subgraph), for comparing against smart-sampling filtered
        # runs. Printed rather than stored on self, since this runs inside
        # DataLoader worker processes (num_workers>0 in eval_task) -- workers
        # are forked copies and never sync attribute writes back to the main
        # process, but they do inherit the same (already-redirected) stdout,
        # which is how the caller's subprocess capture sees everything else
        # this process prints.
        sample_num_nodes = sum(v.shape[0] for v in node.values())
        sample_num_edges = sum(v.shape[1] for v in adj.values())
        print(f"SAMPLE_SIZE task={taskname} nodes={sample_num_nodes} edges={sample_num_edges}", flush=True)

        return node, adj, nodenameemb, edgenameemb, mapping
    
    def fewshot(
        self,
        root_nodetype: str,
        root_nodeidx: torch.LongTensor,
        task_mask: torch.BoolTensor,
        floatemb,
        fanout: int = INF,
        timestamp: Union[list[int], None] = None,
        prefetch_factor: int = 10,
    ):
        assert fanout < INF, "alway sample past"
        # assert node idx in data is sorted by time stamp
        hastimestamp: bool = timestamp is not None

        idx = torch.randint(0, MAXINT64, (root_nodeidx.shape[0], prefetch_factor*fanout))
        idx = idx % (root_nodeidx.clamp_min(1)).reshape(-1, 1)
        
        if prefetch_factor > 1:
            rootfeat = self.nodes[root_nodetype].getfeat(root_nodeidx, floatemb) # (B, C, D)
            rootfeat[task_mask.unsqueeze(-1).expand(-1, -1, rootfeat.shape[-1])] = 0.
            feat = self.nodes[root_nodetype].getfeat(idx.flatten(), floatemb).unflatten(0, (-1, prefetch_factor*fanout)) # （B, FANOUT*PREFETCH, C, D)
            score = feat.flatten(-2, -1) @ rootfeat.flatten(-2, -1).unsqueeze(-1) # (B, FANOUT*PREFETCH, 1)
            score = score.squeeze(-1) # (B, FANOUT*PREFETCH)
            idx = torch.gather(idx, 1, torch.topk(score, k=fanout, dim=-1)[1]).flatten() # (B, FANOUT)
        idx = idx.flatten()
        rootnode = torch.arange(root_nodeidx.shape[0]).repeat_interleave(fanout)
        mask = root_nodeidx[rootnode] > 0
        idx, rootnode = idx[mask], rootnode[mask]
        return idx, rootnode


class Task:
    def __init__(self, path, feat_dim=512) -> None:
        with open(osp.join(path, "metatask.yaml")) as f:
            self.metatask = yaml.safe_load(f)

        self.feat_dim = feat_dim
        self.tasknameemb = torch.load(
            osp.join(path, "tasknameemb.pt"), map_location="cpu", weights_only=True
        )
        self.tasknameemb = {name: emb[:feat_dim] for name, emb in self.tasknameemb.items()}
        self.tasks = {
            taskname: hds.load_from_disk(osp.join(path, "task", taskname)).with_format(
                "torch"
            )
            for taskname in self.metatask
        }

    def get_retrieval(
        self,
        graph: Graph,
        taskname: str,
        split: Literal["train", "valid", "test"],
        idx: torch.Tensor,
    ):
        meta = self.metatask[taskname]
        assert meta["task_type"] == "retrieval"
        target_type = meta["target_type"]

        if split == "train":
            taskinfo = self.tasks[taskname][idx]
        elif split == "valid":
            taskinfo = self.tasks[taskname][meta["split"][0] + idx]
        elif split == "test":
            taskinfo = self.tasks[taskname][meta["split"][0] + meta["split"][1] + idx]
        else:
            raise NotImplementedError

        nodeidx, label = taskinfo["nodeidx"], taskinfo["label"]
        if meta["hastimestamp"]:
            timestamp = taskinfo["timestamp"]
        else:
            timestamp = None

        target_feat_mask = torch.tensor(
            [_ not in meta["masked_feat"] for _ in graph.metanode[target_type]["feat"]],
            dtype=torch.bool,
        )
        return (
            target_type,
            target_feat_mask,
            nodeidx,
            label,
            timestamp,
            self.tasknameemb[taskname],
            (meta["seed_type"], meta["num_class"]),
        )

    def get_regression(
        self,
        graph: Graph,
        taskname: str,
        split: Literal["train", "valid", "test"],
        idx: torch.Tensor,
    ):
        meta = self.metatask[taskname]
        assert meta["task_type"] == "regression"
        target_type = meta["target_type"]

        if split == "train":
            taskinfo = self.tasks[taskname][idx]
        elif split == "valid":
            taskinfo = self.tasks[taskname][meta["split"][0] + idx]
        elif split == "test":
            taskinfo = self.tasks[taskname][meta["split"][0] + meta["split"][1] + idx]
        else:
            raise NotImplementedError

        nodeidx, label = taskinfo["nodeidx"], taskinfo["label"]
        if meta["hastimestamp"]:
            timestamp = taskinfo["timestamp"]
        else:
            timestamp = None

        target_feat_mask = torch.tensor(
            [_ not in meta["masked_feat"] for _ in graph.metanode[target_type]["feat"]],
            dtype=torch.bool,
        )
        return (
            target_type,
            target_feat_mask,
            nodeidx,
            label,
            timestamp,
            self.tasknameemb[taskname],
            (None, None),
        )
