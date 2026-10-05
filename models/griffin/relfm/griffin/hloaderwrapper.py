from typing import Literal

import torch
import torch.nn.functional as F
import numpy as np
import accelerate

from .hdataset import Graph, edgename2head, edgename2tail, Task

def unifyheterograph(rootnodetype, node, nodenameemb, edgenameemb, adj):
    assert rootnodetype in node
    nodetypes = [rootnodetype] + list([_ for _ in node.keys() if _ != rootnodetype])
    nodeptr = [0] + [node[nodetype].shape[0] for nodetype in nodetypes]
    nodeptr = np.cumsum(nodeptr)
    node = [(nodenameemb[nodetype], node[nodetype])  for nodetype in nodetypes]
    
    edgetypes = list(adj.keys())
    if len(edgetypes) > 0:
        edge_attr = torch.stack([edgenameemb[edgetype] for edgetype in edgetypes], dim=0)
        edge_index = []
        edge_attr_type = []
        for i, edgetype in enumerate(edgetypes):
            head, tail = edgename2head(edgetype), edgename2tail(edgetype)
            row = nodeptr[nodetypes.index(head)] + adj[edgetype][0]
            col = nodeptr[nodetypes.index(tail)] + adj[edgetype][1]
            attr = torch.zeros_like(row).fill_(i)
            edge_index.append(torch.stack((row, col), dim=0))
            edge_attr_type.append(attr)
        edge_index = torch.concat(edge_index, dim=1)
        edge_attr_type = torch.concat(edge_attr_type, dim=0)
    else:
        edge_attr, edge_index, edge_attr_type = None, None, None
    return node, edge_index, edge_attr_type, edge_attr

def scalefeat(node, taskfeat, edge_attr, y=None):
    dim = node[0][0].shape[-1]
    shape = (dim,)
    for i in range(len(node)):
        node[i] = (F.layer_norm(node[i][0], shape), F.layer_norm(node[i][1], shape))
        taskfeat[i] = None if taskfeat[i] is None else F.layer_norm(taskfeat[i], shape)
    if edge_attr is not None:
        edge_attr = F.layer_norm(edge_attr, shape)
    if y is not None:
        y = F.layer_norm(y, shape)
    return node, taskfeat, edge_attr, y

def buildindice(shuffle, lens, batch_size):
    ind = []
    if not shuffle:
        for nodetypeidx, (_, num) in enumerate(lens):
            leftnum = num % batch_size
            padnum = 0 if leftnum == 0 else batch_size - leftnum
            idx = torch.arange(num+padnum)
            idx[num:] = -1
            idx = idx.reshape(-1, batch_size)
            idx = torch.concat((idx[:, [0]].clone().fill_(nodetypeidx), idx), dim=-1)
            ind.append(idx)
    else:
        for nodetypeidx, (_, num) in enumerate(lens):
            leftnum = num % batch_size
            padnum = 0 if leftnum == 0 else batch_size - leftnum
            idx = torch.arange(num+padnum)
            idx[num:] = -1
            idx = idx[torch.randperm(idx.shape[0])]
            idx = idx.reshape(-1, batch_size)
            idx = torch.concat((idx[:, [0]].clone().fill_(nodetypeidx), idx), dim=-1)
            ind.append(idx)
    return torch.cat(ind, dim=0)



class LoaderWrapper:

    def __init__(self, graph: Graph, batch_size: int, shuffle: bool, subgraphargs: dict[str]) -> None:
        self.graph = graph
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.subgraphargs = subgraphargs
        self.lens = [(nodetype, self.graph.metanode[nodetype]["num"]) for nodetype in self.graph.metanode]
        self.ind = None

    def rebuild_indice(self, accelerator: accelerate.Accelerator):
        if accelerator.is_main_process:
            self.ind = buildindice(self.shuffle, self.lens, self.batch_size)
        self.ind = accelerate.utils.broadcast_object_list([self.ind], from_process=0)[0]

    def __len__(self):
        assert self.ind is not None, "please rebuild_indice first"
        return self.ind.shape[0]

    def subgraph(self, nodetype, ind, timestamp=None, taskname=None):
        return self.graph.subgraph(nodetype, ind, **self.subgraphargs, timestamp=timestamp, taskname=taskname)


class LoaderWrapperTask(LoaderWrapper):

    def __init__(self, graph: Graph, batch_size: int, shuffle: bool, subgraphargs: dict[str], task: Task, tasknames: list[str], split: Literal["train", "valid", "test"]) -> None:
        super().__init__(graph, batch_size, shuffle, subgraphargs)
        assert len(tasknames) > 0, "please provide task name"
        self.task = task
        self.lens = [(taskname, self.task.metatask[taskname]["split"][{"train":0, "valid": 1, "test": 2}[split]]) for i, taskname in enumerate(tasknames)]
        self.split = split

    def __getitem__(self, idx:int):
        tind = self.ind[idx]
        taskname = self.lens[tind[0].item()][0]
        tind = tind[1:]
        tind = tind[tind>=0]

        is_regression = self.task.metatask[taskname]["task_type"] == "regression"

        if is_regression:
            rootnodetype, target_feat_mask, tind, label, tasktimestamp, tasknameemb, _ = self.task.get_regression(self.graph, taskname, self.split, tind)
            y = None
        else:
            rootnodetype, target_feat_mask, tind, label, tasktimestamp, tasknameemb, (seed_type, num_class) = self.task.get_retrieval(self.graph, taskname, self.split, tind)
            y = self.graph.nodes[seed_type].getfeat(range(num_class), self.subgraphargs["floatemb"])
            assert y.shape[1] == 1
            y = y.squeeze_(1)

        node, adj, nodenameemb, edgenameemb, mapping = self.subgraph(rootnodetype, tind, tasktimestamp, taskname=taskname)

        node, edge_index, edge_attr_type, edge_attr = unifyheterograph(rootnodetype, node, nodenameemb, edgenameemb, adj)

        taskfeat = [tasknameemb for i in range(len(node))]
        mask = [None for i in range(len(node))]
        mask[0] = torch.zeros(node[0][1].shape[:2], dtype=torch.bool)
        mask[0][mapping] = torch.logical_not(target_feat_mask)

        node, taskfeat, edge_attr, y = scalefeat(node, taskfeat, edge_attr, y)

        return node, mask, taskfeat, edge_index, edge_attr_type, edge_attr, label, y, mapping
