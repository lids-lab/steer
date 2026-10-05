from typing import List, Tuple, Union

import torch
import torch.nn as nn
from torch import Tensor
from torch_geometric.nn.aggr import MeanAggregation, MaxAggregation
import numpy as np


class SelfAverageAggregator(nn.Module):
    def __init__(self,
                 hiddim: int,
                 num_heads: int = 8,
                 num_layer: int = 1,
                 dim_feedforward: int = None,
                 dropout: float = 0.1):
        super().__init__()
        if dim_feedforward is None:
            dim_feedforward = hiddim
        self.linq = nn.Linear(hiddim, hiddim, bias=False)
        self.crossattention = nn.MultiheadAttention(
            hiddim, num_heads, dropout, bias=False, batch_first=True
        )
    
    def forward(self, column_name_emb: Tensor, x, mask=None):
        column_name_emb = column_name_emb.unsqueeze(0).expand(x.shape[0], -1, -1)
        ret = self.crossattention(
            column_name_emb,
            column_name_emb,
            x,
            key_padding_mask=mask,
            need_weights=False,
        )[0]
        return self.linq(ret)

class SelfAttentionAggregator(nn.Module):
    def __init__(
        self,
        hiddim: int,
        num_heads=8,
        num_layer=1,
        dim_feedforward: int = None,
        dropout=0.1,
    ):
        super().__init__()
        if dim_feedforward is None:
            dim_feedforward = hiddim
        self.linq = nn.Linear(hiddim, hiddim, bias=False)
        self.crossattention = nn.MultiheadAttention(
            hiddim, num_heads, dropout, bias=False, batch_first=True
        )

    def forward(self, tar: Union[Tensor, None], column_name_emb: Tensor, x, mask=None):
        column_name_emb = column_name_emb.unsqueeze(0)
        ret = self.crossattention(
            tar,
            column_name_emb.expand(x.shape[0], -1, -1),
            x,
            key_padding_mask=mask,
            need_weights=False,
        )[0]
        return ret * self.linq(tar)


def checktaskfeat(taskfeat: List[Tensor], node: List[Tuple[Tensor, Tensor]]):
    for i in range(len(taskfeat)):
        if taskfeat[i] is None:
            taskfeat[i] = node[i][0].mean(dim=0)
        if taskfeat[i].ndim == 1:
            taskfeat[i] = taskfeat[i].unsqueeze(0).expand(node[i][1].shape[0], -1)
        assert taskfeat[i].ndim == 2
    return taskfeat

def _attention_merge_feat_self_attention(
    attention_model: SelfAttentionAggregator,
    feat: Tensor,  # [#node, #feat, hiddim]
    colfeat: Tensor,  # [#feat, hiddim]
    mask: Union[Tensor, None],  # [#node, #feat]
) -> Tensor:
    return torch.mean(attention_model(colfeat, feat, mask), dim=1)

def _attention_merge_feat_with_relation(
    attention_model: SelfAttentionAggregator,
    feat: Tensor,  # [#node, #feat, hiddim]
    colfeat: Tensor,  # [#feat, hiddim]
    taskfeat: Union[Tensor, None],  # [#node, hiddim]
    mask: Union[Tensor, None],  # [#node, #feat]
) -> Tensor:
    return attention_model(taskfeat.unsqueeze(1), colfeat, feat, mask).squeeze(1)

class RMPNN(nn.Module):

    def __init__(self, hiddim) -> None:
        super().__init__()
        self.rellin = nn.Sequential(
            nn.Linear(hiddim, hiddim)
        )
        self.aggr1 = MeanAggregation()
        self.aggr2 = MaxAggregation()

    def forward(
        self,
        x: Tensor,
        edge_index: Tensor,
        edge_attr_type: Tensor,
        edge_attr: Tensor
    ) -> Tensor:
        if edge_index is None or edge_index.shape[1] == 0:
            return 0 * self.rellin(x[0])
        edge_attr = self.rellin(edge_attr)

        center, inv = torch.unique(torch.stack((edge_index[0], edge_attr_type), dim=0), dim=1, return_inverse=True)
        out1 = self.aggr1(x[edge_index[1]], inv, dim_size=center.shape[1])
        out2 = self.aggr2(out1*edge_attr[center[1]], center[0], dim_size=x.shape[0])
        return out2


class GriffinMod(nn.Module):
    def __init__(
        self,
        hiddim: int = 256,
        num_tf: int = 1,
        num_mp: int = 2,
        use_rev: bool = True,
        use_gate: bool = True,
    ):
        super().__init__()
        self.nodefeataggr = nn.ModuleList(
            [
                SelfAverageAggregator(hiddim, num_layer=num_tf)
            ]
            + [SelfAttentionAggregator(hiddim, num_layer=num_tf) for _ in range(num_mp - 1)]
        )
        self.mpnn = nn.ModuleList([RMPNN(hiddim) for _ in range(num_mp)])
        self.lintask = nn.ModuleList(
            [nn.Linear(hiddim, hiddim, bias=False) for _ in range(num_mp - 1)]
        )
        self.mlp = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Linear(hiddim, hiddim, bias=False), nn.SiLU(inplace=True)
                )
                for _ in range(num_mp)
            ]
        )
        self.mlp2 = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Linear(hiddim, hiddim, bias=False), nn.SiLU(inplace=True), nn.Linear(hiddim, hiddim, bias=False),
                )
                for _ in range(num_mp)
            ]
        )
        self.num_mp = num_mp
        self.ln = nn.LayerNorm(hiddim, elementwise_affine=False)
        self.use_rev = use_rev
        self.use_gate = use_gate

        self.gatelin = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Linear(hiddim, int(hiddim**0.5)),
                    nn.SiLU(inplace=True),
                    nn.Linear(int(hiddim**0.5), 1, bias=False),
                )
                for _ in range(num_mp)
            ]
        ) if use_gate else None
        self.revmpnn = nn.ModuleList([RMPNN(hiddim) for _ in range(num_mp)]) if use_rev else None
        self.revgatelin = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Linear(hiddim, int(hiddim**0.5)),
                    nn.SiLU(inplace=True),
                    nn.Linear(int(hiddim**0.5), 1, bias=False),
                )
                for _ in range(num_mp)
            ]
        ) if use_gate and use_rev else None
        with torch.no_grad():
            for _ in range(num_mp):
                if use_gate:
                    self.gatelin[_][2].weight.fill_(0.0)
                    if use_rev:
                        self.revgatelin[_][2].weight.fill_(0.0)

    def forward(
        self,
        node: List[Tuple[Tensor, Tensor]],
        mask: List[Union[Tensor, None]],
        taskfeat: List[Union[Tensor, None]],
        edge_index: Tensor,
        edge_attr_type: Tensor,
        edge_attr: Tensor,
    ) -> Tensor:
        assert len(node) == len(mask)
        assert len(node) == len(taskfeat)
        num_node: int = len(node)
        taskfeat = checktaskfeat(taskfeat, node)
        nodeptr = [0] + [_.shape[0] for _ in taskfeat]
        nodeptr = np.cumsum(nodeptr)

        for i in range(self.num_mp):
            if i == 0:
                x = torch.concat(
                    [
                        _attention_merge_feat_self_attention(
                            self.nodefeataggr[i],
                            feat,
                            colfeat,
                            mask[j],
                        )
                        for j, (colfeat, feat) in enumerate(node)
                    ],
                    dim=0,
                )
            else:
                x = x + torch.concat(
                    [
                        _attention_merge_feat_with_relation(
                            self.nodefeataggr[i],
                            feat,
                            colfeat,
                            self.ln(taskfeat[j]),
                            mask[j],
                        )
                        for j, (colfeat, feat) in enumerate(node)
                    ],
                    dim=0,
                )
            lnx = self.ln(x)
            mlpx = self.mlp[i](lnx)
            x = (
                x
                + self.mlp2[i](lnx)
                + (
                    (self.gatelin[i](lnx) if self.use_gate else 1)
                    * self.mpnn[i](mlpx, edge_index, edge_attr_type, edge_attr)
                )
                + (
                    (
                        (self.revgatelin[i](lnx) if self.use_gate else 1)
                        * self.revmpnn[i](mlpx, None if edge_index is None else edge_index[[1, 0]], edge_attr_type, edge_attr)
                    )
                    if self.use_rev
                    else 0
                )
            )

            if i < self.num_mp - 1:
                ntaskfeat = self.lintask[i](self.ln(x))
                for j in range(num_node):
                    taskfeat[j] = ntaskfeat[nodeptr[j] : nodeptr[j + 1]] + taskfeat[j]
        return self.ln(x)
