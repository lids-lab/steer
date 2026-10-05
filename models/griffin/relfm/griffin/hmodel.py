import os
from typing import List, Tuple, Union

import torch
import torch.nn as nn
from torch import Tensor
from torch_geometric.nn.aggr import MeanAggregation, MaxAggregation
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from safetensors.torch import load_file


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
        """
        self.attention_layer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=hiddim,
                nhead=num_heads,
                dim_feedforward=dim_feedforward,
                dropout=dropout,
                norm_first=True,
                activation="gelu",
                batch_first=True,
            ),
            num_layers=num_layer,
        )
        """
        self.linq = nn.Linear(hiddim, hiddim, bias=False)  # nn.Sequential(nn.Linear(hiddim, out_size, bias=False), nn.SiLU(inplace=True))
        self.crossattention = nn.MultiheadAttention(
            hiddim, num_heads, dropout, bias=False, batch_first=True
        )

    def forward(self, tar: Union[Tensor, None], column_name_emb: Tensor, x, mask=None):
        # tar: (batch_size, q_len, hiddim)
        # x: (batch_size, seq_len, hiddim)
        # column_name_emb: (seq_len, hiddim)
        # return: (batch_size, q_len, hiddim)
        column_name_emb = column_name_emb.unsqueeze(0)
        # column_name_emb = self.attention_layer(column_name_emb)# transformer encoder has res + column_name_emb
        q = tar  # self.linq(tar)
        ret = self.crossattention(
            q,
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

'''
class RMPNN(MessagePassing):

    def __init__(self, hiddim) -> None:
        super().__init__(aggr="mean", flow="target_to_source")
        self.rellin = nn.Sequential(
            nn.Linear(hiddim, hiddim)
        )  # , nn.SiLU(inplace=True))

    def reset_parameters(self) -> None:
        return self.rellin[0].reset_parameters()

    def forward(
        self,
        x: Tensor,
        edge_index: Tensor,
        edge_attr_type: Tensor,
        edge_attr: Tensor,
    ) -> Tensor:
        if edge_index is None or edge_index.shape[1] == 0:
            return 0 * self.rellin(x[0])
        edge_attr = self.rellin(edge_attr)[edge_attr_type]
        out = self.propagate(
            edge_index, edge_attr=edge_attr, x=x, size=(x.shape[0], x.shape[0])
        )
        return out

    def message(self, edge_attr, x_j: Tensor) -> Tensor:
        return edge_attr * x_j
'''


class RMPNN(nn.Module):

    def __init__(self, hiddim) -> None:
        super().__init__()
        self.rellin = nn.Sequential(
            nn.Linear(hiddim, hiddim)
        )  # , nn.SiLU(inplace=True))
        self.aggr1 = MeanAggregation()
        self.aggr2 = MaxAggregation()

    def reset_parameters(self) -> None:
        return self.rellin[0].reset_parameters()

    def forward(
        self,
        x: Tensor,
        edge_index: Tensor,
        edge_attr_type: Tensor,
        edge_attr: Tensor
    ) -> Tensor:
        if edge_index is None or edge_index.shape[1] == 0:
            return 0 * self.rellin(x[0])
        edge_attr = self.rellin(edge_attr) # [edge_attr_type]

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
            # The 0-layer is a simple average aggregator
            [
                SelfAverageAggregator(hiddim, num_layer=num_tf)
            ]
            # The rest are self-attention aggregators
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
                    # nn.Tanh(),
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
                    # nn.Tanh(),
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

    def reset_parameters(self):
        for i in range(len(self.mpnn)):
            self.mpnn[i].reset_parameters()
            if self.gatelin is not None:
                self.gatelin[i][0].reset_parameters()
    
    def forward_with_each_layer_output(
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

        return_x = []

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
            return_x.append(x.detach().cpu())

            if i < self.num_mp - 1:
                ntaskfeat = self.lintask[i](self.ln(x))
                for j in range(num_node):
                    taskfeat[j] = ntaskfeat[nodeptr[j] : nodeptr[j + 1]] + taskfeat[j]
        return return_x

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

def visualize_parameter_heatmaps(model, model_name, figsize=(15, 25)):
    """
    Create heatmap visualizations for GriffinMod parameters with dynamic subplot layout
    """
    # Count total number of plots needed
    num_plots = (
        len(model.nodefeataggr) +  # Attention weights
        len(model.lintask) +      # Linear task weights
        len(model.mpnn) +          # MPNN weights
        len(model.revmpnn) +      # Reverse MPNN weights
        len(model.mlp) +           # MLP weights
        len(model.mlp2) * 2 +          # MLP2 weights, with two layers
        (len(model.gatelin) if model.use_gate else 0) * 2 +   # Gate weights if enabled
        (len(model.revgatelin) if model.use_gate and model.use_rev else 0) * 2
    )
    
    # Calculate optimal grid layout
    num_cols = 4  # You can adjust this
    num_rows = (num_plots + num_cols - 1) // num_cols  # Ceiling division
    
    # Adjust figure size based on number of rows
    figsize = (7.5 * num_cols, 5 * num_rows)
    fig = plt.figure(figsize=figsize)
    
    def plot_weight_heatmap(weight, title, subplot_idx):
        plt.subplot(num_rows, num_cols, subplot_idx)
        # Convert to numpy and handle 2+ dimensional tensors
        if weight.ndim > 2:
            weight = weight.reshape(weight.shape[0], -1)
        weight_np = weight.detach().cpu().numpy()
        
        # Create heatmap with seaborn
        sns.heatmap(
            weight_np,
            cmap='RdBu_r',  # Red-Blue diverging colormap
            center=0,       # Center the colormap at 0
            vmin=-np.abs(weight_np).max(),  # Symmetric scale
            vmax=np.abs(weight_np).max(),
            xticklabels=False,  # Hide x-axis ticks for cleaner visualization
            cbar_kws={'label': 'Weight Value'}
        )
        plt.title(f'{title}\nShape: {tuple(weight.shape)}')
        plt.ylabel('Output Features')
    
    current_plot = 1
    
    # 1. Attention Weights
    for i, aggr in enumerate(model.nodefeataggr):
        weights = aggr.crossattention.in_proj_weight
        plot_weight_heatmap(weights, f'Layer {i} Attention Weights', current_plot)
        current_plot += 1
    
    # 2. Linear task weights
    for i, lintask in enumerate(model.lintask):
        weights = lintask.weight
        plot_weight_heatmap(weights, f'Layer {i} Linear Task Weights', current_plot)
        current_plot += 1
    
    # 3. MPNN Weights
    for i, mp in enumerate(model.mpnn):
        weights = mp.rellin[0].weight
        plot_weight_heatmap(weights, f'Layer {i} MPNN Weights', current_plot)
        current_plot += 1
    
    # 4. Reverse MPNN Weights
    for i, revmp in enumerate(model.revmpnn):
        weights = revmp.rellin[0].weight
        plot_weight_heatmap(weights, f'Layer {i} Reverse MPNN Weights', current_plot)
        current_plot += 1
    
    # 5. MLP Weights
    for i, mlp in enumerate(model.mlp):
        weights = mlp[0].weight
        plot_weight_heatmap(weights, f'Layer {i} MLP Weights', current_plot)
        current_plot += 1
        
    
    # 6. MLP2 Weights
    for i, mlp2 in enumerate(model.mlp2):
        weights = mlp2[0].weight
        plot_weight_heatmap(weights, f'Layer {i} MLP2-0 Weights', current_plot)
        current_plot += 1
        weights = mlp2[2].weight
        plot_weight_heatmap(weights, f'Layer {i} MLP2-2 Weights', current_plot)
        current_plot += 1
    
    # 7. Gate Weights
    if model.use_gate:
        for i, gate in enumerate(model.gatelin):
            weights = gate[0].weight
            plot_weight_heatmap(weights, f'Layer {i} Gate-0 Weights', current_plot)
            current_plot += 1
            weights = gate[2].weight
            plot_weight_heatmap(weights, f'Layer {i} Gate-2 Weights', current_plot)
            current_plot += 1
    
    # 8. Reverse Gate Weights
    if model.use_gate and model.use_rev:
        for i, gate in enumerate(model.revgatelin):
            weights = gate[0].weight
            plot_weight_heatmap(weights, f'Layer {i} Reverse Gate-0 Weights', current_plot)
            current_plot += 1
            weights = gate[2].weight
            plot_weight_heatmap(weights, f'Layer {i} Reverse Gate-2 Weights', current_plot)
            current_plot += 1
    
    plt.tight_layout()
    plt.show()
    plt.savefig(f"{model_name}.png")

if __name__ == "__main__":
    def str2bool(v):
        if isinstance(v, bool):
            return v
        if v.lower() in ("yes", "true", "t", "y", "1"):
            return True
        elif v.lower() in ("no", "false", "f", "n", "0"):
            return False
        else:
            raise argparse.ArgumentTypeError("Boolean value expected.")
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--loadpath", type=str, default="checkpoints/single-pretrain-v3-completion-search/log-25/best_checkpoint/model.safetensors")
    parser.add_argument("--model_name", type=str, default="griffin")
    parser.add_argument("--folder", type=str2bool, default=False)
    args = parser.parse_args()
    model = GriffinMod(hiddim=512, num_mp=4, use_rev=True, use_gate=True)
    model_name = args.model_name
    
    # Load checkpoint directly with torch.load
    if args.folder:
        for file in os.listdir(args.loadpath):
            # file is still a folder, including the model.safetensors
            checkpoint = load_file(os.path.join(args.loadpath, file, "model.safetensors"))
            model.load_state_dict(checkpoint)
            visualize_parameter_heatmaps(model, f"{model_name}_{file}")
    else:
        checkpoint = load_file(os.path.join(args.loadpath, "model.safetensors"))
        model.load_state_dict(checkpoint)
        visualize_parameter_heatmaps(model, args.model_name)