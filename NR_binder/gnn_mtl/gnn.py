import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.nn import Linear, BatchNorm1d, ModuleList
from torch.nn import GRUCell, Parameter

from torch import Tensor
from torch_geometric.typing import Adj, OptTensor
from typing import Optional

# from torch_geometric.nn.models import AttentiveFP
from torch_geometric.nn import (
    TransformerConv, GINEConv, GATConv, MessagePassing,
    GCNConv, global_add_pool,
)
from torch_scatter import scatter_mean

from torch_geometric.utils import softmax

from .dmpnn import DMPNN

class GATEConv(MessagePassing):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        edge_dim: int,
        dropout: float = 0.0,
    ):
        super().__init__(aggr='add', node_dim=0)

        self.dropout = dropout

        self.att_l = Parameter(torch.Tensor(1, out_channels))
        self.att_r = Parameter(torch.Tensor(1, in_channels))

        self.lin1 = Linear(in_channels + edge_dim, out_channels, False)
        self.lin2 = Linear(out_channels, out_channels, False)

        self.bias = Parameter(torch.Tensor(out_channels))


    def forward(self, x: Tensor, edge_index: Adj, edge_attr: Tensor) -> Tensor:
        # propagate_type: (x: Tensor, edge_attr: Tensor)
        out = self.propagate(edge_index, x=x, edge_attr=edge_attr, size=None)
        out = out + self.bias
        return out

    def message(self, x_j: Tensor, x_i: Tensor, edge_attr: Tensor,
                index: Tensor, ptr: OptTensor,
                size_i: Optional[int]) -> Tensor:

        x_j = F.leaky_relu_(self.lin1(torch.cat([x_j, edge_attr], dim=-1)))
        alpha_j = (x_j * self.att_l).sum(dim=-1)
        alpha_i = (x_i * self.att_r).sum(dim=-1)
        alpha = alpha_j + alpha_i
        alpha = F.leaky_relu_(alpha)
        alpha = softmax(alpha, index, ptr, size_i)
        alpha = F.dropout(alpha, p=self.dropout, training=self.training)
        return self.lin2(x_j) * alpha.unsqueeze(-1)
    
class SuperNodeAttention(nn.Module):
    def __init__(self, hidden_channels, num_timesteps=3, dropout=0.2):
        super(SuperNodeAttention, self).__init__()
        self.hidden_channels = hidden_channels
        self.num_timesteps = num_timesteps
        self.dropout = dropout

        self.attention_mlp = nn.Sequential(
            Linear(hidden_channels * 2, hidden_channels),
            nn.LeakyReLU(0.2),
            nn.Dropout(dropout),
            Linear(hidden_channels, 1)
        )

        self.atom_project = Linear(hidden_channels, hidden_channels)

        self.gru = GRUCell(hidden_channels, hidden_channels)

    def forward(self, atom_features, batch):
        super_node = global_add_pool(atom_features, batch)

        all_attention_weights = []

        for t in range(self.num_timesteps):
            super_node_expanded = super_node[batch] 

            attention_input = torch.cat([
                super_node_expanded,
                atom_features
            ], dim=-1)

            attention_scores = self.attention_mlp(attention_input)

            prev = torch.are_deterministic_algorithms_enabled()
            torch.use_deterministic_algorithms(False)

            attention_weights = softmax(attention_scores, batch)
            
            torch.use_deterministic_algorithms(prev)

            all_attention_weights.append(attention_weights.squeeze(-1))

            atom_context = self.atom_project(atom_features)

            molecule_context = global_add_pool(attention_weights * atom_context, batch)

            molecule_context = F.elu(molecule_context)
            super_node = self.gru(molecule_context, super_node)

        final_attention_weights = all_attention_weights[-1]

        return super_node, final_attention_weights


class AttentiveFPReadout(nn.Module):
    def __init__(self, hidden_channels, num_timesteps=3, dropout=0.2):
        super(AttentiveFPReadout, self).__init__()
        self.hidden_channels = hidden_channels
        self.num_timesteps = num_timesteps
        self.dropout = dropout

        self.mol_conv = GATConv(
            hidden_channels, hidden_channels, heads=1,
            dropout=dropout, add_self_loops=False, negative_slope=0.01
        )

        self.mol_gru = GRUCell(hidden_channels, hidden_channels)

    def forward(self, atom_features, batch, return_attention=False):
        row = torch.arange(batch.size(0), device=batch.device)
        edge_index_mol = torch.stack([row, batch], dim=0)

        out = global_add_pool(atom_features, batch).relu_()

        all_attention_weights = []

        for t in range(self.num_timesteps):
            h, (_, alpha) = self.mol_conv(
                (atom_features, out), edge_index_mol,
                return_attention_weights=True
            )
            h = F.elu_(h)
            h = F.dropout(h, p=self.dropout, training=self.training)
            out = self.mol_gru(h, out).relu_()

            if alpha.dim() > 1:
                alpha = alpha.mean(dim=-1)
            all_attention_weights.append(alpha)

        if return_attention:
            final_attention = all_attention_weights[-1]
            return out, final_attention

        return out


class GraphTransformerWithAttention(nn.Module):
    """GraphTransformer + SuperNodeAttention"""
    def __init__(self, in_channels, hidden_channels, out_channels, edge_dim, num_layers,
                 n_heads=1, dropout=0.2, num_timesteps=3, protein_encodings=None):
        super(GraphTransformerWithAttention, self).__init__()

        self.num_layers = num_layers
        self.embedding_size = hidden_channels
        self.n_heads = n_heads
        self.dropout_rate = dropout
        self.num_tasks = out_channels  # 12

        self.conv_layers = ModuleList([
            TransformerConv(
                in_channels if i == 0 else self.embedding_size,
                self.embedding_size,
                heads=self.n_heads,
                dropout=self.dropout_rate,
                edge_dim=edge_dim,
                beta=True
            ) for i in range(self.num_layers)
        ])

        self.transf_layers = ModuleList([
            Linear(self.embedding_size * self.n_heads, self.embedding_size)
            for _ in range(self.num_layers)
        ])
        self.bn_layers = ModuleList([
            BatchNorm1d(self.embedding_size) for _ in range(self.num_layers)
        ])

        self.super_node_attention = SuperNodeAttention(self.embedding_size, num_timesteps, dropout)

        if protein_encodings is not None:
            self.protein_encodings = nn.Parameter(protein_encodings)
            self.protein_dim = protein_encodings.size(1)  # 1418

            self.protein_proj = nn.Linear(self.protein_dim, self.protein_dim // 2)
            self.protein_dim = self.protein_dim // 2 

        else:
            self.protein_encodings = None
            self.protein_dim = 0

            self.protein_proj = None

        input_dim = self.embedding_size + self.protein_dim
        self.task_heads = ModuleList([
            nn.Sequential(
                Linear(input_dim, input_dim // 2),
                nn.ReLU(),
                nn.Dropout(dropout),
                Linear(input_dim // 2, 1)
            ) for _ in range(self.num_tasks)
        ])

    def forward(self, x, edge_index, edge_attr, batch_index, return_attention=False):
        local_representation = []
        for i in range(self.num_layers):
            x = self.conv_layers[i](x, edge_index, edge_attr)
            x = torch.relu(self.transf_layers[i](x))
            x = self.bn_layers[i](x)
            local_representation.append(x)

        x = sum(local_representation)/len(local_representation)

        mol_repr, attention_weights = self.super_node_attention(x, batch_index)
        # mol_repr: [batch_size, embedding_size]

        outputs = []
        for i in range(self.num_tasks):
            if self.protein_encodings is not None:
                protein_feat = self.protein_encodings[i]

                protein_feat = self.protein_proj(protein_feat)

                protein_feat_batch = protein_feat.unsqueeze(0).expand(mol_repr.size(0), -1)
                combined = torch.cat([mol_repr, protein_feat_batch], dim=-1)
            else:
                combined = mol_repr

            task_output = self.task_heads[i](combined)  # [batch_size, 1]
            outputs.append(task_output)

        output = torch.cat(outputs, dim=-1)

        if return_attention:
            return output, attention_weights
        return output


class GINWithAttention(nn.Module):
    """GIN + SuperNodeAttention"""
    def __init__(self, in_channels, hidden_channels, out_channels, edge_dim, num_layers,
                 dropout=0.2, num_timesteps=3, protein_encodings=None):
        super(GINWithAttention, self).__init__()

        self.num_layers = num_layers
        self.embedding_size = hidden_channels
        self.dropout_rate = dropout
        self.num_tasks = out_channels  # 12

        self.conv_layers = ModuleList([
            GINEConv(Linear(in_channels, self.embedding_size), edge_dim=edge_dim)
        ])
        for _ in range(self.num_layers - 1):
            self.conv_layers.append(
                GINEConv(Linear(self.embedding_size, self.embedding_size), edge_dim=edge_dim)
            )

        self.super_node_attention = SuperNodeAttention(self.embedding_size, num_timesteps, dropout)

        if protein_encodings is not None:
            # self.register_buffer('protein_encodings', protein_encodings)  # [12, 1418]
            self.protein_encodings = nn.Parameter(protein_encodings)

            self.protein_dim = protein_encodings.size(1)  # 1418

            self.protein_proj = nn.Linear(self.protein_dim, self.protein_dim // 2)
            self.protein_dim = self.protein_dim // 2 

        else:
            self.protein_encodings = None
            self.protein_dim = 0

            self.protein_proj = None

        input_dim = self.embedding_size + self.protein_dim
        self.task_heads = ModuleList([
            nn.Sequential(
                Linear(input_dim, input_dim // 2),
                nn.ReLU(),
                nn.Dropout(dropout),
                Linear(input_dim // 2, 1)
            ) for _ in range(self.num_tasks)
        ])

    def forward(self, x, edge_index, edge_attr, batch, return_attention=False):
        for conv in self.conv_layers:
            x = conv(x, edge_index, edge_attr)
            x = F.relu(x)

        mol_repr, attention_weights = self.super_node_attention(x, batch)
        # mol_repr: [batch_size, embedding_size]

        outputs = []
        for i in range(self.num_tasks):
            if self.protein_encodings is not None:
                protein_feat = self.protein_encodings[i]

                protein_feat = self.protein_proj(protein_feat)

                protein_feat_batch = protein_feat.unsqueeze(0).expand(mol_repr.size(0), -1)
                combined = torch.cat([mol_repr, protein_feat_batch], dim=-1)
            else:
                combined = mol_repr

            task_output = self.task_heads[i](combined)  # [batch_size, 1]
            outputs.append(task_output)

        output = torch.cat(outputs, dim=-1)

        if return_attention:
            return output, attention_weights
        return output


class GCNWithAttention(nn.Module):
    """GCN + SuperNodeAttention"""
    def __init__(self, in_channels, hidden_channels, out_channels, edge_dim, num_layers,
                 dropout=0.2, num_timesteps=3, protein_encodings=None):
        super(GCNWithAttention, self).__init__()

        self.num_layers = num_layers
        self.embedding_size = hidden_channels
        self.dropout_rate = dropout
        self.num_tasks = out_channels  # 12

        self.conv_layers = ModuleList([GCNConv(in_channels, self.embedding_size)])
        for _ in range(self.num_layers - 1):
            self.conv_layers.append(GCNConv(self.embedding_size, self.embedding_size))

        self.edge_linear = Linear(edge_dim, self.embedding_size)

        self.super_node_attention = SuperNodeAttention(self.embedding_size, num_timesteps, dropout)

        if protein_encodings is not None:
            # self.register_buffer('protein_encodings', protein_encodings)  # [12, 1418]
            self.protein_encodings = nn.Parameter(protein_encodings)

            self.protein_dim = protein_encodings.size(1)  # 1418

            self.protein_proj = nn.Linear(self.protein_dim, self.protein_dim // 2)
            self.protein_dim = self.protein_dim // 2 

        else:
            self.protein_encodings = None
            self.protein_dim = 0

            self.protein_proj = None

        input_dim = self.embedding_size + self.protein_dim
        self.task_heads = ModuleList([
            nn.Sequential(
                Linear(input_dim, input_dim // 2),
                nn.ReLU(),
                nn.Dropout(dropout),
                Linear(input_dim // 2, 1)
            ) for _ in range(self.num_tasks)
        ])

    def forward(self, x, edge_index, edge_attr, batch, return_attention=False):
        edge_features = self.edge_linear(edge_attr)

        for conv in self.conv_layers:
            x = conv(x, edge_index)
            x = F.relu(x)

        x_edge_agg = scatter_mean(edge_features, edge_index[0], dim=0, dim_size=x.size(0))
        x = x + x_edge_agg

        mol_repr, attention_weights = self.super_node_attention(x, batch)
        # mol_repr: [batch_size, embedding_size]

        outputs = []
        for i in range(self.num_tasks):
            if self.protein_encodings is not None:
                protein_feat = self.protein_encodings[i]

                protein_feat = self.protein_proj(protein_feat)

                protein_feat_batch = protein_feat.unsqueeze(0).expand(mol_repr.size(0), -1)
                combined = torch.cat([mol_repr, protein_feat_batch], dim=-1)
            else:
                combined = mol_repr

            task_output = self.task_heads[i](combined)  # [batch_size, 1]
            outputs.append(task_output)

        output = torch.cat(outputs, dim=-1)

        if return_attention:
            return output, attention_weights
        return output


class GATWithAttention(nn.Module):
    """GAT + SuperNodeAttention"""
    def __init__(self, in_channels, hidden_channels, out_channels, edge_dim, num_layers,
                 dropout=0.2, num_timesteps=3, protein_encodings=None):
        super(GATWithAttention, self).__init__()

        self.num_layers = num_layers
        self.embedding_size = hidden_channels
        self.dropout_rate = dropout
        self.num_tasks = out_channels  # 12

        self.conv_layers = ModuleList([
            GATConv(in_channels, self.embedding_size, edge_dim=edge_dim)
        ])
        for _ in range(self.num_layers - 1):
            self.conv_layers.append(
                GATConv(self.embedding_size, self.embedding_size, edge_dim=edge_dim)
            )

        self.super_node_attention = SuperNodeAttention(self.embedding_size, num_timesteps, dropout)

        if protein_encodings is not None:
            # self.register_buffer('protein_encodings', protein_encodings)  # [12, 1418]
            self.protein_encodings = nn.Parameter(protein_encodings)
            self.protein_dim = protein_encodings.size(1)  # 1418
            self.protein_proj = nn.Linear(self.protein_dim, self.protein_dim // 2)
            self.protein_dim = self.protein_dim // 2 
        else:
            self.protein_encodings = None
            self.protein_dim = 0

            self.protein_proj = None

        input_dim = self.embedding_size + self.protein_dim
        self.task_heads = ModuleList([
            nn.Sequential(
                Linear(input_dim, input_dim // 2),
                nn.ReLU(),
                nn.Dropout(dropout),
                Linear(input_dim // 2, 1)
            ) for _ in range(self.num_tasks)
        ])

    def forward(self, x, edge_index, edge_attr, batch, return_attention=False):
        for conv in self.conv_layers:
            x = conv(x, edge_index, edge_attr)
            x = F.relu(x)

        mol_repr, attention_weights = self.super_node_attention(x, batch)
        # mol_repr: [batch_size, embedding_size]

        outputs = []
        for i in range(self.num_tasks):
            if self.protein_encodings is not None:
                protein_feat = self.protein_encodings[i]

                protein_feat = self.protein_proj(protein_feat)

                protein_feat_batch = protein_feat.unsqueeze(0).expand(mol_repr.size(0), -1)
                combined = torch.cat([mol_repr, protein_feat_batch], dim=-1)
            else:
                combined = mol_repr

            task_output = self.task_heads[i](combined)  # [batch_size, 1]
            outputs.append(task_output)

        output = torch.cat(outputs, dim=-1)

        if return_attention:
            return output, attention_weights
        return output


class AttentiveFPWithAttention(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, edge_dim, num_layers,
                 num_timesteps=3, dropout=0.2, protein_encodings=None):
        super(AttentiveFPWithAttention, self).__init__()

        self.in_channels = in_channels
        self.hidden_channels = hidden_channels
        self.out_channels = out_channels
        self.edge_dim = edge_dim
        self.num_layers = num_layers
        self.dropout = dropout
        self.num_tasks = out_channels  # 12

        self.lin1 = Linear(in_channels, hidden_channels)

        self.gate_conv = GATEConv(hidden_channels, hidden_channels, edge_dim, dropout)
        self.gru = GRUCell(hidden_channels, hidden_channels)

        self.atom_convs = ModuleList()
        self.atom_grus = ModuleList()
        for _ in range(num_layers - 1):
            conv = GATConv(hidden_channels, hidden_channels, dropout=dropout,
                          add_self_loops=False, negative_slope=0.01)
            self.atom_convs.append(conv)
            self.atom_grus.append(GRUCell(hidden_channels, hidden_channels))

        self.readout = AttentiveFPReadout(hidden_channels, num_timesteps, dropout)

        if protein_encodings is not None:
            # self.register_buffer('protein_encodings', protein_encodings)  # [12, 1418]

            self.protein_encodings = nn.Parameter(protein_encodings)
            self.protein_dim = protein_encodings.size(1)  # 1418
            self.protein_proj = nn.Linear(self.protein_dim, self.protein_dim // 2)
            self.protein_dim = self.protein_dim // 2 

        else:
            self.protein_encodings = None
            self.protein_dim = 0

            self.protein_proj = None

        input_dim = hidden_channels + self.protein_dim
        self.task_heads = ModuleList([
            nn.Sequential(
                Linear(input_dim, input_dim // 2),
                nn.ReLU(),
                nn.Dropout(dropout),
                Linear(input_dim // 2, 1)
            ) for _ in range(self.num_tasks)
        ])

    def forward(self, x, edge_index, edge_attr, batch, return_attention=False):
        x = F.leaky_relu_(self.lin1(x))
        h = F.elu_(self.gate_conv(x, edge_index, edge_attr))
        h = F.dropout(h, p=self.dropout, training=self.training)
        x = self.gru(h, x).relu_()

        for conv, gru in zip(self.atom_convs, self.atom_grus):
            h = F.elu_(conv(x, edge_index))
            h = F.dropout(h, p=self.dropout, training=self.training)
            x = gru(h, x).relu_()

        if return_attention:
            mol_repr, attention_weights = self.readout(x, batch, return_attention=True)
        else:
            mol_repr = self.readout(x, batch, return_attention=False)
            attention_weights = None
        # mol_repr: [batch_size, hidden_channels]

        outputs = []
        for i in range(self.num_tasks):
            if self.protein_encodings is not None:
                protein_feat = self.protein_encodings[i]

                protein_feat = self.protein_proj(protein_feat)

                protein_feat_batch = protein_feat.unsqueeze(0).expand(mol_repr.size(0), -1)
                combined = torch.cat([mol_repr, protein_feat_batch], dim=-1)
            else:
                combined = mol_repr

            task_output = self.task_heads[i](combined)  # [batch_size, 1]
            outputs.append(task_output)

        output = torch.cat(outputs, dim=-1)

        if return_attention:
            return output, attention_weights
        return output


class DMPNNWithAttention(nn.Module):
    def __init__(self, node_in_channels, edge_in_channels, hidden_channels, out_channels,
                 num_layers, num_timesteps=3, dropout=0.2, protein_encodings=None):
        super(DMPNNWithAttention, self).__init__()

        self.hidden_channels = hidden_channels
        self.dropout = dropout
        self.num_tasks = out_channels  # 12

        self.dmpnn = DMPNN(node_in_channels, edge_in_channels, hidden_channels, num_layers)

        self.super_node_attention = SuperNodeAttention(hidden_channels, num_timesteps, dropout)

        if protein_encodings is not None:
            # self.register_buffer('protein_encodings', protein_encodings)  # [12, 1418]
            self.protein_encodings = nn.Parameter(protein_encodings)
            self.protein_dim = protein_encodings.size(1)  # 1418

            self.protein_proj = nn.Linear(self.protein_dim, self.protein_dim // 2)
            self.protein_dim = self.protein_dim // 2 
        else:
            self.protein_encodings = None
            self.protein_dim = 0
            self.protein_proj = None

        input_dim = hidden_channels + self.protein_dim
        self.task_heads = ModuleList([
            nn.Sequential(
                Linear(input_dim, input_dim // 2),
                nn.ReLU(),
                nn.Dropout(dropout),
                Linear(input_dim // 2, 1)
            ) for _ in range(self.num_tasks)
        ])

    def forward(self, x, edge_index, rev_edge_index, edge_attr, batch, return_attention=False):
        x = self.dmpnn(x, edge_index, rev_edge_index, edge_attr, batch)

        mol_repr, attention_weights = self.super_node_attention(x, batch)
        # mol_repr: [batch_size, hidden_channels]

        outputs = []
        for i in range(self.num_tasks):
            if self.protein_encodings is not None:
                protein_feat = self.protein_encodings[i]

                protein_feat = self.protein_proj(protein_feat)

                protein_feat_batch = protein_feat.unsqueeze(0).expand(mol_repr.size(0), -1)
                combined = torch.cat([mol_repr, protein_feat_batch], dim=-1)
            else:
                combined = mol_repr

            task_output = self.task_heads[i](combined)  # [batch_size, 1]
            outputs.append(task_output)

        output = torch.cat(outputs, dim=-1)

        if return_attention:
            return output, attention_weights
        return output



def create_gnn_model(model_name, in_channels, hidden_channels, out_channels,
                     edge_dim, num_layers, dropout=0.2, num_timesteps=3,
                     protein_encodings=None, **kwargs):
    model_name = model_name.lower()

    if model_name == 'gt':
        n_heads = kwargs.get('n_heads', 1)
        return GraphTransformerWithAttention(
            in_channels, hidden_channels, out_channels, edge_dim,
            num_layers, n_heads, dropout, num_timesteps, protein_encodings
        )
    elif model_name == 'gin':
        return GINWithAttention(
            in_channels, hidden_channels, out_channels, edge_dim,
            num_layers, dropout, num_timesteps, protein_encodings
        )
    elif model_name == 'gcn':
        return GCNWithAttention(
            in_channels, hidden_channels, out_channels, edge_dim,
            num_layers, dropout, num_timesteps, protein_encodings
        )
    elif model_name == 'gat':
        return GATWithAttention(
            in_channels, hidden_channels, out_channels, edge_dim,
            num_layers, dropout, num_timesteps, protein_encodings
        )
    elif model_name == 'afp':
        return AttentiveFPWithAttention(
            in_channels, hidden_channels, out_channels, edge_dim,
            num_layers, num_timesteps, dropout, protein_encodings
        )
    elif model_name == 'dmpnn':
        node_in_channels = kwargs.get('node_in_channels', in_channels)
        edge_in_channels = kwargs.get('edge_in_channels', edge_dim)
        return DMPNNWithAttention(
            node_in_channels, edge_in_channels, hidden_channels, out_channels,
            num_layers, num_timesteps, dropout, protein_encodings
        )
    else:
        raise ValueError(f"Unknown model name: {model_name}. "
                        f"Choose from 'GraphTransformer', 'GIN', 'GCN', 'GAT', 'AttentiveFP', 'DMPNN'")

