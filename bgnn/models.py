import torch
import torch.nn as nn
from torch_geometric.nn import BatchNorm, GCNConv, LayerNorm, SAGEConv, Sequential, GATConv, APPNP
import torch.nn.functional as F
import torch_geometric.transforms as T

class GAT(torch.nn.Module):
    def __init__(self, layer_sizes, heads_counts, layernorm, drop_rate):
        super().__init__()

        self.layernorm = layernorm
        self.drop_rate = drop_rate
        
        self.conv1 = GATConv(layer_sizes[0], layer_sizes[1], heads=heads_counts[0], dropout=drop_rate)
        # On the Pubmed dataset, use heads=8 in conv2.
        self.conv2 = GATConv(layer_sizes[1] * heads_counts[0], layer_sizes[-1], heads=heads_counts[1], concat=False,
                             dropout=drop_rate)

        if self.layernorm:
            self.layer_norms0 = LayerNorm(layer_sizes[1] * heads_counts[0])
            self.layer_norms1 = LayerNorm(layer_sizes[-1])

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        x = F.dropout(x, p=self.drop_rate, training=self.training)
        if self.layernorm:
            x = self.layer_norms0(F.elu(self.conv1(x, edge_index)))
        else:
            x = F.elu(self.conv1(x, edge_index))
        x = F.dropout(x, p=self.drop_rate, training=self.training)
        x = self.conv2(x, edge_index)
        if self.layernorm:
            x = self.layer_norms1(x)
        return F.log_softmax(x, dim=1)
        # return x

    def reset_parameters(self):
        self.conv1.reset_parameters()
        self.conv2.reset_parameters()
        if self.layernorm:
            self.layer_norms0.reset_parameters()
            self.layer_norms1.reset_parameters()

##########################################################
class GAT_link(torch.nn.Module):
    def __init__(self, layer_sizes, heads_counts, layernorm, drop_rate):
        super().__init__()

        self.layernorm = layernorm
        self.drop_rate = drop_rate
        
        self.conv1 = GATConv(layer_sizes[0], layer_sizes[1], heads=heads_counts[0], dropout=drop_rate)
        # On the Pubmed dataset, use heads=8 in conv2.
        self.conv2 = GATConv(layer_sizes[1] * heads_counts[0], layer_sizes[-1], heads=heads_counts[1], concat=False,
                             dropout=drop_rate)

        if self.layernorm:
            self.layer_norms0 = LayerNorm(layer_sizes[1] * heads_counts[0])
            self.layer_norms1 = LayerNorm(layer_sizes[-1])

    def forward(self, data, edge_label, edge_label_index):
        x, edge_index = data.x, data.edge_index
        x = F.dropout(x, p=self.drop_rate, training=self.training)
        if self.layernorm:
            x = self.layer_norms0(F.elu(self.conv1(x, edge_index)))
        else:
            x = F.elu(self.conv1(x, edge_index))
        x = F.dropout(x, p=self.drop_rate, training=self.training)
        x = self.conv2(x, edge_index)
        if self.layernorm:
            x = self.layer_norms1(x)
        #return F.log_softmax(x, dim=1)
        return x

    def reset_parameters(self):
        self.conv1.reset_parameters()
        self.conv2.reset_parameters()
        if self.layernorm:
            self.layer_norms0.reset_parameters()
            self.layer_norms1.reset_parameters()
    
    def decode(self, z, edge_label_index):
        return (z[edge_label_index[0]] * z[edge_label_index[1]]).sum(dim=-1)  # product of a pair of nodes on each edge

    def decode_all(self, z):
        prob_adj = z @ z.t()
        return (prob_adj > 0).nonzero(as_tuple=False).t()
##########################################################

class GCN(torch.nn.Module):
    def __init__(self, layer_sizes, layernorm, drop_rate):
        super().__init__()
        self.layernorm = layernorm
        self.drop_rate = drop_rate

        self.conv1 = GCNConv(layer_sizes[0], layer_sizes[1])
        self.conv2 = GCNConv(layer_sizes[1], layer_sizes[-1])

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        x = F.relu(self.conv1(x, edge_index))
        x = F.dropout(x, p=self.drop_rate, training=self.training)
        x = self.conv2(x, edge_index)
        return F.log_softmax(x, dim=1)
        # return x

    def reset_parameters(self):
        self.conv1.reset_parameters()
        self.conv2.reset_parameters()

##################################################################
class GCN_link(torch.nn.Module):
    def __init__(self, layer_sizes, layernorm, drop_rate):
        super().__init__()
        self.layernorm = layernorm
        self.drop_rate = drop_rate

        self.conv1 = GCNConv(layer_sizes[0], layer_sizes[1])
        self.conv2 = GCNConv(layer_sizes[1], layer_sizes[-1])

    def forward(self, data, edge_label, edge_label_index):
        x, edge_index = data.x, data.edge_index
        x = F.relu(self.conv1(x, edge_index))
        x = F.dropout(x, p=self.drop_rate, training=self.training)
        x = self.conv2(x, edge_index)
        #return F.log_softmax(x, dim=1)
        return x
    
    def decode(self, z, edge_label_index):
        return (z[edge_label_index[0]] * z[edge_label_index[1]]).sum(dim=-1)  # product of a pair of nodes on each edge

    def decode_all(self, z):
        prob_adj = z @ z.t()
        return (prob_adj > 0).nonzero(as_tuple=False).t()

    def reset_parameters(self):
        self.conv1.reset_parameters()
        self.conv2.reset_parameters()
###################################################################

class GraphSage(torch.nn.Module):
    def __init__(self, layer_sizes, layernorm, drop_rate):
        super().__init__()
        self.layernorm = layernorm
        self.drop_rate = drop_rate

        self.conv1 = SAGEConv(layer_sizes[0], layer_sizes[1], normalize=self.layernorm)
        self.conv2 = SAGEConv(layer_sizes[1], layer_sizes[-1], normalize=self.layernorm)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        x = F.relu(self.conv1(x, edge_index))
        x = F.dropout(x, p=self.drop_rate, training=self.training)
        x = self.conv2(x, edge_index)
        return F.log_softmax(x, dim=1)
        # return x

    def reset_parameters(self):
        self.conv1.reset_parameters()
        self.conv2.reset_parameters()

###################################
class GraphSage_link(torch.nn.Module):
    def __init__(self, layer_sizes, layernorm, drop_rate):
        super().__init__()
        self.layernorm = layernorm
        self.drop_rate = drop_rate

        self.conv1 = SAGEConv(layer_sizes[0], layer_sizes[1], normalize=self.layernorm)
        self.conv2 = SAGEConv(layer_sizes[1], layer_sizes[-1], normalize=self.layernorm)

    def forward(self, data, edge_label, edge_label_index): #forward
        x, edge_index = data.x, data.edge_index
        x = F.relu(self.conv1(x, edge_index))
        x = F.dropout(x, p=self.drop_rate, training=self.training)
        x = self.conv2(x, edge_index)
        #return F.log_softmax(x, dim=1)
        return x
    
    def decode(self, z, edge_label_index):
        return (z[edge_label_index[0]] * z[edge_label_index[1]]).sum(dim=-1)  # product of a pair of nodes on each edge

    def decode_all(self, z):
        prob_adj = z @ z.t()
        return (prob_adj > 0).nonzero(as_tuple=False).t()

    def reset_parameters(self):
        self.conv1.reset_parameters()
        self.conv2.reset_parameters()
###################################

class GraphSAGE_GCN(nn.Module):
    def __init__(self, layer_sizes, layernorm):
        super().__init__()
        self.layers = nn.ModuleList()
        self.skip_lins = nn.ModuleList()
        self.activations = nn.ModuleList()
        self.layernorm = layernorm
        if self.layernorm:
            self.layer_norms = nn.ModuleList()
 
        num = 0
        for in_dim, out_dim in zip(layer_sizes[:-1], layer_sizes[1:]):
            self.layers.append(SAGEConv(in_dim, out_dim, root_weight=True))
            if num != (len(layer_sizes)-1):
                self.skip_lins.append(nn.Linear(layer_sizes[0], out_dim, bias=False))
            if self.layernorm:
                self.layer_norms.append(LayerNorm(out_dim))
            self.activations.append(nn.PReLU(1))
            num += 1

        # self.convs = nn.ModuleList([
        #     SAGEConv(input_size, hidden_size, root_weight=True),
        #     SAGEConv(hidden_size, hidden_size, root_weight=True),
        #     SAGEConv(hidden_size, embedding_size, root_weight=True),
        # ])

        # self.skip_lins = nn.ModuleList([
        #     nn.Linear(input_size, hidden_size, bias=False),
        #     nn.Linear(input_size, hidden_size, bias=False),
        #     ])

        # self.layer_norms = nn.ModuleList([
        #     LayerNorm(hidden_size),
        #     LayerNorm(hidden_size),
        #     LayerNorm(embedding_size),
        # ])

        # self.activations = nn.ModuleList([
        #     nn.PReLU(1),
        #     nn.PReLU(1),
        #     nn.PReLU(1),
        # ])

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        batch = data.batch if hasattr(data, 'batch') else None

        hs = []
        for i, conv in enumerate(self.layers):
            if i == 0:
                h = x
            else:
                h = self.skip_lins[i-1](x)
            for samp in hs:
                h += samp
            h = conv(h, edge_index)
            h = self.layer_norms[i](h, batch)
            h = self.activations[i](h)
            hs.append(h)

        return h

        # h1 = self.convs[0](x, edge_index)
        # h1 = self.layer_norms[0](h1, batch)
        # h1 = self.activations[0](h1)

        # x_skip_1 = self.skip_lins[0](x)
        # h2 = self.convs[1](h1 + x_skip_1, edge_index)
        # h2 = self.layer_norms[1](h2, batch)
        # h2 = self.activations[1](h2)

        # x_skip_2 = self.skip_lins[1](x)
        # ret = self.convs[2](h1 + h2 + x_skip_2, edge_index)
        # ret = self.layer_norms[2](ret, batch)
        # ret = self.activations[2](ret)
        # return ret

    def reset_parameters(self):
        for m in self.layers:
            m.reset_parameters()
        for m in self.skip_lins:
            m.reset_parameters()
        for m in self.activations:
            m.weight.data.fill_(0.25)
        if self.layernorm:
            for m in self.layer_norms:
                m.reset_parameters()


