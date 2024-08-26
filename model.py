import torch
from torch import nn as nn

class GraphGenerator(nn.Module):
    def __init__(self, batch_size, dropout, latent_dim, units, bond_dim, n_atoms, atom_dim):
        super(GraphGenerator, self).__init__()
        self.batch_size = batch_size
        self.dropout_rate = dropout
        self.latent_dim = latent_dim
        self.bond_dim = bond_dim
        self.n_atoms = n_atoms
        self.atom_dim = atom_dim

        layers = []
        for i, unit in enumerate(units):
            if i == 0:
                layers.append(nn.Linear(latent_dim, unit))
            else:
                layers.append(nn.Linear(units[i - 1], unit))
            layers.append(nn.Tanh())
            layers.append(nn.Dropout(self.dropout_rate))
        self.input_sequential = nn.Sequential(*layers)

        self.adj_linear = nn.Linear(units[-1], (self.bond_dim * self.n_atoms * self.n_atoms))
        self.feat_linear = nn.Linear(units[-1], (self.n_atoms * self.atom_dim))

        self.softmax_adj = nn.Softmax(dim=1)
        self.softmax_feat = nn.Softmax(dim=2)

    def forward(self, inputs):
        z = self.input_sequential(inputs)
        # generate the adjacency matrix
        adj_input = self.adj_linear(z)
        adj_reshaped = adj_input.view((self.batch_size, self.bond_dim, self.n_atoms, self.n_atoms))
        adj_updated = (adj_reshaped + adj_reshaped.permute(0, 1, 3, 2)) / 2
        adj_final = self.softmax_adj(adj_updated)

        # generate the feature matrix
        feat_input = self.feat_linear(z)
        feat_reshaped = feat_input.view((self.batch_size, self.n_atoms, self.atom_dim))
        feat_final = self.softmax_feat(feat_reshaped)

        return [adj_final, feat_final]

class GraphConv(nn.Module):
    def __init__(self, units, features_dim, edges, dropout_rate):
        super(GraphConv, self).__init__()
        self.units = units
        self.features_dim = features_dim
        self.edges = edges
        self.dropout_rate = dropout_rate

        self.linear1 = [nn.Linear(self.features_dim, self.units) for _ in range(self.edges - 1)]
        self.linear2 = nn.Linear(self.features_dim, self.units)
        self.tanh = nn.Tanh()
        self.dropout = nn.Dropout(self.dropout_rate)

    def forward(self, inputs):
        length = len(inputs)
        adjacency, features = inputs[0], inputs[1]
        if length > 2:
            hidden = inputs[2]
            annotations = torch.cat((hidden, features), dim=-1)
        else:
            annotations = features
        res = torch.stack([linear(annotations) for linear in self.linear1], dim=1)
        matrix = torch.matmul(adjacency[:, :-1, :, :], res)
        total = torch.sum(matrix, dim=1) + self.linear2(annotations)
        return adjacency, features, self.dropout(self.tanh(total))

class MultiGraphConv(nn.Module):
    def __init__(self, units, features_dim, edges, dropout_rate):
        super(MultiGraphConv, self).__init__()
        self.units = units
        self.features_dim = features_dim
        self.edges = edges
        self.dropout_rate = dropout_rate

        self.first_conv = GraphConv(self.units[0], self.features_dim, self.edges, self.dropout_rate)
        self.convs = [
            GraphConv(unit, (self.features_dim + self.units[idx]), self.edges, self.dropout_rate)
            for idx, unit in enumerate(self.units[1:])
        ]

    def forward(self, inputs):
        outputs = self.first_conv(inputs)
        return outputs
        # for layer in self.convs:
        #     outputs = layer(outputs)
        # return outputs[-1]

class GraphAggregation(nn.Module):
    def __init__(self, units, hidden_dim, dropout_rate):
        super(GraphAggregation, self).__init__()
        self.units = units
        self.hidden_dim = hidden_dim
        self.dropout_rate = dropout_rate

        self.linear = nn.Linear(self.hidden_dim, self.units)
        self.sigmoid = nn.Sigmoid()
        self.tanh = nn.Tanh()
        self.dropout = nn.Dropout(self.dropout_rate)

    def forward(self, inputs):
        i, j = inputs, inputs
        linear_i = self.sigmoid(self.linear(i))
        linear_j = self.tanh(self.linear(j))
        result = torch.sum(linear_i * linear_j, dim=1)
        return self.dropout(self.tanh(result))

class Discriminator(nn.Module):
    def __init__(self, gconv_units, units, features_dim, edges, dropout_rate):
        super(Discriminator, self).__init__()
        self.gconv_units = gconv_units
        self.units = units
        self.features_dim = features_dim
        self.edges = edges
        self.hidden_dim = self.gconv_units[0][-1]
        self.dropout_rate = dropout_rate

        self.gconv = MultiGraphConv(self.gconv_units[0], self.features_dim, self.edges, self.dropout_rate)
        self.aggregation = GraphAggregation(self.gconv_units[-1], self.hidden_dim, self.dropout_rate)

        self.linear = nn.Linear(self.gconv_units[-1], self.units[0])
        self.output = nn.Linear(self.units[0], self.units[1])
        self.tanh = nn.Tanh()
        self.dropout = nn.Dropout(self.dropout_rate)

    def forward(self, adjacency, features):
        inputs = [adjacency, features]
        hidden = self.gconv(inputs)
        return hidden
        # aggregation = self.aggregation(hidden)
        # linear = self.dropout(self.tanh(self.linear(aggregation)))
        # output = self.output(linear)
        # return output