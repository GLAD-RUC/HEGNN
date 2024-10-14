import torch
from torch import nn
from models.md17.gcl import GCL, E_GCL, E_GCL_vel, GCL_rf_vel
from models.md17.layer import GMNLayer


class GNN(nn.Module):
    def __init__(self, input_dim, hidden_nf, device='cpu', act_fn=nn.SiLU(), n_layers=4, attention=0, recurrent=False):
        super(GNN, self).__init__()
        self.hidden_nf = hidden_nf
        self.device = device
        self.n_layers = n_layers
        for i in range(0, n_layers):
            self.add_module("gcl_%d" % i, GCL(self.hidden_nf, self.hidden_nf, self.hidden_nf, edges_in_nf=1 + 1,
                                              act_fn=act_fn, attention=attention, recurrent=recurrent))

        self.decoder = nn.Sequential(nn.Linear(hidden_nf, hidden_nf),
                              act_fn,
                              nn.Linear(hidden_nf, 3))
        self.embedding = nn.Sequential(nn.Linear(input_dim, hidden_nf))
        self.to(self.device)

    def forward(self, nodes, edges, edge_attr=None):
        h = self.embedding(nodes)
        for i in range(0, self.n_layers):
            h, _ = self._modules["gcl_%d" % i](h, edges, edge_attr=edge_attr)
        return self.decoder(h)


def get_velocity_attr(loc, vel, rows, cols):
    diff = loc[cols] - loc[rows]
    norm = torch.norm(diff, p=2, dim=1).unsqueeze(1)
    u = diff/norm
    va, vb = vel[rows] * u, vel[cols] * u
    va, vb = torch.sum(va, dim=1).unsqueeze(1), torch.sum(vb, dim=1).unsqueeze(1)
    return va


class EGNN(nn.Module):
    def __init__(self, in_node_nf, in_edge_nf, hidden_nf, device='cpu', act_fn=nn.LeakyReLU(0.2), n_layers=4, coords_weight=1.0):
        super(EGNN, self).__init__()
        self.hidden_nf = hidden_nf
        self.device = device
        self.n_layers = n_layers
        self.embedding = nn.Linear(in_node_nf, self.hidden_nf)
        for i in range(0, n_layers):
            self.add_module("gcl_%d" % i, E_GCL(self.hidden_nf, self.hidden_nf, self.hidden_nf, edges_in_d=in_edge_nf,
                                                act_fn=act_fn, recurrent=True, coords_weight=coords_weight))
        self.to(self.device)


    def forward(self, h, x, edges, edge_attr, vel=None):
        h = self.embedding(h)
        for i in range(0, self.n_layers):
            h, x, _ = self._modules["gcl_%d" % i](h, edges, x, edge_attr=edge_attr)
        return x


class EGNN_vel(nn.Module):
    def __init__(self, in_node_nf, in_edge_nf, hidden_nf, device='cpu', act_fn=nn.SiLU(), n_layers=4, coords_weight=1.0,
                 recurrent=False, norm_diff=False, tanh=False):
        super(EGNN_vel, self).__init__()
        self.hidden_nf = hidden_nf
        self.device = device
        self.n_layers = n_layers
        self.embedding = nn.Linear(in_node_nf, self.hidden_nf)
        for i in range(0, n_layers):
            self.add_module("gcl_%d" % i, E_GCL_vel(self.hidden_nf, self.hidden_nf, self.hidden_nf, edges_in_d=in_edge_nf,
                                                    act_fn=act_fn, coords_weight=coords_weight, recurrent=recurrent,
                                                    norm_diff=norm_diff, tanh=tanh))
        self.to(self.device)

    def forward(self, h, x, edges, vel, edge_attr):
        h = self.embedding(h)
        for i in range(0, self.n_layers):
            h, x, _ = self._modules["gcl_%d" % i](h, edges, x, vel, edge_attr=edge_attr)
        return x


class GMN(nn.Module):
    def __init__(self, in_node_nf, in_edge_nf, hidden_nf, device='cpu', act_fn=nn.SiLU(), n_layers=4, coords_weight=1.0,
                 recurrent=False, norm_diff=False, tanh=False, learnable=False):
        """
        Graph Mechanics Networks.
        :param in_node_nf: input node feature dimension
        :param in_edge_nf: input edge feature dimension
        :param hidden_nf: hidden dimension
        :param device: device
        :param act_fn: activation function
        :param n_layers: the number of layers
        :param coords_weight: coords weight, inherited from EGNN
        :param recurrent: residual connection on x
        :param norm_diff: normalize the distance, inherited
        :param tanh: Tanh activation, inherited
        :param learnable: use learnable FK
        """
        super(GMN, self).__init__()
        self.hidden_nf = hidden_nf
        self.device = device
        self.n_layers = n_layers
        self.learnable = learnable

        self.embedding = nn.Linear(in_node_nf, self.hidden_nf)
        for i in range(n_layers):
            self.add_module("gcl_%d" % i, GMNLayer(self.hidden_nf, self.hidden_nf, self.hidden_nf,
                                                   edges_in_d=in_edge_nf, act_fn=act_fn,
                                                   coords_weight=coords_weight,
                                                   recurrent=recurrent, norm_diff=norm_diff, tanh=tanh,
                                                   learnable=learnable
                                                   )
                            )

        self.to(self.device)

    def forward(self, h, x, edges, v, c_edge_index, edge_attr):
        h = self.embedding(h)
        for i in range(0, self.n_layers):
            h, x, v, _ = self._modules["gcl_%d" % i](h, edges, x, v, c_edge_index, edge_attr=edge_attr)

        return x, v


class RF_vel(nn.Module):
    def __init__(self, hidden_nf, edge_attr_nf=0, device='cpu', act_fn=nn.SiLU(), n_layers=4):
        super(RF_vel, self).__init__()
        self.hidden_nf = hidden_nf
        self.device = device
        self.n_layers = n_layers

        for i in range(0, n_layers):
            self.add_module("gcl_%d" % i, GCL_rf_vel(nf=hidden_nf, edge_attr_nf=edge_attr_nf, act_fn=act_fn))
        self.to(self.device)

    def forward(self, vel_norm, x, edges, vel, edge_attr):
        for i in range(0, self.n_layers):
            x, _ = self._modules["gcl_%d" % i](x, vel_norm, vel, edges, edge_attr)
        return x


class Baseline(nn.Module):
    def __init__(self, device='cpu'):
        super(Baseline, self).__init__()
        self.dummy = nn.Linear(1, 1)
        self.device = device
        self.to(self.device)

    def forward(self, loc):
        return loc


class Linear(nn.Module):
    def __init__(self, input_nf, output_nf, device='cpu'):
        super(Linear, self).__init__()
        self.linear = nn.Linear(input_nf, output_nf)
        self.device = device
        self.to(self.device)

    def forward(self, input):
        return self.linear(input)


class Linear_dynamics(nn.Module):
    def __init__(self, device='cpu'):
        super(Linear_dynamics, self).__init__()
        self.time = nn.Parameter(torch.ones(1)*0.7)
        self.device = device
        self.to(self.device)

    def forward(self, x, v):
        return x + v*self.time