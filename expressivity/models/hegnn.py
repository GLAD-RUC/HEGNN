from functools import partial

import torch
from torch import nn
from torch_scatter import scatter
from torch_geometric.nn import global_add_pool, global_mean_pool

import e3nn

class BaseMLP(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, activation=nn.SiLU(), residual=False, last_act=False):
        super(BaseMLP, self).__init__()
        self.residual = residual
        if residual:
            assert output_dim == input_dim
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            activation,
            nn.Linear(hidden_dim, output_dim),
            activation if last_act else nn.Identity()
        )
    def forward(self, x):
        return x + self.mlp(x) if self.residual else self.mlp(x)

class HEGNN_Layer(nn.Module):
    def __init__(self, edge_attr_dim, hidden_dim, sh_irreps, activation=nn.SiLU()):
        super(HEGNN_Layer, self).__init__()
        self.sh_irreps = sh_irreps
        MLP = partial(BaseMLP, hidden_dim=hidden_dim, activation=activation)
        self.mlp_msg = MLP(input_dim=2 * hidden_dim + edge_attr_dim + 1 + sh_irreps.lmax + 1, output_dim=hidden_dim, last_act=True)
        self.mlp_pos = MLP(input_dim=hidden_dim, output_dim=1)
        self.mlp_node_feat = MLP(input_dim=hidden_dim + hidden_dim, output_dim=hidden_dim)
        # self.mlp_vel = MLP(input_dim=hidden_dim, output_dim=1)

        self.sh_msg = SH_Msg(sh_irreps)
        self.sh_coff = e3nn.o3.FullyConnectedTensorProduct(
            self.sh_irreps, '1x0e', self.sh_irreps, shared_weights=False
        )
        self.mlp_sh = MLP(input_dim=hidden_dim, output_dim=self.sh_coff.weight_numel)

    def forward(self, node_feat, node_pos, node_sh, edge_index, edge_attr=None):
        msg, diff_pos, diff_sh = self.Msg(edge_index, edge_attr, node_feat, node_pos, node_sh)
        msg_agg, pos_agg, sh_agg = self.Agg(edge_index, node_feat.size(0), msg, diff_pos, diff_sh)
        node_feat, node_pos, node_sh = self.Upd(node_feat, node_pos, node_sh, msg_agg, pos_agg, sh_agg)
        return node_feat, node_pos, node_sh
        
    def Msg(self, edge_index, edge_attr, node_feat, node_pos, node_sh):
        row, col = edge_index
        diff_pos = node_pos[row] - node_pos[col]
        dist = torch.norm(diff_pos, p=2, dim=-1).unsqueeze(-1) ** 2
        sh_ip = self.sh_msg(edge_index, node_sh) 
        
        if edge_attr is None:
            msg = torch.cat([node_feat[row], node_feat[col], dist, sh_ip], dim=-1)
        else:
            msg = torch.cat([node_feat[row], node_feat[col], edge_attr, dist, sh_ip], dim=-1)
        msg = self.mlp_msg(msg)
        diff_pos = diff_pos * self.mlp_pos(msg)

        diff_sh = node_sh[row] - node_sh[col]

        one = torch.ones([diff_sh.size(0), 1], device=diff_sh.device)
        diff_sh = self.sh_coff(diff_sh, one, self.mlp_sh(msg))

        return msg, diff_pos, diff_sh
    
    def Agg(self, edge_index, dim_size, msg, diff_pos, diff_sh):
        row, col = edge_index
        msg_agg = scatter(src=msg, index=row, dim=0, dim_size=dim_size, reduce='mean')
        pos_agg = scatter(src=diff_pos, index=row, dim=0, dim_size=dim_size, reduce='mean')
        sh_agg = scatter(src=diff_sh, index=row, dim=0, dim_size=dim_size, reduce='mean')
        return msg_agg, pos_agg, sh_agg
    
    def Upd(self, node_feat, node_pos, node_sh, msg_agg, pos_agg, sh_agg):
        node_sh = node_sh + sh_agg
        node_pos = node_pos + pos_agg # + self.mlp_vel(node_feat) * node_vel
        node_feat = torch.cat([node_feat, msg_agg], dim=-1)
        node_feat = self.mlp_node_feat(node_feat)
        return node_feat, node_pos, node_sh
    
class SH_Msg(nn.Module):
    def __init__(self, sh_irreps):
        super(SH_Msg, self).__init__()
        self.sh_irreps = sh_irreps

    def forward(self, edge_index, node_sh):
        assert node_sh.size(1) == self.sh_irreps.dim
        row, col = edge_index
        temp = node_sh[row] * node_sh[col]

        idx = 0
        ip = torch.zeros([node_sh[row].size(0), self.sh_irreps.lmax + 1], device=node_sh.device)
        for (mul, ir) in self.sh_irreps:
            ip[:, ir.l] = torch.sum(temp[:, idx:idx + ir.dim], dim=-1)
            idx = idx + ir.dim
        return ip

class SH_INIT(nn.Module):
    def __init__(self, edge_attr_dim, hidden_dim, max_ell, activation=nn.SiLU()):
        super(SH_INIT, self).__init__()
        self.sh_irreps = e3nn.o3.Irreps.spherical_harmonics(max_ell)
        self.spherical_harmonics = e3nn.o3.SphericalHarmonics(
            self.sh_irreps, normalize=True, normalization="norm"
        )

        self.sh_coff = e3nn.o3.FullyConnectedTensorProduct(
            self.sh_irreps, '1x0e', self.sh_irreps, shared_weights=False
        )

        MLP = partial(BaseMLP, hidden_dim=hidden_dim, activation=activation)
        self.mlp_sh = MLP(input_dim=2 * hidden_dim + edge_attr_dim + 1, output_dim=self.sh_coff.weight_numel, last_act=False)

        # Use Spherical Harmonics init by pos removed center of mass
        self.sh_coff_CoM = e3nn.o3.FullyConnectedTensorProduct(
            self.sh_irreps, '1x0e', self.sh_irreps, shared_weights=False
        )
        self.mlp_sh_CoM = MLP(input_dim=hidden_dim, output_dim=self.sh_coff.weight_numel, last_act=False)


    def forward(self, node_feat, node_pos, edge_index, edge_attr=None, data_batch=None):
        row, col = edge_index
        diff_pos = node_pos[row] - node_pos[col]
        dist = torch.norm(diff_pos, dim=-1, keepdim=True)
        
        if edge_attr is None:
            msg = torch.cat([dist, node_feat[row], node_feat[col]], dim=-1)
        else:
            msg = torch.cat([dist, node_feat[row], node_feat[col], edge_attr], dim=-1)
        msg = self.mlp_sh(msg)
        diff_sh = self.spherical_harmonics(diff_pos).detach()
        one = torch.ones([diff_sh.size(0), 1], device=diff_sh.device).detach()
        diff_sh = self.sh_coff(diff_sh, one, msg)

        node_sh = scatter(diff_sh, index=row, dim=0, dim_size=node_feat.size(0), reduce='mean')

        # Use Spherical Harmonics init by pos removed center
        pos_CoM = global_mean_pool(node_pos, data_batch)[data_batch]
        node_sh_CoM = self.spherical_harmonics(node_pos - pos_CoM).detach()

        one = torch.ones([node_sh_CoM.size(0), 1], device=diff_sh.device).detach()
        node_sh_CoM = self.sh_coff(node_sh_CoM, one, self.mlp_sh_CoM(node_feat))

        return node_sh + node_sh_CoM

class HEGNNModel(nn.Module):
    def __init__(
            self, 
            num_layers: int = 5, 
            emb_dim: int = 128,
            in_dim: int = 1,
            out_dim: int = 1,
            edge_attr_dim: int = 0,
            max_ell: int = 2,
            all_ell: bool = False,
            activation: nn.Module = nn.SiLU(), 
            # aggr: str = "sum",
            pool: str = "sum",
            residual: bool = True,
            equivariant_pred: bool = True,
            device='cpu'
        ):
        super(HEGNNModel, self).__init__()
        self.num_layer = num_layers
        self.embedding = nn.Embedding(in_dim, emb_dim)
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.max_ell = max_ell
        self.all_ell = all_ell
        self.sh_init = SH_INIT(edge_attr_dim, emb_dim, max_ell, activation)
        self.equivariant_pred = equivariant_pred
        self.layers = nn.ModuleList()
        self.pool = {"mean": global_mean_pool, "sum": global_add_pool}[pool]
        for _ in range(self.num_layer):
            layer = HEGNN_Layer(edge_attr_dim, emb_dim, self.sh_init.sh_irreps, activation=activation)
            self.layers.append(layer)

        MLP = partial(BaseMLP, hidden_dim=emb_dim, output_dim=self.out_dim, activation=activation)
        if self.equivariant_pred:
            # Linear predictor for equivariant tasks using geometric features
            self.pred = nn.Linear(emb_dim + 3 + (max_ell+1)**2, out_dim)
        else:
            # MLP predictor for invariant tasks using only scalar features
            self.pred = MLP(input_dim=emb_dim)
        self.to(device)        

    # def forward(self, node_feat, node_pos, node_vel, edge_index, edge_attr):
    def forward(self, batch):
        node_feat = self.embedding(batch.atoms)
        node_pos = batch.pos
        edge_index = batch.edge_index

        node_sh = self.sh_init(node_feat, node_pos, edge_index, data_batch=batch.batch)
        if not self.all_ell:
            node_sh[:, 0:self.max_ell ** 2] = 0

        for layer in self.layers:
            node_feat, node_pos, node_sh = layer(node_feat, node_pos, node_sh, edge_index)

        if not self.equivariant_pred:
            out = self.pool(node_feat, batch.batch)  # (n, d) -> (batch_size, d)
        else:
            out = self.pool(torch.cat([node_feat, node_pos, node_sh], dim=-1), batch.batch)

        return self.pred(out)