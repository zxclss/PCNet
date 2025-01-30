import math
from typing import Dict, List, Mapping, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_cluster import radius
from torch_cluster import radius_graph
from torch_geometric.data import Batch
from torch_geometric.data import HeteroData
from torch_geometric.utils import dense_to_sparse

from layers import AttentionLayer
from layers import FourierEmbedding
from layers import MLPLayer
from utils import angle_between_2d_vectors
from utils import bipartite_dense_to_sparse
from utils import weight_init
from utils import wrap_angle
from config import PivotConfig

class PCNetDecoder(nn.Module):

    def __init__(self,
                 dataset: str,
                 input_dim: int,
                 hidden_dim: int,
                 output_dim: int,
                 output_head: bool,
                 num_historical_steps: int,
                 num_future_steps: int,
                 num_modes: int,
                 num_recurrent_steps: int,
                 num_t2m_steps: Optional[int],
                 pl2m_radius: float,
                 a2m_radius: float,
                 num_freq_bands: int,
                 num_layers: int,
                 num_heads: int,
                 head_dim: int,
                 dropout: float) -> None:
        super(PCNetDecoder, self).__init__()
        self.dataset = dataset
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.output_head = output_head
        self.num_historical_steps = num_historical_steps
        self.num_future_steps = num_future_steps
        self.num_modes = num_modes
        self.num_recurrent_steps = num_recurrent_steps
        self.num_t2m_steps = num_t2m_steps if num_t2m_steps is not None else num_historical_steps
        self.pl2m_radius = pl2m_radius
        self.a2m_radius = a2m_radius
        self.num_freq_bands = num_freq_bands
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.dropout = dropout

        input_dim_r_t = 4
        input_dim_r_pl2m = 3
        input_dim_r_a2m = 3

        self.mode_emb = nn.Embedding(num_modes, hidden_dim)
        self.r_t2m_emb = FourierEmbedding(input_dim=input_dim_r_t, hidden_dim=hidden_dim, num_freq_bands=num_freq_bands)
        self.r_pl2m_emb = FourierEmbedding(input_dim=input_dim_r_pl2m, hidden_dim=hidden_dim,
                                           num_freq_bands=num_freq_bands)
        self.r_a2m_emb = FourierEmbedding(input_dim=input_dim_r_a2m, hidden_dim=hidden_dim,
                                          num_freq_bands=num_freq_bands)
        self.y_emb = FourierEmbedding(input_dim=output_dim + output_head, hidden_dim=hidden_dim,
                                      num_freq_bands=num_freq_bands)
        self.traj_emb = nn.GRU(input_size=hidden_dim, hidden_size=hidden_dim, num_layers=1, bias=True,
                               batch_first=False, dropout=0.0, bidirectional=False)
        self.traj_emb_h0 = nn.Parameter(torch.zeros(1, hidden_dim))
        self.t2m_propose_attn_layers = nn.ModuleList(
            [AttentionLayer(hidden_dim=hidden_dim, num_heads=num_heads, head_dim=head_dim, dropout=dropout,
                            bipartite=True, has_pos_emb=True) for _ in range(num_layers)]
        )
        self.pl2m_propose_attn_layers = nn.ModuleList(
            [AttentionLayer(hidden_dim=hidden_dim, num_heads=num_heads, head_dim=head_dim, dropout=dropout,
                            bipartite=True, has_pos_emb=True) for _ in range(num_layers)]
        )
        self.a2m_propose_attn_layers = nn.ModuleList(
            [AttentionLayer(hidden_dim=hidden_dim, num_heads=num_heads, head_dim=head_dim, dropout=dropout,
                            bipartite=True, has_pos_emb=True) for _ in range(num_layers)]
        )
        self.m2m_propose_attn_layer = AttentionLayer(hidden_dim=hidden_dim, num_heads=num_heads, head_dim=head_dim,
                                                    dropout=dropout, bipartite=False, has_pos_emb=False)
        if PivotConfig.traj_pred_recurrent:
            self.to_loc_propose_pos = MLPLayer(input_dim=hidden_dim, hidden_dim=hidden_dim,
                                            output_dim=num_future_steps * output_dim // num_recurrent_steps)
            self.to_scale_propose_pos = MLPLayer(input_dim=hidden_dim, hidden_dim=hidden_dim,
                                                output_dim=num_future_steps * output_dim // num_recurrent_steps)
        else:
            self.to_loc_propose_pos = MLPLayer(input_dim=hidden_dim, hidden_dim=hidden_dim,
                                            output_dim=num_future_steps * output_dim)
            self.to_scale_propose_pos = MLPLayer(input_dim=hidden_dim, hidden_dim=hidden_dim,
                                                output_dim=num_future_steps * output_dim)
        if PivotConfig.trajectory_refine:
            self.t2m_refine_attn_layers = nn.ModuleList(
                [AttentionLayer(hidden_dim=hidden_dim, num_heads=num_heads, head_dim=head_dim, dropout=dropout,
                                bipartite=True, has_pos_emb=True) for _ in range(num_layers)]
            )
            self.pl2m_refine_attn_layers = nn.ModuleList(
                [AttentionLayer(hidden_dim=hidden_dim, num_heads=num_heads, head_dim=head_dim, dropout=dropout,
                                bipartite=True, has_pos_emb=True) for _ in range(num_layers)]
            )
            self.a2m_refine_attn_layers = nn.ModuleList(
                [AttentionLayer(hidden_dim=hidden_dim, num_heads=num_heads, head_dim=head_dim, dropout=dropout,
                                bipartite=True, has_pos_emb=True) for _ in range(num_layers)]
            )
            self.m2m_refine_attn_layer = AttentionLayer(hidden_dim=hidden_dim, num_heads=num_heads, head_dim=head_dim,
                                                        dropout=dropout, bipartite=False, has_pos_emb=False)
        self.t2m_propose_pivot_attn_layers = nn.ModuleList(
            [AttentionLayer(hidden_dim=hidden_dim, num_heads=num_heads, head_dim=head_dim, dropout=dropout,
                            bipartite=True, has_pos_emb=True) for _ in range(num_layers)]
        )
        self.pl2m_propose_pivot_attn_layers = nn.ModuleList(
            [AttentionLayer(hidden_dim=hidden_dim, num_heads=num_heads, head_dim=head_dim, dropout=dropout,
                            bipartite=True, has_pos_emb=True) for _ in range(num_layers)]
        )
        self.a2m_propose_pivot_attn_layers = nn.ModuleList(
            [AttentionLayer(hidden_dim=hidden_dim, num_heads=num_heads, head_dim=head_dim, dropout=dropout,
                            bipartite=True, has_pos_emb=True) for _ in range(num_layers)]
        )
        self.m2m_propose_pivot_attn_layer = AttentionLayer(hidden_dim=hidden_dim, num_heads=num_heads, head_dim=head_dim,
                                                     dropout=dropout, bipartite=False, has_pos_emb=False)
        self.num_of_pivot = PivotConfig.pivot_recurrent_num_config[-1]
        if PivotConfig.pivot_refine:
            self.t2m_refine_pivot_attn_layers = nn.ModuleList(
                [AttentionLayer(hidden_dim=hidden_dim, num_heads=num_heads, head_dim=head_dim, dropout=dropout,
                                bipartite=True, has_pos_emb=True) for _ in range(num_layers)]
            )
            self.pl2m_refine_pivot_attn_layers = nn.ModuleList(
                [AttentionLayer(hidden_dim=hidden_dim, num_heads=num_heads, head_dim=head_dim, dropout=dropout,
                                bipartite=True, has_pos_emb=True) for _ in range(num_layers)]
            )
            self.a2m_refine_pivot_attn_layers = nn.ModuleList(
                [AttentionLayer(hidden_dim=hidden_dim, num_heads=num_heads, head_dim=head_dim, dropout=dropout,
                                bipartite=True, has_pos_emb=True) for _ in range(num_layers)]
            )
            self.m2m_refine_pivot_attn_layer = AttentionLayer(hidden_dim=hidden_dim, num_heads=num_heads, head_dim=head_dim,
                                                        dropout=dropout, bipartite=False, has_pos_emb=False)
            self.to_loc_pivot_refine_pos = MLPLayer(input_dim=hidden_dim, hidden_dim=hidden_dim,
                                            output_dim=self.num_of_pivot * output_dim)
            self.to_scale_pivot_refine_pos = MLPLayer(input_dim=hidden_dim, hidden_dim=hidden_dim,
                                            output_dim=self.num_of_pivot * output_dim)
        self.multiscale_num_of_pivot = PivotConfig.pivot_recurrent_num_config
        self.to_loc_pivot_propose_pos_recurrent1 = MLPLayer(input_dim=hidden_dim, hidden_dim=hidden_dim,
                                                output_dim=self.multiscale_num_of_pivot[0] * output_dim)
        self.to_scale_pivot_propose_pos_recurrent1 = MLPLayer(input_dim=hidden_dim, hidden_dim=hidden_dim,
                                                output_dim=self.multiscale_num_of_pivot[0] * output_dim)
        self.to_loc_pivot_propose_pos_recurrent2 = MLPLayer(input_dim=hidden_dim, hidden_dim=hidden_dim,
                                                output_dim=self.multiscale_num_of_pivot[1] * output_dim)
        self.to_scale_pivot_propose_pos_recurrent2 = MLPLayer(input_dim=hidden_dim, hidden_dim=hidden_dim,
                                                output_dim=self.multiscale_num_of_pivot[1] * output_dim)
        self.to_loc_pivot_propose_pos_recurrent3 = MLPLayer(input_dim=hidden_dim, hidden_dim=hidden_dim,
                                                output_dim=self.multiscale_num_of_pivot[2] * output_dim)
        self.to_scale_pivot_propose_pos_recurrent3 = MLPLayer(input_dim=hidden_dim, hidden_dim=hidden_dim,
                                                output_dim=self.multiscale_num_of_pivot[2] * output_dim)
        self.to_loc_pivot_propose_pos_recurrent = [
            self.to_loc_pivot_propose_pos_recurrent1,
            self.to_loc_pivot_propose_pos_recurrent2,
            self.to_loc_pivot_propose_pos_recurrent3
        ]
        self.to_scale_pivot_propose_pos_recurrent = [
            self.to_scale_pivot_propose_pos_recurrent1, self.to_scale_pivot_propose_pos_recurrent2, self.to_scale_pivot_propose_pos_recurrent3
        ]
        # self.to_loc_pivot_propose_pos = MLPLayer(input_dim=hidden_dim, hidden_dim=hidden_dim,
        #                                         output_dim=self.num_of_pivot * output_dim)
        # self.to_scale_pivot_propose_pos = MLPLayer(input_dim=hidden_dim, hidden_dim=hidden_dim,
        #                                         output_dim=self.num_of_pivot * output_dim)
        if PivotConfig.trajectory_refine:
            self.to_loc_refine_pos = MLPLayer(input_dim=hidden_dim, hidden_dim=hidden_dim,
                                            output_dim=num_future_steps * output_dim)
            self.to_scale_refine_pos = MLPLayer(input_dim=hidden_dim, hidden_dim=hidden_dim,
                                                output_dim=num_future_steps * output_dim)
        if output_head:
            self.to_loc_propose_head = MLPLayer(input_dim=hidden_dim, hidden_dim=hidden_dim,
                                                output_dim=num_future_steps // num_recurrent_steps)
            self.to_conc_propose_head = MLPLayer(input_dim=hidden_dim, hidden_dim=hidden_dim,
                                                 output_dim=num_future_steps // num_recurrent_steps)
            self.to_loc_refine_head = MLPLayer(input_dim=hidden_dim, hidden_dim=hidden_dim, output_dim=num_future_steps)
            self.to_conc_refine_head = MLPLayer(input_dim=hidden_dim, hidden_dim=hidden_dim,
                                                output_dim=num_future_steps)
        else:
            self.to_loc_propose_head = None
            self.to_conc_propose_head = None
            self.to_loc_refine_head = None
            self.to_conc_refine_head = None
        self.to_pi = MLPLayer(input_dim=hidden_dim, hidden_dim=hidden_dim, output_dim=1)
        self.apply(weight_init)

    def forward(self,
                data: HeteroData,
                scene_enc: Mapping[str, torch.Tensor],
                current_epoch) -> Dict[str, torch.Tensor]:
        pos_m = data['agent']['position'][:, self.num_historical_steps - 1, :self.input_dim]
        head_m = data['agent']['heading'][:, self.num_historical_steps - 1]
        head_vector_m = torch.stack([head_m.cos(), head_m.sin()], dim=-1)

        x_t = scene_enc['x_a'].reshape(-1, self.hidden_dim)
        x_pl = scene_enc['x_pl'][:, self.num_historical_steps - 1].repeat(self.num_modes, 1)
        x_a = scene_enc['x_a'][:, -1].repeat(self.num_modes, 1)
        m = self.mode_emb.weight.repeat(scene_enc['x_a'].size(0), 1)

        mask_src = data['agent']['valid_mask'][:, :self.num_historical_steps].contiguous()
        mask_src[:, :self.num_historical_steps - self.num_t2m_steps] = False
        mask_dst = data['agent']['predict_mask'].any(dim=-1, keepdim=True).repeat(1, self.num_modes)

        pos_t = data['agent']['position'][:, :self.num_historical_steps, :self.input_dim].reshape(-1, self.input_dim)
        head_t = data['agent']['heading'][:, :self.num_historical_steps].reshape(-1)
        edge_index_t2m = bipartite_dense_to_sparse(mask_src.unsqueeze(2) & mask_dst[:, -1:].unsqueeze(1))
        rel_pos_t2m = pos_t[edge_index_t2m[0]] - pos_m[edge_index_t2m[1]]
        rel_head_t2m = wrap_angle(head_t[edge_index_t2m[0]] - head_m[edge_index_t2m[1]])
        r_t2m = torch.stack(
            [torch.norm(rel_pos_t2m[:, :2], p=2, dim=-1),
             angle_between_2d_vectors(ctr_vector=head_vector_m[edge_index_t2m[1]], nbr_vector=rel_pos_t2m[:, :2]),
             rel_head_t2m,
             (edge_index_t2m[0] % self.num_historical_steps) - self.num_historical_steps + 1], dim=-1)
        r_t2m = self.r_t2m_emb(continuous_inputs=r_t2m, categorical_embs=None)
        edge_index_t2m = bipartite_dense_to_sparse(mask_src.unsqueeze(2) & mask_dst.unsqueeze(1))
        r_t2m = r_t2m.repeat_interleave(repeats=self.num_modes, dim=0)

        pos_pl = data['map_polygon']['position'][:, :self.input_dim]
        orient_pl = data['map_polygon']['orientation']
        edge_index_pl2m = radius(
            x=pos_m[:, :2],
            y=pos_pl[:, :2],
            r=self.pl2m_radius,
            batch_x=data['agent']['batch'] if isinstance(data, Batch) else None,
            batch_y=data['map_polygon']['batch'] if isinstance(data, Batch) else None,
            max_num_neighbors=300)
        edge_index_pl2m = edge_index_pl2m[:, mask_dst[edge_index_pl2m[1], 0]]
        rel_pos_pl2m = pos_pl[edge_index_pl2m[0]] - pos_m[edge_index_pl2m[1]]
        rel_orient_pl2m = wrap_angle(orient_pl[edge_index_pl2m[0]] - head_m[edge_index_pl2m[1]])
        r_pl2m = torch.stack(
            [torch.norm(rel_pos_pl2m[:, :2], p=2, dim=-1),
             angle_between_2d_vectors(ctr_vector=head_vector_m[edge_index_pl2m[1]], nbr_vector=rel_pos_pl2m[:, :2]),
             rel_orient_pl2m], dim=-1)
        r_pl2m = self.r_pl2m_emb(continuous_inputs=r_pl2m, categorical_embs=None)
        edge_index_pl2m = torch.cat([edge_index_pl2m + i * edge_index_pl2m.new_tensor(
            [[data['map_polygon']['num_nodes']], [data['agent']['num_nodes']]]) for i in range(self.num_modes)], dim=1)
        r_pl2m = r_pl2m.repeat(self.num_modes, 1)

        edge_index_a2m = radius_graph(
            x=pos_m[:, :2],
            r=self.a2m_radius,
            batch=data['agent']['batch'] if isinstance(data, Batch) else None,
            loop=False,
            max_num_neighbors=300)
        edge_index_a2m = edge_index_a2m[:, mask_src[:, -1][edge_index_a2m[0]] & mask_dst[edge_index_a2m[1], 0]]
        rel_pos_a2m = pos_m[edge_index_a2m[0]] - pos_m[edge_index_a2m[1]]
        rel_head_a2m = wrap_angle(head_m[edge_index_a2m[0]] - head_m[edge_index_a2m[1]])
        r_a2m = torch.stack(
            [torch.norm(rel_pos_a2m[:, :2], p=2, dim=-1),
             angle_between_2d_vectors(ctr_vector=head_vector_m[edge_index_a2m[1]], nbr_vector=rel_pos_a2m[:, :2]),
             rel_head_a2m], dim=-1)
        r_a2m = self.r_a2m_emb(continuous_inputs=r_a2m, categorical_embs=None)
        edge_index_a2m = torch.cat(
            [edge_index_a2m + i * edge_index_a2m.new_tensor([data['agent']['num_nodes']]) for i in
             range(self.num_modes)], dim=1)
        r_a2m = r_a2m.repeat(self.num_modes, 1)

        edge_index_m2m = dense_to_sparse(mask_dst.unsqueeze(2) & mask_dst.unsqueeze(1))[0]

        locs_propose_pos: List[Optional[torch.Tensor]] = [None] * self.num_recurrent_steps
        scales_propose_pos: List[Optional[torch.Tensor]] = [None] * self.num_recurrent_steps
        locs_propose_head: List[Optional[torch.Tensor]] = [None] * self.num_recurrent_steps
        concs_propose_head: List[Optional[torch.Tensor]] = [None] * self.num_recurrent_steps
        locs_pivot_propose_pos: List[Optional[torch.Tensor]] = [None] * self.num_recurrent_steps
        scales_pivot_propose_pos: List[Optional[torch.Tensor]] = [None] * self.num_recurrent_steps
        if PivotConfig.using_pivot_learning:
            for t in range(self.num_recurrent_steps):
                for i in range(self.num_layers):
                    m = m.reshape(-1, self.hidden_dim)
                    m = self.t2m_propose_pivot_attn_layers[i]((x_t, m), r_t2m, edge_index_t2m)
                    m = m.reshape(-1, self.num_modes, self.hidden_dim).transpose(0, 1).reshape(-1, self.hidden_dim)
                    m = self.pl2m_propose_pivot_attn_layers[i]((x_pl, m), r_pl2m, edge_index_pl2m)
                    m = self.a2m_propose_pivot_attn_layers[i]((x_a, m), r_a2m, edge_index_a2m)
                    m = m.reshape(self.num_modes, -1, self.hidden_dim).transpose(0, 1).reshape(-1, self.hidden_dim)
                m = self.m2m_propose_pivot_attn_layer(m, None, edge_index_m2m)
                m = m.reshape(-1, self.num_modes, self.hidden_dim)
                locs_pivot_propose_pos[t] = self.to_loc_pivot_propose_pos_recurrent[t](m)
                scales_pivot_propose_pos[t] = self.to_scale_pivot_propose_pos_recurrent[t](m)

                if t == 0:
                    locs_pivot_propose_pos[t] = torch.cumsum(
                        locs_pivot_propose_pos[t].view(-1, self.num_modes, self.multiscale_num_of_pivot[t], self.output_dim),
                        dim=-2
                    )
                    scales_pivot_propose_pos[t] = torch.cumsum(
                        F.elu_(
                            scales_pivot_propose_pos[t].view(-1, self.num_modes, self.multiscale_num_of_pivot[t], self.output_dim),
                            alpha=1.0
                        ) + 1.0,
                        dim=-2
                    ) + 0.1
                elif t == 1:
                    locs_pivot_propose_pos[t] = torch.cumsum(
                        locs_pivot_propose_pos[t].view(-1, self.num_modes, self.multiscale_num_of_pivot[t], self.output_dim),
                        dim=-2
                    )
                    scales_pivot_propose_pos[t] = torch.cumsum(
                        F.elu_(
                            scales_pivot_propose_pos[t].view(-1, self.num_modes, self.multiscale_num_of_pivot[t], self.output_dim),
                            alpha=1.0
                        ) + 1.0,
                        dim=-2
                    ) + 0.1
                elif t == 2:
                    locs_pivot_propose_pos[t] = torch.cumsum(
                        locs_pivot_propose_pos[t].view(-1, self.num_modes, self.multiscale_num_of_pivot[t], self.output_dim),
                        dim=-2
                    )
                    scales_pivot_propose_pos[t] = torch.cumsum(
                        F.elu_(
                            scales_pivot_propose_pos[t].view(-1, self.num_modes, self.multiscale_num_of_pivot[t], self.output_dim),
                            alpha=1.0
                        ) + 1.0,
                        dim=-2
                    ) + 0.1
                if PivotConfig.pivot_recurrent_fusing_query:
                    pass
                    # m = self.y_emb(locs_pivot_propose_pos[t].detach().view(-1, self.output_dim))    # [BNKP, 2] -> [BNKP, 128]
                    # m = m.reshape(-1, self.multiscale_num_of_pivot[t], self.hidden_dim).transpose(0, 1)   # [BNK, P, 128] -> [P, BNK, 128]
                    # m = self.traj_emb(m, self.traj_emb_h0.unsqueeze(1).repeat(1, m.size(1), 1))[1].squeeze(0) # GRU: [BNK, 128] 时序融合
            m = self.y_emb(locs_pivot_propose_pos[-1].detach().view(-1, self.output_dim))
            m = m.reshape(-1, self.num_of_pivot, self.hidden_dim).transpose(0, 1)   # [BNK, P, 128]
            m = self.traj_emb(m, self.traj_emb_h0.unsqueeze(1).repeat(1, m.size(1), 1))[1].squeeze(0) # GRU: [BNK, 128] 时序融合
            if PivotConfig.pivot_refine:
                for i in range(self.num_layers):
                    m = self.t2m_refine_pivot_attn_layers[i]((x_t, m), r_t2m, edge_index_t2m)
                    m = m.reshape(-1, self.num_modes, self.hidden_dim).transpose(0, 1).reshape(-1, self.hidden_dim)
                    m = self.pl2m_refine_pivot_attn_layers[i]((x_pl, m), r_pl2m, edge_index_pl2m)
                    m = self.a2m_refine_pivot_attn_layers[i]((x_a, m), r_a2m, edge_index_a2m)
                    m = m.reshape(self.num_modes, -1, self.hidden_dim).transpose(0, 1).reshape(-1, self.hidden_dim)
                m = self.m2m_refine_pivot_attn_layer(m, None, edge_index_m2m)
                m = m.reshape(-1, self.num_modes, self.hidden_dim)  # [BN, K, 128]
                loc_pivot_refine_pos = self.to_loc_pivot_refine_pos(m).view(-1, self.num_modes, self.num_of_pivot, self.output_dim)
                loc_pivot_refine_pos = loc_pivot_refine_pos + locs_pivot_propose_pos[-1].detach()
                scale_pivot_refine_pos = F.elu_(
                    self.to_scale_pivot_refine_pos(m).view(-1, self.num_modes, self.num_of_pivot, self.output_dim),
                    alpha=1.0) + 1.0 + 0.1
                m = self.y_emb(loc_pivot_refine_pos.detach().view(-1, self.output_dim))    # [BNKP, 2] -> [BNKP, 128]
                m = m.reshape(-1, self.num_of_pivot, self.hidden_dim).transpose(0, 1)   # [BNK, P, 128] -> [P, BNK, 128]
                m = self.traj_emb(m, self.traj_emb_h0.unsqueeze(1).repeat(1, m.size(1), 1))[1].squeeze(0) # GRU: [BNK, 128] 时序融合
            else:
                loc_pivot_refine_pos = locs_pivot_propose_pos[-1].detach()
                scale_pivot_refine_pos = scales_pivot_propose_pos[-1].detach()
            if PivotConfig.traj_pred_recurrent:
                for t in range(self.num_recurrent_steps):
                    for i in range(self.num_layers):
                        m = m.reshape(-1, self.hidden_dim)
                        m = self.t2m_propose_attn_layers[i]((x_t, m), r_t2m, edge_index_t2m)
                        m = m.reshape(-1, self.num_modes, self.hidden_dim).transpose(0, 1).reshape(-1, self.hidden_dim)
                        m = self.pl2m_propose_attn_layers[i]((x_pl, m), r_pl2m, edge_index_pl2m)
                        m = self.a2m_propose_attn_layers[i]((x_a, m), r_a2m, edge_index_a2m)
                        m = m.reshape(self.num_modes, -1, self.hidden_dim).transpose(0, 1).reshape(-1, self.hidden_dim)
                    m = self.m2m_propose_attn_layer(m, None, edge_index_m2m)
                    m = m.reshape(-1, self.num_modes, self.hidden_dim)
                    locs_propose_pos[t] = self.to_loc_propose_pos(m)
                    scales_propose_pos[t] = self.to_scale_propose_pos(m)
                loc_propose_pos = torch.cumsum(torch.cat(locs_propose_pos, dim=-1).view(-1, self.num_modes, self.num_of_pivot, self.num_future_steps // self.num_of_pivot, self.output_dim), dim=-2)
                loc_propose_pos[..., 1:, :, :] += loc_pivot_refine_pos[..., :-1, None, :].detach()
                loc_propose_pos = loc_propose_pos.view(-1, self.num_modes, self.num_future_steps, self.output_dim)
                scale_propose_pos = (torch.cumsum(
                    F.elu_(
                        torch.cat(scales_propose_pos, dim=-1).view(-1, self.num_modes, self.num_of_pivot, self.num_future_steps // self.num_of_pivot, self.output_dim),
                        alpha=1.0
                    ) + 1.0,
                dim=-2) + 0.1).view(-1, self.num_modes, self.num_future_steps, self.output_dim)
            else:
                for i in range(self.num_layers):
                    m = m.reshape(-1, self.hidden_dim)
                    m = self.t2m_propose_attn_layers[i]((x_t, m), r_t2m, edge_index_t2m)
                    m = m.reshape(-1, self.num_modes, self.hidden_dim).transpose(0, 1).reshape(-1, self.hidden_dim)
                    m = self.pl2m_propose_attn_layers[i]((x_pl, m), r_pl2m, edge_index_pl2m)
                    m = self.a2m_propose_attn_layers[i]((x_a, m), r_a2m, edge_index_a2m)
                    m = m.reshape(self.num_modes, -1, self.hidden_dim).transpose(0, 1).reshape(-1, self.hidden_dim)
                m = self.m2m_propose_attn_layer(m, None, edge_index_m2m)
                m = m.reshape(-1, self.num_modes, self.hidden_dim)
                locs_propose_pos = self.to_loc_propose_pos(m)
                scales_propose_pos = self.to_scale_propose_pos(m)
                loc_propose_pos = torch.cumsum(locs_propose_pos.view(-1, self.num_modes, self.num_of_pivot, self.num_future_steps // self.num_of_pivot, self.output_dim), dim=-2)
                loc_propose_pos[..., 1:, :, :] += loc_pivot_refine_pos[..., :-1, None, :].detach()
                loc_propose_pos = loc_propose_pos.view(-1, self.num_modes, self.num_future_steps, self.output_dim)
                scale_propose_pos = (torch.cumsum(
                    F.elu_(
                        scales_propose_pos.view(-1, self.num_modes, self.num_of_pivot, self.num_future_steps // self.num_of_pivot, self.output_dim),
                        alpha=1.0
                    ) + 1.0,
                dim=-2) + 0.1).view(-1, self.num_modes, self.num_future_steps, self.output_dim)
            if PivotConfig.trajectory_refine:
                m = self.y_emb(loc_propose_pos.detach().view(-1, self.output_dim))
                m = m.reshape(-1, self.num_future_steps, self.hidden_dim).transpose(0, 1)
                m = self.traj_emb(m, self.traj_emb_h0.unsqueeze(1).repeat(1, m.size(1), 1))[1].squeeze(0)
                for i in range(self.num_layers):
                    m = self.t2m_refine_attn_layers[i]((x_t, m), r_t2m, edge_index_t2m)
                    m = m.reshape(-1, self.num_modes, self.hidden_dim).transpose(0, 1).reshape(-1, self.hidden_dim)
                    m = self.pl2m_refine_attn_layers[i]((x_pl, m), r_pl2m, edge_index_pl2m)
                    m = self.a2m_refine_attn_layers[i]((x_a, m), r_a2m, edge_index_a2m)
                    m = m.reshape(self.num_modes, -1, self.hidden_dim).transpose(0, 1).reshape(-1, self.hidden_dim)
                m = self.m2m_refine_attn_layer(m, None, edge_index_m2m)
                m = m.reshape(-1, self.num_modes, self.hidden_dim)
                loc_refine_pos_delta = self.to_loc_refine_pos(m).view(-1, self.num_modes, self.num_future_steps, self.output_dim)
                loc_refine_pos = loc_refine_pos_delta + loc_propose_pos.detach()
                scale_refine_pos = F.elu_(
                    self.to_scale_refine_pos(m).view(-1, self.num_modes, self.num_future_steps, self.output_dim),
                    alpha=1.0) + 1.0 + 0.1
            else:
                loc_refine_pos = loc_propose_pos.detach()
                scale_refine_pos = scale_propose_pos.detach()
        pi = self.to_pi(m).squeeze(-1)

        return {
            'locs_pivot_propose_pos': locs_pivot_propose_pos,
            'scales_pivot_propose_pos': scales_pivot_propose_pos,
            'loc_pivot_refine_pos': loc_pivot_refine_pos,
            'scale_pivot_refine_pos': scale_pivot_refine_pos,
            'loc_propose_pos': loc_propose_pos,
            'scale_propose_pos': scale_propose_pos,
            'loc_refine_pos': loc_refine_pos,
            'scale_refine_pos': scale_refine_pos,
            'pi': pi,
        }
