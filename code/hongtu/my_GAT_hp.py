import torch
from torch import nn
from torch.nn import functional as F
import dgl.function as fn
from dgl.base import DGLError
from dgl.utils import expand_as_pair, check_eq_shape, dgl_warning
from dgl.ops import edge_softmax
import torch.distributed as dist
from .embedding_swap_op import HP_T2D, HP_D2T ,comm_hp_d2t, comm_hp_t2d



class GATConvHP(nn.Module):
    def __init__(self,
                 in_feats,
                 out_feats,
                 num_heads,
                 proc_id,
                 device,
                 nprocs,
                 final_layer = False,
                 norm=None,
                 activation=None
                 ):
        super(GATConvHP, self).__init__()

        self._in_src_feats, self._in_dst_feats = expand_as_pair(in_feats)
        self._out_feats = out_feats
        self.norm = norm
        self.device=device
        self.proc_id=proc_id
        self.nprocs=nprocs
        self.final_layer=final_layer
        self.activation = activation
        self.num_heads = num_heads

        # self.num_heads = num_heads
        # self.fc = nn.Linear(self._in_src_feats, out_feats * num_heads, bias=False) #把输入映射到 num_heads * out_feats 维度
        # self.fc = nn.Linear(in_feats, out_feats, bias=False)
        if isinstance(in_feats, tuple):
            self.fc_src = nn.Linear(self._in_src_feats, out_feats * num_heads, bias=False)
            self.fc_dst = nn.Linear(self._in_dst_feats, out_feats * num_heads, bias=False)
        else:
            self.fc = nn.Linear(self._in_src_feats, out_feats * num_heads, bias=False)

        # 注意力权重参数
        self.attn_l = nn.Parameter(torch.FloatTensor(size=(1, num_heads, out_feats)))
        self.attn_r = nn.Parameter(torch.FloatTensor(size=(1, num_heads, out_feats)))

        self.leaky_relu = nn.LeakyReLU(negative_slope=0.2)
        self.attn_drop = nn.Dropout(0.6)
        
        self.reset_parameters()

    def reset_parameters(self):
        gain = nn.init.calculate_gain('relu')
        if hasattr(self, 'fc'):
            nn.init.xavier_uniform_(self.fc.weight, gain=gain)
        else:
            nn.init.xavier_uniform_(self.fc_src.weight, gain=gain)
            nn.init.xavier_uniform_(self.fc_dst.weight, gain=gain)
        nn.init.xavier_uniform_(self.attn_l, gain=gain)
        nn.init.xavier_uniform_(self.attn_r, gain=gain)

    def forward(self, graph, feat, edge_weight=None):
        with graph.local_scope():
            if isinstance(feat, tuple):
                feat_src = feat[0]
                feat_dst = feat[1]
            else:
                feat_src = feat_dst = feat
                if graph.is_block:
                    feat_dst = feat_src[:graph.number_of_dst_nodes()]

            vertex_split_index = [graph.number_of_dst_nodes() // self.nprocs] * self.nprocs
            vertex_split_index[-1] += graph.number_of_dst_nodes() % self.nprocs
            dim_in_split_index_1 = [(self._in_src_feats * self.num_heads) // self.nprocs] * self.nprocs
            dim_in_split_index_1[-1] += (self._in_src_feats * self.num_heads) % self.nprocs

            dim_in_split_index_2 = [(self._in_src_feats) // self.nprocs] * self.nprocs
            dim_in_split_index_2[-1] += (self._in_src_feats) % self.nprocs

            dim_out_split_index_1 = [(self._out_feats * self.num_heads) // self.nprocs] * self.nprocs
            dim_out_split_index_1[-1] += (self._out_feats * self.num_heads) % self.nprocs

            dim_out_split_index_2 = [(self._out_feats) // self.nprocs] * self.nprocs
            dim_out_split_index_2[-1] += (self._out_feats) % self.nprocs

            if hasattr(self, 'fc'):
                feat_src = self.fc(feat_src).view(-1, self.num_heads, self._out_feats)
                feat_dst = self.fc(feat_dst).view(-1, self.num_heads, self._out_feats)
            else:
                feat_src = self.fc_src(feat_src).view(-1, self.num_heads, self._out_feats)
                feat_dst = self.fc_dst(feat_dst).view(-1, self.num_heads, self._out_feats)

            el = (feat_src * self.attn_l).sum(dim=-1).unsqueeze(-1)
            er = (feat_dst * self.attn_r).sum(dim=-1).unsqueeze(-1)
            graph.srcdata.update({'ft': feat_src, 'el': el})
            graph.dstdata.update({'er': er})
            graph.apply_edges(fn.u_add_v('el', 'er', 'e'))
            e = self.leaky_relu(graph.edata.pop('e'))
            graph.edata['a'] = self.attn_drop(edge_softmax(graph, e))
            graph.update_all(fn.u_mul_e('ft', 'a', 'm'), fn.sum('m', 'ft'))
            
            if self.final_layer:
                rst = graph.dstdata['ft'].mean(1)
                # graph.dstdata['ft'] = graph.dstdata['ft'].mean(1)
                # rst=comm_hp_t2d(graph.dstdata['ft'], vertex_split_index, dim_in_split_index_2, self.proc_id,self.device)
                if self.activation is not None:
                    rst = self.activation(rst)
                if self.norm is not None:
                    rst = self.norm(rst)
                # rst = comm_hp_d2t(rst, vertex_split_index, dim_out_split_index_2, self.proc_id, self.device)
            else:
                rst = graph.dstdata['ft'].flatten(1)
                # graph.dstdata['ft'] = graph.dstdata['ft'].flatten(1)
                # rst=comm_hp_t2d(graph.dstdata['ft'], vertex_split_index, dim_in_split_index_1, self.proc_id,self.device)
                if self.activation is not None:
                    rst = self.activation(rst)
                if self.norm is not None:
                    rst = self.norm(rst)
                # rst = comm_hp_d2t(rst, vertex_split_index, dim_out_split_index_1, self.proc_id, self.device)
            # print(f"Rank {self.proc_id}: rst shape = {rst.shape}")

            return rst