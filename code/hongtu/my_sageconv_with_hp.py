"""Torch Module for GraphSAGE layer"""
# pylint: disable= no-member, arguments-differ, invalid-name
import torch
from torch import nn
from torch.nn import functional as F
import dgl.function as fn
from dgl.base import DGLError
from dgl.utils import expand_as_pair, check_eq_shape, dgl_warning
import torch.distributed as dist
from .embedding_swap_op import HP_T2D, HP_D2T ,comm_hp_d2t, comm_hp_t2d



class SAGEConvHP(nn.Module):
    r"""GraphSAGE layer from `Inductive Representation Learning on
    Large Graphs <https://arxiv.org/pdf/1706.02216.pdf>`__

    .. math::
        h_{\mathcal{N}(i)}^{(l+1)} &= \mathrm{aggregate}
        \left(\{h_{j}^{l}, \forall j \in \mathcal{N}(i) \}\right)

        h_{i}^{(l+1)} &= \sigma \left(W \cdot \mathrm{concat}
        (h_{i}^{l}, h_{\mathcal{N}(i)}^{l+1}) \right)

        h_{i}^{(l+1)} &= \mathrm{norm}(h_{i}^{(l+1)})

    If a weight tensor on each edge is provided, the aggregation becomes:

    .. math::
        h_{\mathcal{N}(i)}^{(l+1)} = \mathrm{aggregate}
        \left(\{e_{ji} h_{j}^{l}, \forall j \in \mathcal{N}(i) \}\right)

    where :math:`e_{ji}` is the scalar weight on the edge from node :math:`j` to node :math:`i`.
    Please make sure that :math:`e_{ji}` is broadcastable with :math:`h_j^{l}`.

    Parameters
    ----------
    in_feats : int, or pair of ints
        Input feature size; i.e, the number of dimensions of :math:`h_i^{(l)}`.

        SAGEConv can be applied on homogeneous graph and unidirectional
        `bipartite graph <https://docs.dgl.ai/generated/dgl.bipartite.html?highlight=bipartite>`__.
        If the layer applies on a unidirectional bipartite graph, ``in_feats``
        specifies the input feature size on both the source and destination nodes.  If
        a scalar is given, the source and destination node feature size would take the
        same value.

        If aggregator type is ``gcn``, the feature size of source and destination nodes
        are required to be the same.
    out_feats : int
        Output feature size; i.e, the number of dimensions of :math:`h_i^{(l+1)}`.
    aggregator_type : str
        Aggregator type to use (``mean``, ``gcn``, ``pool``, ``lstm``).
    feat_drop : float
        Dropout rate on features, default: ``0``.
    bias : bool
        If True, adds a learnable bias to the output. Default: ``True``.
    norm : callable activation function/layer or None, optional
        If not None, applies normalization to the updated node features.
    activation : callable activation function/layer or None, optional
        If not None, applies an activation function to the updated node features.
        Default: ``None``.

    Examples
    --------
    >>> import dgl
    >>> import numpy as np
    >>> import torch as th
    >>> from dgl.nn import SAGEConv

    >>> # Case 1: Homogeneous graph
    >>> g = dgl.graph(([0,1,2,3,2,5], [1,2,3,4,0,3]))
    >>> g = dgl.add_self_loop(g)
    >>> feat = th.ones(6, 10)
    >>> conv = SAGEConv(10, 2, 'pool')
    >>> res = conv(g, feat)
    >>> res
    tensor([[-1.0888, -2.1099],
            [-1.0888, -2.1099],
            [-1.0888, -2.1099],
            [-1.0888, -2.1099],
            [-1.0888, -2.1099],
            [-1.0888, -2.1099]], grad_fn=<AddBackward0>)

    >>> # Case 2: Unidirectional bipartite graph
    >>> u = [0, 1, 0, 0, 1]
    >>> v = [0, 1, 2, 3, 2]
    >>> g = dgl.bipartite((u, v))
    >>> u_fea = th.rand(2, 5)
    >>> v_fea = th.rand(4, 10)
    >>> conv = SAGEConv((5, 10), 2, 'mean')
    >>> res = conv(g, (u_fea, v_fea))
    >>> res
    tensor([[ 0.3163,  3.1166],
            [ 0.3866,  2.5398],
            [ 0.5873,  1.6597],
            [-0.2502,  2.8068]], grad_fn=<AddBackward0>)
    """
    def __init__(self,
                 in_feats,
                 out_feats,
                 aggregator_type,
                 proc_id,
                 device,
                 nprocs,
                 final_layer = False,
                 feat_drop=0.,
                 bias=True,
                 norm=None,
                 activation=None):
        super(SAGEConvHP, self).__init__()
        valid_aggre_types = {'mean', 'gcn', 'pool', 'lstm'}
        if aggregator_type not in valid_aggre_types:
            raise DGLError(
                'Invalid aggregator_type. Must be one of {}. '
                'But got {!r} instead.'.format(valid_aggre_types, aggregator_type)
            )

        self._in_src_feats, self._in_dst_feats = expand_as_pair(in_feats)
        self._out_feats = out_feats
        self._aggre_type = aggregator_type
        self.norm = norm
        self.device=device
        self.proc_id=proc_id
        self.nprocs=nprocs
        self.final_layer=final_layer
        self.feat_drop = nn.Dropout(feat_drop)
        self.activation = activation
        # aggregator type: mean/pool/lstm/gcn
        # if aggregator_type == 'pool':
        #     self.fc_pool = nn.Linear(self._in_src_feats, self._in_src_feats)
        # if aggregator_type == 'lstm':
        #     self.lstm = nn.LSTM(self._in_src_feats, self._in_src_feats, batch_first=True)
        if aggregator_type != 'gcn': 
            self.fc_self = nn.Linear(self._in_dst_feats, out_feats, bias=False)
        self.fc_neigh = nn.Linear(self._in_src_feats, out_feats, bias=False)
        if bias:
            self.bias = nn.parameter.Parameter(torch.zeros(self._out_feats))
        else:
            self.register_buffer('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        r"""

        Description
        -----------
        Reinitialize learnable parameters.

        Note
        ----
        The linear weights :math:`W^{(l)}` are initialized using Glorot uniform initialization.
        The LSTM module is using xavier initialization method for its weights.
        """
        gain = nn.init.calculate_gain('relu')
        # if self._aggre_type == 'pool':
        #     nn.init.xavier_uniform_(self.fc_pool.weight, gain=gain)
        # if self._aggre_type == 'lstm':
        #     self.lstm.reset_parameters()
        if self._aggre_type != 'gcn':
            nn.init.xavier_uniform_(self.fc_self.weight, gain=gain)
        nn.init.xavier_uniform_(self.fc_neigh.weight, gain=gain)

    def _compatibility_check(self):
        """Address the backward compatibility issue brought by #2747"""
        if not hasattr(self, 'bias'):
            dgl_warning("You are loading a GraphSAGE model trained from a old version of DGL, "
                        "DGL automatically convert it to be compatible with latest version.")
            bias = self.fc_neigh.bias
            self.fc_neigh.bias = None
            if hasattr(self, 'fc_self'):
                if bias is not None:
                    bias = bias + self.fc_self.bias
                    self.fc_self.bias = None
            self.bias = bias

    # def _lstm_reducer(self, nodes):
    #     """LSTM reducer
    #     NOTE(zihao): lstm reducer with default schedule (degree bucketing)
    #     is slow, we could accelerate this with degree padding in the future.
    #     """
    #     m = nodes.mailbox['m'] # (B, L, D)
    #     batch_size = m.shape[0]
    #     h = (m.new_zeros((1, batch_size, self._in_src_feats)),
    #          m.new_zeros((1, batch_size, self._in_src_feats)))
    #     _, (rst, _) = self.lstm(m, h)
    #     return {'neigh': rst.squeeze(0)}

    def forward(self, graph, feat, edge_weight=None):
        r"""

        Description
        -----------
        Compute GraphSAGE layer.

        Parameters
        ----------
        graph : DGLGraph
            The graph.
        feat : torch.Tensor or pair of torch.Tensor
            If a torch.Tensor is given, it represents the input feature of shape
            :math:`(N, D_{in})`
            where :math:`D_{in}` is size of input feature, :math:`N` is the number of nodes.
            If a pair of torch.Tensor is given, the pair must contain two tensors of shape
            :math:`(N_{in}, D_{in_{src}})` and :math:`(N_{out}, D_{in_{dst}})`.
        edge_weight : torch.Tensor, optional
            Optional tensor on the edge. If given, the convolution will weight
            with regard to the message.

        Returns
        -------
        torch.Tensor
            The output feature of shape :math:`(N_{dst}, D_{out})`
            where :math:`N_{dst}` is the number of destination nodes in the input graph,
            :math:`D_{out}` is the size of the output feature.
        """
        self._compatibility_check()
        with graph.local_scope():
            if isinstance(feat, tuple):
                feat_src = self.feat_drop(feat[0])
                feat_dst = self.feat_drop(feat[1])
            else:
                feat_src = feat_dst = self.feat_drop(feat)
                if graph.is_block:
                    feat_dst = feat_src[:graph.number_of_dst_nodes()]
            msg_fn = fn.copy_u('h', 'm')
            # if edge_weight is not None:
            #     assert edge_weight.shape[0] == graph.number_of_edges()
            #     graph.edata['_edge_weight'] = edge_weight
            #     msg_fn = fn.u_mul_e('h', '_edge_weight', 'm')
            vertex_split_index = [graph.number_of_dst_nodes() // self.nprocs] * self.nprocs
            vertex_split_index[-1] += graph.number_of_dst_nodes() % self.nprocs
            dim_in_split_index = [self._in_src_feats // self.nprocs] * self.nprocs
            dim_in_split_index[-1] += self._in_src_feats % self.nprocs
            dim_out_split_index = [self._out_feats // self.nprocs] * self.nprocs
            dim_out_split_index[-1] += self._out_feats % self.nprocs

            h_self = feat_dst

            # Handle the case of graphs without edges
            #if graph.number_of_edges() == 0:
            #    graph.dstdata['neigh'] = torch.zeros(
            #        feat_dst.shape[0], self._in_src_feats).to(feat_dst)

            # Determine whether to apply linear transformation before message passing A(XW)
            #lin_before_mp = self._in_src_feats > self._out_feats

            # Message Passing
            #if self._aggre_type == 'mean':

            graph.srcdata['h'] = feat_src
            graph.update_all(msg_fn, fn.mean('m', 'neigh'))

            #dist.barrier()
           
            #forwarda2a
            #h_nbr=comm_hp_t2d(graph.dstdata['neigh'])
            h_nbr=comm_hp_t2d(graph.dstdata['neigh'], vertex_split_index, dim_in_split_index, self.proc_id,self.device)
            h_neigh = self.fc_neigh(h_nbr)
            h_self=comm_hp_t2d(h_self, vertex_split_index, dim_in_split_index, self.proc_id,self.device)
            #else:
            #    raise KeyError('Aggregator type {} not recognized.'.format(self._aggre_type))

            # GraphSAGE GCN does not require fc_self.
            #if self._aggre_type == 'gcn':
            #    rst = h_neigh
            #else:

            rst = self.fc_self(h_self) + h_neigh
            rst = rst + self.bias
            # print("we come here")
            # exit(0);
            if not self.final_layer:
                rst = comm_hp_d2t(rst, vertex_split_index, dim_out_split_index, self.proc_id, self.device)
            # if self.final_layer:
            #     rst1=rst    
            # bias term
            #if self.bias is not None:
            # activation
            # if self.activation is not None:
            #     rst = self.activation(rst)
            # # normalization
            # if self.norm is not None:
            #     rst = self.norm(rst)
            #print("we come here")
            #exit(0);
            return rst