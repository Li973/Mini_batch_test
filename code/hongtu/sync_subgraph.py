"""Subgraph serialize and sync"""
# pylint: disable= no-member, arguments-differ, invalid-name
import torch
from dgl.base import DGLError
from dgl.utils import expand_as_pair, check_eq_shape, dgl_warning
import torch.distributed as dist
import dgl
import torch
import torch.distributed as dist
import dgl


sync_subgraph_print_the_log_= False
def multi_layer_graph_to_tensor(blocks,rank):
    """
    将多层子图转换为张量。
    参数:
        graphs: 包含多层子图的列表
    返回:
        多层节点特征和边索引的列表
    """
    eid_list=[]
    edge_index_list = []
    num_layers = len(blocks)
    edges_of_each_layer=[]
    nodes_of_each_layer=[]
    #src_data_of_each_layer=[]
    for block in blocks:
        edge_index = block.edges() 
        if sync_subgraph_print_the_log_:
            if rank==1:
                #print(f"rank [{rank}]: - TEST edge_index{edge_index}") 
                print(f" rank [{rank}]: - TEST - dst {block.number_of_dst_nodes()} src - {block.number_of_src_nodes()} edges {block.num_edges()}") 
                #print(f"rank [{rank}]:  in_degree {block.in_degrees()} out_degree {block.out_degrees()}")
                # src_feat = block.srcdata['feat']
                # dst_feat = block.dstdata['feat']
                # #print(f"rank [{rank}]:  block.srcdata['feat'] {src_feat.shape} dst_feat {dst_feat.shape}")
                distinct_src = torch.unique(edge_index[0], dim=0).size(0)
                distinct_dst = torch.unique(edge_index[1], dim=0).size(0)
                print(f"rank [{rank}]: - TEST INPUT - Distinct edges in layer: dst {distinct_dst} src {distinct_src}") 
                #print(f"rank [{rank}]: - TEST EID - src {block.edata[dgl.EID]}")
                print(f"rank [{rank}]: - TEST block.srcdata[NID] - {block.srcdata[dgl.NID]} - {block.srcdata[dgl.NID].shape}")
                print(f"rank [{rank}]: - TEST block.dstdata[NID] - {block.dstdata[dgl.NID]} - {block.dstdata[dgl.NID].shape}")
        # dist.barrier()
        # exit(0)
        edge_index_list.append(edge_index)
        edges_of_each_layer.append(block.num_edges())
        nodes_of_each_layer.append(block.number_of_dst_nodes())
        eid_list.append(block.edata[dgl.EID])
        #src_data_of_each_layer.append(block.srcdata[dgl.NID])
    nodes_of_each_layer.append(blocks[-1].number_of_src_nodes())

    send_edges_of_each_layer = torch.tensor(edges_of_each_layer, device=block.device)
    send_nodes_of_each_layer = torch.tensor(nodes_of_each_layer, device=block.device)
    #send_src_data=torch.tensor(src_data_of_each_layer, device=block.device)
    send_edge_index_tensor = torch.cat([
        torch.cat(
            (torch.cat(edge_index_list[i], dim=0), eid_list[i]), dim=0
        ) for i in range(len(edge_index_list))
    ], dim=0)

    return send_nodes_of_each_layer, send_edges_of_each_layer, send_edge_index_tensor

def tensor_to_multi_layer_graph(output_nodes, edge_in_each_layer,node_in_each_layer, edge_index,rank,nproc,num_layers):
    """
    将张量重新转换为多层子图。
    参数:
        node_features_list: 节点特征列表
        edge_index_list: 边索引列表
    返回:
        多层子图列表
    """
    blocks=[]
    start =0
    node_indicator=1
    for edges in edge_in_each_layer:
        g = dgl.create_block((edge_index[start:start+edges], edge_index[start+edges:start+edges*2]))
        g.edata[dgl.EID] = edge_index[start+2*edges:start+3*edges]
        if sync_subgraph_print_the_log_:
            if rank==0:
                distinct_src = torch.unique(edge_index[start:start+edges], dim=0).size(0)
                distinct_dst = torch.unique(edge_index[start+edges:start+2*edges], dim=0).size(0)

                print(f"rank [{rank}]: - TEST OUTPUT - dst {g.number_of_dst_nodes()} ,src {g.number_of_src_nodes()} edges {g.num_edges()} | Distinct edges in layer: dst {distinct_dst} src {distinct_src} edge {edges}") 
                #print(f"rank [{rank}]:  in_degree {g.in_degrees()} out_degree {g.out_degrees()}")
                #print(f"rank [{rank}]: - TEST EID - src {g.edata[dgl.EID]}")
                #src_feat = g.srcdata['feat']
                #dst_feat = g.dstdata['feat']
                #print(f"rank [{rank}]:  block.srcdata['feat'] {src_feat.shape} dst_feat {dst_feat.shape}")
                #print(f"rank [{rank}]: - TEST block.srcdata[NID] -  {g.srcdata[dgl.NID]} - {g.srcdata[dgl.NID].shape}")
                #print(f"rank [{rank}]: - TEST block.dstdata[NID] -  {g.dstdata[dgl.NID]} - {g.dstdata[dgl.NID].shape}")
                #print(f"rank [{rank}]: - output edge_index{(edge_index[start:start+edges], edge_index[start+edges:start+edges*2])}") 
        blocks.append(g)
        start+=edges*3
        node_indicator+=1
    return blocks

def synchronize_graph(blocks, input_nodes,output_nodes):
    check_layer=len(blocks)-3
    nproc = dist.get_world_size()
    rank = dist.get_rank()
    meta_data_sizes = len(blocks)*2+3

    send_nodes_of_each_layer, send_edges_of_each_layer, send_edge_index_tensor = multi_layer_graph_to_tensor(blocks,rank)
    edge_size_sync_recv_tensor_list = [torch.zeros(meta_data_sizes,device=send_edge_index_tensor.device).to(torch.int64) for _ in range(nproc)]
    edge_size_sync_send = torch.cat((send_edges_of_each_layer,send_nodes_of_each_layer,\
                                     torch.tensor([input_nodes.size(0)],device=send_edge_index_tensor.device),\
                                        torch.tensor([output_nodes.size(0)],device=send_edge_index_tensor.device)), dim=0)
    dist.all_gather(edge_size_sync_recv_tensor_list, edge_size_sync_send)

    input_graph_edge_size_list = [int(torch.sum(edge_size_sync_recv_tensor_list[i][0:len(blocks)]).item()) for i in range(nproc)]
    #print(f"rank: {rank} - metda_data_sizes{meta_data_sizes} - edge_size_sync_recv[0]{edge_size_sync_recv_tensor_list[0]} edge_size_sync_recv[1]{edge_size_sync_recv_tensor_list[1]} -edge_size_sync_send {edge_size_sync_send}")
    #print(f"rank: {rank} - meta{meta_data_sizes} - input_graph_size_list{input_graph_size_list} send_edge_index_tensor.shape{send_edge_index_tensor.shape}")
    #print(f"rank: {rank} - blocks[-1].dst{blocks[check_layer].number_of_dst_nodes()} - blocks[-1].dst{blocks[check_layer].number_of_src_nodes()} - blocks[-1].edges {blocks[check_layer].num_edges()}")
    #dist.barrier()
    padding_edge_size = max(input_graph_edge_size_list)*3
    recv_edge_index_data_list= [torch.zeros(padding_edge_size,device=send_edge_index_tensor.device,dtype=send_edge_index_tensor.dtype) for _ in range(nproc) ]
    send_edge_index_data= torch.zeros(padding_edge_size,device=send_edge_index_tensor.device,dtype=send_edge_index_tensor.dtype)
    send_edge_index_data[:send_edge_index_tensor.numel()] = send_edge_index_tensor
    dist.all_gather(recv_edge_index_data_list, send_edge_index_data)

    
    recv_input_nodes_size_list = [edge_size_sync_recv_tensor_list[i][meta_data_sizes-2].item() for i in range(nproc)]
    padding_input_node_size = max(recv_input_nodes_size_list)
    send_input_nodes_tensor = torch.zeros(padding_input_node_size, dtype=torch.int64, device=input_nodes.device)
    recv_input_nodes_tensor_list=  [torch.zeros(padding_input_node_size,device=input_nodes.device,dtype=input_nodes.dtype) for _ in range(nproc) ]
    send_input_nodes_tensor[:input_nodes.numel()] = input_nodes
    dist.all_gather(recv_input_nodes_tensor_list, send_input_nodes_tensor)

    recv_output_nodes_size_list = [edge_size_sync_recv_tensor_list[i][-1].item() for i in range(nproc)]
    padding_output_node_size = max(recv_output_nodes_size_list)
    send_output_nodes_tensor = torch.zeros(padding_output_node_size, dtype=torch.int64, device=input_nodes.device)
    recv_output_nodes_tensor_list=  [torch.zeros(padding_output_node_size,device=input_nodes.device,dtype=input_nodes.dtype) for _ in range(nproc) ]
    send_output_nodes_tensor[:output_nodes.numel()] = output_nodes
    dist.all_gather(recv_output_nodes_tensor_list, send_output_nodes_tensor)

    #validation code
    # val_recv_edge_index_data_list= [torch.zeros(padding_edge_size,device=send_edge_index_tensor.device,dtype=send_edge_index_tensor.dtype) for _ in range(nproc) ]
    # val_send_edge_index_data=recv_edge_index_data_list[(rank+1)%nproc]
    # dist.all_gather(val_recv_edge_index_data_list, val_send_edge_index_data)
    
    # num_diff = torch.sum(send_edge_index_data != val_recv_edge_index_data_list[(rank+1)%nproc]).item()
    # print(f"rank:++++++++++++++++++++++++++++++++++++++++++++++ output_nodes{output_nodes}")

    graphs=[]
    input_node_list=[]
    output_node_list=[]
    for it in range(nproc):
        if it!=rank:
            graphs.append(tensor_to_multi_layer_graph(output_nodes, edge_size_sync_recv_tensor_list[it][0:len(blocks)].tolist(),\
                                                      edge_size_sync_recv_tensor_list[it][len(blocks):meta_data_sizes-1].tolist(),\
                                                          recv_edge_index_data_list[it],rank,nproc,len(blocks)))
            input_node_list.append(recv_input_nodes_tensor_list[it][0:recv_input_nodes_size_list[it]])
            output_node_list.append(recv_output_nodes_tensor_list[it][0:recv_output_nodes_size_list[it]])
        else:
            graphs.append(blocks)
            input_node_list.append(input_nodes)
            output_node_list.append(output_nodes)
        dist.barrier()
        if sync_subgraph_print_the_log_:
            print(f"rank: {rank} - graphs[{it}].dst {graphs[it][check_layer].number_of_dst_nodes()} graphs[{it}].src - {graphs[it][check_layer].number_of_src_nodes()} - blocks[-1].edges {graphs[it][check_layer].num_edges()}")

    return graphs, input_node_list, output_node_list
# 假设我们有4个进程

# 根据接收到的张量重新构建多层
