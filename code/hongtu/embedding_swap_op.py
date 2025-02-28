
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchmetrics.functional as MF
import torch.distributed as dist
import sys

# P: partial
# F: full
# V: vertex
# D: dimension
do_print_the_log_= False
class HP_T2D(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input_FVPD, vertex_split_index, dim_split_index, rank, device_):
        ctx.rank=rank
        ctx.device=device_
        ctx.vertex_split_index=vertex_split_index
        ctx.dim_split_index=dim_split_index
        num_procs=dist.get_world_size()
        local_dim= input_FVPD.size(1)
        local_vertex_range= vertex_split_index[rank]
        
        #print(f"HP_T2D {rank} - local_dim: {local_dim} - dim_split_index[rank]: {dim_split_index[rank]} ")
        assert dim_split_index[rank]== local_dim
        input_size_list = [vertex_range * local_dim for vertex_range in vertex_split_index]
        output_size_list = [local_vertex_range * dim_range for dim_range in dim_split_index]
        output_size = sum(output_size_list)
        
        send_data = input_FVPD.view(-1)
        recv_data = torch.zeros(output_size,device=device_,dtype=torch.float32) 
        
        #print(f"Rank## {rank} - send_data: {send_data}")
        t0 = time.time()
        dist.barrier()
        dist.all_to_all_single(recv_data, send_data, output_split_sizes=output_size_list, input_split_sizes=input_size_list)
        dist.barrier()
        
        start_ = 0
        end_  = 0
        recv_data_list=[]
        for dim in dim_split_index:
            end_ = end_ + dim * local_vertex_range
            split_tensor = recv_data[start_ :end_].view(-1, dim)
            recv_data_list.append(split_tensor)
            start_=start_ + dim *local_vertex_range
        #exit(0)
        t1 = time.time()
        #sys.exit(0)
        #print(f"Rank## {rank} - Recv: {recv_data}")
        if do_print_the_log_:
            print(f"HP_T2D FWD {rank} - Communication time: {t1 - t0:.4f} seconds")
        output_PVFD=torch.cat(recv_data_list, dim=1)
        #print(f"Rank## {rank} - output: {output}")
        #print(f"Rank {rank} - output: {output.shape}")
        #dist.barrier()
        #sys.exit(0)
        return output_PVFD

    @staticmethod
    def backward(ctx, grad_output_PVFD):
        #input, = ctx.saved_tensors
        rank = ctx.rank
        vertex_split_index= ctx.vertex_split_index
        dim_split_index =ctx.dim_split_index
        device_=ctx.device

        local_dim= dim_split_index[rank]
        local_vertex_range= vertex_split_index[rank]
        input_size_list = [vertex_range * local_dim for vertex_range in vertex_split_index]
        output_size_list = [local_vertex_range * dim_range for dim_range in dim_split_index]
        input_size= sum(input_size_list)

        grad_output_PVFD_list = torch.split(grad_output_PVFD, dim_split_index, dim=1)
        grad_output_PVFD_list = [tensor.contiguous().view(-1) for tensor in grad_output_PVFD_list]
        send_data = torch.cat(grad_output_PVFD_list, dim=0)
        recv_data = torch.zeros(input_size,device=device_,dtype=torch.float32) 

        #print(f"Rank## {rank} - send_data: {send_data}")
        t0 = time.time()
        dist.barrier()
        dist.all_to_all_single(recv_data, send_data, output_split_sizes=input_size_list, input_split_sizes=output_size_list)
        dist.barrier()
        t1 = time.time()
        #sys.exit(0)
        #print(f"Rank## {rank} - Recv: {recv_data}")
        if do_print_the_log_:
            print(f"HP_T2D BACK Rank {rank} - Communication time: {t1 - t0:.4f} seconds")
        grad_input_FVPD=recv_data.view(-1,local_dim)
        #print(f"BACK Rank## {rank} - output: {grad_input}")

        return grad_input_FVPD, None, None, None, None
comm_hp_t2d = HP_T2D.apply

class HP_D2T(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input_PVFD, vertex_split_index, dim_split_index, rank,device_): #input_by_dim
        ctx.rank=rank
        ctx.device=device_
        ctx.vertex_split_index=vertex_split_index
        ctx.dim_split_index=dim_split_index
        num_procs=dist.get_world_size()
        
        local_vertex_range= input_PVFD.size(0)
        local_dim= dim_split_index[rank]
        assert local_vertex_range == vertex_split_index[rank]
        if do_print_the_log_:
            print(f"Rank## {rank} - dim_split_index: {dim_split_index} - vertex_split_index {vertex_split_index}")
        input_size_list= [local_vertex_range * dim_range for dim_range in dim_split_index]
        output_size_list= [vertex_range * local_dim for vertex_range in vertex_split_index]
        output_size= sum(output_size_list)
        #print(output_size_list)

        input_PVFD_list = torch.split(input_PVFD, dim_split_index, dim=1)
        input_PVFD_list = [tensor.contiguous().view(-1) for tensor in input_PVFD_list]
        send_data = torch.cat(input_PVFD_list, dim=0)
        #send_data = input_partialVtx.view(-1)
        recv_data = torch.zeros(output_size,device=device_,dtype=torch.float32) 
        #print(f"FWD local_vertex_range {send_data.size(0)} {input_size}")
        #dist.barrier()
        if do_print_the_log_:
            print(f"Rank## {rank} - send_data: {send_data.shape} - local_dim {local_dim} - local_vertex_range {local_vertex_range}")
            print(f"Rank## {rank} - recv_data: {recv_data.shape} - local_dim {local_dim} - local_vertex_range {local_vertex_range}")
        t0 = time.time()
        dist.barrier()
        dist.all_to_all_single(recv_data, send_data, output_split_sizes=output_size_list, input_split_sizes=input_size_list)
        dist.barrier()
        t1 = time.time()
        #sys.exit(0)
        #print(f"Rank## {rank} - Recv: {recv_data}")
        if do_print_the_log_:
            print(f"HP_D2T FWD {rank} - Communication time: {t1 - t0:.4f} seconds")
        output_FVPD=recv_data.view(-1,local_dim)

        return output_FVPD
        

    @staticmethod
    def backward(ctx, grad_output_FVPD): #return grad_by_vtx
        #input, = ctx.saved_tensors
        rank = ctx.rank
        vertex_split_index= ctx.vertex_split_index
        dim_split_index =ctx.dim_split_index
        device_=ctx.device

        local_dim= dim_split_index[rank]
        local_vertex_range= vertex_split_index[rank]
        input_size_list = [vertex_range * local_dim for vertex_range in vertex_split_index]
        output_size_list = [local_vertex_range * dim_range for dim_range in dim_split_index]
        # print(output_size_list)
        # dist.barrier()
        # exit(0)
        output_size_ = sum(output_size_list)
        send_data = grad_output_FVPD.view(-1)
        #grad_by_dim_list = torch.split(grad_output_FVPD, vertex_split_index, dim=0)
        #input_partialVtx_list = [tensor.contiguous().view(-1) for tensor in input_partialVtx_list]
        #send_data = torch.cat(grad_output_partialVtx_list, dim=0)
        recv_data = torch.zeros(output_size_,device=device_,dtype=torch.float32) 
        
        #print(f"Rank## {rank} - send_data: {send_data}")
        t0 = time.time()
        dist.barrier()
        dist.all_to_all_single(recv_data, send_data, output_split_sizes=output_size_list, input_split_sizes=input_size_list)
        dist.barrier()
        start_ = 0
        end_  = 0
        recv_data_list=[]
        for dim in dim_split_index:
            end_ = end_ + dim * local_vertex_range
            split_tensor = recv_data[start_ :end_].view(-1, dim)
            recv_data_list.append(split_tensor)
            start_=start_ + dim *local_vertex_range
        #exit(0)
        t1 = time.time()
        #sys.exit(0)
        #print(f"Rank## {rank} - Recv: {recv_data}")
        #print(f"Rank {rank} - Communication time: {t1 - t0:.4f} seconds")
        grad_input_PVFD=torch.cat(recv_data_list, dim=1)
        if do_print_the_log_:
            print(f"HP_D2T BACK {rank} - grad_input_PVFD {grad_input_PVFD.shape}")
        return grad_input_PVFD, None, None, None, None

comm_hp_d2t = HP_D2T.apply




def test_shuffle(device, proc_id):
        # Generate input tensor
    input_data = torch.ones(9, 2, device=device, requires_grad=True)
    input_data1= torch.ones(9, 2, device=device, requires_grad=False)
    if proc_id ==1:
        input_data = torch.ones(9, 3, device=device, requires_grad=True)
        input_data1= torch.ones(9, 3, device=device, requires_grad=False)
    for i in range(input_data1.size(0)):  # 遍历行
        for j in range(input_data1.size(1)):  # 遍历列
            input_data1[i, j] = i * 10 + j+proc_id*2  # 根据行列索引赋值
    input_datas=input_data+input_data1
    
    print(f"BACK Rank## {proc_id} - input_data: {input_data}")
    # # Define split indices as integers
    # # dim_split_index = [100,100] 
    # # vertex_split_index = [400, 500]
    # dim_split_index = [200,200,300] 
    # vertex_split_index = [200000, 300000, 400000]

    # # print(f"Rank {proc_id} - dim_split_index: {dim_split_index}")
    # # print(f"Rank {proc_id} - vertex_split_index: {vertex_split_index}")
    # #dist.barrier() 
    # #sys.exit(0)
    # # Apply blSwappingFunction
    # output = forwarda2a(input_data, vertex_split_index, dim_split_index, proc_id,device)
    # input_=output.backward(output)
    # output = forwarda2a(input_data, vertex_split_index, dim_split_index, proc_id,device)
    # input_=output.backward(output)
    # output = forwarda2a(input_data, vertex_split_index, dim_split_index, proc_id,device)
    # input_=output.backward(output)
    # # Print the arrays and output to verify
    # #print(f"Rank## {proc_id} - Output shape: {output.shape}")

    # # Synchronize processes
    # dist.barrier()
    # #sys.exit(0)
            # Generate input tensor
    #input_data = torch.ones(400000, 500, device=device, requires_grad=True)
    #input_data1= torch.ones(9000000, 200, device=device, requires_grad=False)
    #if proc_id ==1:
        #input_data = torch.ones(500000, 500, device=device, requires_grad=True)
    #    input_data1= torch.ones(9000000, 300, device=device, requires_grad=False)
    # for i in range(input_data1.size(0)):  # 遍历行
    #     for j in range(input_data1.size(1)):  # 遍历列
    #         input_data1[i, j] = i * 10 + j+proc_id*2  # 根据行列索引赋值
    # input_data=input_data+input_data1
    
    print(f"BACK Rank## {proc_id} - before: {input_data.shape}")
    # Define split indices as integers
    # dim_split_index = [100,100] 
    # vertex_split_index = [400, 500]
    dim_split_index = [2,3] 
    vertex_split_index = [4, 5]

    # print(f"Rank {proc_id} - dim_split_index: {dim_split_index}")
    # print(f"Rank {proc_id} - vertex_split_index: {vertex_split_index}")
    #dist.barrier() 
    #sys.exit(0)
    # Apply blSwappingFunction
    #input_datas.retain_grad()
    input_data2=comm_hp_t2d(input_datas, vertex_split_index, dim_split_index, proc_id,device)
    print(f"Rank## {proc_id} - input_data2 : {input_data2}")
    input_data2.retain_grad()
    output = comm_hp_d2t(input_data2, vertex_split_index, dim_split_index, proc_id,device)
    print(f"Rank## {proc_id} - Output: {output}")
    output.backward(output)
    print(f"Rank## {proc_id} - Input grad: {input_data2.grad}")
    # Print the arrays and output to verify
    #print(f"Rank## {proc_id} - Output shape: {output.shape}")

    # Synchronize processes
    dist.barrier()
    #sys.exit(0)