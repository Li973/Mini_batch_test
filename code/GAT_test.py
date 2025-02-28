import argparse
import os
import time
import random
import dgl
import dgl.nn as dglnn
import sys 
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.nn as nn
import torch.nn.functional as F
import torchmetrics.functional as MF
import tqdm
from dgl.data import AsNodePredDataset
from dgl.dataloading import (
    DataLoader,
    MultiLayerFullNeighborSampler,
    NeighborSampler,
)
from dgl.multiprocessing import shared_tensor
from ogb.nodeproppred import DglNodePropPredDataset
from torch.nn.parallel import DistributedDataParallel
from dgl.data import register_data_args, load_data
from dgl.data import CiteseerGraphDataset, CoraGraphDataset, PubmedGraphDataset,RedditDataset
import numpy as np
from dgl.nn import GATConv

from hongtu.embedding_swap_op import HP_T2D, HP_D2T ,comm_hp_d2t, comm_hp_t2d
from hongtu.my_GAT_hp import GATConvHP
from hongtu.sync_subgraph import synchronize_graph


def set_seed(seed):
    dgl.seed(seed)
    torch.manual_seed(seed)  # 设置CPU上所有操作的随机种子
    torch.cuda.manual_seed(seed)  # 为所有GPU设置随机种子（如果使用多个GPU）
    random.seed(seed)  # 设置Python的随机模块种子
    np.random.seed(seed)
    dgl.random.seed(seed)            # 设置 DGL 随机种子
    torch.cuda.manual_seed_all(seed)  # 对所有 GPU 设置相同的种子
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

class GAT(nn.Module):
    def __init__(self, in_size, hid_size, out_size, num_heads, proc_id, device, nprocs):
        super().__init__()
        self.layers = nn.ModuleList()

        # self.layers.append(GATConv(in_size, hid_size, num_heads, allow_zero_in_degree=True))
        # self.layers.append(GATConv(hid_size * num_heads, hid_size, num_heads, allow_zero_in_degree=True))
        # self.layers.append(GATConv(hid_size * num_heads, out_size, 1, allow_zero_in_degree=True))

        self.layers.append(GATConvHP(in_size, hid_size, num_heads,proc_id=proc_id, device=device, nprocs=nprocs))
        self.layers.append(GATConvHP(hid_size * num_heads, hid_size, num_heads,proc_id=proc_id, device=device, nprocs=nprocs))
        self.layers.append(GATConvHP(hid_size * num_heads, out_size, 1,proc_id=proc_id, device=device, nprocs=nprocs,final_layer = True))

        self.dropout = nn.Dropout(0.5)
        self.hid_size = hid_size
        self.out_size = out_size
        self.proc_id = proc_id
        self.device = device
        self.num_heads = num_heads

    def forward(self, blocks, x):
        h = x
        for l, (layer, block) in enumerate(zip(self.layers, blocks)):
            h = layer(block, h)
            # if l != len(self.layers) - 1:
            #     h = F.relu(h)
            #     h = self.dropout(h)
            if l != len(self.layers) - 1:
                # h = h.flatten(1)
                h = F.relu(h)
                h = self.dropout(h)
            # else:
                # h = h.mean(dim=1)

        return h

    def inference(self, g, device, batch_size, use_uva, nprocs, proc_id):
        g.ndata["h"] = g.ndata["feat"]
        sampler = MultiLayerFullNeighborSampler(1, prefetch_node_feats=["h"])
        y_out = shared_tensor(
                (
                    g.num_nodes(),
                    self.out_size,
                )
            )
        for l, layer in enumerate(self.layers):
            dataloader = DataLoader(
                g,
                torch.arange(g.num_nodes(), device=device),
                sampler,
                device=device,
                batch_size=batch_size,
                shuffle=False,
                drop_last=False,
                num_workers=0,
                use_ddp=False,
                use_uva=use_uva,
            )
            # in order to prevent running out of GPU memory, allocate a
            # shared output tensor 'y' in host memory
            # y = shared_tensor(
            #     (
            #         g.num_nodes(),
            #         self.hid_size
            #         if l != len(self.layers) - 1
            #         else self.out_size,
            #     )
            # )
            
            cur_size=self.hid_size if l != len(self.layers) - 1 else self.out_size
            #print(f"Process : x = {cur_size}")
            dim_split_index = [cur_size// nprocs] * nprocs
            dim_split_index[-1] += cur_size % nprocs

            # y_cpu=torch.empty((g.num_nodes(), dim_split_index[proc_id]), pin_memory=True)
            cur_size = self.hid_size * self.num_heads if l != len(self.layers) - 1 else self.out_size
            dim_split_index = [cur_size // nprocs] * nprocs
            dim_split_index[-1] += cur_size % nprocs

            y_cpu = torch.empty((g.num_nodes(), dim_split_index[proc_id]), pin_memory=True)
            
            y = shared_tensor((1,1))
            for input_nodes, output_nodes, blocks in (
                tqdm.tqdm(dataloader) if dist.get_rank() == 0 else dataloader
            ):
                x = blocks[0].srcdata["h"]
                h = layer(blocks[0], x)  # len(blocks) = 1
                if l != len(self.layers) - 1:
                    h = F.relu(h)
                    h = self.dropout(h)
                # non_blocking (with pinned memory) to accelerate data transfer
                    # h = h.flatten(1)
                    y_cpu[output_nodes] = h.to(y.device, non_blocking=True)
                else:
                    # h = h.mean(dim=1)
                    start = proc_id * output_nodes.size(0) // nprocs
                    end = start + (output_nodes.size(0) // nprocs if proc_id < nprocs - 1 else output_nodes.size(0))

                    y_out[output_nodes[start:end]]=h.to(y.device, non_blocking=True)
            # make sure all GPUs are done writing to 'y'
            #print(f"Process : x = {x.shape} y_hats shape {y_cpu.shape}")
            if l != len(self.layers) - 1:
                g.ndata["h"] = y_cpu if use_uva else y_cpu.to(device)
            dist.barrier()
            #print(f"Process : x = {x.shape} y_out shape {y_out.shape}")
        #exit(0)

        g.ndata.pop("h")
        #exit(0)
        return y_out

def layerwise_infer(
    proc_id, nprocs, device, g, num_classes, nid, model, use_uva, batch_size=2**10):
    model.eval()
    with torch.no_grad():
        pred = model.module.inference(g, device, batch_size, use_uva, nprocs, proc_id)
        pred = pred[nid]
        labels = g.ndata["label"][nid].to(pred.device)
    if proc_id == 0:
        acc = MF.accuracy(
            pred, labels, task="multiclass", num_classes=num_classes
        )
        print("Test accuracy {:.4f}".format(acc.item()))



def evaluate(model, g, num_classes, dataloader, nprocs, proc_id):
    model.eval()
    ys = []
    y_hats = []
    for it, (input_nodes, output_nodes, blocks) in enumerate(dataloader):
        with torch.no_grad():
            t_cosamp_s=time.time()
            list_of_co_blocks, input_node_list, output_node_list=synchronize_graph(blocks, input_nodes, output_nodes)
            t_cosamp_e=time.time()
            for local_iter in range(nprocs):
                blocks=list_of_co_blocks[local_iter]
                input_nodes=input_node_list[local_iter]
                output_nodes= output_node_list[local_iter]
                x = g.get_node_storage('feat').fetch(input_nodes, output_nodes.device, pin_memory=True)
                k = g.get_node_storage('label').fetch(output_nodes, output_nodes.device, pin_memory=True).to(torch.int64)            
                start_ = proc_id * output_nodes.size(0) // nprocs
                end_ = start_ + (output_nodes.size(0) // nprocs if proc_id < nprocs - 1 else output_nodes.size(0))
                #print(f"Process {proc_id}:  blocks[-1].dstdata[label] = {k.shape}")
                dist.barrier()
                ys.append(k[start_:end_])
                y_hats.append(model(blocks, x))
                #print(f"Process {proc_id}:  blocks[-1].dstdata[label] = {k[start_:end_].shape} y_hats shape {y_hats[-1].shape}")
    return MF.accuracy(
        torch.cat(y_hats),
        torch.cat(ys),
        task="multiclass",
        num_classes=num_classes,
    )


def train(
    proc_id,
    nprocs,
    device,
    g,
    num_classes,
    train_idx,
    val_idx,
    model,
    use_uva,
    num_epochs,
):
    #test_shuffle(device, proc_id)
    #dist.barrier()
    #exit(0)
    set_seed(42)
    sampler = NeighborSampler(
        [3, 3, 3], prefetch_labels=["label"], prefetch_node_feats=["feat"]
    )
    train_sampler = NeighborSampler(
        [3, 3, 3] #prefetch_labels=[], prefetch_node_feats=[], #prefetch_node_feats=[],
    )

    train_dataloader = DataLoader(
        g,
        train_idx,
        train_sampler,
        device=device,
        batch_size=1024,
        shuffle=True,
        drop_last=False,
        num_workers=0,
        use_ddp=True,
        use_uva=use_uva,
    )
    #if epoch==0:
    print(f"Process {proc_id}: train_vtx = {train_idx.size(0)}")
    val_dataloader = DataLoader(
        g,
        val_idx,
        sampler,
        device=device,
        batch_size=1024,
        shuffle=True,
        drop_last=False,
        num_workers=0,
        use_ddp=False,
        use_uva=use_uva,
    )
    opt = torch.optim.Adam(model.parameters(), lr=0.003, weight_decay=5e-4)
    for epoch in range(num_epochs):
        t0 = time.time()
        model.train()
        total_loss = 0
        t3=0.
        t4=0.
        t_sync_graph=0.
        # Process in batches
        batch_size = 1024*2*2*2*2
        it=0

        train_size= train_idx.size(0)
        train_offset = [i * (train_size // nprocs) for i in range(nprocs)]
        train_offset.append(train_size)
        local_train_vertices=train_offset[proc_id+1]-train_offset[proc_id]
        local_train_offset=train_offset[proc_id]

        for start in range(0, local_train_vertices, batch_size):
            it+=1
            t_start=local_train_offset+start
            t_end = t_start+min(batch_size, local_train_vertices-start)

            batch_idx = train_idx[t_start:t_end]
            #print(f"rank: {proc_id} - train_idx.size(0) {train_idx.size(0)} - start {start} ")
            input_nodes, output_nodes, blocks = train_sampler.sample(g, batch_idx)

            # validated.
            # num_diff = torch.sum(output_nodes != batch_idx).item()
            # print(f"rank:+++++++++++dddd+++++++++++++++++++++++++++++++++++ num_diff{num_diff}")
            t_cosamp_s=time.time()
            list_of_co_blocks, input_node_list, output_node_list=synchronize_graph(blocks, input_nodes, output_nodes)
            t_cosamp_e=time.time()
            t_sync_graph+=t_cosamp_e-t_cosamp_s
            #print(f"rank: {proc_id} - TEST_co_sample - Time {t_cosamp_e-t_cosamp_s}")   
            dist.barrier()
            #print(f"rank: {proc_id} - input_nodes{input_nodes}  {input_nodes.shape}  - input_node_list {input_node_list}") 
            #dist.barrier()
            #exit(0)
            for local_iter in range(nprocs):

                blocks=list_of_co_blocks[local_iter]
                input_nodes=input_node_list[local_iter]
                output_nodes= output_node_list[local_iter]
                t3=time.time()  
                # Fetch features using FeatureStorage
                x = g.get_node_storage('feat').fetch(input_nodes, device, pin_memory=True)
                y = g.get_node_storage('label').fetch(output_nodes, device, pin_memory=True).to(torch.int64)
                # Select feat_partitions[proc_id]
                start = proc_id * y.size(0) // nprocs
                end = start + (y.size(0) // nprocs if proc_id < nprocs - 1 else y.size(0))
                y = y[start:end]

                #print(f"y shape: {y.size()}")

                # x = blocks[0].srcdata["feat"]
                # y = blocks[-1].dstdata["label"].to(torch.int64)
                # input_nodes = input_nodes.to(g.ndata['feat'].device)
                # x = g.ndata['feat'][input_nodes].to(device)
                #sys.exit(0)

                y_hat = model(blocks, x)

                loss = F.cross_entropy(y_hat, y)
                opt.zero_grad()
                loss.backward()
                opt.step()
                total_loss += loss
                t5=time.time()
                t4+=t5-t3
                dist.barrier()
                #print(f"rank: {proc_id} - Local_iter({local_iter}) success")   
            #print(f"rank: {proc_id} - iter({it}) success")   
            #exit(0)
        acc = (
            evaluate(model, g, num_classes, val_dataloader, nprocs, proc_id).to(device) / nprocs
        )
        #acc =torch.tensor([0])
        t1 = time.time()
        #dist.reduce(acc, 0)
       
        if proc_id == 0:
            print(
                "Epoch {:05d} | Loss {:.4f} | Accuracy {:.4f} | "
                "Time {:.4f} | Time_remove_sample {:.4f} | Time sync graph {:.4f}".format(
                    epoch, total_loss / (it + 1), acc.item(), t1 - t0, t4,t_sync_graph #acc.item()
                )
            )
        t4=0.0
        t_sync_graph=0.0


def run(proc_id, nprocs, devices, g, data, mode, num_epochs):
    # find corresponding device for my rank
    device = devices[proc_id]
    torch.cuda.set_device(device)
    # initialize process group and unpack data for sub-processes
    dist.init_process_group(
        backend="nccl",
        init_method="tcp://127.0.0.1:12354",
        world_size=nprocs,
        rank=proc_id,
    )

    num_classes, train_idx, val_idx, test_idx = data
    train_idx = train_idx.to(device)

    set_seed(42)
    perm = torch.randperm(train_idx.size(0), device=device)
    train_idx_shuffled = train_idx[perm]
    val_idx = val_idx.to(device)
    g = g.to(device if mode == "puregpu" else "cpu")
    # create GraphSAGE model (distributed)
    in_size = g.ndata["feat"].shape[1]
    print("g.ndata {:03d}".format(in_size))
    #debug

    nproc = dist.get_world_size()
    # dim_split_index = [in_size // nproc] * nproc
    # dim_split_index[-1] += in_size% nproc
    # # Select feat_partitions[proc_id]
    # g.ndata["feat"] = g.ndata["feat"][:, :dim_split_index[proc_id]].contiguous()
    # 如果使用 GAT，中间层的特征维度会变为 hid_size * num_heads
    if proc_id == 0:
        print("Splitting features for GAT...")
    dim_split_index = [(in_size * 4) // nproc] * nproc
    dim_split_index[-1] += (in_size * 4) % nproc
    g.ndata["feat"] = g.ndata["feat"][:, :dim_split_index[proc_id]].contiguous()
    print("Enable Partition Slice")
    print("g.ndata {:03d}".format(in_size))

    model = GAT(in_size, 256, num_classes, num_heads=4, proc_id=proc_id, device=device, nprocs=nproc).to(device)


    model = DistributedDataParallel(
        model, device_ids=[device], output_device=device
    )
    #DEBUG
    #DEBUG_END
    # training + testing
    use_uva = mode == "mixed"
    train(
        proc_id,
        nprocs,
        device,
        g,
        num_classes,
        train_idx_shuffled,
        #local_train_idx,#debug
        val_idx,
        model,
        use_uva,
        num_epochs,
    )
    layerwise_infer(proc_id, nprocs, device, g, num_classes, test_idx, model, use_uva)
    # cleanup process group
    dist.destroy_process_group()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--mode",
        default="mixed",
        choices=["mixed", "puregpu"],
        help="Training mode. 'mixed' for CPU-GPU mixed training, "
        "'puregpu' for pure-GPU training.",
    )
    parser.add_argument(
        "--gpu",
        type=str,
        default="1",
        help="GPU(s) in use. Can be a list of gpu ids for multi-gpu training,"
        " e.g., 0,1,2,3.",
    )
    parser.add_argument(
        "--num_epochs",
        type=int,
        default=10,
        help="Number of epochs for train.",
    )
    parser.add_argument(
        "--dataset_name",
        type=str,
        default="ogbn-products",
        help="Dataset name.",
    )
    parser.add_argument(
        "--dataset_dir",
        type=str,
        default="dataset",
        help="Root directory of dataset.",
    )
    args = parser.parse_args()
    devices = list(map(int, args.gpu.split(",")))
    nprocs = len(devices)
    assert (
        torch.cuda.is_available()
    ), f"Must have GPUs to enable multi-gpu training."
    print(f"Training in {args.mode} mode using {nprocs} GPU(s)")

    # load and preprocess dataset
    print("Loading data")
    dataset = AsNodePredDataset(
        DglNodePropPredDataset(args.dataset_name, root=args.dataset_dir)
    )#NO
    #dataset = RedditDataset()
    # dataset = RedditDataset(raw_dir ='/data/wangqg/dgl/')YES
    g = dataset[0]
    # avoid creating certain graph formats in each sub-process to save momory
    g.create_formats_()
    # if args.dataset_name == "ogbn-arxiv":YES
    g = dgl.to_bidirected(g, copy_ndata=True)
    g = dgl.add_self_loop(g)#TAB
    # thread limiting to avoid resource competition
    os.environ["OMP_NUM_THREADS"] = str(mp.cpu_count() // 2 // nprocs)
    # masks = g.ndata["train_mask"], g.ndata["val_mask"], g.ndata["test_mask"]YES
    # print("generating masks")YES
    # v_size=len(masks[0])YES

    # set_seed(42)YES
    # for i in range(0, v_size):
    #     rand_val=random.random()
    #     if(rand_val<0.25):
    #         masks[0][i]=False
    #         masks[1][i]=True
    #     if(rand_val>0.25 and rand_val<0.75):
    #         masks[0][i]=False
    #         masks[2][i]=True

    # print("masks generated")
    # train_idx= torch.tensor([index for index, value in enumerate(masks[0]) if value == True])
    # print("train_idx generated")
    # val_idx= torch.tensor([index for index, value in enumerate(masks[1]) if value == True])
    # print("val_idx generated")
    # test_idx= torch.tensor([index for index, value in enumerate(masks[2]) if value == True])
    # print("test_idx generated")YES

    # print (torch.max(train_idx))  
    # print (torch.max(val_idx))  
    # print (torch.max(test_idx))  

    # print(train_idx.dtype)YES
    
    # print(val_idx.dim())  
    # print(test_idx.dim())
    
    # print(train_idx.size(0))
    # print(val_idx.size(0))  
    # print(test_idx.size(0))YES
    
    #sys.exit(0)

    # data = (
    #     dataset.num_classes,
    #     train_idx,
    #     val_idx,
    #     test_idx,
    # )YES
    data = (
        dataset.num_classes,
        dataset.train_idx,
        dataset.val_idx,
        dataset.test_idx,
    )#NO

    mp.spawn(
        run,
        args=(nprocs, devices, g, data, args.mode, args.num_epochs),
        nprocs=nprocs,
    )

    # all_index= torch.arange(0, len(masks), 1)
    # shuffled_index = all_index[torch.randperm(len(all_index))]
    # x0=0
    # x1=int(len(masks)*0.25)
    # x2=x1+int(len(masks)*0.25)
    # x3=len(masks)
    # train_idx= shuffled_index[x0:x1]
    # val_idx= shuffled_index[x1:x2]
    # test_idx= shuffled_index[x2:x3]
    # print(train_idx.size())
    # print(val_idx.size())  
    # print(test_idx.size())