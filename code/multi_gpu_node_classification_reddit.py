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

from hongtu.embedding_swap_op import blSwappingFunction
from hongtu.embedding_swap_op import forwardshuffle
from hongtu.embedding_swap_op import test_shuffle



class SAGE(nn.Module):
    def __init__(self, in_size, hid_size, out_size):
        super().__init__()
        self.layers = nn.ModuleList()
        # three-layer GraphSAGE-mean
       # self.layers.append(dglnn.GraphConv(in_size, hid_size, activation=F.relu))
        self.layers.append(dglnn.SAGEConv(in_size, hid_size, "mean"))
        self.layers.append(dglnn.SAGEConv(hid_size, hid_size, "mean"))
        self.layers.append(dglnn.SAGEConv(hid_size, out_size, "mean"))
        # self.layers.append(dglnn.SAGEConv(hid_size, hid_size, "mean"))
        #self.layers.append(dglnn.GraphConv(hid_size, out_size, activation=F.relu))
        self.dropout = nn.Dropout(0.5)
        self.hid_size = hid_size
        self.out_size = out_size

    def forward(self, blocks, x):
        h = x
        for l, (layer, block) in enumerate(zip(self.layers, blocks)):
            h = layer(block, h)
            if l != len(self.layers) - 1:
                h = F.relu(h)
                h = self.dropout(h)
        return h

    def inference(self, g, device, batch_size, use_uva):
        g.ndata["h"] = g.ndata["feat"]
        sampler = MultiLayerFullNeighborSampler(1, prefetch_node_feats=["h"])
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
                use_ddp=True,
                use_uva=use_uva,
            )
            # in order to prevent running out of GPU memory, allocate a
            # shared output tensor 'y' in host memory
            y = shared_tensor(
                (
                    g.num_nodes(),
                    self.hid_size
                    if l != len(self.layers) - 1
                    else self.out_size,
                )
            )
            for input_nodes, output_nodes, blocks in (
                tqdm.tqdm(dataloader) if dist.get_rank() == 0 else dataloader
            ):
                x = blocks[0].srcdata["h"]
                h = layer(blocks[0], x)  # len(blocks) = 1
                if l != len(self.layers) - 1:
                    h = F.relu(h)
                    h = self.dropout(h)
                # non_blocking (with pinned memory) to accelerate data transfer
                y[output_nodes] = h.to(y.device, non_blocking=True)
            # make sure all GPUs are done writing to 'y'
            dist.barrier()
            g.ndata["h"] = y if use_uva else y.to(device)

        g.ndata.pop("h")
        return y


def evaluate(model, g, num_classes, dataloader):
    model.eval()
    ys = []
    y_hats = []
    for it, (input_nodes, output_nodes, blocks) in enumerate(dataloader):
        with torch.no_grad():
            x = blocks[0].srcdata["feat"]
            ys.append(blocks[-1].dstdata["label"])
            y_hats.append(model(blocks, x))
    return MF.accuracy(
        torch.cat(y_hats),
        torch.cat(ys),
        task="multiclass",
        num_classes=num_classes,
    )


def layerwise_infer(
    proc_id, device, g, num_classes, nid, model, use_uva, batch_size=2**10
):
    model.eval()
    with torch.no_grad():
        pred = model.module.inference(g, device, batch_size, use_uva)
        pred = pred[nid]
        labels = g.ndata["label"][nid].to(pred.device)
    if proc_id == 0:
        acc = MF.accuracy(
            pred, labels, task="multiclass", num_classes=num_classes
        )
        print("Test accuracy {:.4f}".format(acc.item()))


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
    test_shuffle(device, proc_id)
    sampler = NeighborSampler(
        [5, 5, 5], prefetch_node_feats=["feat"], prefetch_labels=["label"]
    )
    train_dataloader = DataLoader(
        g,
        train_idx,
        sampler,
        device=device,
        batch_size=2048,
        shuffle=True,
        drop_last=False,
        num_workers=0,
        use_ddp=True,
        use_uva=use_uva,
    )
    val_dataloader = DataLoader(
        g,
        val_idx,
        sampler,
        device=device,
        batch_size=2048,
        shuffle=True,
        drop_last=False,
        num_workers=0,
        use_ddp=True,
        use_uva=use_uva,
    )
    opt = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=5e-4)
    for epoch in range(num_epochs):
        t0 = time.time()
        model.train()
        total_loss = 0
        for it, (_, _, blocks) in enumerate(train_dataloader):
            x = blocks[0].srcdata["feat"]
            y = blocks[-1].dstdata["label"].to(torch.int64)
            y_hat = model(blocks, x)
            loss = F.cross_entropy(y_hat, y)
            opt.zero_grad()
            loss.backward()
            opt.step()
            total_loss += loss
        acc = (
            evaluate(model, g, num_classes, val_dataloader).to(device) / nprocs
        )
        t1 = time.time()
        dist.reduce(acc, 0)
        if proc_id == 0:
            print(
                "Epoch {:05d} | Loss {:.4f} | Accuracy {:.4f} | "
                "Time {:.4f}".format(
                    epoch, total_loss / (it + 1), acc.item(), t1 - t0
                )
            )


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
    val_idx = val_idx.to(device)
    g = g.to(device if mode == "puregpu" else "cpu")
    # create GraphSAGE model (distributed)
    in_size = g.ndata["feat"].shape[1]
    model = SAGE(in_size, 256, num_classes).to(device)
    model = DistributedDataParallel(
        model, device_ids=[device], output_device=device
    )
    # training + testing
    use_uva = mode == "mixed"
    train(
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
    )
    layerwise_infer(proc_id, device, g, num_classes, test_idx, model, use_uva)
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
        default="0",
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
    # dataset = AsNodePredDataset(
    #     DglNodePropPredDataset(args.dataset_name, root=args.dataset_dir)
    # )
    #dataset = RedditDataset()
    dataset = RedditDataset(raw_dir ='/data/wangqg/dgl/')
    g = dataset[0]
    # avoid creating certain graph formats in each sub-process to save momory
    g.create_formats_()
    if args.dataset_name == "ogbn-arxiv":
        g = dgl.to_bidirected(g, copy_ndata=True)
        g = dgl.add_self_loop(g)
    # thread limiting to avoid resource competition
    os.environ["OMP_NUM_THREADS"] = str(mp.cpu_count() // 2 // nprocs)
    masks = g.ndata["train_mask"], g.ndata["val_mask"], g.ndata["test_mask"]
    print("generating masks")
    v_size=len(masks[0])
    # for i in range(0, v_size):
    #     rand_val=random.random()
    #     if(rand_val<0.25):
    #         masks[0][i]=False
    #         masks[1][i]=True
    #     if(rand_val>0.25 and rand_val<0.75):
    #         masks[0][i]=False
    #         masks[2][i]=True
    print("masks generated")
    train_idx= torch.tensor([index for index, value in enumerate(masks[0]) if value == True])
    print("train_idx generated")
    val_idx= torch.tensor([index for index, value in enumerate(masks[1]) if value == True])
    print("val_idx generated")
    test_idx= torch.tensor([index for index, value in enumerate(masks[2]) if value == True])
    print("test_idx generated")
    # print (torch.max(train_idx))  
    # print (torch.max(val_idx))  
    # print (torch.max(test_idx))  
    print(train_idx.dtype)
    # print(val_idx.dim())  
    # print(test_idx.dim())
    print(train_idx.size(0))
    print(val_idx.size(0))  
    print(test_idx.size(0))
    #sys.exit(0)
    data = (
        dataset.num_classes,
        train_idx,
        val_idx,
        test_idx,
    )

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