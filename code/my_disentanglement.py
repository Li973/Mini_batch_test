import argparse
import os
import time

import dgl
import dgl.nn as dglnn

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


class SAGE(nn.Module):
    def __init__(self, in_size, hid_size, out_size):
        super().__init__()
        self.layers = nn.ModuleList()
        # three-layer GraphSAGE-mean
        self.layers.append(dglnn.GraphConv(in_size, hid_size, activation=F.relu))
        self.layers.append(dglnn.GraphConv(hid_size, hid_size, activation=F.relu))
        #self.layers.append(dglnn.SAGEConv(hid_size, hid_size, "mean"))
        # self.layers.append(dglnn.SAGEConv(hid_size, hid_size, "mean"))
        self.layers.append(dglnn.GraphConv(hid_size, out_size))
        print("4")
        self.dropout = nn.Dropout(0.5)
        self.hid_size = hid_size
        self.out_size = out_size

    def forward(self, blocks, x):
        h = x
        for l, (layer, block) in enumerate(zip(self.layers, blocks)):
            if l != len(self.layers) - 1:
                #h = F.relu(h)
                h = self.dropout(h)
            h = layer(block, h)
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
#                 if l != len(self.layers) - 1:
# #                    h = F.relu(h)
#                     kk = self.dropout(x)
#                 h = layer(blocks[0], kk)  # len(blocks) = 1
                h = layer(blocks[0], x)  # len(blocks) = 1
                if l != len(self.layers) - 1:
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
    acc_x=0
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
        acc_x=acc
    return acc_x


def train(proc_id, nprocs, device, g, num_classes, train_idx, val_idx, test_idx, model, use_uva, num_epochs, args):
    sampler = NeighborSampler([10, 5, 5])
    train_dataloader = DataLoader(
        g,
        train_idx,
        sampler,
        device=device,
        batch_size=1024,  # 默认 batch size
        shuffle=True,
        drop_last=False,
        num_workers=0,
        use_uva=use_uva,
    )

    opt = torch.optim.Adam(model.parameters(), lr=0.0001, weight_decay=5e-4)

    for epoch in range(num_epochs):
        model.train()
        total_loss = 0
        data_iter = iter(train_dataloader)
        
        with tqdm.tqdm(total=len(train_dataloader)) as pbar:
            while True:
                buffer_blocks = []
                buffer_x = []
                buffer_y = []

                # Step 1: 显式从 CPU 中缓存 num_part 个 batch 的数据
                try:
                    for _ in range(args.num_parts):
                        input_nodes, output_nodes, blocks = next(data_iter)
                        input_nodes = input_nodes.cpu()
                        output_nodes = output_nodes.cpu()
                        x = g.ndata['feat'][input_nodes]
                        y = g.ndata['label'][output_nodes].to(torch.int64)
                        buffer_blocks.append(blocks)
                        buffer_x.append(x)
                        buffer_y.append(y)
                except StopIteration:
                    break  # 当前 epoch 所有 batch 已处理完

                # Step 2: 将这些 batch 统一加载到 GPU
                buffer_blocks_gpu = [[block.to(device) for block in b] for b in buffer_blocks]
                buffer_x_gpu = [x.to(device) for x in buffer_x]
                buffer_y_gpu = [y.to(device) for y in buffer_y]

                # Step 3: 对每个 batch 进行训练
                for i in range(len(buffer_blocks)):
                    x_batch = buffer_x_gpu[i]
                    y_batch = buffer_y_gpu[i]
                    blocks_batch = buffer_blocks_gpu[i]

                    y_hat = model(blocks_batch, x_batch)
                    loss = F.cross_entropy(y_hat, y_batch)
                    opt.zero_grad()
                    loss.backward()
                    opt.step()

                    total_loss += loss.item()

                pbar.update(args.num_parts)

        avg_loss = total_loss / len(train_dataloader)
        print(f"Epoch {epoch+1}, Loss: {avg_loss:.4f}")


def run(proc_id, nprocs, devices, g, data, mode, num_epochs, args):
    # find corresponding device for my rank
    device = devices[proc_id]
    torch.cuda.set_device(device)
    # initialize process group and unpack data for sub-processes
    dist.init_process_group(
        backend="nccl",
        init_method="tcp://127.0.0.1:12345",
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
        test_idx,
        model,
        use_uva,
        num_epochs,
        args #加个输入
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
    parser.add_argument(
        "--num_parts",
        type=int,
        default=8,
        help="Number of batches to load onto GPU at once.",
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
    )
    g = dataset[0]
    # avoid creating certain graph formats in each sub-process to save momory
    g.create_formats_()
    #if args.dataset_name == "ogbn-arxiv":
    g = dgl.to_bidirected(g, copy_ndata=True)
    g = dgl.add_self_loop(g)
    # thread limiting to avoid resource competition
    os.environ["OMP_NUM_THREADS"] = str(mp.cpu_count() // 2 // nprocs)
    data = (
        dataset.num_classes,
        dataset.train_idx,
        dataset.val_idx,
        dataset.test_idx,
    )

    mp.spawn(
        run,
        args=(nprocs, devices, g, data, args.mode, args.num_epochs, args),
        nprocs=nprocs,
    )
