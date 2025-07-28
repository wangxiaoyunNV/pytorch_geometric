import time
import argparse
import torch
import torch.nn.functional as F
from torch.nn import Linear, BatchNorm1d, ModuleList
from torch_geometric.nn import GINConv
from torch_geometric.data import Batch
from torch_geometric.datasets import TUDataset

from cugraph_pyg.data import GraphStore, FeatureStore
from cugraph_pyg.loader import NeighborLoader
from cugraph_pyg.tensor import DistTensor
import torch.distributed as dist

def setup_distributed():
    if not dist.is_initialized():
        dist.init_process_group(
            backend="nccl",  # or "gloo" for CPU-only, but "nccl" is recommended for GPU
            init_method="env://"
        )

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--hidden_dim", type=int, default=64)
    parser.add_argument("--num_layers", type=int, default=2)
    parser.add_argument("--dropout", type=float, default=0.5)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--fan_out", type=int, nargs='+', default=[10, 10])
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--dataset", type=str, default="MUTAG")
    parser.add_argument("--data_root", type=str, default="../../data/TU")
    return parser.parse_args()

class GIN(torch.nn.Module):
    def __init__(self, num_features, num_classes, hidden_dim=64, num_layers=5, dropout=0.5):
        super().__init__()
        self.convs = ModuleList()
        self.bns = ModuleList()
        self.dropout = dropout

        self.convs.append(GINConv(nn=Linear(num_features, hidden_dim)))
        self.bns.append(BatchNorm1d(hidden_dim))
        for _ in range(num_layers - 2):
            self.convs.append(GINConv(nn=Linear(hidden_dim, hidden_dim)))
            self.bns.append(BatchNorm1d(hidden_dim))
        self.convs.append(GINConv(nn=Linear(hidden_dim, num_classes)))

    def forward(self, x, edge_index):
        for conv, bn in zip(self.convs[:-1], self.bns):
            x = conv(x, edge_index)
            x = bn(x)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.convs[-1](x, edge_index)
        return x

def load_data(dataset_name, dataset_root, device, device_id):
    dataset = TUDataset(dataset_root, name=dataset_name).shuffle()
    data = Batch.from_data_list(dataset)
    # Create a DistTensor for the edge index (edgelist)
    num_edges = data.edge_index.size(1)
    dist_edge_index = DistTensor(
        data= data.edge_index.t().to(f"cuda:{device_id}"),
        shape=[num_edges, 2],
        dtype=data.edge_index.dtype,
        device_id=device_id,
        name="dist_edge_index"
    )
    print(dist_edge_index)
    print (data.edge_index.t())
    # print the first 10 elements of the dist_edge_index
    idx = torch.tensor([0,1,2,3,4,5,6,7,8,9], device=dist_edge_index.device)

    print(dist_edge_index[idx])

   
    feature_store = FeatureStore()

    feature_store["node", "x", None] = data.x
    feature_store["graph", "y", None] = data.y

    # 80/10/10 split
    num_nodes = data.num_nodes
    indices = torch.randperm(num_nodes)
    n_train = int(0.8 * num_nodes)
    n_val = int(0.1 * num_nodes)
    split_idx = {
        "train": indices[:n_train],
        "valid": indices[n_train:n_train+n_val],
        "test": indices[n_train+n_val:],
    }
    num_features = data.x.size(1)
    num_classes = int(data.y.max().item()) + 1
    return (feature_store, dist_edge_index), split_idx, num_features, num_classes

def create_loader(feature_store, dist_edge_index, input_nodes, num_neighbors, batch_size, shuffle=True):
    return NeighborLoader(
        data=(feature_store, dist_edge_index),
        num_neighbors=num_neighbors,
        input_nodes=input_nodes,
        batch_size=batch_size,
        shuffle=shuffle,
    )

def train(model, loader, optimizer, device):
    model.train()
    for batch in loader:
        batch = batch.to(device)
        x = batch.x
        edge_index = batch.edge_index
        y = batch.y
        optimizer.zero_grad()
        out = model(x, edge_index)
        loss = F.cross_entropy(out, y)
        loss.backward()
        optimizer.step()

@torch.no_grad()
def test(model, loader, device):
    model.eval()
    total_correct = total_examples = 0
    for batch in loader:
        batch = batch.to(device)
        x = batch.x
        edge_index = batch.edge_index
        y = batch.y
        out = model(x, edge_index)
        pred = out.argmax(dim=-1)
        total_correct += int((pred == y).sum())
        total_examples += y.size(0)
    return total_correct / total_examples if total_examples > 0 else 0.0

def main():
    setup_distributed()
    args = parse_args()
    if args.device.startswith("cuda") and ":" in args.device:
        device = torch.device(args.device)
        device_id = device.index
    elif args.device.startswith("cuda"):
        device = torch.device("cuda:0")
        device_id = 0
    else:
        device = torch.device(args.device)
        device_id = 0
    print(f"Device: {device}, {torch.cuda.is_available()} Device ID: {device_id}")

    (feature_store, dist_edge_index), split_idx, num_features, num_classes = load_data(
        args.dataset, args.data_root, device, device_id
    )

    train_loader = create_loader(feature_store, dist_edge_index, split_idx["train"], args.fan_out, args.batch_size, shuffle=True)
    val_loader = create_loader(feature_store, dist_edge_index, split_idx["valid"], args.fan_out, args.batch_size, shuffle=False)
    test_loader = create_loader(feature_store, dist_edge_index, split_idx["test"], args.fan_out, args.batch_size, shuffle=False)

    model = GIN(num_features, num_classes, args.hidden_dim, args.num_layers, args.dropout).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

    for epoch in range(1, args.epochs + 1):
        train(model, train_loader, optimizer, device)
        val_acc = test(model, val_loader, device)
        print(f"Epoch: {epoch:03d}, Val Acc: {val_acc:.4f}")

    test_acc = test(model, test_loader, device)
    print(f"Test Acc: {test_acc:.4f}")

if __name__ == '__main__':
    main()

if dist.is_initialized():
    dist.destroy_process_group() 