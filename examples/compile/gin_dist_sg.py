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
import numpy as np

from torch.utils.data import Dataset, DataLoader
from torch_geometric.data import Data

class DistTensorGraphDataset(Dataset):
    def __init__(self, dist_edge_index, feature_store, start_indices, num_edges):
        self.dist_edge_index = dist_edge_index  # DistTensor for edge_index [2, num_edges]
        self.feature_store = feature_store     # featurestore for node features [num_nodes, num_features] and graph labels [num_graphs, num_classes]
        self.start_indices = start_indices      # List/array of start indices for each small graph
        self.num_edges = num_edges              # Total number of edges in the big graph

    def __len__(self):
        return len(self.start_indices)

    def __getitem__(self, idx):
        edge_start = self.start_indices[idx]
        # Compute node_end: for last graph, go to end; else, next start - 1
        if idx < len(self.start_indices) - 1:
            edge_end = self.start_indices[idx + 1]
        else:
            edge_end = self.num_edges
        # edge indices for this small graph: [edge_start, edge_end)
        edge_indices = torch.tensor(range(edge_start, edge_end), device= self.dist_edge_index.device)

        # get subgraph edge index
        sub_edge_index = self.dist_edge_index[edge_indices]

        # Get node features
        x = self.feature_store["node", "x", None]

        # Get edge indices for this small graph
        edge_index = self.dist_edge_index  # shape [num_edges, 2]
        # Mask: both source and target in [node_start, node_end)
        mask = (edge_index[0] >= edge_start) & (edge_index[0] < edge_end) & \
               (edge_index[1] >= edge_start) & (edge_index[1] < edge_end)
        sub_edge_index = edge_index[mask] - edge_start  # reindex to local

        return Data(x=x, edge_index=sub_edge_index, y=self.feature_store["graph", "y", None][idx])

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



# batch.batch: shape [num_nodes], value is graph index for each node
    batch_vector = data.batch.cpu().numpy()

# Find where the graph index changes
    change_indices = np.where(np.diff(batch_vector) != 0)[0] + 1

# Start indices: 0 plus all change_indices
    start_indices = np.concatenate(([0], change_indices))
# End indices: all change_indices minus 1, plus the last index
    end_indices = np.concatenate((change_indices - 1, [len(batch_vector) - 1]))

# Print the node index range for each small graph
    for i, (start, end) in enumerate(zip(start_indices, end_indices)):
        print(f"Graph {i}: nodes {start} to {end}")
    
    print (start_indices)
    print (end_indices)

    # Create a DistTensor for the edge index (edgelist)
    num_edges = data.edge_index.size(1)
    dist_edge_index = DistTensor.from_tensor(tensor=data.edge_index.t().to(f"cuda:{device_id}"))
    #print(dist_edge_index)
    #print (data.edge_index.t())
    #print the first 10 elements of the dist_edge_index
    #idx = torch.tensor([0,1,2,3,4,5,6,7,8,9], device=dist_edge_index.device)
    #print(dist_edge_index)
    #print(dist_edge_index[idx])
    # dist_edges_index all 0s  not correct. 


    feature_store = FeatureStore()

    feature_store["node", "x", None] = data.x
    feature_store["graph", "y", None] = data.y

    dataset_test = DistTensorGraphDataset(dist_edge_index, feature_store, start_indices, num_edges)
    # Create the DataLoader (batch_size=128, or any batch size you want)
    loader = DataLoader(dataset_test, batch_size=128, shuffle=False)

    print(f"Total number of small graphs: {len(dataset_test)}")
    for i, batch in enumerate(loader):
        print(f"\nBatch {i}:")
        print(f"  Number of small graphs in this batch: {len(batch)}")
        for j, data in enumerate(batch):
            print(f"    Small graph {j}:")
            print(f"      x shape: {data.x.shape}")
            print(f"      edge_index shape: {data.edge_index.shape}")
        # Optionally, break after first batch for brevity
        break

    # some thing is wrong with feature store. 
   

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