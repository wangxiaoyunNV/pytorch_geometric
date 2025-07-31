import time
import argparse
import torch
import torch.nn.functional as F

from torch_geometric.nn import MLP, GINConv, global_add_pool
from torch_geometric.data import Batch, Data
from torch_geometric.datasets import TUDataset
from torch.utils.data import Dataset, DataLoader

from cugraph_pyg.data import FeatureStore
from cugraph_pyg.tensor import DistTensor
import torch.distributed as dist

class DistTensorGraphDataset(Dataset):
    """
    Dataset for extracting individual graphs from distributed tensors.
    Avoids double reindexing by returning raw tensors instead of Data objects.
    """
    def __init__(self, dist_edge_index, feature_store, batch_vector, graph_indices=None):
        self.dist_edge_index = dist_edge_index
        self.feature_store = feature_store
        self.batch_vector = batch_vector
        
        if graph_indices is not None:
            self.graph_indices = graph_indices
        else:
            num_graphs = int(self.batch_vector.max().item()) + 1
            self.graph_indices = list(range(num_graphs))

    def __len__(self):
        return len(self.graph_indices)

    def __getitem__(self, idx):
        """Extract a single graph and return raw tensors to avoid double reindexing."""
        actual_graph_idx = self.graph_indices[idx]
        
        # Find all nodes belonging to this graph
        graph_node_mask = (self.batch_vector == actual_graph_idx)
        nodes_in_graph = torch.where(graph_node_mask)[0]
        
        # Extract node features
        sub_x = self.feature_store["node", "x", None][nodes_in_graph]
        
        # Get all edges from DistTensor
        num_total_edges = self.dist_edge_index._tensor.shape[0]
        all_edge_indices = torch.arange(num_total_edges, device=self.batch_vector.device)
        all_edges = self.dist_edge_index[all_edge_indices]
        
        # Find edges where both endpoints belong to this graph
        node_in_graph_mask = torch.zeros(self.batch_vector.shape[0], dtype=torch.bool, device=all_edges.device)
        node_in_graph_mask[nodes_in_graph] = True
        
        src_in_graph = node_in_graph_mask[all_edges[:, 0]]
        dst_in_graph = node_in_graph_mask[all_edges[:, 1]]
        edge_in_graph_mask = src_in_graph & dst_in_graph
        graph_edges = all_edges[edge_in_graph_mask]
        
        # Reindex nodes from global to local coordinates
        global_to_local = torch.full((self.batch_vector.shape[0],), -1, dtype=torch.long, device=all_edges.device)
        nodes_in_graph_gpu = nodes_in_graph.to(all_edges.device)
        global_to_local[nodes_in_graph_gpu] = torch.arange(nodes_in_graph.size(0), device=all_edges.device)
        local_edges = global_to_local[graph_edges]
        
        # Get graph label
        actual_graph_idx_gpu = torch.tensor([actual_graph_idx], device=self.feature_store["graph", "y", None].device)
        graph_y = self.feature_store["graph", "y", None][actual_graph_idx_gpu]
        
        return {
            'x': sub_x,
            'edge_index': local_edges.t(),
            'y': graph_y,
            'num_nodes': sub_x.size(0)
        }

def custom_collate_fn(batch):
    """
    Custom collate function to batch graphs.
    Takes raw tensors and creates a single PyG Data object with proper indexing.
    """
    x_list = [item['x'] for item in batch]
    edge_index_list = [item['edge_index'] for item in batch]
    y_list = [item['y'] for item in batch]
    num_nodes_list = [item['num_nodes'] for item in batch]
    
    # Concatenate features and labels
    x_batch = torch.cat(x_list, dim=0)
    y_batch = torch.cat(y_list, dim=0)
    
    # Create batch vector for graph membership tracking
    batch_vector = []
    for i, num_nodes in enumerate(num_nodes_list):
        batch_vector.extend([i] * num_nodes)
    batch_tensor = torch.tensor(batch_vector, dtype=torch.long)
    
    # Reindex edges with proper node offsets
    edge_index_batch = []
    node_offset = 0
    for i, edge_index in enumerate(edge_index_list):
        if edge_index.size(1) > 0:
            reindexed_edges = edge_index + node_offset
            edge_index_batch.append(reindexed_edges)
        node_offset += num_nodes_list[i]
    
    edge_index_final = torch.cat(edge_index_batch, dim=1) if edge_index_batch else torch.empty((2, 0), dtype=torch.long)
    
    # Create PyG Data object
    batch_data = Data(x=x_batch, edge_index=edge_index_final, y=y_batch, batch=batch_tensor)
    batch_data.num_graphs = len(batch)
    return batch_data

def setup_distributed():
    """Initialize distributed processing for cugraph-pyg."""
    if not dist.is_initialized():
        dist.init_process_group(backend="nccl", init_method="env://")

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="GIN training with cugraph-pyg distributed backend")
    parser.add_argument("--hidden_dim", type=int, default=64, help="Hidden dimension size")
    parser.add_argument("--num_layers", type=int, default=2, help="Number of GIN layers")
    parser.add_argument("--epochs", type=int, default=20, help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=128, help="Batch size for training")
    parser.add_argument("--device", type=str, default="cuda", help="Device to use for training")
    parser.add_argument("--dataset", type=str, default="MUTAG", help="Dataset name")
    parser.add_argument("--data_root", type=str, default="../../data/TU", help="Data root directory")
    return parser.parse_args()

class GIN(torch.nn.Module):
    """Graph Isomorphism Network for graph classification."""
    def __init__(self, in_channels, hidden_channels, out_channels, num_layers):
        super().__init__()

        self.convs = torch.nn.ModuleList()
        for _ in range(num_layers):
            mlp = MLP([in_channels, hidden_channels, hidden_channels])
            self.convs.append(GINConv(nn=mlp, train_eps=False))
            in_channels = hidden_channels

        self.mlp = MLP([hidden_channels, hidden_channels, out_channels], norm=None, dropout=0.5)

    def forward(self, x, edge_index, batch, batch_size):
        """Forward pass with graph-level pooling."""
        for conv in self.convs:
            x = conv(x, edge_index).relu()
        x = global_add_pool(x, batch, size=batch_size)
        return self.mlp(x)

def load_data(dataset_name, dataset_root, device, device_id):
    """
    Load dataset and set up distributed storage.
    Returns feature store, distributed edge tensor, and metadata.
    """
    dataset = TUDataset(dataset_root, name=dataset_name).shuffle()
    data = Batch.from_data_list(dataset)

    # Create distributed storage
    dist_edge_index = DistTensor.from_tensor(tensor=data.edge_index.t().to(f"cuda:{device_id}"))
    feature_store = FeatureStore()
    feature_store["node", "x", None] = data.x
    feature_store["graph", "y", None] = data.y

    split_idx = {"batch_vector": data.batch, "num_graphs": len(dataset)}
    num_features = data.x.size(1)
    num_classes = int(data.y.max().item()) + 1
    
    return (feature_store, dist_edge_index), split_idx, num_features, num_classes

def train(model, loader, optimizer, device):
    """Train the model for one epoch."""
    model.train()
    total_loss = 0
    for batch in loader:
        batch = batch.to(device)
        optimizer.zero_grad()
        out = model(batch.x, batch.edge_index, batch.batch, batch.num_graphs)
        loss = F.cross_entropy(out, batch.y)
        loss.backward()
        optimizer.step()
        total_loss += float(loss.detach()) * batch.num_graphs
    return total_loss / len(loader.dataset)

@torch.no_grad()
def test(model, loader, device):
    """Evaluate the model on the given loader."""
    model.eval()
    total_correct = 0
    for batch in loader:
        batch = batch.to(device)
        out = model(batch.x, batch.edge_index, batch.batch, batch.num_graphs)
        pred = out.argmax(dim=-1)
        total_correct += int((pred == batch.y).sum())
    return total_correct / len(loader.dataset)

def main():
    """Main training function."""
    setup_distributed()
    args = parse_args()
 
    # Setup device
    device = torch.device("cuda:0") if args.device == "cuda" and torch.cuda.is_available() else torch.device(args.device)
    device_id = 0
    print(f"Device: {device}, CUDA available: {torch.cuda.is_available()}")

    # Load data and create distributed storage
    (feature_store, dist_edge_index), split_idx, num_features, num_classes = load_data(
        args.dataset, args.data_root, device, device_id
    )

    # Create train/test split (90%/10%)
    batch_vector = split_idx["batch_vector"]
    num_graphs = split_idx["num_graphs"]
    train_size = int(0.9 * num_graphs)
    
    train_graph_indices = list(range(0, train_size))
    test_graph_indices = list(range(train_size, num_graphs))
    
    # Create datasets and loaders
    train_dataset = DistTensorGraphDataset(dist_edge_index, feature_store, batch_vector, train_graph_indices)
    test_dataset = DistTensorGraphDataset(dist_edge_index, feature_store, batch_vector, test_graph_indices)
    
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, collate_fn=custom_collate_fn)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, collate_fn=custom_collate_fn)

    # Create model and optimizer
    model = GIN(num_features, args.hidden_dim, num_classes, args.num_layers).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

    # Training loop
    times = []
    for epoch in range(1, args.epochs + 1):
        start = time.time()
        loss = train(model, train_loader, optimizer, device)
        train_acc = test(model, train_loader, device)
        test_acc = test(model, test_loader, device)
        times.append(time.time() - start)
        print(f'Epoch: {epoch:03d}, Loss: {loss:.4f}, Train: {train_acc:.4f}, Test: {test_acc:.4f}')
    
    print(f'Median time per epoch: {torch.tensor(times).median():.4f}s')

if __name__ == '__main__':
    main()

if dist.is_initialized():
    dist.destroy_process_group()