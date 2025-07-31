import time
import argparse
import os
import torch
import torch.nn.functional as F
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler

from torch_geometric.nn import MLP, GINConv, global_add_pool
from torch_geometric.data import Batch
from torch_geometric.datasets import TUDataset

from cugraph_pyg.data import FeatureStore
from cugraph_pyg.tensor import DistTensor
import torch.distributed as dist
import numpy as np

from torch.utils.data import Dataset
from torch_geometric.loader import DataLoader
from torch_geometric.data import Data

class DistTensorGraphDataset(Dataset):
    def __init__(self, dist_edge_index, feature_store, start_indices, num_edges, graph_indices=None):
        self.dist_edge_index = dist_edge_index  # DistTensor for edge_index [2, num_edges]
        self.feature_store = feature_store     # featurestore for node features [num_nodes, num_features] and graph labels [num_graphs, num_classes]
        self.start_indices = start_indices      # List/array of start indices for each small graph
        self.num_edges = num_edges              # Total number of edges in the big graph
        
        # If graph_indices is provided, only use those specific graphs
        if graph_indices is not None:
            self.graph_indices = graph_indices
        else:
            self.graph_indices = list(range(len(start_indices)))  # Use all graphs

    def __len__(self):
        return len(self.graph_indices)

    def __getitem__(self, idx):
        # Map the dataset index to the actual graph index
        actual_graph_idx = self.graph_indices[idx]
        edge_start = self.start_indices[actual_graph_idx]
        # Compute edge_end: for last graph, go to end; else, next start
        if actual_graph_idx < len(self.start_indices) - 1:
            edge_end = self.start_indices[actual_graph_idx + 1]
        else:
            edge_end = self.num_edges
            
        # Get edge indices for this small graph: [edge_start, edge_end)
        edge_indices = torch.arange(edge_start, edge_end, device=self.dist_edge_index.device)
        
        # Get subgraph edge index - shape [num_subgraph_edges, 2]
        sub_edge_index = self.dist_edge_index[edge_indices]  
        
        # 1. Get all unique node indices in the subgraph
        nodes_in_subgraph = torch.unique(sub_edge_index)
        
        # 2. Extract node features for these nodes
        sub_x = self.feature_store["node", "x", None][nodes_in_subgraph]
        
        # 3. Reindex edge indices to local node indices
        global_to_local = -torch.ones(self.feature_store["node", "x", None].shape[0], 
                                     dtype=torch.long, device=sub_edge_index.device)
        global_to_local[nodes_in_subgraph] = torch.arange(nodes_in_subgraph.size(0), 
                                                         device=sub_edge_index.device)
        local_sub_edge_index = global_to_local[sub_edge_index]
        actual_graph_idx_gpu = torch.tensor([actual_graph_idx], device=self.feature_store["graph", "y", None].device)

        # 4. Return Data object with correct format
        return Data(x=sub_x, 
                   edge_index=local_sub_edge_index.t(),  # Transpose to [2, num_edges] format
                   y=self.feature_store["graph", "y", None][actual_graph_idx_gpu])

def setup_distributed(rank, world_size, master_addr='localhost', master_port='12355', num_nodes=1, node_rank=0):
    """Setup distributed training environment for given rank and world_size"""
    os.environ['MASTER_ADDR'] = master_addr
    os.environ['MASTER_PORT'] = master_port
    
    # Calculate global rank and local rank for multi-node setup
    local_world_size = world_size // num_nodes
    global_rank = node_rank * local_world_size + rank
    local_rank = rank
    
    # Set environment variables required by cugraph-pyg FeatureStore
    os.environ['RANK'] = str(global_rank)
    os.environ['WORLD_SIZE'] = str(world_size)
    os.environ['LOCAL_RANK'] = str(local_rank)
    os.environ['LOCAL_WORLD_SIZE'] = str(local_world_size)
    
    if not dist.is_initialized():
        dist.init_process_group(
            backend="nccl",  # NCCL backend for GPU communication
            rank=global_rank,
            world_size=world_size,
            init_method="env://"
        )
    
    # Set the current device for this process
    torch.cuda.set_device(local_rank)
    print(f"Node {node_rank}, Local Rank {local_rank}, Global Rank {global_rank}/{world_size} initialized on device cuda:{local_rank}")

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--hidden_dim", type=int, default=64)
    parser.add_argument("--num_layers", type=int, default=2)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--dataset", type=str, default="MUTAG")
    parser.add_argument("--data_root", type=str, default="../../data/TU")
    # Distributed training arguments
    parser.add_argument("--master_addr", type=str, default="localhost")
    parser.add_argument("--master_port", type=str, default="12355")
    parser.add_argument("--world_size", type=int, default=None, help="Total number of processes (auto-detected if None)")
    parser.add_argument("--num_nodes", type=int, default=1, help="Number of nodes")
    parser.add_argument("--node_rank", type=int, default=0, help="Rank of this node")
    return parser.parse_args()

class GIN(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, num_layers):
        super().__init__()

        self.convs = torch.nn.ModuleList()
        for _ in range(num_layers):
            mlp = MLP([in_channels, hidden_channels, hidden_channels])
            self.convs.append(GINConv(nn=mlp, train_eps=False))
            in_channels = hidden_channels

        self.mlp = MLP([hidden_channels, hidden_channels, out_channels],
                       norm=None, dropout=0.5)

    def forward(self, x, edge_index, batch, batch_size):
        for conv in self.convs:
            x = conv(x, edge_index).relu()
        # Pass the batch size to avoid CPU communication/graph breaks:
        x = global_add_pool(x, batch, size=batch_size)
        return self.mlp(x)

def load_data(dataset_name, dataset_root, device, device_id):
    dataset = TUDataset(dataset_root, name=dataset_name).shuffle()
    data = Batch.from_data_list(dataset)

    # batch.batch: shape [num_nodes], value is graph index for each node
    batch_vector = data.batch.cpu().numpy()

    # Find where the graph index changes
    change_indices = np.where(np.diff(batch_vector) != 0)[0] + 1

    # Start indices: 0 plus all change_indices
    start_indices = np.concatenate(([0], change_indices))

    # Create a DistTensor for the edge index (edgelist)
    num_edges = data.edge_index.size(1)
    dist_edge_index = DistTensor.from_tensor(tensor=data.edge_index.t().to(f"cuda:{device_id}"))

    feature_store = FeatureStore()

    feature_store["node", "x", None] = data.x
    feature_store["graph", "y", None] = data.y

    # Return the information needed for graph-level training
    split_idx = {
        "start_indices": start_indices,
        "num_edges": num_edges,
    }
    num_features = data.x.size(1)
    num_classes = int(data.y.max().item()) + 1
    return (feature_store, dist_edge_index), split_idx, num_features, num_classes

def train(model, loader, optimizer, device, local_rank, global_rank):
    model.train()
    total_loss = 0
    total_samples = 0
    
    # Only local rank 0 on each node prints to avoid too much output
    if local_rank == 0:
        print(f"Node local rank {local_rank} (global rank {global_rank}): Starting training with {len(loader)} batches")
    
    for batch_idx, batch in enumerate(loader):
        batch = batch.to(device)
        x = batch.x
        edge_index = batch.edge_index
        y = batch.y
        optimizer.zero_grad()
        out = model(x, edge_index, batch.batch, batch.num_graphs)  # Pass batch indices and size for pooling
        loss = F.cross_entropy(out, y)
        loss.backward()
        optimizer.step()
        total_loss += float(loss.detach()) * batch.num_graphs
        total_samples += batch.num_graphs
        
        # Optional: Detailed logging from local rank 0 only (one per node)
        if local_rank == 0 and batch_idx % 10 == 0:
            print(f"Local rank {local_rank} (global rank {global_rank}): Batch {batch_idx}, Loss: {loss.item():.4f}")
    
    # Average loss across all ranks
    total_loss_tensor = torch.tensor(total_loss, device=device)
    total_samples_tensor = torch.tensor(total_samples, device=device)
    
    dist.all_reduce(total_loss_tensor, op=dist.ReduceOp.SUM)
    dist.all_reduce(total_samples_tensor, op=dist.ReduceOp.SUM)
    
    if local_rank == 0:
        print(f"Local rank {local_rank} (global rank {global_rank}): Training completed. Global avg loss: {total_loss_tensor.item() / total_samples_tensor.item():.4f}")
    
    return total_loss_tensor.item() / total_samples_tensor.item()

@torch.no_grad()
def test(model, loader, device, local_rank, global_rank):
    model.eval()
    total_correct = total_examples = 0
    
    if local_rank == 0:
        print(f"Local rank {local_rank} (global rank {global_rank}): Starting evaluation with {len(loader)} batches")
    
    for batch_idx, batch in enumerate(loader):
        batch = batch.to(device)
        x = batch.x
        edge_index = batch.edge_index
        y = batch.y
        out = model(x, edge_index, batch.batch, batch.num_graphs)
        pred = out.argmax(dim=-1)
        total_correct += int((pred == y).sum())
        total_examples += y.size(0)
        
        # Optional: Progress logging from local rank 0 only
        if local_rank == 0 and batch_idx % 5 == 0:
            local_acc = total_correct / total_examples if total_examples > 0 else 0
            print(f"Local rank {local_rank} (global rank {global_rank}): Eval batch {batch_idx}, Local accuracy so far: {local_acc:.4f}")
    
    # Aggregate results across all ranks
    total_correct_tensor = torch.tensor(total_correct, device=device)
    total_examples_tensor = torch.tensor(total_examples, device=device)
    
    dist.all_reduce(total_correct_tensor, op=dist.ReduceOp.SUM)
    dist.all_reduce(total_examples_tensor, op=dist.ReduceOp.SUM)
    
    global_accuracy = total_correct_tensor.item() / total_examples_tensor.item() if total_examples_tensor.item() > 0 else 0.0
    
    if local_rank == 0:
        print(f"Local rank {local_rank} (global rank {global_rank}): Evaluation completed. Global accuracy: {global_accuracy:.4f}")
    
    return global_accuracy

def run_worker(rank, world_size, args):
    """Worker function that runs on each GPU process"""
    # Setup distributed training for this rank
    setup_distributed(rank, world_size, args.master_addr, args.master_port, args.num_nodes, args.node_rank)
    
    # Calculate actual local rank and device for multi-node setup
    local_world_size = world_size // args.num_nodes
    local_rank = rank
    global_rank = args.node_rank * local_world_size + rank
    
    # Set device for this rank
    device = torch.device(f"cuda:{local_rank}")
    device_id = local_rank
    
    # Load data (each rank loads the same data but will get different subsets via DistributedSampler)
    (feature_store, dist_edge_index), split_idx, num_features, num_classes = load_data(
        args.dataset, args.data_root, device, device_id
    )

    # Calculate splits (90% train, 10% test)
    start_indices = split_idx["start_indices"]
    num_graphs = len(start_indices)
    train_size = int(0.9 * num_graphs)
    
    # Create graph index ranges for each split
    train_graph_indices = list(range(0, train_size))
    test_graph_indices = list(range(train_size, num_graphs))
    
    # Create separate DistTensorGraphDataset instances for train/test
    train_dataset = DistTensorGraphDataset(
        dist_edge_index=dist_edge_index,
        feature_store=feature_store,
        start_indices=start_indices,
        num_edges=split_idx["num_edges"],
        graph_indices=train_graph_indices
    )
    
    test_dataset = DistTensorGraphDataset(
        dist_edge_index=dist_edge_index,
        feature_store=feature_store,
        start_indices=start_indices,
        num_edges=split_idx["num_edges"],
        graph_indices=test_graph_indices
    )
    
    # Create distributed samplers to split data across ranks
    train_sampler = DistributedSampler(
        train_dataset, 
        num_replicas=world_size,
        rank=rank,
        shuffle=True
    )
    
    test_sampler = DistributedSampler(
        test_dataset,
        num_replicas=world_size,
        rank=rank,
        shuffle=False
    )
    
    # Create DataLoaders with distributed samplers
    train_loader = DataLoader(
        train_dataset, 
        batch_size=args.batch_size, 
        sampler=train_sampler,
        num_workers=2,
        pin_memory=True
    )
    test_loader = DataLoader(
        test_dataset, 
        batch_size=args.batch_size, 
        sampler=test_sampler,
        num_workers=2,
        pin_memory=True
    )

    # Create model and wrap with DistributedDataParallel
    model = GIN(num_features, args.hidden_dim, num_classes, args.num_layers).to(device)
    model = DDP(model, device_ids=[local_rank], find_unused_parameters=True)
    
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

    # Training loop
    times = []
    for epoch in range(1, args.epochs + 1):
        # Set epoch for DistributedSampler to ensure different shuffling across epochs
        train_sampler.set_epoch(epoch)
        
        start = time.time()
        loss = train(model, train_loader, optimizer, device, local_rank, global_rank)
        train_acc = test(model, train_loader, device, local_rank, global_rank)
        test_acc = test(model, test_loader, device, local_rank, global_rank)
        times.append(time.time() - start)
        
        # Synchronize all processes before printing
        dist.barrier()
        
        # Only global rank 0 prints to avoid duplicate outputs
        if global_rank == 0:
            print(f'Epoch: {epoch:03d}, Loss: {loss:.4f}, Train: {train_acc:.4f}, '
                  f'Test: {test_acc:.4f}')
        
        # Synchronize again after printing
        dist.barrier()
    
    if global_rank == 0:
        print(f'Median time per epoch: {torch.tensor(times).median():.4f}s')
    
    # Cleanup
    dist.destroy_process_group()

def main():
    args = parse_args()
    
    # Determine local world size (GPUs per node)
    local_world_size = torch.cuda.device_count()
    if local_world_size == 0:
        raise RuntimeError("No CUDA devices available for distributed training")
    
    # Calculate total world size
    if args.world_size is None:
        world_size = local_world_size * args.num_nodes
    else:
        world_size = args.world_size
    
    print(f"Starting distributed training:")
    print(f"  Total nodes: {args.num_nodes}")
    print(f"  Node rank: {args.node_rank}")
    print(f"  GPUs per node: {local_world_size}")
    print(f"  Total processes: {world_size}")
    
    # Use torch.multiprocessing to spawn multiple processes (one per local GPU)
    mp.spawn(
        run_worker,
        args=(world_size, args),
        nprocs=local_world_size,  # Only spawn processes for local GPUs
        join=True
    )

if __name__ == '__main__':
    main() 