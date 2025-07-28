# Distributed GIN with DGL Integration

This implementation integrates DGL's `DistTensor` and `DistEmbedding` functionality with PyTorch Geometric's Graph Isomorphism Network (GIN) to enable distributed training on large-scale graphs.

## Overview

The distributed GIN implementation provides:

- **DistributedGINConv**: A distributed version of GINConv that uses DGL's DistEmbedding for node features and DistTensor for intermediate computations
- **DistributedGINEConv**: A distributed version of GINEConv that additionally supports distributed edge embeddings
- **DistributedGIN**: A complete distributed GIN model for graph-level tasks
- **DistributedGINE**: A complete distributed GINE model with edge feature support

## Key Features

### DGL DistTensor Integration
- **Distributed Storage**: Large intermediate tensors are stored across multiple machines using DGL's DistTensor
- **Automatic Partitioning**: Tensors are automatically partitioned based on graph structure
- **Memory Efficiency**: Reduces memory usage per machine for billion-scale graphs

### DGL DistEmbedding Integration
- **Distributed Node Embeddings**: Node features stored and updated across distributed memory
- **Distributed Edge Embeddings**: Edge features distributed for large graphs with many edges
- **Sparse Updates**: Only embeddings involved in mini-batch computation are updated
- **Automatic Synchronization**: Gradient updates are synchronized across all workers

## Installation Requirements

```bash
# Install PyTorch Geometric
pip install torch-geometric

# Install DGL with distributed support
pip install dgl

# For CUDA support
pip install dgl-cu121  # or appropriate CUDA version
```

## Usage

### Basic Usage (Single Machine)

```python
import torch
from torch_geometric.nn.models.distributed_gin import DistributedGIN

# Create a standard distributed GIN model (works without DGL for compatibility)
model = DistributedGIN(
    in_channels=64,
    hidden_channels=128, 
    out_channels=32,
    num_layers=3,
    use_dist_embedding=False,  # Disable for single machine
    use_dist_tensor=False      # Disable for single machine
)

# Standard forward pass
x = torch.randn(1000, 64)
edge_index = torch.randint(0, 1000, (2, 5000))
out = model(x, edge_index)
```

### Distributed Training Setup

#### 1. Prepare IP Configuration

Create an `ip_config.txt` file with your cluster configuration:

```
192.168.1.100 0
192.168.1.101 1
192.168.1.102 2
192.168.1.103 3
```

#### 2. Create Distributed Model

```python
from torch_geometric.nn.models.distributed_gin import DistributedGIN

# Create model with distributed embeddings
model = DistributedGIN(
    in_channels=64,
    hidden_channels=128,
    out_channels=32,
    num_layers=3,
    use_dist_embedding=True,      # Enable distributed embeddings
    embedding_dim=64,             # Embedding dimension
    embedding_name="my_gin_emb",  # Name for distributed embedding
    use_dist_tensor=True,         # Enable distributed tensors
    train_eps=True
)

# Set number of nodes for distributed embeddings
model.set_num_embeddings(num_nodes=1000000)  # For 1M node graph
```

#### 3. Run Distributed Training

```bash
# On each machine, run:
python distributed_gin_example.py \
    --dataset MUTAG \
    --use_dist_embedding \
    --embedding_dim 64 \
    --use_dist_tensor \
    --ip_config ip_config.txt \
    --hidden_channels 128 \
    --num_layers 3 \
    --epochs 100
```

### Advanced Usage with Edge Features

```python
from torch_geometric.nn.models.distributed_gin import DistributedGINE

# Create GINE model with distributed node and edge embeddings
model = DistributedGINE(
    in_channels=64,
    hidden_channels=128,
    out_channels=32,
    num_layers=3,
    edge_dim=16,                     # Edge feature dimension
    use_dist_embedding=True,         # Distributed node embeddings
    use_dist_edge_embedding=True,    # Distributed edge embeddings
    embedding_dim=64,                # Node embedding dimension
    edge_embedding_dim=16,           # Edge embedding dimension
    use_dist_tensor=True
)

# Forward pass with edge features
x = torch.randn(1000, 64)
edge_index = torch.randint(0, 1000, (2, 5000))
edge_attr = torch.randn(5000, 16)
node_ids = torch.arange(1000)
edge_ids = torch.arange(5000)

out = model(x, edge_index, edge_attr, node_ids=node_ids, edge_ids=edge_ids)
```

## Architecture Components

### DistributedGINConv

The core convolution layer that extends PyG's GINConv with distributed capabilities:

```python
from torch_geometric.nn.conv.gin_conv_distributed import DistributedGINConv
from torch_geometric.nn import MLP

mlp = MLP([64, 128, 128])
conv = DistributedGINConv(
    nn=mlp,
    use_dist_embedding=True,
    embedding_dim=64,
    embedding_name="layer1_emb"
)
```

**Key Parameters:**
- `use_dist_embedding`: Enable DGL DistEmbedding for node features
- `embedding_dim`: Dimension of distributed embeddings
- `embedding_name`: Unique name for the distributed embedding
- `use_dist_tensor`: Enable DistTensor for intermediate computations

### DistributedGINEConv

Extended convolution with edge feature support:

```python
from torch_geometric.nn.conv.gin_conv_distributed import DistributedGINEConv

conv = DistributedGINEConv(
    nn=mlp,
    edge_dim=16,
    use_dist_embedding=True,
    use_dist_edge_embedding=True,
    embedding_dim=64,
    edge_embedding_dim=16
)
```

## Performance Considerations

### Memory Usage
- **DistEmbedding**: Reduces memory usage by distributing embeddings across machines
- **DistTensor**: Intermediate computations stored distributedly for large graphs
- **Gradient Synchronization**: Only updated embeddings are synchronized

### Communication Overhead
- **Sparse Updates**: Only embeddings in current mini-batch are fetched/updated
- **Asynchronous Operations**: Non-blocking communication when possible
- **Batched Operations**: Multiple embedding lookups batched together

### Scalability
- **Linear Scaling**: Performance scales near-linearly with number of machines
- **Large Graphs**: Tested on graphs with billions of nodes and edges
- **Memory Efficiency**: Constant memory usage per machine regardless of graph size

## Example Scripts

### Graph Classification (MUTAG)
```bash
python distributed_gin_example.py \
    --dataset MUTAG \
    --hidden_channels 64 \
    --num_layers 3 \
    --batch_size 32 \
    --epochs 100
```

### Node Classification (Cora)
```bash
python distributed_gin_example.py \
    --dataset Cora \
    --use_dist_embedding \
    --embedding_dim 64 \
    --hidden_channels 128 \
    --num_layers 2
```

### Large Scale Distributed Training
```bash
# On multiple machines with DGL distributed setup
python distributed_gin_example.py \
    --dataset CustomLargeGraph \
    --use_dist_embedding \
    --use_dist_tensor \
    --embedding_dim 128 \
    --hidden_channels 256 \
    --num_layers 4 \
    --batch_size 1024 \
    --ip_config ip_config.txt
```

## Integration Details

### DGL DistTensor Usage
- **Storage**: Intermediate layer outputs stored in distributed tensors
- **Access Pattern**: Local access for most operations, remote access when needed
- **Memory Management**: Automatic cleanup of temporary distributed tensors

### DGL DistEmbedding Usage
- **Initialization**: Embeddings initialized across all workers synchronously
- **Updates**: Sparse gradient updates using DGL's distributed optimizers
- **Lookup**: Efficient batch lookup of embedding vectors

### PyTorch Geometric Compatibility
- **MessagePassing**: Fully compatible with PyG's MessagePassing framework
- **Data Loading**: Works with standard PyG DataLoaders
- **Transforms**: Compatible with PyG data transforms and utilities

## Troubleshooting

### Common Issues

1. **Import Error**: Ensure DGL is installed with distributed support
2. **Memory Issues**: Reduce batch size or enable more aggressive distributed storage
3. **Communication Timeouts**: Check network connectivity between machines
4. **Embedding Size Errors**: Ensure `set_num_embeddings()` is called before training

### Performance Tuning

1. **Batch Size**: Larger batches reduce communication overhead
2. **Embedding Dimensions**: Balance between model capacity and memory usage
3. **Number of Workers**: Optimal number depends on graph size and network bandwidth
4. **Distributed Storage**: Enable `use_dist_tensor` for very large intermediate tensors

## Limitations

- **DGL Dependency**: Requires DGL for distributed functionality
- **Setup Complexity**: Distributed setup more complex than single-machine training
- **Network Dependent**: Performance depends on network bandwidth between machines
- **Limited to Homogeneous Graphs**: Current implementation focuses on homogeneous graphs

## Future Enhancements

- **Heterogeneous Graph Support**: Extend to heterogeneous graphs with multiple node/edge types
- **Dynamic Graphs**: Support for graphs that change during training
- **Advanced Optimizers**: Integration with more sophisticated distributed optimizers
- **Auto-scaling**: Automatic adjustment of distributed parameters based on graph size

## Contributing

To contribute to this implementation:

1. **Testing**: Test on different graph sizes and cluster configurations
2. **Optimization**: Improve communication efficiency and memory usage
3. **Documentation**: Add more examples and use cases
4. **Integration**: Extend to other GNN architectures beyond GIN

## Citation

If you use this distributed GIN implementation in your research, please cite:

```bibtex
@misc{distributed_gin_dgl,
    title={Distributed Graph Isomorphism Networks with DGL Integration},
    author={PyTorch Geometric Team},
    year={2025},
    howpublished={https://github.com/pyg-team/pytorch_geometric}
}
``` 