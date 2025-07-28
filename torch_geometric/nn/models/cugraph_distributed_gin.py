"""
Distributed GIN model implementation that integrates cugraph-pyg's DistTensor and DistEmbedding
with PyTorch Geometric's GIN architecture for large-scale graph processing.

Based on: https://github.com/rapidsai/cugraph-gnn/blob/branch-25.08/python/cugraph-pyg/cugraph_pyg/tensor/dist_tensor.py
"""

from typing import Optional, Union, List
import warnings

import torch
import torch.nn.functional as F
from torch import Tensor

from torch_geometric.nn import MLP, global_add_pool
from torch_geometric.nn.conv.gin_conv_cugraph_distributed import (
    CuGraphDistributedGINConv, CuGraphDistributedGINEConv
)

try:
    # Import cugraph-pyg's distributed tensor components
    from cugraph_pyg.tensor import DistTensor, DistEmbedding
    CUGRAPH_AVAILABLE = True
except ImportError:
    try:
        # Fallback for older cugraph-pyg versions or different module structure
        from cugraph_pyg.data import DistTensor, DistEmbedding
        CUGRAPH_AVAILABLE = True
    except ImportError:
        CUGRAPH_AVAILABLE = False
        warnings.warn(
            "cugraph-pyg distributed components are not available. "
            "Install cugraph-pyg with distributed support for full functionality."
        )


class CuGraphDistributedGIN(torch.nn.Module):
    r"""A distributed Graph Isomorphism Network (GIN) model that leverages
    cugraph-pyg's DistTensor and DistEmbedding for large-scale distributed training.

    This implementation extends the standard GIN architecture with cugraph-pyg's
    distributed capabilities, enabling training on graphs with billions of nodes 
    and edges across multiple GPUs and machines.

    Args:
        in_channels (int): Size of each input sample.
        hidden_channels (int): Size of each hidden sample.
        out_channels (int): Size of each output sample.
        num_layers (int): Number of message passing layers.
        dropout (float, optional): Dropout probability. (default: :obj:`0.`)
        act (str, optional): The non-linear activation function to use.
            (default: :obj:`"relu"`)
        use_dist_embedding (bool, optional): Whether to use cugraph-pyg's DistEmbedding
            for node features. (default: :obj:`False`)
        embedding_dim (int, optional): Dimension of distributed embeddings.
            Required if use_dist_embedding is True. (default: :obj:`None`)
        embedding_name (str, optional): Name for the distributed embedding.
            (default: :obj:`None`)
        use_dist_tensor (bool, optional): Whether to use cugraph-pyg's DistTensor for
            intermediate computations. (default: :obj:`False`)
        device_id (int, optional): GPU device ID for distributed operations.
            (default: :obj:`None`)
        train_eps (bool, optional): If set to :obj:`True`, :math:`\epsilon`
            will be trainable parameters. (default: :obj:`False`)
        eps (float, optional): Initial :math:`\epsilon` value. (default: :obj:`0.`)
        **kwargs (optional): Additional arguments.

    Examples:
    ---------
    >>> # Standard usage with cugraph-pyg distributed features
    >>> model = CuGraphDistributedGIN(
    ...     in_channels=64, 
    ...     hidden_channels=128, 
    ...     out_channels=32,
    ...     num_layers=3,
    ...     use_dist_embedding=True,
    ...     embedding_dim=64,
    ...     device_id=0
    ... )
    >>> 
    >>> # Usage with distributed tensors for large-scale processing
    >>> model = CuGraphDistributedGIN(
    ...     in_channels=64, 
    ...     hidden_channels=128, 
    ...     out_channels=32,
    ...     num_layers=3,
    ...     use_dist_embedding=True,
    ...     use_dist_tensor=True,
    ...     embedding_dim=64,
    ...     device_id=0
    ... )
    """

    def __init__(
        self, 
        in_channels: int, 
        hidden_channels: int, 
        out_channels: int, 
        num_layers: int,
        dropout: float = 0.,
        act: str = "relu",
        use_dist_embedding: bool = False,
        embedding_dim: Optional[int] = None,
        embedding_name: Optional[str] = None,
        use_dist_tensor: bool = False,
        device_id: Optional[int] = None,
        train_eps: bool = False,
        eps: float = 0.,
        **kwargs
    ):
        super().__init__()

        if (use_dist_embedding or use_dist_tensor) and not CUGRAPH_AVAILABLE:
            raise ImportError(
                "cugraph-pyg is required for distributed functionality. "
                "Install with: pip install cugraph-pyg"
            )

        self.in_channels = in_channels
        self.hidden_channels = hidden_channels
        self.out_channels = out_channels
        self.num_layers = num_layers
        self.dropout = dropout
        self.use_dist_embedding = use_dist_embedding
        self.use_dist_tensor = use_dist_tensor
        self.device_id = device_id or 0

        # Activation function
        if act == "relu":
            self.act = F.relu
        elif act == "gelu":
            self.act = F.gelu
        elif act == "tanh":
            self.act = torch.tanh
        else:
            raise ValueError(f"Unsupported activation function: {act}")

        # Build GIN layers using cugraph-pyg distributed convolutions
        self.convs = torch.nn.ModuleList()
        
        for i in range(num_layers):
            if i == 0:
                input_dim = embedding_dim if use_dist_embedding else in_channels
            else:
                input_dim = hidden_channels
                
            # Create MLP for this layer
            mlp = MLP([input_dim, hidden_channels, hidden_channels], 
                     dropout=dropout, act=act)
            
            # Create cugraph-pyg distributed GIN convolution
            conv = CuGraphDistributedGINConv(
                nn=mlp,
                eps=eps,
                train_eps=train_eps,
                use_dist_embedding=use_dist_embedding and i == 0,  # Only first layer uses embedding
                embedding_dim=embedding_dim if i == 0 else None,
                embedding_name=f"{embedding_name}_layer_{i}" if embedding_name and i == 0 else None,
                device_id=self.device_id,
                **kwargs
            )
            self.convs.append(conv)

        # Final MLP for classification/regression
        self.classifier = MLP(
            [hidden_channels, hidden_channels, out_channels],
            norm=None, 
            dropout=dropout,
            act=act
        )

        # Initialize distributed embedding for input features if needed
        self.input_embedding = None
        if use_dist_embedding and CUGRAPH_AVAILABLE:
            if embedding_dim is None:
                raise ValueError("embedding_dim must be specified when use_dist_embedding=True")
            
            self.input_embedding = DistEmbedding(
                num_embeddings=0,  # Will be set based on graph size
                embedding_dim=embedding_dim,
                device_id=self.device_id,
                name=embedding_name or "cugraph_gin_input_embeddings"
            )

    def set_num_embeddings(self, num_nodes: int):
        """Set the number of embeddings for distributed embeddings.
        
        This should be called after the graph size is known but before training.
        
        Args:
            num_nodes (int): Number of nodes in the graph.
        """
        if self.use_dist_embedding and self.input_embedding is not None:
            # Update the input embedding with the correct size
            self.input_embedding.resize_(num_nodes)
            
        # Also update the first convolution layer if it uses embeddings
        if len(self.convs) > 0 and hasattr(self.convs[0], 'dist_embedding'):
            if self.convs[0].dist_embedding is not None:
                self.convs[0].dist_embedding.resize_(num_nodes)

    def forward(
        self, 
        x: Tensor, 
        edge_index: Tensor, 
        batch: Optional[Tensor] = None,
        node_ids: Optional[Tensor] = None
    ) -> Tensor:
        """
        Forward pass of the distributed GIN model.
        
        Args:
            x: Node feature matrix or node IDs (if using distributed embeddings)
            edge_index: Edge indices
            batch: Batch vector for graph-level prediction
            node_ids: Node IDs for distributed embedding lookup
            
        Returns:
            Output tensor
        """
        
        # Handle input features with cugraph-pyg distributed embeddings
        if self.use_dist_embedding and self.input_embedding is not None:
            if node_ids is None:
                # Assume x contains node IDs
                node_ids = x.long()
            x = self.input_embedding(node_ids)
        
        # Apply GIN layers with cugraph-pyg distributed convolutions
        for i, conv in enumerate(self.convs):
            x = conv(
                x, 
                edge_index, 
                node_ids=node_ids if i == 0 else None,  # Only first layer needs node_ids
                use_dist_tensor=self.use_dist_tensor
            )
            x = self.act(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        
        # Graph-level pooling (for graph classification)
        if batch is not None:
            x = global_add_pool(x, batch)
        
        # Apply final classifier
        return self.classifier(x)

    def __repr__(self) -> str:
        return (f'{self.__class__.__name__}('
                f'in_channels={self.in_channels}, '
                f'hidden_channels={self.hidden_channels}, '
                f'out_channels={self.out_channels}, '
                f'num_layers={self.num_layers}, '
                f'use_dist_embedding={self.use_dist_embedding}, '
                f'device_id={self.device_id})')


class CuGraphDistributedGINE(torch.nn.Module):
    r"""A distributed Graph Isomorphism Network with Edge features (GINE) model
    that leverages cugraph-pyg's DistTensor and DistEmbedding for large-scale 
    distributed training.

    Args:
        in_channels (int): Size of each input node feature.
        hidden_channels (int): Size of each hidden sample.
        out_channels (int): Size of each output sample.
        num_layers (int): Number of message passing layers.
        edge_dim (int, optional): Edge feature dimensionality.
        dropout (float, optional): Dropout probability. (default: :obj:`0.`)
        act (str, optional): The non-linear activation function. (default: :obj:`"relu"`)
        use_dist_embedding (bool, optional): Whether to use distributed node embeddings.
        use_dist_edge_embedding (bool, optional): Whether to use distributed edge embeddings.
        embedding_dim (int, optional): Dimension of distributed node embeddings.
        edge_embedding_dim (int, optional): Dimension of distributed edge embeddings.
        device_id (int, optional): GPU device ID for distributed operations.
        train_eps (bool, optional): If trainable epsilon. (default: :obj:`False`)
        eps (float, optional): Initial epsilon value. (default: :obj:`0.`)
    """

    def __init__(
        self, 
        in_channels: int, 
        hidden_channels: int, 
        out_channels: int, 
        num_layers: int,
        edge_dim: Optional[int] = None,
        dropout: float = 0.,
        act: str = "relu",
        use_dist_embedding: bool = False,
        use_dist_edge_embedding: bool = False,
        embedding_dim: Optional[int] = None,
        edge_embedding_dim: Optional[int] = None,
        embedding_name: Optional[str] = None,
        edge_embedding_name: Optional[str] = None,
        use_dist_tensor: bool = False,
        device_id: Optional[int] = None,
        train_eps: bool = False,
        eps: float = 0.,
        **kwargs
    ):
        super().__init__()

        if (use_dist_embedding or use_dist_edge_embedding or use_dist_tensor) and not CUGRAPH_AVAILABLE:
            raise ImportError(
                "cugraph-pyg is required for distributed functionality. "
                "Install with: pip install cugraph-pyg"
            )

        self.in_channels = in_channels
        self.hidden_channels = hidden_channels
        self.out_channels = out_channels
        self.num_layers = num_layers
        self.edge_dim = edge_dim
        self.dropout = dropout
        self.use_dist_embedding = use_dist_embedding
        self.use_dist_edge_embedding = use_dist_edge_embedding
        self.use_dist_tensor = use_dist_tensor
        self.device_id = device_id or 0

        # Activation function
        if act == "relu":
            self.act = F.relu
        elif act == "gelu":
            self.act = F.gelu
        elif act == "tanh":
            self.act = torch.tanh
        else:
            raise ValueError(f"Unsupported activation function: {act}")

        # Build GINE layers using cugraph-pyg distributed convolutions
        self.convs = torch.nn.ModuleList()
        
        for i in range(num_layers):
            if i == 0:
                input_dim = embedding_dim if use_dist_embedding else in_channels
            else:
                input_dim = hidden_channels
                
            # Create MLP for this layer
            mlp = MLP([input_dim, hidden_channels, hidden_channels], 
                     dropout=dropout, act=act)
            
            # Create cugraph-pyg distributed GINE convolution
            conv = CuGraphDistributedGINEConv(
                nn=mlp,
                eps=eps,
                train_eps=train_eps,
                edge_dim=edge_dim,
                use_dist_embedding=use_dist_embedding and i == 0,
                use_dist_edge_embedding=use_dist_edge_embedding,
                embedding_dim=embedding_dim if i == 0 else None,
                edge_embedding_dim=edge_embedding_dim,
                embedding_name=f"{embedding_name}_layer_{i}" if embedding_name and i == 0 else None,
                edge_embedding_name=edge_embedding_name,
                device_id=self.device_id,
                **kwargs
            )
            self.convs.append(conv)

        # Final classifier
        self.classifier = MLP(
            [hidden_channels, hidden_channels, out_channels],
            norm=None, 
            dropout=dropout,
            act=act
        )

    def forward(
        self, 
        x: Tensor, 
        edge_index: Tensor, 
        edge_attr: Optional[Tensor] = None,
        batch: Optional[Tensor] = None,
        node_ids: Optional[Tensor] = None,
        edge_ids: Optional[Tensor] = None
    ) -> Tensor:
        """
        Forward pass of the distributed GINE model.
        
        Args:
            x: Node feature matrix
            edge_index: Edge indices
            edge_attr: Edge feature matrix
            batch: Batch vector for graph-level prediction
            node_ids: Node IDs for distributed embedding lookup
            edge_ids: Edge IDs for distributed edge embedding lookup
            
        Returns:
            Output tensor
        """
        
        # Apply GINE layers with cugraph-pyg distributed convolutions
        for i, conv in enumerate(self.convs):
            x = conv(
                x, 
                edge_index, 
                edge_attr=edge_attr,
                node_ids=node_ids if i == 0 else None,
                edge_ids=edge_ids,
                use_dist_tensor=self.use_dist_tensor
            )
            x = self.act(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        
        # Graph-level pooling
        if batch is not None:
            x = global_add_pool(x, batch)
        
        # Apply final classifier
        return self.classifier(x)

    def __repr__(self) -> str:
        return (f'{self.__class__.__name__}('
                f'in_channels={self.in_channels}, '
                f'hidden_channels={self.hidden_channels}, '
                f'out_channels={self.out_channels}, '
                f'num_layers={self.num_layers}, '
                f'edge_dim={self.edge_dim}, '
                f'use_dist_embedding={self.use_dist_embedding}, '
                f'use_dist_edge_embedding={self.use_dist_edge_embedding}, '
                f'device_id={self.device_id})')


# Convenience aliases for backward compatibility
DistributedGIN = CuGraphDistributedGIN
DistributedGINE = CuGraphDistributedGINE 