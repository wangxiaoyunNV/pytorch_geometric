"""
Distributed GIN model implementation that integrates DGL's DistTensor and DistEmbedding
with PyTorch Geometric's GIN architecture for large-scale graph processing.
"""

from typing import Optional, Union, List
import warnings

import torch
import torch.nn.functional as F
from torch import Tensor

from torch_geometric.nn import MLP, global_add_pool
from torch_geometric.nn.conv.gin_conv_distributed import DistributedGINConv, DistributedGINEConv

try:
    import dgl.distributed as dgl_dist
    from dgl.distributed import DistTensor, DistEmbedding
    DGL_AVAILABLE = True
except ImportError:
    DGL_AVAILABLE = False
    warnings.warn(
        "DGL is not available. DistributedGIN requires DGL for distributed functionality."
    )


class DistributedGIN(torch.nn.Module):
    r"""A distributed Graph Isomorphism Network (GIN) model that leverages
    DGL's DistTensor and DistEmbedding for large-scale distributed training.

    This implementation extends the standard GIN architecture with distributed
    capabilities, enabling training on graphs with billions of nodes and edges
    across multiple machines.

    Args:
        in_channels (int): Size of each input sample.
        hidden_channels (int): Size of each hidden sample.
        out_channels (int): Size of each output sample.
        num_layers (int): Number of message passing layers.
        dropout (float, optional): Dropout probability. (default: :obj:`0.`)
        act (str, optional): The non-linear activation function to use.
            (default: :obj:`"relu"`)
        use_dist_embedding (bool, optional): Whether to use DGL's DistEmbedding
            for node features. (default: :obj:`False`)
        embedding_dim (int, optional): Dimension of distributed embeddings.
            Required if use_dist_embedding is True. (default: :obj:`None`)
        embedding_name (str, optional): Name for the distributed embedding.
            (default: :obj:`None`)
        use_dist_tensor (bool, optional): Whether to use DistTensor for
            intermediate computations. (default: :obj:`False`)
        train_eps (bool, optional): If set to :obj:`True`, :math:`\epsilon`
            will be trainable parameters. (default: :obj:`False`)
        eps (float, optional): Initial :math:`\epsilon` value. (default: :obj:`0.`)
        **kwargs (optional): Additional arguments.

    Examples:
    ---------
    >>> # Standard usage with regular tensors
    >>> model = DistributedGIN(
    ...     in_channels=64, 
    ...     hidden_channels=128, 
    ...     out_channels=32,
    ...     num_layers=3
    ... )
    >>> 
    >>> # Usage with distributed embeddings
    >>> model = DistributedGIN(
    ...     in_channels=64, 
    ...     hidden_channels=128, 
    ...     out_channels=32,
    ...     num_layers=3,
    ...     use_dist_embedding=True,
    ...     embedding_dim=64
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
        train_eps: bool = False,
        eps: float = 0.,
        **kwargs
    ):
        super().__init__()

        if not DGL_AVAILABLE:
            raise ImportError("DGL is required for DistributedGIN")

        self.in_channels = in_channels
        self.hidden_channels = hidden_channels
        self.out_channels = out_channels
        self.num_layers = num_layers
        self.dropout = dropout
        self.use_dist_embedding = use_dist_embedding
        self.use_dist_tensor = use_dist_tensor

        # Activation function
        if act == "relu":
            self.act = F.relu
        elif act == "gelu":
            self.act = F.gelu
        elif act == "tanh":
            self.act = torch.tanh
        else:
            raise ValueError(f"Unsupported activation function: {act}")

        # Build GIN layers
        self.convs = torch.nn.ModuleList()
        
        for i in range(num_layers):
            if i == 0:
                input_dim = embedding_dim if use_dist_embedding else in_channels
            else:
                input_dim = hidden_channels
                
            # Create MLP for this layer
            mlp = MLP([input_dim, hidden_channels, hidden_channels], 
                     dropout=dropout, act=act)
            
            # Create distributed GIN convolution
            conv = DistributedGINConv(
                nn=mlp,
                eps=eps,
                train_eps=train_eps,
                use_dist_embedding=use_dist_embedding and i == 0,  # Only first layer uses embedding
                embedding_dim=embedding_dim if i == 0 else None,
                embedding_name=embedding_name if i == 0 else None,
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
        if use_dist_embedding:
            if embedding_dim is None:
                raise ValueError("embedding_dim must be specified when use_dist_embedding=True")
            
            self.input_embedding = DistEmbedding(
                num_embeddings=0,  # Will be set based on graph size
                embedding_dim=embedding_dim,
                name=embedding_name or "gin_input_embeddings"
            )

    def set_num_embeddings(self, num_nodes: int):
        """Set the number of embeddings for distributed embeddings.
        
        This should be called after the graph size is known but before training.
        
        Args:
            num_nodes (int): Number of nodes in the graph.
        """
        if self.use_dist_embedding and self.input_embedding is not None:
            # Create a new embedding with the correct size
            self.input_embedding = DistEmbedding(
                num_embeddings=num_nodes,
                embedding_dim=self.input_embedding._embedding_dim,
                name=self.input_embedding._name
            )
            
        # Also update the first convolution layer if it uses embeddings
        if len(self.convs) > 0 and hasattr(self.convs[0], 'dist_embedding'):
            if self.convs[0].dist_embedding is not None:
                self.convs[0].dist_embedding = DistEmbedding(
                    num_embeddings=num_nodes,
                    embedding_dim=self.convs[0].dist_embedding._embedding_dim,
                    name=self.convs[0].dist_embedding._name
                )

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
        
        # Handle input features
        if self.use_dist_embedding and self.input_embedding is not None:
            if node_ids is None:
                # Assume x contains node IDs
                node_ids = x.long()
            x = self.input_embedding(node_ids)
        
        # Apply GIN layers
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
                f'use_dist_embedding={self.use_dist_embedding})')


class DistributedGINE(torch.nn.Module):
    r"""A distributed Graph Isomorphism Network with Edge features (GINE) model
    that leverages DGL's DistTensor and DistEmbedding for large-scale distributed training.

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
        train_eps: bool = False,
        eps: float = 0.,
        **kwargs
    ):
        super().__init__()

        if not DGL_AVAILABLE:
            raise ImportError("DGL is required for DistributedGINE")

        self.in_channels = in_channels
        self.hidden_channels = hidden_channels
        self.out_channels = out_channels
        self.num_layers = num_layers
        self.edge_dim = edge_dim
        self.dropout = dropout
        self.use_dist_embedding = use_dist_embedding
        self.use_dist_edge_embedding = use_dist_edge_embedding
        self.use_dist_tensor = use_dist_tensor

        # Activation function
        if act == "relu":
            self.act = F.relu
        elif act == "gelu":
            self.act = F.gelu
        elif act == "tanh":
            self.act = torch.tanh
        else:
            raise ValueError(f"Unsupported activation function: {act}")

        # Build GINE layers
        self.convs = torch.nn.ModuleList()
        
        for i in range(num_layers):
            if i == 0:
                input_dim = embedding_dim if use_dist_embedding else in_channels
            else:
                input_dim = hidden_channels
                
            # Create MLP for this layer
            mlp = MLP([input_dim, hidden_channels, hidden_channels], 
                     dropout=dropout, act=act)
            
            # Create distributed GINE convolution
            conv = DistributedGINEConv(
                nn=mlp,
                eps=eps,
                train_eps=train_eps,
                edge_dim=edge_dim,
                use_dist_embedding=use_dist_embedding and i == 0,
                use_dist_edge_embedding=use_dist_edge_embedding,
                embedding_dim=embedding_dim if i == 0 else None,
                edge_embedding_dim=edge_embedding_dim,
                embedding_name=embedding_name if i == 0 else None,
                edge_embedding_name=edge_embedding_name,
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
        
        # Apply GINE layers
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
                f'use_dist_edge_embedding={self.use_dist_edge_embedding})') 