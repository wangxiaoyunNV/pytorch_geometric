"""
Distributed GIN implementation integrating DGL's DistTensor and DistEmbedding 
with PyTorch Geometric's GIN architecture.

This module provides distributed versions of GINConv and GINEConv that leverage
DGL's distributed tensor and embedding capabilities for large-scale graph processing.
"""

from typing import Callable, Optional, Union
import warnings

import torch
from torch import Tensor

from torch_geometric.nn.conv import MessagePassing
from torch_geometric.nn.dense.linear import Linear
from torch_geometric.nn.inits import reset
from torch_geometric.typing import (
    Adj,
    OptPairTensor,
    OptTensor,
    Size,
    SparseTensor,
)
from torch_geometric.utils import spmm

try:
    import dgl.distributed as dgl_dist
    from dgl.distributed import DistTensor, DistEmbedding
    DGL_AVAILABLE = True
except ImportError:
    DGL_AVAILABLE = False
    warnings.warn(
        "DGL is not available. DistributedGINConv requires DGL for distributed functionality."
    )


class DistributedGINConv(MessagePassing):
    r"""A distributed version of the Graph Isomorphism Network (GIN) operator
    that integrates DGL's DistTensor and DistEmbedding for large-scale distributed
    graph neural network training.

    This implementation extends the original GIN operator from 
    `"How Powerful are Graph Neural Networks?" <https://arxiv.org/abs/1810.00826>`_ 
    with distributed capabilities:

    .. math::
        \mathbf{x}^{\prime}_i = h_{\mathbf{\Theta}} \left( (1 + \epsilon) \cdot
        \mathbf{x}_i + \sum_{j \in \mathcal{N}(i)} \mathbf{x}_j \right)

    Args:
        nn (torch.nn.Module): A neural network :math:`h_{\mathbf{\Theta}}` that
            maps node features of shape :obj:`[-1, in_channels]` to
            shape :obj:`[-1, out_channels]`.
        eps (float, optional): (Initial) :math:`\epsilon`-value. (default: :obj:`0.`)
        train_eps (bool, optional): If set to :obj:`True`, :math:`\epsilon`
            will be a trainable parameter. (default: :obj:`False`)
        use_dist_embedding (bool, optional): Whether to use DGL's DistEmbedding 
            for node features. (default: :obj:`False`)
        embedding_dim (int, optional): Dimension of distributed embeddings when
            use_dist_embedding is True. (default: :obj:`None`)
        embedding_name (str, optional): Name for the distributed embedding.
            (default: :obj:`None`)
        **kwargs (optional): Additional arguments of
            :class:`torch_geometric.nn.conv.MessagePassing`.
    """

    def __init__(
        self, 
        nn: Callable, 
        eps: float = 0., 
        train_eps: bool = False,
        use_dist_embedding: bool = False,
        embedding_dim: Optional[int] = None,
        embedding_name: Optional[str] = None,
        **kwargs
    ):
        if not DGL_AVAILABLE:
            raise ImportError("DGL is required for DistributedGINConv")
            
        kwargs.setdefault('aggr', 'add')
        super().__init__(**kwargs)
        
        self.nn = nn
        self.initial_eps = eps
        self.use_dist_embedding = use_dist_embedding
        
        if train_eps:
            self.eps = torch.nn.Parameter(torch.empty(1))
        else:
            self.register_buffer('eps', torch.empty(1))
            
        # Initialize distributed embedding if requested
        self.dist_embedding = None
        if use_dist_embedding:
            if embedding_dim is None:
                raise ValueError("embedding_dim must be specified when use_dist_embedding=True")
            
            # Initialize DistEmbedding for node features
            self.dist_embedding = DistEmbedding(
                num_embeddings=0,  # Will be set based on graph size
                embedding_dim=embedding_dim,
                name=embedding_name or "gin_node_embeddings"
            )
            
        self.reset_parameters()

    def reset_parameters(self):
        super().reset_parameters()
        reset(self.nn)
        self.eps.data.fill_(self.initial_eps)

    def _get_distributed_features(self, x: Tensor, node_ids: Optional[Tensor] = None) -> Tensor:
        """Get features from distributed embedding or regular tensor."""
        if self.use_dist_embedding and self.dist_embedding is not None:
            if node_ids is None:
                raise ValueError("node_ids must be provided when using distributed embeddings")
            return self.dist_embedding(node_ids)
        return x

    def forward(
        self,
        x: Union[Tensor, OptPairTensor],
        edge_index: Adj,
        size: Size = None,
        node_ids: Optional[Tensor] = None,
        use_dist_tensor: bool = False,
    ) -> Tensor:
        """
        Forward pass of distributed GIN convolution.
        
        Args:
            x: Node features or tuple of source/target node features
            edge_index: Edge indices
            size: Size of the source and target nodes
            node_ids: Node IDs for distributed embedding lookup
            use_dist_tensor: Whether to use DistTensor for intermediate computations
        """
        
        if isinstance(x, Tensor):
            # Handle distributed embeddings
            if self.use_dist_embedding:
                x = self._get_distributed_features(x, node_ids)
            x = (x, x)
        
        # Propagate messages
        out = self.propagate(edge_index, x=x, size=size)
        
        x_r = x[1]
        if x_r is not None:
            out = out + (1 + self.eps) * x_r

        # Apply neural network
        result = self.nn(out)
        
        # Optionally store result in DistTensor for large-scale processing
        if use_dist_tensor and DGL_AVAILABLE:
            # Create a DistTensor for the output if needed
            # This allows for distributed storage of intermediate results
            result_shape = result.shape
            dist_result = DistTensor(
                shape=result_shape,
                dtype=result.dtype,
                name=f"gin_output_{id(self)}",
                persistent=False
            )
            # Store local result in distributed tensor
            dist_result[:result.shape[0]] = result
            return result  # Return local result for now
            
        return result

    def message(self, x_j: Tensor) -> Tensor:
        return x_j

    def message_and_aggregate(self, adj_t: Adj, x: OptPairTensor) -> Tensor:
        if isinstance(adj_t, SparseTensor):
            adj_t = adj_t.set_value(None, layout=None)
        return spmm(adj_t, x[0], reduce=self.aggr)

    def __repr__(self) -> str:
        return (f'{self.__class__.__name__}(nn={self.nn}, '
                f'use_dist_embedding={self.use_dist_embedding})')


class DistributedGINEConv(MessagePassing):
    r"""A distributed version of the GINE (Graph Isomorphism Network with Edge features) 
    operator that integrates DGL's DistTensor and DistEmbedding for large-scale 
    distributed graph neural network training.

    This implementation extends the modified GINConv operator from 
    `"Strategies for Pre-training Graph Neural Networks" <https://arxiv.org/abs/1905.12265>`_
    with distributed capabilities:

    .. math::
        \mathbf{x}^{\prime}_i = h_{\mathbf{\Theta}} \left( (1 + \epsilon) \cdot
        \mathbf{x}_i + \sum_{j \in \mathcal{N}(i)} \mathrm{ReLU}
        ( \mathbf{x}_j + \mathbf{e}_{j,i} ) \right)

    Args:
        nn (torch.nn.Module): A neural network :math:`h_{\mathbf{\Theta}}`.
        eps (float, optional): (Initial) :math:`\epsilon`-value. (default: :obj:`0.`)
        train_eps (bool, optional): If set to :obj:`True`, :math:`\epsilon`
            will be a trainable parameter. (default: :obj:`False`)
        edge_dim (int, optional): Edge feature dimensionality. (default: :obj:`None`)
        use_dist_embedding (bool, optional): Whether to use DGL's DistEmbedding 
            for node features. (default: :obj:`False`)
        use_dist_edge_embedding (bool, optional): Whether to use DGL's DistEmbedding 
            for edge features. (default: :obj:`False`)
        embedding_dim (int, optional): Dimension of distributed node embeddings.
        edge_embedding_dim (int, optional): Dimension of distributed edge embeddings.
        **kwargs (optional): Additional arguments of MessagePassing.
    """

    def __init__(
        self, 
        nn: torch.nn.Module, 
        eps: float = 0.,
        train_eps: bool = False, 
        edge_dim: Optional[int] = None,
        use_dist_embedding: bool = False,
        use_dist_edge_embedding: bool = False,
        embedding_dim: Optional[int] = None,
        edge_embedding_dim: Optional[int] = None,
        embedding_name: Optional[str] = None,
        edge_embedding_name: Optional[str] = None,
        **kwargs
    ):
        if not DGL_AVAILABLE:
            raise ImportError("DGL is required for DistributedGINEConv")
            
        kwargs.setdefault('aggr', 'add')
        super().__init__(**kwargs)
        
        self.nn = nn
        self.initial_eps = eps
        self.use_dist_embedding = use_dist_embedding
        self.use_dist_edge_embedding = use_dist_edge_embedding
        
        if train_eps:
            self.eps = torch.nn.Parameter(torch.empty(1))
        else:
            self.register_buffer('eps', torch.empty(1))

        # Handle edge dimension transformation
        if edge_dim is not None:
            if isinstance(self.nn, torch.nn.Sequential):
                nn_first = self.nn[0]
            else:
                nn_first = self.nn
                
            if hasattr(nn_first, 'in_features'):
                in_channels = nn_first.in_features
            elif hasattr(nn_first, 'in_channels'):
                in_channels = nn_first.in_channels
            else:
                raise ValueError("Could not infer input channels from `nn`.")
            self.lin = Linear(edge_dim, in_channels)
        else:
            self.lin = None

        # Initialize distributed embeddings
        self.dist_embedding = None
        self.dist_edge_embedding = None
        
        if use_dist_embedding:
            if embedding_dim is None:
                raise ValueError("embedding_dim must be specified when use_dist_embedding=True")
            self.dist_embedding = DistEmbedding(
                num_embeddings=0,  # Will be set based on graph size
                embedding_dim=embedding_dim,
                name=embedding_name or "gine_node_embeddings"
            )
            
        if use_dist_edge_embedding:
            if edge_embedding_dim is None:
                raise ValueError("edge_embedding_dim must be specified when use_dist_edge_embedding=True")
            self.dist_edge_embedding = DistEmbedding(
                num_embeddings=0,  # Will be set based on number of edges
                embedding_dim=edge_embedding_dim,
                name=edge_embedding_name or "gine_edge_embeddings"
            )

        self.reset_parameters()

    def reset_parameters(self):
        reset(self.nn)
        self.eps.data.fill_(self.initial_eps)
        if self.lin is not None:
            self.lin.reset_parameters()

    def _get_distributed_node_features(self, x: Tensor, node_ids: Optional[Tensor] = None) -> Tensor:
        """Get node features from distributed embedding or regular tensor."""
        if self.use_dist_embedding and self.dist_embedding is not None:
            if node_ids is None:
                raise ValueError("node_ids must be provided when using distributed node embeddings")
            return self.dist_embedding(node_ids)
        return x

    def _get_distributed_edge_features(self, edge_attr: Tensor, edge_ids: Optional[Tensor] = None) -> Tensor:
        """Get edge features from distributed embedding or regular tensor."""
        if self.use_dist_edge_embedding and self.dist_edge_embedding is not None:
            if edge_ids is None:
                raise ValueError("edge_ids must be provided when using distributed edge embeddings")
            return self.dist_edge_embedding(edge_ids)
        return edge_attr

    def forward(
        self,
        x: Union[Tensor, OptPairTensor],
        edge_index: Adj,
        edge_attr: OptTensor = None,
        size: Size = None,
        node_ids: Optional[Tensor] = None,
        edge_ids: Optional[Tensor] = None,
        use_dist_tensor: bool = False,
    ) -> Tensor:
        """
        Forward pass of distributed GINE convolution.
        
        Args:
            x: Node features or tuple of source/target node features
            edge_index: Edge indices
            edge_attr: Edge features
            size: Size of the source and target nodes
            node_ids: Node IDs for distributed embedding lookup
            edge_ids: Edge IDs for distributed edge embedding lookup
            use_dist_tensor: Whether to use DistTensor for intermediate computations
        """

        if isinstance(x, Tensor):
            # Handle distributed node embeddings
            if self.use_dist_embedding:
                x = self._get_distributed_node_features(x, node_ids)
            x = (x, x)

        # Handle distributed edge embeddings
        if edge_attr is not None and self.use_dist_edge_embedding:
            edge_attr = self._get_distributed_edge_features(edge_attr, edge_ids)

        # Propagate messages
        out = self.propagate(edge_index, x=x, edge_attr=edge_attr, size=size)

        x_r = x[1]
        if x_r is not None:
            out = out + (1 + self.eps) * x_r

        result = self.nn(out)
        
        # Optionally store result in DistTensor
        if use_dist_tensor and DGL_AVAILABLE:
            result_shape = result.shape
            dist_result = DistTensor(
                shape=result_shape,
                dtype=result.dtype,
                name=f"gine_output_{id(self)}",
                persistent=False
            )
            dist_result[:result.shape[0]] = result
            return result
            
        return result

    def message(self, x_j: Tensor, edge_attr: Tensor) -> Tensor:
        if self.lin is None and x_j.size(-1) != edge_attr.size(-1):
            raise ValueError("Node and edge feature dimensionalities do not "
                           "match. Consider setting the 'edge_dim' "
                           "attribute of 'DistributedGINEConv'")

        if self.lin is not None:
            edge_attr = self.lin(edge_attr)

        return (x_j + edge_attr).relu()

    def __repr__(self) -> str:
        return (f'{self.__class__.__name__}(nn={self.nn}, '
                f'use_dist_embedding={self.use_dist_embedding}, '
                f'use_dist_edge_embedding={self.use_dist_edge_embedding})') 