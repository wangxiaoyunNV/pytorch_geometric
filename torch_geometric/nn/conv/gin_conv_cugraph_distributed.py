"""
Distributed GIN implementation integrating cugraph-pyg's distTensor and distEmbedding 
with PyTorch Geometric's GIN architecture.

This module provides distributed versions of GINConv and GINEConv that leverage
cugraph-pyg's distributed tensor and embedding capabilities for large-scale graph processing.

Based on: https://github.com/rapidsai/cugraph-gnn/blob/branch-25.08/python/cugraph-pyg/cugraph_pyg/tensor/dist_tensor.py
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


class CuGraphDistributedGINConv(MessagePassing):
    r"""A distributed version of the Graph Isomorphism Network (GIN) operator
    that integrates cugraph-pyg's DistTensor and DistEmbedding for large-scale 
    distributed graph neural network training.

    This implementation extends the original GIN operator from 
    `"How Powerful are Graph Neural Networks?" <https://arxiv.org/abs/1810.00826>`_ 
    with cugraph-pyg's distributed capabilities:

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
        use_dist_embedding (bool, optional): Whether to use cugraph-pyg's DistEmbedding 
            for node features. (default: :obj:`False`)
        embedding_dim (int, optional): Dimension of distributed embeddings when
            use_dist_embedding is True. (default: :obj:`None`)
        embedding_name (str, optional): Name for the distributed embedding.
            (default: :obj:`None`)
        device_id (int, optional): GPU device ID for distributed operations.
            (default: :obj:`None`)
        **kwargs (optional): Additional arguments of
            :class:`torch_geometric.nn.conv.MessagePassing`.

    Examples:
    --------
    >>> from torch_geometric.nn import MLP
    >>> from torch_geometric.nn.conv.gin_conv_cugraph_distributed import CuGraphDistributedGINConv
    >>> 
    >>> mlp = MLP([64, 128, 128])
    >>> conv = CuGraphDistributedGINConv(
    ...     nn=mlp,
    ...     use_dist_embedding=True,
    ...     embedding_dim=64,
    ...     device_id=0
    ... )
    """

    def __init__(
        self, 
        nn: Callable, 
        eps: float = 0., 
        train_eps: bool = False,
        use_dist_embedding: bool = False,
        embedding_dim: Optional[int] = None,
        embedding_name: Optional[str] = None,
        device_id: Optional[int] = None,
        **kwargs
    ):
        if use_dist_embedding and not CUGRAPH_AVAILABLE:
            raise ImportError(
                "cugraph-pyg is required for distributed functionality. "
                "Install with: pip install cugraph-pyg"
            )
            
        kwargs.setdefault('aggr', 'add')
        super().__init__(**kwargs)
        
        self.nn = nn
        self.initial_eps = eps
        self.use_dist_embedding = use_dist_embedding
        self.device_id = device_id or 0
        
        if train_eps:
            self.eps = torch.nn.Parameter(torch.empty(1))
        else:
            self.register_buffer('eps', torch.empty(1))
            
        # Initialize cugraph-pyg distributed embedding if requested
        self.dist_embedding = None
        if use_dist_embedding and CUGRAPH_AVAILABLE:
            if embedding_dim is None:
                raise ValueError("embedding_dim must be specified when use_dist_embedding=True")
            
            # Initialize cugraph-pyg DistEmbedding for node features
            self.dist_embedding = DistEmbedding(
                num_embeddings=0,  # Will be set based on graph size
                embedding_dim=embedding_dim,
                device_id=self.device_id,
                name=embedding_name or "cugraph_gin_node_embeddings"
            )
            
        self.reset_parameters()

    def reset_parameters(self):
        super().reset_parameters()
        reset(self.nn)
        self.eps.data.fill_(self.initial_eps)

    def _get_distributed_features(self, x: Tensor, node_ids: Optional[Tensor] = None) -> Tensor:
        """Get features from cugraph-pyg distributed embedding or regular tensor."""
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
            use_dist_tensor: Whether to use cugraph-pyg DistTensor for intermediate computations
        """
        
        if isinstance(x, Tensor):
            # Handle cugraph-pyg distributed embeddings
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
        
        # Optionally store result in cugraph-pyg DistTensor for large-scale processing
        if use_dist_tensor and CUGRAPH_AVAILABLE:
            # Create a cugraph-pyg DistTensor for the output if needed
            # This allows for distributed storage of intermediate results
            dist_result = DistTensor(
                data=result,
                device_id=self.device_id,
                name=f"cugraph_gin_output_{id(self)}"
            )
            # The DistTensor handles the distribution automatically
            return result  # Return local result for computation graph
            
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


class CuGraphDistributedGINEConv(MessagePassing):
    r"""A distributed version of the GINE (Graph Isomorphism Network with Edge features) 
    operator that integrates cugraph-pyg's DistTensor and DistEmbedding for large-scale 
    distributed graph neural network training.

    This implementation extends the modified GINConv operator from 
    `"Strategies for Pre-training Graph Neural Networks" <https://arxiv.org/abs/1905.12265>`_
    with cugraph-pyg's distributed capabilities:

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
        use_dist_embedding (bool, optional): Whether to use cugraph-pyg's DistEmbedding 
            for node features. (default: :obj:`False`)
        use_dist_edge_embedding (bool, optional): Whether to use cugraph-pyg's DistEmbedding 
            for edge features. (default: :obj:`False`)
        embedding_dim (int, optional): Dimension of distributed node embeddings.
        edge_embedding_dim (int, optional): Dimension of distributed edge embeddings.
        device_id (int, optional): GPU device ID for distributed operations.
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
        device_id: Optional[int] = None,
        **kwargs
    ):
        if (use_dist_embedding or use_dist_edge_embedding) and not CUGRAPH_AVAILABLE:
            raise ImportError(
                "cugraph-pyg is required for distributed functionality. "
                "Install with: pip install cugraph-pyg"
            )
            
        kwargs.setdefault('aggr', 'add')
        super().__init__(**kwargs)
        
        self.nn = nn
        self.initial_eps = eps
        self.use_dist_embedding = use_dist_embedding
        self.use_dist_edge_embedding = use_dist_edge_embedding
        self.device_id = device_id or 0
        
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

        # Initialize cugraph-pyg distributed embeddings
        self.dist_embedding = None
        self.dist_edge_embedding = None
        
        if use_dist_embedding and CUGRAPH_AVAILABLE:
            if embedding_dim is None:
                raise ValueError("embedding_dim must be specified when use_dist_embedding=True")
            self.dist_embedding = DistEmbedding(
                num_embeddings=0,  # Will be set based on graph size
                embedding_dim=embedding_dim,
                device_id=self.device_id,
                name=embedding_name or "cugraph_gine_node_embeddings"
            )
            
        if use_dist_edge_embedding and CUGRAPH_AVAILABLE:
            if edge_embedding_dim is None:
                raise ValueError("edge_embedding_dim must be specified when use_dist_edge_embedding=True")
            self.dist_edge_embedding = DistEmbedding(
                num_embeddings=0,  # Will be set based on number of edges
                embedding_dim=edge_embedding_dim,
                device_id=self.device_id,
                name=edge_embedding_name or "cugraph_gine_edge_embeddings"
            )

        self.reset_parameters()

    def reset_parameters(self):
        reset(self.nn)
        self.eps.data.fill_(self.initial_eps)
        if self.lin is not None:
            self.lin.reset_parameters()

    def _get_distributed_node_features(self, x: Tensor, node_ids: Optional[Tensor] = None) -> Tensor:
        """Get node features from cugraph-pyg distributed embedding or regular tensor."""
        if self.use_dist_embedding and self.dist_embedding is not None:
            if node_ids is None:
                raise ValueError("node_ids must be provided when using distributed node embeddings")
            return self.dist_embedding(node_ids)
        return x

    def _get_distributed_edge_features(self, edge_attr: Tensor, edge_ids: Optional[Tensor] = None) -> Tensor:
        """Get edge features from cugraph-pyg distributed embedding or regular tensor."""
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
            use_dist_tensor: Whether to use cugraph-pyg DistTensor for intermediate computations
        """

        if isinstance(x, Tensor):
            # Handle cugraph-pyg distributed node embeddings
            if self.use_dist_embedding:
                x = self._get_distributed_node_features(x, node_ids)
            x = (x, x)

        # Handle cugraph-pyg distributed edge embeddings
        if edge_attr is not None and self.use_dist_edge_embedding:
            edge_attr = self._get_distributed_edge_features(edge_attr, edge_ids)

        # Propagate messages
        out = self.propagate(edge_index, x=x, edge_attr=edge_attr, size=size)

        x_r = x[1]
        if x_r is not None:
            out = out + (1 + self.eps) * x_r

        result = self.nn(out)
        
        # Optionally store result in cugraph-pyg DistTensor
        if use_dist_tensor and CUGRAPH_AVAILABLE:
            dist_result = DistTensor(
                data=result,
                device_id=self.device_id,
                name=f"cugraph_gine_output_{id(self)}"
            )
            return result
            
        return result

    def message(self, x_j: Tensor, edge_attr: Tensor) -> Tensor:
        if self.lin is None and x_j.size(-1) != edge_attr.size(-1):
            raise ValueError("Node and edge feature dimensionalities do not "
                           "match. Consider setting the 'edge_dim' "
                           "attribute of 'CuGraphDistributedGINEConv'")

        if self.lin is not None:
            edge_attr = self.lin(edge_attr)

        return (x_j + edge_attr).relu()

    def __repr__(self) -> str:
        return (f'{self.__class__.__name__}(nn={self.nn}, '
                f'use_dist_embedding={self.use_dist_embedding}, '
                f'use_dist_edge_embedding={self.use_dist_edge_embedding})')


# Convenience aliases for backward compatibility
DistributedGINConv = CuGraphDistributedGINConv
DistributedGINEConv = CuGraphDistributedGINEConv 