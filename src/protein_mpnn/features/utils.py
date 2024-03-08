from torch import nn
from typing import Any


def gather_edges(edges: Any, neighbor_idx: Any) -> Any: ...
def gather_nodes(nodes: Any, neighbor_idx: Any) -> Any: ...
def gather_nodes_t(nodes: Any, neighbor_idx: Any) -> Any: ...
def cat_neighbors_nodes(h_nodes: Any, h_neighbors: Any, E_idx: Any) -> Any: ...


class PositionWiseFeedForward(nn.Module):
    pass

class EncLayer(nn.Module):
    pass

class DecLauer(nn.Module):
    pass

class PositionalEncodings(nn.Module):
    pass
