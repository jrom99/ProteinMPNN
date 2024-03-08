from typing import Any
from torch import nn

def tied_featurize(*args, **kwargs) -> Any: ...
def scores(*args, **kwargs) -> Any: ...
def S_to_seq(*args, **kwargs) -> Any: ...


class CA_ProteinFeatures(nn.Module):
    pass

class ProteinFeatures(nn.Module):
    pass
