"""Policy represents practical decisions about from
which source to obtain the prediction for each
item. The heart of the policy is simple array,
but it carries additional information about the
number of classes and indices which correspond
to humans."""

from typing import Union

import torch

class Policy:

    def __init__(self, data: torch.tensor,
                 n_executors: int,
                 human_executors: [int]):
        assert len(data.shape) == 1 or (len(data.shape) == 2 and data.shape[1] == 1)
        assert not torch.is_floating_point(data) and not torch.is_complex(data)
        self.data = data
        self.n_executors = n_executors
        self.human_executors = human_executors
    
def as_policy(data: Union[Policy, torch.tensor],
              n_executors: int = None,
              human_executors: [int] = None) -> Policy:
    if isinstance(data, Policy):
        return data
    elif isinstance(data, torch.tensor):
        assert torch.is_floating_point(data) and not torch.is_complex(data)
        n_executors = n_executors or (data.max() - 1)
        human_executors = human_executors or [n_executors - 1]
        return Policy(data, n_executors, human_executors)
    else:
        raise ValueError(f'Only Policy and torch.tensor are supported, got {type(data)}')
