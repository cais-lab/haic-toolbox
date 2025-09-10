from contextlib import contextmanager

import torch
import torch.nn as nn
import torch.nn.functional as F


class MLP(torch.nn.Module):

    def __init__(self, layers: list[int],
                       activation: str = 'relu'):
        super().__init__()
        net = []
        if activation == 'relu':
            activation = torch.nn.ReLU
        elif activation == 'tanh':
            activation = torch.nn.Tanh
        else:
            raise BadValue('Activation must be either "relu" or "tanh"')
        for idx, (i, o) in enumerate(zip(layers[:-1], layers[1:])):
            net.append(torch.nn.Linear(i, o))
            net.append(activation())
        net.pop()
        self.net = torch.nn.Sequential(*net)
        self.n_outputs = layers[-1]
    
    def forward(self, x: torch.tensor) -> torch.tensor:
        return self.net(x)

# Connectors are nothing special, but modules intended to 
# receive input from the inner layers of the network and
# feed the result to the interpretation network.
# Types of connectors are adapted to the output of the 
# layers they are connected to (e.g., 2D images require
# 2d connectors, etc.)

class GlobalAvgPool2dConnector(nn.Module):
    """Connector transforming 2d tensor by GlobalAvgPooling it."""

    def __init__(self, in_channels, add_conv=False):
        super().__init__()
        layers = []
        if add_conv:
            layers.append(nn.Conv2d(in_channels, in_channels, 1))
        # Global average pooling
        layers.append(nn.AdaptiveAvgPool2d(1))
        # Flatten 2d structure
        layers.append(nn.Flatten(start_dim=-3))
        self.net = nn.Sequential(*layers)
        
    def forward(self, x):
        return self.net(x)

class InterceptorNetwork(nn.Module):
    """Fully-connected interpretation network (MLP)."""
    
    def __init__(self, backbone: nn.Module,
                       connectors: list[tuple[nn.Module, nn.Module]],
                       return_connectors_only: bool=False):
        """Constructs a fully-connected interpretation network.
        
        Parameters
        ----------
        backbone
            Backbone model. Typically, a deep one.
        connectors : list
            A list of pairs, describing what layers of the
            backbone model should be streamed to the interpretation
            network.
        return_connectors_only: bool
            Controls whether forward should return tuple (backbone and
            connectors) or just connectors.
        """
        super().__init__()
        
        assert len(connectors) > 0

        self.backbone = backbone
        self.return_connectors_only = return_connectors_only

        # Make connectors part of this model's state
        self.connectors = nn.ModuleList(
            m for (l, m) in connectors
        )
        # Backbone's modules, to which hooks must be attached
        self.connection_points = [
            l for (l, m) in connectors
        ]

    @contextmanager
    def _multiple_hooks(self, connection_points):

        def hook(module, input, output):
            # Look for the index of the module
            # in the connectors
            for i, m in enumerate(connection_points):
                if module == m:
                    self._glue[i] = output

        hook_handles = [
            module.register_forward_hook(hook)
            for module in connection_points
        ]
        self._reset_glue()
        
        try:
            yield
        finally:
            for h in hook_handles:
                h.remove()
        
    def _reset_glue(self):
        self._glue = [None] * len(self.connectors)
        
    def forward(self, x):
        with self._multiple_hooks(self.connection_points):
            backbone_output = self.backbone(x)
        # Collect all tensors obtained via connections
        # (applying to them respective transformations)
        x = [f(t) for (f, t) in zip(self.connectors, self._glue)]
        x = torch.cat(x, -1)
        if self.return_connectors_only:
            return x
        else:
            return backbone_output, x
