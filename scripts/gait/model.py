import torch
import torch.nn as nn
import numpy as np
from typing import Any, Dict, Tuple, Union, Optional, Sequence

## Gait Controller network
class GaitModel(nn.Module):
    """
    Gait Controller network of PRELUDE
    """

    def __init__(
        self,
        sizeState: int,
        sizeAction: int = 0,
        lenHistory: int = 0,
        sizeEmbedding: int = 32,
        sizeFeature: int = 8,
        activation: Optional[nn.Module] = nn.ReLU,
        device: Union[str, int, torch.device] = "cpu",
    ) -> None:
        super().__init__()
        """
        Generate a Gait Controller network.

        Args:
            sizeState: size of states: recent velocity command + recent robot states
            sizeAction: size of joint-space actions
            lenHistory: size of history buffer
            sizeEmbedding: size of embedding input at History Encoder 
            sizeFeature: size of History Encoder output
            activation: activation function type at Policy MLP
            device: computing device

        Returns:
            None
        """

        self.device = device

        self.sizeState = sizeState
        self.sizeAction = sizeAction

        self.output_dim = 256

        self.modelEncoder = HistoryEncoder(numChannelInput = sizeState + sizeAction,
                                              numChannelEmbedding = sizeEmbedding,
                                              lenHistoryInput = lenHistory,
                                              sizeOutput = sizeFeature,
                                              device=device)

        self.modelPolicy = MLP(sizeFeature + sizeState + sizeAction, hidden_sizes=(256, 256), activation=activation, device=device)


    def forward(self, s: Union[np.ndarray, torch.Tensor], state: Any = None, info: Dict[str, Any] = {}, ) -> Tuple[torch.Tensor, Any]:
        """
        Run Gait Controller.

        Args:
            s: observation
            state: not used
            info: not used

        Returns:
            action: output joint-space action
            state: not used
        """

        s = torch.as_tensor(s, device=self.device, dtype=torch.float32)
        o1 = s.narrow(1, 0, self.sizeState + self.sizeAction)
        s = s.view(-1, self.modelEncoder.lenHistoryInput, self.modelEncoder.numChannelInput)
        o2 = self.modelEncoder(s)
        o = torch.cat((o1, o2), dim=1)

        action = self.modelPolicy(o)

        return action, state


class HistoryEncoder(nn.Module):
    """
    History Encoder at Gait Controller
    """

    def __init__(self,  numChannelInput: int = 62,
                        numChannelEmbedding: int = 32,
                        lenHistoryInput: int = 50,
                        sizeOutput: int = 8, 
                        device: Union[str, int, torch.device] = "cpu",) -> None:
        super().__init__()
        """
        Generate a History Encoder for Gait Controller.

        Args:
            numChannelInput: size of states at each time step: recent velocity command + recent robot states + previous joint-space action
            numChannelEmbedding: size of embedding input at History Encoder
            lenHistory: size of history buffer
            sizeOutput: size of History Encoder output
            device: computing device

        Returns:
            None
        """

        self.device = device

        self.numChannelInput = numChannelInput
        self.numChannelEmbedding = numChannelEmbedding
        self.lenHistoryInput = lenHistoryInput
        self.sizeOutput = sizeOutput

        sizeProj = self.lenHistoryInput
        raConv = []
        for input, output, kernel, stride in ((self.numChannelEmbedding, self.numChannelEmbedding, 4, 2),
                                              (self.numChannelEmbedding, self.numChannelEmbedding, 3, 1),
                                              (self.numChannelEmbedding, self.numChannelEmbedding, 3, 1)):

            sizeProj = int(np.floor((sizeProj-kernel)/stride + 1))
            raConv += [nn.Conv1d(self.numChannelEmbedding, self.numChannelEmbedding, kernel_size=kernel, stride=stride)]
            raConv += [nn.ReLU(inplace=True)]

        self.embd = nn.Sequential(nn.Linear(self.numChannelInput, self.numChannelEmbedding),  nn.ReLU(inplace=True))
        self.conv = nn.Sequential(*raConv)
        self.proj = nn.Linear(sizeProj*self.numChannelEmbedding, self.sizeOutput)


    def forward(self, x: Union[np.ndarray, torch.Tensor]) -> torch.Tensor:
        """
        Run History Encoder.

        Args:
            x: observation: recent velocity command + recent robot states + previous joint-space action
            device: computing device

        Returns:
            out: History Encoder output
        """

        x = torch.as_tensor(x, device=self.device, dtype=torch.float32)
        h1 = self.embd(x)
        h2 = self.conv(h1.transpose(1,2))
        out = self.proj(h2.flatten(1))

        return out


class MLP(nn.Module):
    """
    Policy MLP at Gait Controller
    """

    def __init__(
        self,
        input_dim: int,
        output_dim: int = 0,
        hidden_sizes: Sequence[int] = (),
        activation: Optional[Union[nn.Module, Sequence[nn.Module]]] = nn.ReLU,
        device: Optional[Union[str, int, torch.device]] = None,
    ) -> None:
        super().__init__()
        """
        Generate Policy MLP at Gait Controller.

        Args:
            input_dim: size of input
            output_dim: size of output
            hidden_sizes: hidden layer sizes
            activation: activation function type
            device: computing device

        Returns:
            None
        """

        self.device = device

        if activation:
            if isinstance(activation, list):
                assert len(activation) == len(hidden_sizes)
                activation_list = activation
            else:
                activation_list = [
                    activation for _ in range(len(hidden_sizes))]
        else:
            activation_list = [None] * len(hidden_sizes)

        hidden_sizes = [input_dim] + list(hidden_sizes)

        model = []
        for in_dim, out_dim, activ in zip(
                hidden_sizes[:-1], hidden_sizes[1:], activation_list):
            model += [nn.Linear(in_dim, out_dim)]

            if activ is not None:
                model += [activation()]
        if output_dim > 0:
            model += [nn.Linear(hidden_sizes[-1], output_dim)]
            
        self.output_dim = output_dim or hidden_sizes[-1]
        self.model = nn.Sequential(*model)

    def forward(self, x: Union[np.ndarray, torch.Tensor]) -> torch.Tensor:
        """
        Run Policy MLP.

        Args:
            x: observation: output of History Encoder
            device: computing device

        Returns:
            out: Policy MLP output
        """

        x = torch.as_tensor(x, device=self.device, dtype=torch.float32)
        return self.model(x.flatten(1))