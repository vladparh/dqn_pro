import numpy as np
import torch
import torch.nn as nn


class DQNAgent(nn.Module):
    def __init__(self, n_actions, epsilon=0):

        super().__init__()
        self.epsilon = epsilon
        self.n_actions = n_actions

        # Define your network body here. Please make sure agent is fully contained here
        # nn.Flatten() can be useful
        self.model = nn.Sequential(
            nn.Conv2d(in_channels=4, out_channels=32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1),
            nn.Flatten(),
            nn.ReLU(),
            nn.Linear(in_features=64*7*7 , out_features=512),
            nn.ReLU(),
            nn.Linear(in_features=512, out_features=n_actions)
        )

    def forward(self, state_t):
        """
        takes agent's observation (tensor), returns qvalues (tensor)
        :param state_t: a batch of 4-frame buffers, shape = [batch_size, 4, h, w]
        """
        # Use your network to compute qvalues for given state
        qvalues = self.model(state_t)
        return qvalues

    @torch.inference_mode()
    def get_qvalues(self, states: np.ndarray) -> np.ndarray:
        """
        like forward, but works on numpy arrays, not tensors
        """
        states_t = torch.tensor(states, device=next(self.model.parameters()).device)
        qvalues = self.forward(states_t)
        return qvalues.numpy(force=True)

    def sample_actions_by_qvalues(self, qvalues: np.ndarray, greedy: bool = False) -> np.ndarray:
        """pick actions given qvalues based on epsilon-greedy exploration strategy."""
        batch_size, n_actions = qvalues.shape
        eps = self.epsilon
        actions = qvalues.argmax(axis=-1)
        if greedy:
            return actions
        else:
            if np.random.rand() < eps:
                actions = np.random.choice(n_actions, size=batch_size)
        return actions

    def sample_actions(self, states: np.ndarray, greedy: bool = False) -> np.ndarray:
        qvalues = self.get_qvalues(states)
        actions = self.sample_actions_by_qvalues(qvalues, greedy)
        return actions
