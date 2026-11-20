import torch
import torch.nn as nn
import torch.nn.functional as F

class QNetwork(nn.Module):
    """Actor (Policy) Model."""

    def __init__(self, state_size, action_size, seed):
        """Initialize parameters and build model.
        Params
        ======
            state_size (int): Dimension of each state
            action_size (int): Dimension of each action
            seed (int): Random seed
        """
        super(QNetwork, self).__init__()
        self.seed = torch.manual_seed(seed)
        self.state_size = state_size
        self.action_size = action_size
        # First fully connected layer
        self.fc1 = nn.Linear(self.state_size, 128)
        # Second fully connected layer that outputs our 10 labels
        self.fc2_vstream = nn.Linear(128, 128)
        self.fc2_astream = nn.Linear(128, 128)
        
        self.output_value = nn.Linear(128, 1)
        self.output_advantage = nn.Linear(128, self.action_size)

    # In the main network this method produces predictions and in the target network it produces target values.
    def forward(self, state):
        """Build a network that maps state -> action values."""
        x = self.fc1(state)
        x = F.relu(x)
        
        x_value = self.fc2_vstream(x)
        x_advantage = self.fc2_astream(x)
        x_value = F.relu(x_value)
        x_advantage = F.relu(x_advantage)

        x_value = self.output_value(x_value)
        x_advantage = self.output_advantage(x_advantage)
        
        output = x_value + x_advantage - x_advantage.mean(dim=-1).unsqueeze(dim=1)
        
        return output
    
