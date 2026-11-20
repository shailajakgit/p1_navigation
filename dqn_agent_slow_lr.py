import numpy as np
import random
from collections import namedtuple, deque

from model import QNetwork

import torch
import torch.nn.functional as F
import torch.nn as nn
import torch.optim as optim

BUFFER_SIZE = int(1e5)  # replay buffer size
BATCH_SIZE = 64         # minibatch size
GAMMA = 0.99            # discount factor
TAU = 1e-3              # for soft update of target parameters
LR = 1e-4               # learning rate, started with 5e-4 but it's leading to late stage training instability.
UPDATE_EVERY = 4        # how often to update the network
K=3                     #Length of steps in multi-step-returns

#device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
device = torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")
#print(f'device:{device}')

class Agent():
    """Interacts with and learns from the environment."""

    def __init__(self, state_size, action_size, seed):
        """Initialize an Agent object.
        
        Params
        ======
            state_size (int): dimension of each state
            action_size (int): dimension of each action
            seed (int): random seed
        """
        self.state_size = state_size
        self.action_size = action_size
        self.seed = random.seed(seed)

        # Q-Network
        self.qnetwork_local = QNetwork(state_size, action_size, seed).to(device)
        self.qnetwork_target = QNetwork(state_size, action_size, seed).to(device)
        self.optimizer = optim.Adam(self.qnetwork_local.parameters(), lr=LR)
        # Loss function
        self.loss_fn = nn.MSELoss(reduction='none')

        # Replay memory
        self.memory = ReplayBufferSumTree(BUFFER_SIZE, BATCH_SIZE, seed)
        # Initialize time step (for updating every UPDATE_EVERY steps)
        self.t_step = 0
        # Initialize cumulative time step for beta annealing
        self.cum_t_step = 0
        self.multi_step_memory = deque(maxlen=K)
        self.ms_experience = namedtuple("Experience", field_names=["state", "action", "reward", "next_state", "done"])

    def add_to_per(self):
        # Get cumulative discounted reward
        l = len(self.multi_step_memory)
        cum_disc_reward = sum([self.multi_step_memory[i].reward*(GAMMA**i) for i in range(l)])
        oldest_exp = self.multi_step_memory[0]
        # Add to the PER buffer
        self.memory.add(oldest_exp.state, oldest_exp.action, cum_disc_reward, self.multi_step_memory[-1].next_state, self.multi_step_memory[-1].done, l)
        
    def rollout_multi_step_memory_on_termination(self):
        while self.multi_step_memory:
            self.add_to_per()
            # Evict the oldest experience
            self.multi_step_memory.popleft()
            
    def step(self, state, action, reward, next_state, done):
        # Save experience in multi-step-return memory
        ee = self.ms_experience(state, action, reward, next_state, done)
        self.multi_step_memory.append(ee)
        # Rollout upon termination
        if done:
            self.rollout_multi_step_memory_on_termination()
        # Add to PER buffer otherwise
        elif len(self.multi_step_memory)==K:
            self.add_to_per()

        # Increment cumulative time step
        self.cum_t_step +=1
        # Learn every UPDATE_EVERY time steps.
        self.t_step = (self.t_step + 1) % UPDATE_EVERY
        if self.t_step == 0:
            # If enough samples are available in memory, get random subset and learn
            if len(self.memory) > BATCH_SIZE:
                batch_idx, is_weights_norm, experiences = self.memory.sample(self.cum_t_step)
                new_priorities = self.learn(experiences, GAMMA, is_weights_norm)
                # Update the batch priorities
                self.memory.update_priority(batch_idx, new_priorities.to("cpu").numpy())
    
    def act(self, state, eps=0.):
        """Returns actions for given state as per current policy.
        
        Params
        ======
            state (array_like): current state
            eps (float): epsilon, for epsilon-greedy action selection
        """
        state = torch.from_numpy(state).float().unsqueeze(0).to(device)
        self.qnetwork_local.eval()
        with torch.no_grad():
            action_values = self.qnetwork_local(state)
            #print(f'action values from forward pass: {action_values}')
        self.qnetwork_local.train()

        # Epsilon-greedy action selection
        if random.random() > eps:
            return np.argmax(action_values.cpu().data.numpy())
        else:
            return random.choice(np.arange(self.action_size))

    def learn(self, experiences, gamma, is_weights_norm):
        """Update value parameters using given batch of experience tuples.

        Params
        ======
            experiences (Tuple[torch.Tensor]): tuple of (s, a, r, s', done) tuples 
            gamma (float): discount factor
            is_weights_norm (bool): importance sampling weights normalized
        """
        states, actions, rewards, next_states, dones, k_s = experiences
        targets = self.compute_td_target(experiences, gamma)
        # Get predictions
        self.qnetwork_local.train()
        
        action_values = self.qnetwork_local(states)

        # Compute loss
        squared_errors = self.loss_fn(torch.gather(action_values, 1, actions),targets)

        weighted_squared_errors = squared_errors*is_weights_norm
        weighted_loss_sum = torch.mean(weighted_squared_errors)
        
        # Backprop
        self.optimizer.zero_grad()  # Clear existing gradients from previous steps

        weighted_loss_sum.backward()
        self.optimizer.step()

        # ------------------- update target network ------------------- #
        self.soft_update(self.qnetwork_local, self.qnetwork_target, TAU) 

        # re-compute TD error with updated weights to set priority in experience replay
        self.qnetwork_local.eval()
        
        with torch.no_grad():
            new_action_values = self.qnetwork_local(states)
        new_targets = self.compute_td_target(experiences, gamma)
        td_error = new_targets - torch.gather(new_action_values, 1, actions)
        self.qnetwork_local.train()
        return td_error.squeeze(dim=1)
        

    def compute_td_target(self, experiences, gamma):
        self.qnetwork_target.eval()
        self.qnetwork_local.eval()
        states, actions, rewards, next_states, dones, k_s = experiences
        with torch.no_grad():
            next_action_values_local = self.qnetwork_local(next_states)
            next_action_values_target = self.qnetwork_target(next_states)

        optimal_next_action_values_local, optimal_next_action_indices_local = torch.max(next_action_values_local, dim=1)
        
        optimal_next_action_values = torch.gather(next_action_values_target,1,optimal_next_action_indices_local.unsqueeze(dim=1))
        
        targets = torch.where(dones, rewards, rewards + ((GAMMA**k_s) * optimal_next_action_values))
        
        return targets

        

    def soft_update(self, local_model, target_model, tau):
        """Soft update model parameters.
        θ_target = τ*θ_local + (1 - τ)*θ_target

        Params
        ======
            local_model (PyTorch model): weights will be copied from
            target_model (PyTorch model): weights will be copied to
            tau (float): interpolation parameter 
        """
        for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
            target_param.data.copy_(tau*local_param.data + (1.0-tau)*target_param.data)


class ReplayBufferSumTree:
    """Fixed-size buffer to store experience tuples."""

    def __init__(self, buffer_size, batch_size, seed, alpha=0.6, beta=0.4, epsilon=1e-5, beta_anneal_steps=5000000):
        """Initialize a ReplayBuffer object.

        Params
        ======
            action_size (int): dimension of each action
            buffer_size (int): maximum size of buffer
            batch_size (int): size of each training batch
            seed (int): random seed
        """ 
        self.batch_size = batch_size
        self.seed = random.seed(seed)
        self.rng = np.random.default_rng(seed=seed)
        self.alpha = alpha
        self.beta_0 = beta
        self.epsilon = epsilon

        self.buffer_size = buffer_size
        # Total nodes in a full binary tree with N leaves is 2N - 1. It just stores priorities.
        self.tree = np.zeros(2 * buffer_size - 1)
        # Array to store the actual experience data objects corresponding to leaves
        self.data = [None]* self.buffer_size
        self.experience = namedtuple("Experience", field_names=["state", "action", "reward", "next_state", "done",  "k"])
        self.data_pointer = 0
        self.size = 0
        self.beta_anneal_steps = beta_anneal_steps

    def add(self, state, action, reward, next_state, done, k):
        """Adds a new experience and its initial priority to the ring buffer.
        Params
        ======
            state (tuple): state
            action (int): action
            reward (float): reward
            next_state (tuple): next_state
            done (bool): done
            k (int): roll out
        
        """
        e = self.experience(state, action, reward, next_state, done, k)
        # New experiences are allocated maximum priority

        priority = np.max(self.tree[self.buffer_size - 1:]) if self.size > 0 else 1.0

        # Find the leaf index in the flat tree array
        tree_idx = self.data_pointer + self.buffer_size - 1
        
        # Store data and update priority
        self.data[self.data_pointer] = e
        self.update(tree_idx, priority)
        
        # Advance ring buffer pointer circularly
        self.data_pointer = (self.data_pointer + 1) % self.buffer_size
        self.size = min(self.size + 1, self.buffer_size)

    def update(self, tree_idx: int, priority: float):
        """Updates a leaf node's priority and propagates the delta up the tree."""
        change = priority - self.tree[tree_idx]
        self.tree[tree_idx] = priority
        
        # Propagate the change up to the root
        while tree_idx != 0:
            tree_idx = (tree_idx - 1) // 2
            self.tree[tree_idx] += change

    def get_leaf(self, value: float):
        """
        Finds the leaf node corresponding to a sampled value dynamically.
        Time Complexity: O(log N)
        """
        parent_idx = 0
        
        while True:
            left_child_idx = 2 * parent_idx + 1
            right_child_idx = left_child_idx + 1
            
            # If we've reached the leaf layer, terminate search
            if left_child_idx >= len(self.tree):
                leaf_idx = parent_idx
                break
            else:
                # Traverse downward based on cumulative weights
                if value <= self.tree[left_child_idx]:
                    parent_idx = left_child_idx
                else:
                    value -= self.tree[left_child_idx]
                    parent_idx = right_child_idx
                    
        data_idx = leaf_idx - self.buffer_size + 1
        return leaf_idx, self.tree[leaf_idx], self.data[data_idx]

    @property
    def total_priority(self) -> float:
        """Returns the root value, which is the sum of all priorities."""
        return self.tree[0]

    def __len__(self):
        """Return the current size of internal memory."""
        return self.size

    def sample(self, t_step):
        """Randomly sample a batch of experiences from memory."""
        segment = self.total_priority / self.batch_size
        batch_idx, batch_priorities, experiences = [], [], []
        
        for i in range(self.batch_size):
            a = segment * i
            b = segment * (i + 1)
            #value = np.random.uniform(a, b)
            value = self.rng.uniform(a, b)
            
            idx, priority, data = self.get_leaf(value)
            batch_idx.append(idx)
            batch_priorities.append(priority)
            experiences.append(data)

        states = torch.from_numpy(np.vstack([e.state for e in experiences if e is not None])).float().to(device)
        actions = torch.from_numpy(np.vstack([e.action for e in experiences if e is not None])).long().to(device)
        rewards = torch.from_numpy(np.vstack([e.reward for e in experiences if e is not None])).float().to(device)
        next_states = torch.from_numpy(np.vstack([e.next_state for e in experiences if e is not None])).float().to(device)
        dones = torch.from_numpy(np.vstack([e.done for e in experiences if e is not None])).to(device)
        
        k_s = torch.from_numpy(np.vstack([e.k for e in experiences if e is not None])).to(device)

        # Compute importance sampling weights
        beta_t = min(1, self.beta_0 + (1-self.beta_0)*(t_step/self.beta_anneal_steps))
        probabilities = batch_priorities/self.total_priority
        is_weights = [(1/(self.size*p))**beta_t for p in probabilities]

        # Occupied leaves
        min_occ_probability=np.min(self.tree[(self.buffer_size - 1):(self.buffer_size + self.size - 1)])/self.total_priority
        
        #max_probability = (np.max(self.tree[self.buffer_size - 1:]) if self.size > 0 else 1.0)/self.total_priority
        max_is_weight = 1/((self.size*min_occ_probability)**beta_t)
        is_weights_normalized = torch.from_numpy(np.vstack(is_weights/max_is_weight)).float().to(device)
        
        return (batch_idx, is_weights_normalized, (states, actions, rewards, next_states, dones, k_s))

    def update_priority(self, batch_idx, td_errors):
        for idx, error in zip(batch_idx, td_errors):
            new_priority = (abs(error) + self.epsilon) ** self.alpha
            self.update(idx, new_priority)

