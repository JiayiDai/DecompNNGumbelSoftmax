import copy
import torch
import numpy as np
import torch.nn as nn
from typing import Union
from environment.environments_combogrid_gym import ComboGym
from gymnasium.vector import SyncVectorEnv
from torch.distributions.categorical import Categorical

class Trajectory:
    def __init__(self):
        self._sequence = []
        self.logits = []

    def add_pair(self, state, action, logits=None, detach=False):
        if isinstance(action, torch.Tensor):
            action = action.item()
        self._sequence.append((state, action))
        if logits is not None:
            self.logits.append(copy.deepcopy(logits.cpu().detach()) if detach else logits)

    def concat(self, other):
        self._sequence = self._sequence + copy.deepcopy(other._sequence)
        self.logits = self.logits + copy.deepcopy(other.logits)

    def slice(self, start, stop=None, n=None):
        if stop:
            end = stop
        elif n:
            end = start + n
        else:
            end = len(self._sequence)
        new = copy.deepcopy(self)
        new._sequence = self._sequence[start:end]
        new.logits = self.logits[start:end]
        return new

    def get_length(self):
        return len(self._sequence)
    
    def get_trajectory(self):
        return self._sequence
    
    def get_logits_sequence(self):
        return self.logits
    
    def get_action_sequence(self):
        return [pair[1] for pair in self._sequence]
    
    def get_state_sequence(self):
        return [pair[0] for pair in self._sequence]
    
    def __repr__(self):
        return f"Trajectory(sequence={self._sequence})"
    

def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer

class PPOAgent(nn.Module):
    def __init__(self, envs, hidden_size=6):
        super().__init__()
        if isinstance(envs, ComboGym):
            observation_space_size = envs.get_observation_space()
            action_space_size = envs.get_action_space()
        elif isinstance(envs, SyncVectorEnv):
            observation_space_size = envs.observation_space.shape[1]
            action_space_size = envs.action_space[0].n.item()
        else:
            raise NotImplementedError

        self.critic = nn.Sequential(
                layer_init(nn.Linear(observation_space_size, 64)),
                nn.Tanh(),
                layer_init(nn.Linear(64, 64)),
                nn.Tanh(),
                layer_init(nn.Linear(64, 1)),
            )
        self.actor = nn.Sequential(
            layer_init(nn.Linear(observation_space_size, hidden_size)),
            nn.ReLU(),
            layer_init(nn.Linear(hidden_size, action_space_size)),
        )

        # Option attributes
        self.mask = None
        self.option_size = None
        self.problem_id = None
        self.environment_args = None
        
    def get_value(self, x):
        return self.critic(x)
    
    def get_action_and_value(self, x, action=None):
        logits = self.actor(x)
        probs = Categorical(logits=logits)
        if action is None:
            action = probs.sample()
        return action, probs.log_prob(action), probs.entropy(), self.critic(x), logits
    
    def _masked_neuron_operation_softmax(self, logits, mask):
        relu_out = torch.relu(logits)
        return (mask[0] * 0) + (mask[1] * logits) + (mask[2] * relu_out)

    def run(self, env: Union[ComboGym], length_cap=None, detach_tensors=True, verbose=False):

        trajectory = Trajectory()
        current_length = 0
        self.actor.requires_grad = False

        o, _ = env.reset()
        
        done = False

        while not done:
            o = torch.tensor(o, dtype=torch.float32)
            a, _, _, _, logits = self.get_action_and_value(o)
            trajectory.add_pair(copy.deepcopy(env), a.item(), logits, detach=detach_tensors)

            next_o, _, terminal, truncated, _ = env.step(a.item())
            
            current_length += 1
            if (length_cap is not None and current_length > length_cap) or \
                terminal or truncated:
                done = True     

            o = next_o   
        
        self._h = None
        return trajectory
   
    def _get_action_with_mask(self, x_tensor, mask=None):
        hidden_logits = self.actor[0](x_tensor)
        if mask is None:
            hidden_relu = self.actor[1](hidden_logits)
        else:
            hidden_relu = self._masked_neuron_operation_softmax(hidden_logits, mask)
        logits = self.actor[2](hidden_relu)
        prob_actions = Categorical(logits=logits).probs
        a = torch.argmax(prob_actions).item()
        return a, logits


    def run_with_mask(self, envs, mask, max_size_sequence):
        trajectory = Trajectory()

        length = 0
        if isinstance(envs, list):
            env = envs[length]
            while not env.is_over():
                env = envs[length]
                x_tensor = torch.tensor(env.get_observation(), dtype=torch.float32).view(1, -1)
                a, logits = self._get_action_with_mask(x_tensor, mask)
                trajectory.add_pair(copy.deepcopy(env), a, logits=logits[0])

                length += 1

                if length >= max_size_sequence:
                    return trajectory
        else:
            while not envs.is_over():
                x_tensor = torch.tensor(envs.get_observation(), dtype=torch.float32).view(1, -1)

                a, logits = self._get_action_with_mask(x_tensor, mask)
                
                trajectory.add_pair(copy.deepcopy(envs), a, logits=logits[0])
                envs.step(a)

                length += 1

                if length >= max_size_sequence:
                    return trajectory

        return trajectory