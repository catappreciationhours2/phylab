import os
import sys
import json
import numpy as np
import torch
import matplotlib.pyplot as plt
import torch.nn as nn
import torch.optim as optim

import motornet as mn

print('All packages imported.')
print('pytorch version: ' + torch.__version__)
print('numpy version: ' + np.__version__)
print('motornet version: ' + mn.__version__)

from motornet.effector import RigidTendonArm26
from motornet.muscle import RigidTendonHillMuscle
from motornet.environment import RandomTargetReach

# ----------------
# Define controller
# ----------------
class RNNController(nn.Module):
    def __init__(self, obs_size, act_size, hidden_size=128):
        super().__init__()
        self.rnn = nn.GRU(obs_size, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, act_size)
        self.act_size = act_size

    def forward(self, obs):
        # obs shape: (batch, T, obs_size)
        h, _ = self.rnn(obs)
        out = torch.sigmoid(self.fc(h))  # map to [0,1] activations
        return out

# ----------------
# Set up plant + task
# ----------------
plant = RigidTendonArm26(muscle=RigidTendonHillMuscle(), timestep=0.01)   # 10ms timestep
task = RandomTargetReach(plant, max_ep_duration=1.)

obs_size = task.observation_space.shape[0]
act_size = task.action_space.shape[0]
controller = RNNController(obs_size, act_size)

optimizer = optim.Adam(controller.parameters(), lr=1e-3)

def rollout_episode(task, controller, batch_size=32, max_steps=100, noise_level=0.0):
    """Manually rollout an episode through the environment"""
    obs, info = task.reset(options={"batch_size": batch_size})
    
    # Extract target from the observation (first space_dim elements)
    target = obs[:, :task.space_dim]  # Target is at the beginning of obs
    
    observations = [obs]
    actions_list = []
    end_positions = []
    
    # Convert obs to tensor if needed
    if not torch.is_tensor(obs):
        obs = torch.tensor(obs, dtype=torch.float32)
    
    hidden = None
    
    for step in range(max_steps):
        # Get action from controller (single step)
        obs_input = obs.unsqueeze(1)  # Add time dimension: (batch, 1, obs_size)
        if hidden is not None:
            action_output, hidden = controller.rnn(obs_input, hidden)
            action = torch.sigmoid(controller.fc(action_output.squeeze(1)))
        else:
            action_output, hidden = controller.rnn(obs_input)
            action = torch.sigmoid(controller.fc(action_output.squeeze(1)))
        
        # Add motor noise if specified
        if noise_level > 0:
            noise = torch.randn_like(action) * noise_level
            action = torch.clamp(action * (1 + noise), 0, 1)
        
        actions_list.append(action)
        
        # Step the environment
        obs, reward, terminated, truncated, info = task.step(action.detach().numpy() if not task.differentiable else action)
        
        # Convert obs back to tensor if needed
        if not torch.is_tensor(obs):
            obs = torch.tensor(obs, dtype=torch.float32)
        
        observations.append(obs)
        
        # Get end-effector position from states
        end_pos = info['states']['fingertip']
        if not torch.is_tensor(end_pos):
            end_pos = torch.tensor(end_pos, dtype=torch.float32)
        end_positions.append(end_pos)
        
        if terminated or truncated:
            break
    
    # Stack results
    actions = torch.stack(actions_list, dim=1)  # (batch, T, act_size)
    end_positions = torch.stack(end_positions, dim=1)  # (batch, T, space_dim)
    
    return {
        'actions': actions,
        'end_positions': end_positions,
        'target': target,
        'observations': observations
    }

def add_motor_noise(u, noise_level):
    noise = torch.randn_like(u) * noise_level
    return torch.clamp(u * (1 + noise), 0, 1)  # still in [0,1]

# ----------------
# Training loop
# ----------------
print("Starting training...")

for step in range(5000):  # Reduced for testing
    
    result = rollout_episode(task, controller, batch_size=32, max_steps=int(task.max_ep_duration / task.dt))
    
    actions = result['actions']
    end_positions = result['end_positions']
    target = result['target']
    
    # Loss: endpoint error + smoothness
    final_end_pos = end_positions[:, -1, :]  # Final position
    end_err = torch.mean((final_end_pos - target)**2)
    
    # Jerk penalty (smoothness)
    if actions.shape[1] >= 3:
        jerk_penalty = torch.mean((actions[:,2:] - 2*actions[:,1:-1] + actions[:,:-2])**2)
    else:
        jerk_penalty = torch.tensor(0.0)
    
    loss = end_err + 0.1*jerk_penalty

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if step % 500 == 0:
        print(f"Step {step}, loss {loss.item():.4f}, endpoint {end_err.item():.4f}")

print("Training with noise curriculum...")

noise_start, noise_end = 0.4, 0.0
total_steps = 5000  # Reduced for testing

for step in range(total_steps):
    # Anneal noise linearly
    noise_level = noise_start * (1 - step / total_steps) + noise_end * (step / total_steps)

    result = rollout_episode(task, controller, batch_size=32, 
                           max_steps=int(task.max_ep_duration / task.dt),
                           noise_level=noise_level)
    
    actions = result['actions']
    end_positions = result['end_positions']
    target = result['target']

    final_end_pos = end_positions[:, -1, :]
    end_err = torch.mean((final_end_pos - target)**2)
    
    if actions.shape[1] >= 3:
        jerk_penalty = torch.mean((actions[:,2:] - 2*actions[:,1:-1] + actions[:,:-2])**2)
    else:
        jerk_penalty = torch.tensor(0.0)
    
    loss = end_err + 0.1*jerk_penalty

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if step % 500 == 0:
        print(f"Step {step}, noise {noise_level:.3f}, loss {loss.item():.4f}")

print("Training completed!")