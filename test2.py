import os
import sys
import json
import numpy as np
import torch
import matplotlib.pyplot as plt
import torch.nn as nn
import torch.optim as optim
from collections import defaultdict

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
controller = RNNController(obs_size, act_size, hidden_size=256)  # Increased capacity

# FIXED: Lower learning rate for more stable training
optimizer = optim.Adam(controller.parameters(), lr=3e-4)

# ----------------
# Loss tracking
# ----------------
class LossTracker:
    def __init__(self):
        self.losses = defaultdict(list)
        
    def add(self, **kwargs):
        for key, value in kwargs.items():
            self.losses[key].append(value)
    
    def plot(self, save_path='loss_curves.png'):
        fig, axes = plt.subplots(2, 2, figsize=(12, 8))
        fig.suptitle('Training Metrics Over Time')
        
        # Total loss
        axes[0, 0].plot(self.losses['total_loss'], alpha=0.7)
        axes[0, 0].set_title('Total Loss')
        axes[0, 0].set_xlabel('Step')
        axes[0, 0].set_ylabel('Loss')
        axes[0, 0].grid(True)
        
        # Endpoint error
        axes[0, 1].plot(self.losses['endpoint_error'], color='orange', alpha=0.7)
        axes[0, 1].set_title('Endpoint Error')
        axes[0, 1].set_xlabel('Step')
        axes[0, 1].set_ylabel('Error (m²)')
        axes[0, 1].grid(True)
        
        # Jerk penalty
        axes[1, 0].plot(self.losses['jerk_penalty'], color='green', alpha=0.7)
        axes[1, 0].set_title('Jerk Penalty (Smoothness)')
        axes[1, 0].set_xlabel('Step')
        axes[1, 0].set_ylabel('Jerk')
        axes[1, 0].grid(True)
        
        # Noise level (if applicable)
        if 'noise_level' in self.losses:
            axes[1, 1].plot(self.losses['noise_level'], color='red', alpha=0.7)
            axes[1, 1].set_title('Motor Noise Level')
            axes[1, 1].set_xlabel('Step')
            axes[1, 1].set_ylabel('Noise σ')
            axes[1, 1].grid(True)
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=150)
        print(f"Loss curves saved to {save_path}")
        plt.close()
    
    def plot_moving_average(self, save_path='loss_curves_smoothed.png', window=50):
        """Plot smoothed loss curves"""
        fig, axes = plt.subplots(2, 2, figsize=(12, 8))
        fig.suptitle('Training Metrics (Moving Average)')
        
        def smooth(data, window):
            return np.convolve(data, np.ones(window)/window, mode='valid')
        
        # Total loss
        if len(self.losses['total_loss']) > window:
            smoothed = smooth(self.losses['total_loss'], window)
            axes[0, 0].plot(smoothed, alpha=0.7)
            axes[0, 0].set_title('Total Loss (smoothed)')
            axes[0, 0].set_xlabel('Step')
            axes[0, 0].set_ylabel('Loss')
            axes[0, 0].grid(True)
        
        # Endpoint error
        if len(self.losses['endpoint_error']) > window:
            smoothed = smooth(self.losses['endpoint_error'], window)
            axes[0, 1].plot(smoothed, color='orange', alpha=0.7)
            axes[0, 1].set_title('Endpoint Error (smoothed)')
            axes[0, 1].set_xlabel('Step')
            axes[0, 1].set_ylabel('Error (m²)')
            axes[0, 1].grid(True)
        
        # Jerk penalty
        if len(self.losses['jerk_penalty']) > window:
            smoothed = smooth(self.losses['jerk_penalty'], window)
            axes[1, 0].plot(smoothed, color='green', alpha=0.7)
            axes[1, 0].set_title('Jerk Penalty (smoothed)')
            axes[1, 0].set_xlabel('Step')
            axes[1, 0].set_ylabel('Jerk')
            axes[1, 0].grid(True)
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=150)
        print(f"Smoothed loss curves saved to {save_path}")
        plt.close()

tracker = LossTracker()

# ----------------
# Proprioception perturbation functions
# ----------------
def perturb_proprioception(obs, perturb_type='none', perturb_strength=0.0):
    """
    Perturb proprioceptive signals in the observation.
    
    Args:
        obs: observation tensor (batch, obs_size)
        perturb_type: 'noise', 'bias', 'gain', 'delay', or 'none'
        perturb_strength: magnitude of perturbation
    
    Returns:
        perturbed observation
    """
    if perturb_type == 'none' or perturb_strength == 0.0:
        return obs
    
    obs_perturbed = obs.clone()
    
    # Assume proprioceptive info is in the observation after target position
    space_dim = task.space_dim
    proprio_start = space_dim  # After target position
    
    if perturb_type == 'noise':
        # Add Gaussian noise to proprioceptive signals
        noise = torch.randn_like(obs_perturbed[:, proprio_start:]) * perturb_strength
        obs_perturbed[:, proprio_start:] = obs_perturbed[:, proprio_start:] + noise
        
    elif perturb_type == 'bias':
        # Add systematic bias
        bias = torch.ones_like(obs_perturbed[:, proprio_start:]) * perturb_strength
        obs_perturbed[:, proprio_start:] = obs_perturbed[:, proprio_start:] + bias
        
    elif perturb_type == 'gain':
        # Scale proprioceptive signals (gain error)
        gain = 1.0 + perturb_strength
        obs_perturbed[:, proprio_start:] = obs_perturbed[:, proprio_start:] * gain
    
    return obs_perturbed

# ----------------
# Motor noise functions
# ----------------
def add_motor_noise(actions, noise_type='gaussian', noise_level=0.0):
    """
    Add noise to motor commands.
    
    Args:
        actions: action tensor (batch, T, act_size)
        noise_type: 'gaussian', 'signal_dependent', or 'uniform'
        noise_level: magnitude of noise
    
    Returns:
        noisy actions
    """
    if noise_level == 0.0:
        return actions
    
    if noise_type == 'gaussian':
        # Constant Gaussian noise
        noise = torch.randn_like(actions) * noise_level
        return torch.clamp(actions + noise, 0, 1)
    
    elif noise_type == 'signal_dependent':
        # Signal-dependent noise (scales with action magnitude)
        noise = torch.randn_like(actions) * noise_level * (actions + 0.1)  # Added baseline
        return torch.clamp(actions + noise, 0, 1)
    
    elif noise_type == 'uniform':
        # Uniform noise
        noise = (torch.rand_like(actions) - 0.5) * 2 * noise_level
        return torch.clamp(actions + noise, 0, 1)
    
    return actions

# ----------------
# Rollout function with perturbations
# ----------------
def rollout_episode(task, controller, batch_size=32, max_steps=100, 
                   motor_noise_type='gaussian', motor_noise_level=0.0,
                   proprio_perturb_type='none', proprio_perturb_strength=0.0):
    """Manually rollout an episode through the environment with perturbations"""
    obs, info = task.reset(options={"batch_size": batch_size})
    
    # Extract target from the observation
    target = obs[:, :task.space_dim]
    
    observations = [obs]
    actions_list = []
    end_positions = []
    
    # Convert obs to tensor if needed
    if not torch.is_tensor(obs):
        obs = torch.tensor(obs, dtype=torch.float32)
    
    hidden = None
    
    for step in range(max_steps):
        # Perturb proprioception
        obs_perturbed = perturb_proprioception(obs, proprio_perturb_type, proprio_perturb_strength)
        
        # Get action from controller (single step)
        obs_input = obs_perturbed.unsqueeze(1)  # Add time dimension
        if hidden is not None:
            action_output, hidden = controller.rnn(obs_input, hidden)
            action = torch.sigmoid(controller.fc(action_output.squeeze(1)))
        else:
            action_output, hidden = controller.rnn(obs_input)
            action = torch.sigmoid(controller.fc(action_output.squeeze(1)))
        
        actions_list.append(action)
        
        # Apply motor noise after collecting clean action
        action_noisy = add_motor_noise(action.unsqueeze(1), motor_noise_type, motor_noise_level).squeeze(1)
        
        # Step the environment with noisy action
        obs, reward, terminated, truncated, info = task.step(
            action_noisy.detach().numpy() if not task.differentiable else action_noisy
        )
        
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

# ----------------
# Diagnostic visualization
# ----------------
def visualize_trajectories(task, controller, n_samples=5, save_path='trajectories.png'):
    """Visualize sample reaching trajectories"""
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    for i in range(n_samples):
        result = rollout_episode(task, controller, batch_size=1, max_steps=100)
        
        traj = result['end_positions'][0].detach().numpy()
        target = result['target'][0].detach().numpy()
        actions = result['actions'][0].detach().numpy()
        
        # Plot trajectory
        axes[0].plot(traj[:, 0], traj[:, 1], alpha=0.6, label=f'Trial {i+1}')
        axes[0].plot(target[0], target[1], 'r*', markersize=15)
        axes[0].plot(traj[0, 0], traj[0, 1], 'go', markersize=10)  # Start
        axes[0].plot(traj[-1, 0], traj[-1, 1], 'bs', markersize=8)  # End
    
    axes[0].set_xlabel('X position (m)')
    axes[0].set_ylabel('Y position (m)')
    axes[0].set_title('Sample Reaching Trajectories')
    axes[0].legend()
    axes[0].grid(True)
    axes[0].axis('equal')
    
    # Plot one sample's muscle activations
    result = rollout_episode(task, controller, batch_size=1, max_steps=100)
    actions = result['actions'][0].detach().numpy()
    
    for muscle_idx in range(min(6, actions.shape[1])):
        axes[1].plot(actions[:, muscle_idx], label=f'Muscle {muscle_idx+1}')
    
    axes[1].set_xlabel('Time step')
    axes[1].set_ylabel('Activation [0,1]')
    axes[1].set_title('Muscle Activations (Sample Trial)')
    axes[1].legend()
    axes[1].grid(True)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    print(f"Trajectories saved to {save_path}")
    plt.close()

# ----------------
# Training loop 1: Clean training with better hyperparameters
# ----------------
print("\n=== Phase 1: Clean Training ===")
print("FIXED: Lower LR (3e-4), larger batch (64), gradient clipping")

for step in range(3000):  # More steps for better learning
    
    result = rollout_episode(task, controller, batch_size=64,  # FIXED: Larger batch
                           max_steps=int(task.max_ep_duration / task.dt))
    
    actions = result['actions']
    end_positions = result['end_positions']
    target = result['target']
    
    # Loss: endpoint error + smoothness
    final_end_pos = end_positions[:, -1, :]
    end_err = torch.mean((final_end_pos - target)**2)
    
    # FIXED: Better jerk calculation
    if actions.shape[1] >= 3:
        action_diff2 = actions[:,2:] - 2*actions[:,1:-1] + actions[:,:-2]
        jerk_penalty = torch.mean(action_diff2**2)
    else:
        jerk_penalty = torch.tensor(0.0)
    
    loss = end_err + 0.01*jerk_penalty  # FIXED: Lower jerk weight

    optimizer.zero_grad()
    loss.backward()
    
    # FIXED: Add gradient clipping
    torch.nn.utils.clip_grad_norm_(controller.parameters(), max_norm=1.0)
    
    optimizer.step()
    
    # Track losses
    tracker.add(
        total_loss=loss.item(),
        endpoint_error=end_err.item(),
        jerk_penalty=jerk_penalty.item(),
        noise_level=0.0
    )

    if step % 300 == 0:
        # DIAGNOSTICS
        with torch.no_grad():
            action_mean = actions.mean().item()
            action_std = actions.std().item()
            action_min = actions.min().item()
            action_max = actions.max().item()
            
            # Distance traveled
            movement = torch.sqrt(torch.sum((end_positions[:, -1] - end_positions[:, 0])**2, dim=1))
            avg_movement = movement.mean().item()
        
        print(f"Step {step:4d} | Loss: {loss.item():.4f} | Endpoint: {end_err.item():.4f} | Jerk: {jerk_penalty.item():.5f}")
        print(f"  Actions: μ={action_mean:.3f}, σ={action_std:.3f}, range=[{action_min:.3f}, {action_max:.3f}]")
        print(f"  Avg movement distance: {avg_movement:.3f}m")

# Visualize after Phase 1
print("\nGenerating trajectory visualizations...")
visualize_trajectories(task, controller, n_samples=5, save_path='trajectories_phase1.png')

# ----------------
# Training loop 2: Motor noise curriculum
# ----------------
print("\n=== Phase 2: Motor Noise Curriculum ===")

noise_start, noise_end = 0.2, 0.0  # FIXED: Lower starting noise
total_steps = 2000

for step in range(total_steps):
    # Anneal noise linearly
    noise_level = noise_start * (1 - step / total_steps) + noise_end * (step / total_steps)

    result = rollout_episode(task, controller, batch_size=64, 
                           max_steps=int(task.max_ep_duration / task.dt),
                           motor_noise_type='signal_dependent',
                           motor_noise_level=noise_level)
    
    actions = result['actions']
    end_positions = result['end_positions']
    target = result['target']

    final_end_pos = end_positions[:, -1, :]
    end_err = torch.mean((final_end_pos - target)**2)
    
    if actions.shape[1] >= 3:
        action_diff2 = actions[:,2:] - 2*actions[:,1:-1] + actions[:,:-2]
        jerk_penalty = torch.mean(action_diff2**2)
    else:
        jerk_penalty = torch.tensor(0.0)
    
    loss = end_err + 0.01*jerk_penalty

    optimizer.zero_grad()
    loss.backward()
    torch.nn.utils.clip_grad_norm_(controller.parameters(), max_norm=1.0)
    optimizer.step()
    
    # Track losses
    tracker.add(
        total_loss=loss.item(),
        endpoint_error=end_err.item(),
        jerk_penalty=jerk_penalty.item(),
        noise_level=noise_level
    )

    if step % 300 == 0:
        print(f"Step {step:4d} | Noise: {noise_level:.3f} | Loss: {loss.item():.4f} | Endpoint: {end_err.item():.4f}")

visualize_trajectories(task, controller, n_samples=5, save_path='trajectories_phase2.png')

# ----------------
# Training loop 3: Proprioceptive perturbation
# ----------------
print("\n=== Phase 3: Proprioceptive Perturbation Training ===")

proprio_strength_start, proprio_strength_end = 0.15, 0.0  # FIXED: Lower starting perturbation
total_steps = 2000

for step in range(total_steps):
    # Anneal proprioceptive noise
    proprio_strength = proprio_strength_start * (1 - step / total_steps) + proprio_strength_end * (step / total_steps)

    result = rollout_episode(task, controller, batch_size=64, 
                           max_steps=int(task.max_ep_duration / task.dt),
                           proprio_perturb_type='noise',
                           proprio_perturb_strength=proprio_strength)
    
    actions = result['actions']
    end_positions = result['end_positions']
    target = result['target']

    final_end_pos = end_positions[:, -1, :]
    end_err = torch.mean((final_end_pos - target)**2)
    
    if actions.shape[1] >= 3:
        action_diff2 = actions[:,2:] - 2*actions[:,1:-1] + actions[:,:-2]
        jerk_penalty = torch.mean(action_diff2**2)
    else:
        jerk_penalty = torch.tensor(0.0)
    
    loss = end_err + 0.01*jerk_penalty

    optimizer.zero_grad()
    loss.backward()
    torch.nn.utils.clip_grad_norm_(controller.parameters(), max_norm=1.0)
    optimizer.step()
    
    # Track losses
    tracker.add(
        total_loss=loss.item(),
        endpoint_error=end_err.item(),
        jerk_penalty=jerk_penalty.item(),
        noise_level=proprio_strength
    )

    if step % 300 == 0:
        print(f"Step {step:4d} | Proprio: {proprio_strength:.3f} | Loss: {loss.item():.4f} | Endpoint: {end_err.item():.4f}")

visualize_trajectories(task, controller, n_samples=5, save_path='trajectories_phase3.png')

print("\n=== Training completed! ===")

# Generate loss visualizations
tracker.plot('training_loss_curves.png')
tracker.plot_moving_average('training_loss_curves_smoothed.png', window=50)

# ----------------
# Save model
# ----------------
torch.save({
    'model_state_dict': controller.state_dict(),
    'optimizer_state_dict': optimizer.state_dict(),
    'losses': dict(tracker.losses)
}, 'trained_controller.pt')
print("Model saved to trained_controller.pt")

print("\n=== Summary ===")
print(f"Final total loss: {tracker.losses['total_loss'][-1]:.4f}")
print(f"Final endpoint error: {tracker.losses['endpoint_error'][-1]:.4f}")
print(f"Final jerk penalty: {tracker.losses['jerk_penalty'][-1]:.6f}")
print(f"\nInitial endpoint error: {tracker.losses['endpoint_error'][0]:.4f}")
print(f"Improvement: {(tracker.losses['endpoint_error'][0] - tracker.losses['endpoint_error'][-1]) / tracker.losses['endpoint_error'][0] * 100:.1f}%")