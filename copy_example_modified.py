import os
import sys
import json
import numpy as np
import torch as th
import matplotlib.pyplot as plt
import motornet as mn

print('All packages imported.')
print(f'pytorch version: {th.__version__}')
print(f'numpy version: {np.__version__}')
print(f'motornet version: {mn.__version__}')


class Policy(th.nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int, device):
        super().__init__()
        self.device = device
        self.hidden_dim = hidden_dim
        self.n_layers = 1
        
        self.gru = th.nn.GRU(input_dim, hidden_dim, 1, batch_first=True)
        self.fc = th.nn.Linear(hidden_dim, output_dim)
        self.sigmoid = th.nn.Sigmoid()

        # Initialize weights
        for name, param in self.named_parameters():
            if name == "gru.weight_ih_l0":
                th.nn.init.xavier_uniform_(param)
            elif name == "gru.weight_hh_l0":
                th.nn.init.orthogonal_(param)
            elif name == "gru.bias_ih_l0":
                th.nn.init.zeros_(param)
            elif name == "gru.bias_hh_l0":
                th.nn.init.zeros_(param)
            elif name == "fc.weight":
                th.nn.init.xavier_uniform_(param)
            elif name == "fc.bias":
                th.nn.init.constant_(param, -5.)
            else:
                raise ValueError(f"Unexpected parameter: {name}")
        
        self.to(device)

    def forward(self, x, h0):
        y, h = self.gru(x[:, None, :], h0)
        u = self.sigmoid(self.fc(y)).squeeze(dim=1)
        return u, h
    
    def init_hidden(self, batch_size):
        weight = next(self.parameters()).data
        hidden = weight.new(self.n_layers, batch_size, self.hidden_dim).zero_().to(self.device)
        return hidden


def l1(x, y):
    """L1 loss"""
    return th.mean(th.sum(th.abs(x - y), dim=-1))


# ----------------
# Perturbation functions
# ----------------
def perturb_proprioception(obs, space_dim, perturb_strength=0.0):
    """
    Add Gaussian noise to proprioceptive signals in the observation.
    
    Args:
        obs: observation tensor (batch, obs_size)
        space_dim: dimensionality of task space (target is first space_dim elements)
        perturb_strength: magnitude of noise
    
    Returns:
        perturbed observation
    """
    if perturb_strength == 0.0:
        return obs
    
    obs_perturbed = obs.clone()
    
    # Proprioceptive info is everything after target position
    proprio_start = space_dim
    noise = th.randn_like(obs_perturbed[:, proprio_start:]) * perturb_strength
    obs_perturbed[:, proprio_start:] = obs_perturbed[:, proprio_start:] + noise
    
    return obs_perturbed


def add_motor_noise(actions, noise_level=0.0):
    """
    Add Gaussian noise to motor commands.
    
    Args:
        actions: action tensor (batch, act_size)
        noise_level: magnitude of noise
    
    Returns:
        noisy actions (clamped to [0, 1])
    """
    if noise_level == 0.0:
        return actions
    
    noise = th.randn_like(actions) * noise_level
    return th.clamp(actions + noise, 0, 1)


def get_noise_schedule(batch_idx, n_batch, initial_noise, final_noise=0.0):
    """
    Linear noise schedule from initial_noise to final_noise.
    
    Args:
        batch_idx: current batch number
        n_batch: total number of batches
        initial_noise: starting noise level
        final_noise: ending noise level (default 0)
    
    Returns:
        current noise level
    """
    progress = batch_idx / n_batch
    return initial_noise * (1 - progress) + final_noise * progress


def plot_training_log(logs_dict, save_path=None):
    """Plot training logs for all conditions"""
    fig, axs = plt.subplots(1, 1)
    fig.set_tight_layout(True)
    fig.set_size_inches((10, 4))

    for condition, log in logs_dict.items():
        axs.semilogy(log, label=condition, alpha=0.8)
    
    axs.set_ylabel("Loss")
    axs.set_xlabel("Batch #")
    axs.legend()
    axs.grid(True, alpha=0.3)
    
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Training log plot saved to {save_path}")
    plt.show()


def plot_simulations(xy_dict, target_xy, save_path=None):
    """Plot simulations for all conditions side by side"""
    n_conditions = len(xy_dict)
    target_x = target_xy[:, -1, 0]
    target_y = target_xy[:, -1, 1]

    fig = plt.figure(figsize=(6*n_conditions, 6))
    
    for idx, (condition, xy) in enumerate(xy_dict.items()):
        # Trajectories
        ax1 = plt.subplot(2, n_conditions, idx + 1)
        ax1.set_ylim([-1.1, 1.1])
        ax1.set_xlim([-1.1, 1.1])
        mn.plotor.plot_pos_over_time(axis=ax1, cart_results=xy)
        ax1.scatter(target_x, target_y, c='red', s=50, alpha=0.6)
        ax1.set_title(f"{condition}\nTrajectories")
        ax1.set_aspect('equal')
        
        # Distance to target
        ax2 = plt.subplot(2, n_conditions, n_conditions + idx + 1)
        ax2.set_ylim([-2, 2])
        ax2.set_xlim([-2, 2])
        mn.plotor.plot_pos_over_time(axis=ax2, cart_results=xy - target_xy)
        ax2.axhline(0, c="grey", linestyle='--', alpha=0.5)
        ax2.axvline(0, c="grey", linestyle='--', alpha=0.5)
        ax2.set_xlabel("X distance to target")
        ax2.set_ylabel("Y distance to target")
        ax2.set_title("Distance to Target")
        ax2.set_aspect('equal')
    
    plt.tight_layout()
    
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Simulation plot saved to {save_path}")
    plt.show()


def plot_final_comparison(final_losses_dict, save_path=None):
    """Bar plot comparing final performance across conditions"""
    fig, ax = plt.subplots(1, 1, figsize=(8, 5))
    
    conditions = list(final_losses_dict.keys())
    losses = list(final_losses_dict.values())
    
    bars = ax.bar(conditions, losses, color=['#1f77b4', '#ff7f0e', '#2ca02c'], alpha=0.7, edgecolor='black')
    ax.set_ylabel('Final Loss (L1 Distance)', fontsize=12)
    ax.set_xlabel('Training Condition', fontsize=12)
    ax.set_title('Final Performance Comparison', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3, axis='y')
    
    # Add value labels on bars
    for bar, loss in zip(bars, losses):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{loss:.4f}',
                ha='center', va='bottom', fontsize=10, fontweight='bold')
    
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Comparison plot saved to {save_path}")
    plt.show()


def train_model_with_conditions(batch_size=32, n_batch=6000, learning_rate=1e-3,
                                 initial_motor_noise=0.2, initial_proprio_noise=0.15):
    """Train three policies: baseline, motor noise, and proprioception perturbation"""
    
    # Create environments (one for each condition)
    effector = mn.effector.ReluPointMass24()
    envs = {
        'baseline': mn.environment.RandomTargetReach(effector=effector, max_ep_duration=1.),
        'motor_noise': mn.environment.RandomTargetReach(effector=effector, max_ep_duration=1.),
        'proprio_perturb': mn.environment.RandomTargetReach(effector=effector, max_ep_duration=1.)
    }
    
    # Create policies and optimizers
    device = th.device("cpu")
    policies = {}
    optimizers = {}
    
    for condition in envs.keys():
        env = envs[condition]
        policies[condition] = Policy(env.observation_space.shape[0], 32, env.n_muscles, device=device)
        optimizers[condition] = th.optim.Adam(policies[condition].parameters(), lr=learning_rate)
    
    # Storage for losses
    losses_dict = {condition: [] for condition in envs.keys()}
    interval = 250
    space_dim = 2  # 2D task space

    print(f"\nStarting training for {n_batch} batches with 3 conditions...")
    print(f"Initial motor noise: {initial_motor_noise}")
    print(f"Initial proprio perturbation: {initial_proprio_noise}")
    print(f"Noise will decay linearly to 0 by the end of training\n")
    
    for batch in range(n_batch):
        # Calculate current noise levels (linear decay)
        motor_noise_level = get_noise_schedule(batch, n_batch, initial_motor_noise)
        proprio_noise_level = get_noise_schedule(batch, n_batch, initial_proprio_noise)
        
        # Train each condition
        for condition in envs.keys():
            env = envs[condition]
            policy = policies[condition]
            optimizer = optimizers[condition]
            
            # Initialize batch
            h = policy.init_hidden(batch_size=batch_size)
            obs, info = env.reset(options={"batch_size": batch_size})
            terminated = False

            # Initial positions and targets
            xy = [info["states"]["fingertip"][:, None, :]]
            tg = [info["goal"][:, None, :]]

            # Simulate whole episode with condition-specific perturbations
            while not terminated:
                # Apply proprioception perturbation if needed
                if condition == 'proprio_perturb':
                    obs_input = perturb_proprioception(obs, space_dim, proprio_noise_level)
                else:
                    obs_input = obs
                
                # Get action from policy
                action, h = policy(obs_input, h)
                
                # Apply motor noise if needed
                if condition == 'motor_noise':
                    action = add_motor_noise(action, motor_noise_level)
                
                # Step environment
                obs, reward, terminated, truncated, info = env.step(action=action)

                xy.append(info["states"]["fingertip"][:, None, :])
                tg.append(info["goal"][:, None, :])

            # Concatenate into tensors
            xy = th.cat(xy, axis=1)
            tg = th.cat(tg, axis=1)
            loss = l1(xy, tg)
            
            # Backward pass & update weights
            optimizer.zero_grad() 
            loss.backward()
            th.nn.utils.clip_grad_norm_(policy.parameters(), max_norm=1.)
            optimizer.step()
            losses_dict[condition].append(loss.item())

        # Print progress
        if (batch % interval == 0) and (batch != 0):
            print(f"Batch {batch}/{n_batch} | Motor noise: {motor_noise_level:.4f} | Proprio noise: {proprio_noise_level:.4f}")
            for condition in envs.keys():
                mean_loss = sum(losses_dict[condition][-interval:]) / interval
                print(f"  {condition:20s}: mean loss = {mean_loss:.6f}")
            print()
    
    print("Training complete for all conditions!\n")
    
    # Store final xy and tg for last batch of baseline
    xy_train = th.detach(xy)
    tg_train = th.detach(tg)
    
    return policies, envs, losses_dict, xy_train, tg_train


def evaluate_all_models(policies, envs, batch_size=32):
    """Evaluate all trained models without any perturbations"""
    
    print("Evaluating all models on clean test batch (no perturbations)...")
    
    xy_dict = {}
    tg_eval = None
    
    for condition in policies.keys():
        policy = policies[condition]
        env = envs[condition]
        
        h = policy.init_hidden(batch_size=batch_size)
        obs, info = env.reset(options={"batch_size": batch_size})
        terminated = False

        xy = [info["states"]["fingertip"][:, None, :]]
        tg = [info["goal"][:, None, :]]

        while not terminated:
            action, h = policy(obs, h)
            obs, reward, terminated, truncated, info = env.step(action=action)

            xy.append(info["states"]["fingertip"][:, None, :])
            tg.append(info["goal"][:, None, :])

        xy_dict[condition] = th.detach(th.cat(xy, axis=1))
        tg_eval = th.detach(th.cat(tg, axis=1))
    
    # Calculate final losses
    final_losses = {}
    for condition in xy_dict.keys():
        final_loss = l1(xy_dict[condition], tg_eval).item()
        final_losses[condition] = final_loss
        print(f"  {condition:20s}: final L1 loss = {final_loss:.6f}")
    
    print()
    return xy_dict, tg_eval, final_losses


def save_all_models(policies, envs, losses_dict, final_losses, save_dir="save"):
    """Save all models and training logs"""
    
    os.makedirs(save_dir, exist_ok=True)
    
    for condition in policies.keys():
        condition_dir = os.path.join(save_dir, condition)
        os.makedirs(condition_dir, exist_ok=True)
        
        # Save weights
        weight_file = os.path.join(condition_dir, "weights")
        th.save(policies[condition].state_dict(), weight_file)
        
        # Save training log
        log_file = os.path.join(condition_dir, "log.json")
        with open(log_file, 'w') as file:
            json.dump(losses_dict[condition], file)
        
        # Save config
        cfg_file = os.path.join(condition_dir, "cfg.json")
        cfg = envs[condition].get_save_config()
        with open(cfg_file, 'w') as file:
            json.dump(cfg, file)
    
    # Save comparison results
    comparison_file = os.path.join(save_dir, "final_comparison.json")
    with open(comparison_file, 'w') as file:
        json.dump(final_losses, file, indent=2)
    
    print(f"All models saved to {save_dir}/")


def main():
    """Main training and evaluation pipeline"""
    
    # Train all models with different conditions
    policies, envs, losses_dict, xy_train, tg_train = train_model_with_conditions(
        batch_size=32, 
        n_batch=6000,
        initial_motor_noise=0.2,
        initial_proprio_noise=0.15
    )
    
    # Plot training logs
    print("Plotting training logs...")
    plot_training_log(losses_dict, save_path="save/training_comparison.png")
    
    # Evaluate all models
    xy_eval_dict, tg_eval, final_losses = evaluate_all_models(policies, envs, batch_size=32)
    
    # Plot evaluation results
    print("Plotting evaluation results...")
    plot_simulations(xy_eval_dict, tg_eval, save_path="save/evaluation_comparison.png")
    
    # Plot final comparison
    print("Plotting final performance comparison...")
    plot_final_comparison(final_losses, save_path="save/final_comparison.png")
    
    # Save all models
    print("Saving all models...")
    save_all_models(policies, envs, losses_dict, final_losses)
    
    # Print summary
    print("\n" + "="*60)
    print("TRAINING SUMMARY")
    print("="*60)
    sorted_conditions = sorted(final_losses.items(), key=lambda x: x[1])
    for rank, (condition, loss) in enumerate(sorted_conditions, 1):
        print(f"{rank}. {condition:20s}: {loss:.6f}")
    print("="*60)
    print(f"\nBest performing model: {sorted_conditions[0][0]}")
    print("\nAll done!")


if __name__ == "__main__":
    main()