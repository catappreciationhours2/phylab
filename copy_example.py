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


def plot_training_log(log, save_path=None):
    fig, axs = plt.subplots(1, 1)
    fig.set_tight_layout(True)
    fig.set_size_inches((8, 3))

    axs.semilogy(log)
    axs.set_ylabel("Loss")
    axs.set_xlabel("Batch #")
    
    if save_path:
        plt.savefig(save_path)
        print(f"Training log plot saved to {save_path}")
    plt.show()


def plot_simulations(xy, target_xy, save_path=None):
    target_x = target_xy[:, -1, 0]
    target_y = target_xy[:, -1, 1]

    plt.figure(figsize=(10, 3))

    plt.subplot(1, 2, 1)
    plt.ylim([-1.1, 1.1])
    plt.xlim([-1.1, 1.1])
    mn.plotor.plot_pos_over_time(axis=plt.gca(), cart_results=xy)
    plt.scatter(target_x, target_y)
    plt.title("Trajectories")

    plt.subplot(1, 2, 2)
    plt.ylim([-2, 2])
    plt.xlim([-2, 2])
    mn.plotor.plot_pos_over_time(axis=plt.gca(), cart_results=xy - target_xy)
    plt.axhline(0, c="grey")
    plt.axvline(0, c="grey")
    plt.xlabel("X distance to target")
    plt.ylabel("Y distance to target")
    plt.title("Distance to Target")
    
    if save_path:
        plt.savefig(save_path)
        print(f"Simulation plot saved to {save_path}")
    plt.show()


def train_model(batch_size=32, n_batch=6000, learning_rate=1e-3):
    """Train the policy network"""
    
    # Create effector and environment
    effector = mn.effector.ReluPointMass24()
    env = mn.environment.RandomTargetReach(effector=effector, max_ep_duration=1.)
    
    # Create policy and optimizer
    device = th.device("cpu")
    policy = Policy(env.observation_space.shape[0], 32, env.n_muscles, device=device)
    optimizer = th.optim.Adam(policy.parameters(), lr=learning_rate)
    
    losses = []
    interval = 250

    print(f"\nStarting training for {n_batch} batches...")
    
    for batch in range(n_batch):
        # Initialize batch
        h = policy.init_hidden(batch_size=batch_size)
        obs, info = env.reset(options={"batch_size": batch_size})
        terminated = False

        # Initial positions and targets
        xy = [info["states"]["fingertip"][:, None, :]]
        tg = [info["goal"][:, None, :]]

        # Simulate whole episode
        while not terminated:
            action, h = policy(obs, h)
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
        losses.append(loss.item())

        if (batch % interval == 0) and (batch != 0):
            mean_loss = sum(losses[-interval:]) / interval
            print(f"Batch {batch}/{n_batch} Done, mean policy loss: {mean_loss}")
    
    print("\nTraining complete!")
    
    return policy, env, losses, xy, tg


def evaluate_model(policy, env, batch_size=32):
    """Evaluate the trained model"""
    
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

    xy = th.detach(th.cat(xy, axis=1))
    tg = th.detach(th.cat(tg, axis=1))
    
    return xy, tg


def save_model(policy, env, losses, save_dir="save"):
    """Save model weights, training log, and configuration"""
    
    os.makedirs(save_dir, exist_ok=True)
    
    weight_file = os.path.join(save_dir, "weights")
    log_file = os.path.join(save_dir, "log.json")
    cfg_file = os.path.join(save_dir, "cfg.json")

    # Save model weights
    th.save(policy.state_dict(), weight_file)
    print(f"Model weights saved to {weight_file}")

    # Save training history
    with open(log_file, 'w') as file:
        json.dump(losses, file)
    print(f"Training log saved to {log_file}")

    # Save environment configuration
    cfg = env.get_save_config()
    with open(cfg_file, 'w') as file:
        json.dump(cfg, file)
    print(f"Configuration saved to {cfg_file}")


def load_model(save_dir="save"):
    """Load a trained model"""
    
    weight_file = os.path.join(save_dir, "weights")
    log_file = os.path.join(save_dir, "log.json")
    
    # Recreate environment and policy
    effector = mn.effector.ReluPointMass24()
    env = mn.environment.RandomTargetReach(effector=effector, max_ep_duration=1.)
    device = th.device("cpu")
    policy = Policy(env.observation_space.shape[0], 32, env.n_muscles, device=device)
    
    # Load weights
    policy.load_state_dict(th.load(weight_file))
    print(f"Model loaded from {weight_file}")
    
    # Load training log
    with open(log_file, 'r') as file:
        losses = json.load(file)
    print(f"Training log loaded from {log_file}")
    
    return policy, env, losses


def main():
    """Main training and evaluation pipeline"""
    
    # Train the model
    policy, env, losses, xy, tg = train_model(batch_size=32, n_batch=6000)
    
    # Plot training log
    print("\nPlotting training log...")
    plot_training_log(losses, save_path="save/training_log.png")
    
    # Plot final batch simulations
    print("\nPlotting final batch simulations...")
    plot_simulations(th.detach(xy), th.detach(tg), save_path="save/final_simulation.png")
    
    # Save the model
    print("\nSaving model...")
    save_model(policy, env, losses)
    
    # Evaluate on new batch
    print("\nEvaluating model on new batch...")
    xy_eval, tg_eval = evaluate_model(policy, env, batch_size=32)
    plot_simulations(xy_eval, tg_eval, save_path="save/evaluation_simulation.png")
    
    print("\nAll done!")


if __name__ == "__main__":
    main()