import numpy as np
import matplotlib.pyplot as plt
import gymnasium as gym
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from collections import deque, namedtuple
import random
from tqdm import tqdm

# Define the transition tuple
Transition = namedtuple('Transition', 
                        ('state', 'action', 'reward', 'next_state', 'done'))

class ReplayBuffer:
    """Experience replay buffer to store and sample transitions"""
    def __init__(self, capacity):
        self.memory = deque(maxlen=capacity)
        
    def add(self, state, action, reward, next_state, done):
        self.memory.append(Transition(state, action, reward, next_state, done))
        
    def sample(self, batch_size):
        batch = random.sample(self.memory, batch_size)
        return Transition(*zip(*batch))
    
    def __len__(self):
        return len(self.memory)

class QNetwork(nn.Module):
    """Neural network for Q-function approximation"""
    def __init__(self, state_dim, action_dim, hidden_dim=128):
        super(QNetwork, self).__init__()
        self.fc1 = nn.Linear(state_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, action_dim)
        
    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x)  # Q-values for each action

class DQN:
    """Deep Q-Network agent implementation"""
    def __init__(self, state_dim, action_dim, 
                 gamma=0.99, lr=0.0003, buffer_size=10_000, batch_size=32,
                 target_update_freq=200, eps_start=1.0, eps_end=0.01, eps_decay=0.995):
        
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.gamma = gamma
        self.batch_size = batch_size
        self.target_update_freq = target_update_freq
        self.eps = eps_start
        self.eps_end = eps_end
        self.eps_decay = eps_decay
        
        # Device setup
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Q-networks
        self.q_network = QNetwork(state_dim, action_dim).to(self.device)
        self.target_network = QNetwork(state_dim, action_dim).to(self.device)
        self.target_network.load_state_dict(self.q_network.state_dict())
        
        # Optimizer
        self.optimizer = optim.Adam(self.q_network.parameters(), lr=lr)
        
        # Replay buffer
        self.replay_buffer = ReplayBuffer(buffer_size)
        
        # Step counter
        self.steps_done = 0
        
    def select_action(self, state, eval_mode=False):
        """Select action using epsilon-greedy policy"""
        if eval_mode:
            epsilon = 0.05  # Small epsilon for evaluation
        else:
            epsilon = max(self.eps_end, self.eps * self.eps_decay)
            self.eps = epsilon
            
        if random.random() > epsilon:
            with torch.no_grad():
                state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
                q_values = self.q_network(state_tensor)
                return q_values.argmax().item()
        else:
            return random.randrange(self.action_dim)
        
    def update(self):
        """Update Q-network using a batch of transitions"""
        if len(self.replay_buffer) < self.batch_size:
            return 0.0  # Not enough samples
        
        # Sample batch from replay buffer
        transitions = self.replay_buffer.sample(self.batch_size)
        
        # Convert to tensors
        state_batch = torch.FloatTensor(np.array(transitions.state)).to(self.device)
        action_batch = torch.LongTensor(transitions.action).unsqueeze(1).to(self.device)
        reward_batch = torch.FloatTensor(transitions.reward).unsqueeze(1).to(self.device)
        non_final_mask = torch.tensor([not done for done in transitions.done], 
                                      dtype=torch.bool, device=self.device)
        non_final_next_states = torch.FloatTensor(np.array([s for s, d in zip(transitions.next_state, 
                                                                           transitions.done) 
                                                         if not d])).to(self.device)
        
        # Get Q(s,a) values
        q_values = self.q_network(state_batch).gather(1, action_batch)
        
        # Compute target Q values
        next_q_values = torch.zeros(self.batch_size, 1, device=self.device)
        if len(non_final_next_states) > 0:
            with torch.no_grad():
                next_q_values[non_final_mask] = self.target_network(non_final_next_states).max(1, keepdim=True)[0]
        
        # Compute expected Q values
        expected_q_values = reward_batch + (self.gamma * next_q_values)
        
        # Compute loss
        loss = F.smooth_l1_loss(q_values, expected_q_values)
        
        # Optimize
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        # Update target network periodically
        self.steps_done += 1
        if self.steps_done % self.target_update_freq == 0:
            self.target_network.load_state_dict(self.q_network.state_dict())
            
        return loss.item()

class iDQN(DQN):
    """Iterated Deep Q-Network implementation"""
    def __init__(self, state_dim, action_dim, K=4, D=10, T=150, **kwargs):
        """
        K: Number of consecutive Bellman updates to learn
        D: Frequency to update target networks to their respective online networks
        T: Frequency to shift the window of Bellman updates
        """
        super(iDQN, self).__init__(state_dim, action_dim, **kwargs)
        
        self.K = K  # Number of consecutive Bellman updates
        self.D = D  # Update target to online frequency
        self.T = T  # Window shift frequency
        
        # Create K online networks and K target networks
        # L_ON = [Q_1, Q_2, ..., Q_K]
        self.online_networks = [QNetwork(state_dim, action_dim).to(self.device) for _ in range(K)]
        # L_TN = [\bar{Q}_0, \bar{Q}_1, ..., \bar{Q}_{K-1}]
        self.target_networks = [QNetwork(state_dim, action_dim).to(self.device) for _ in range(K)]
        
        # Initialize target networks with their respective online networks
        for k in range(1, K):
            # \bar{Q}_k <- Q_k, i.e., L_TN[k] <- L_ON[k-1]
            self.target_networks[k].load_state_dict(self.online_networks[k-1].state_dict())
            
        # Create an optimizer for each online network
        self.optimizers = [optim.Adam(net.parameters(), lr=kwargs.get('lr', 1e-3)) 
                           for net in self.online_networks]
        
    def select_action(self, state, eval_mode=False):
        """Select action by sampling a network uniformly and using epsilon-greedy"""
        epsilon = max(self.eps_end, self.eps * self.eps_decay)
        self.eps = epsilon
            
        if random.random() > epsilon or eval_mode:
            with torch.no_grad():
                state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
                
                # Uniformly sample an online network as per the paper
                k = random.randint(0, self.K - 1)
                q_values = self.online_networks[k](state_tensor)
                return q_values.argmax().item()
        else:
            return random.randrange(self.action_dim)
        
    def update(self):
        """Update all online networks using their respective targets"""
        if len(self.replay_buffer) < self.batch_size:
            return 0.0  # Not enough samples
        
        # Sample from replay buffer
        transitions = self.replay_buffer.sample(self.batch_size)
        
        # Convert to tensors
        state_batch = torch.FloatTensor(np.array(transitions.state)).to(self.device)
        action_batch = torch.LongTensor(transitions.action).unsqueeze(1).to(self.device)
        reward_batch = torch.FloatTensor(transitions.reward).unsqueeze(1).to(self.device)
        non_final_mask = torch.tensor([not done for done in transitions.done], 
                                      dtype=torch.bool, device=self.device)
        non_final_next_states = torch.FloatTensor(np.array([s for s, d in zip(transitions.next_state, 
                                                                           transitions.done) 
                                                         if not d])).to(self.device)
        
        total_loss = 0.0
        
        # Update each online network
        for k in range(self.K):
            # Compute Q_k(s,a), i.e., L_TN[k] for current online network
            q_values = self.online_networks[k](state_batch).gather(1, action_batch)
            
            # Compute target using the corresponding target network \bar{Q}_{k-1}, i.e., L_ON[k]
            target_network = self.target_networks[k]
            
            # Compute target Q values
            next_q_values = torch.zeros(self.batch_size, 1, device=self.device)
            if len(non_final_next_states) > 0:
                with torch.no_grad():
                    next_q_values[non_final_mask] = target_network(non_final_next_states).max(1, keepdim=True)[0]
            
            # Compute expected Q values
            expected_q_values = reward_batch + (self.gamma * next_q_values)
            
            # Compute loss
            loss = F.smooth_l1_loss(q_values, expected_q_values)
            
            # Optimize
            self.optimizers[k].zero_grad()
            loss.backward()
            self.optimizers[k].step()
            
            total_loss += loss.item()
        
        # Increment step counter
        self.steps_done += 1
        
        # Shift window every T steps
        # \bar{Q}_{k - 1} <- Q_k, i.e., L_TN[k] <- L_ON[k]
        # AND 
        # Q_k <- Q_{k + 1}, i.e., L_ON[k] <- L_ON[k + 1]
        if self.steps_done % self.T == 0:
            for k in range(self.K - 1):
                self.target_networks[k].load_state_dict(self.online_networks[k].state_dict())
                self.online_networks[k].load_state_dict(self.online_networks[k + 1].state_dict())
        
        # Update target networks to their respective online networks every D steps
        # \bar{Q}_k <- Q_k, i.e., L_TN[k] <- L_ON[k-1]
        if self.steps_done % self.D == 0:
            for k in range(1, self.K):
                self.target_networks[k].load_state_dict(self.online_networks[k - 1].state_dict())
        
        return total_loss / self.K

def train_agent(env_name, agent_type='dqn', K=4, D=10, T=150, 
                num_episodes=400, max_steps=1_000, gamma=0.99, 
                lr=0.0003, batch_size=32, target_update_freq=200, 
                buffer_size=10_000, ma_window=100):
    """
    Train a DQN or iDQN agent
    
    Args:
        env_name: Gym environment name
        agent_type: 'dqn' or 'idqn'
        K: Number of consecutive Bellman updates for iDQN
        D: Target update frequency for iDQN
        T: Window shift frequency for iDQN
        num_episodes: Number of training episodes
        max_steps: Maximum steps per episode
        gamma: Discount factor
        lr: Learning rate
        batch_size: Batch size for updates
        target_update_freq: Target network update frequency for DQN
        buffer_size: Replay buffer capacity
        ma_window: Moving average window size
        
    Returns:
        rewards: List of episode rewards
        ma_rewards: Moving average rewards
    """
    # Set seed for reproducibility 
    SEED = 42
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(SEED)

    # Create environment
    env = gym.make(env_name)
    
    # Get state and action dimensions
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n
    
    # Create agent
    if agent_type.lower() == 'dqn':
        agent = DQN(state_dim, action_dim, gamma=gamma, lr=lr, 
                    buffer_size=buffer_size, batch_size=batch_size,
                    target_update_freq=target_update_freq)
    elif agent_type.lower() == 'idqn':
        agent = iDQN(state_dim, action_dim, K=K, D=D, T=T, gamma=gamma, 
                     lr=lr, buffer_size=buffer_size, batch_size=batch_size,
                     target_update_freq=target_update_freq)
    else:
        raise ValueError(f"Unknown agent type: {agent_type}. Use 'dqn' or 'idqn'.")
        
    # Training variables
    all_rewards = []
    ma_rewards = []
    
    # Training loop
    for episode in (pbar := tqdm(range(num_episodes))):
        state, _ = env.reset(seed=episode)
        episode_reward = 0
        done = False
        truncated = False
        
        for t in range(max_steps):
            # Select action
            action = agent.select_action(state)
            
            # Execute action
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            
            # Store transition
            agent.replay_buffer.add(state, action, reward, next_state, done)
            
            # Update agent
            loss = agent.update()
            
            # Update state and reward
            state = next_state
            episode_reward += reward
            
            if done:
                break
                
        # Store and log results
        all_rewards.append(episode_reward)
        
        # Calculate moving average
        window_size = min(ma_window, len(all_rewards))
        ma_reward = np.mean(all_rewards[-window_size:])
        ma_rewards.append(ma_reward)
        
        # Update progress bar
        pbar.set_description(f"Episode {episode+1}, Reward: {episode_reward:.2f}, MA: {ma_reward:.2f}, Epsilon: {agent.eps}")
        
    env.close()
    return all_rewards, ma_rewards

def plot_comparison(dqn_rewards, idqn_rewards, ma_window, env_name, K):
    """
    Plot comparison between DQN and iDQN results
    
    Args:
        dqn_rewards: List of DQN episode rewards
        idqn_rewards: List of iDQN episode rewards
        ma_window: Moving average window size
        env_name: Environment name
        K: Number of consecutive Bellman updates in iDQN
    """
    # Calculate moving averages
    dqn_ma = [np.mean(dqn_rewards[max(0, i-ma_window):i+1]) for i in range(len(dqn_rewards))]
    idqn_ma = [np.mean(idqn_rewards[max(0, i-ma_window):i+1]) for i in range(len(idqn_rewards))]
    
    # Create figure
    plt.figure(figsize=(15, 6))
    
    # Plot raw rewards
    plt.subplot(1, 2, 1)
    plt.plot(dqn_rewards, alpha=0.3, color='blue')
    plt.plot(idqn_rewards, alpha=0.3, color='red')
    plt.xlabel('Episode')
    plt.ylabel('Reward')
    plt.title(f'Episode Rewards ({env_name})')
    
    # Plot moving average
    plt.subplot(1, 2, 2)
    plt.plot(dqn_ma, color='blue', label='DQN')
    plt.plot(idqn_ma, color='red', label=f'iDQN (K={K})')
    plt.xlabel('Episode')
    plt.ylabel(f'MA Reward (window={ma_window})')
    plt.title(f'Moving Average Rewards ({env_name})')
    plt.legend()
    
    plt.tight_layout()
    
    # Save figure
    plt.savefig(f'{env_name}_K{K}_comparison.png')
    plt.show()
    
    # Print final performance
    print(f"DQN Final MA Reward: {dqn_ma[-1]:.2f}")
    print(f"iDQN (K={K}) Final MA Reward: {idqn_ma[-1]:.2f}")
    print(f"Improvement: {((idqn_ma[-1] - dqn_ma[-1]) / abs(dqn_ma[-1]) * 100):.2f}%")

# Example usage
if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Train and compare DQN and iDQN')
    parser.add_argument('--env', type=str, default='CartPole-v1', help='Gym environment')
    parser.add_argument('--episodes', type=int, default=500, help='Number of episodes')
    parser.add_argument('--K', type=int, default=3, help='Number of consecutive Bellman updates for iDQN')
    parser.add_argument('--D', type=int, default=10, help='Target update frequency for iDQN')
    parser.add_argument('--T', type=int, default=150, help='Window shift frequency for iDQN')
    parser.add_argument('--ma-window', type=int, default=100, help='Moving average window size')
    
    args = parser.parse_args()
    
    # Train DQN
    print(f"Training DQN on {args.env}...")
    dqn_rewards, dqn_ma = train_agent(args.env, 'dqn', num_episodes=args.episodes, ma_window=args.ma_window)
    
    # Train iDQN
    print(f"\nTraining iDQN (K={args.K}) on {args.env}...")
    idqn_rewards, idqn_ma = train_agent(args.env, 'idqn', K=args.K, D=args.D, T=args.T, 
                                       num_episodes=args.episodes, ma_window=args.ma_window)
    
    # Plot comparison
    plot_comparison(dqn_rewards, idqn_rewards, args.ma_window, args.env, args.K)