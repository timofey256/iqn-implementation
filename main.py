from implementation import train_agent, plot_comparison
import argparse

def main():
    parser = argparse.ArgumentParser(description='Run experiment comparing DQN and iDQN')
    
    # Environment settings
    parser.add_argument('--env', type=str, default='LunarLander-v3', 
                        choices=['CartPole-v1', 'LunarLander-v2', 'Acrobot-v1'],
                        help='Gym environment to use')
    
    # Training settings
    parser.add_argument('--episodes', type=int, default=200, help='Number of training episodes')
    parser.add_argument('--max-steps', type=int, default=500, help='Maximum steps per episode')
    
    # Agent settings
    parser.add_argument('--gamma', type=float, default=0.99, help='Discount factor')
    parser.add_argument('--lr', type=float, default=1e-3, help='Learning rate')
    parser.add_argument('--batch-size', type=int, default=64, help='Batch size')
    parser.add_argument('--buffer-size', type=int, default=10000, help='Replay buffer size')
    parser.add_argument('--target-update-freq', type=int, default=1000, 
                       help='Target network update frequency')
    
    # iDQN specific settings
    parser.add_argument('--K', type=int, default=4, help='Number of consecutive Bellman updates for iDQN')
    parser.add_argument('--D', type=int, default=30, help='Target update frequency for iDQN')
    parser.add_argument('--T', type=int, default=750, help='Window shift frequency for iDQN')
    
    # Evaluation settings
    parser.add_argument('--ma-window', type=int, default=100, help='Moving average window size')
    
    args = parser.parse_args()
    
    print("=" * 50)
    print(f"Starting experiment on {args.env}")
    print(f"Training for {args.episodes} episodes, max {args.max_steps} steps per episode")
    print(f"iDQN parameters: K={args.K}, D={args.D}, T={args.T}")
    print("=" * 50)
    
    # Train DQN
    print(f"\nTraining DQN...")
    dqn_rewards, dqn_ma = train_agent(
        env_name=args.env,
        agent_type='dqn',
        num_episodes=args.episodes,
        max_steps=args.max_steps,
        gamma=args.gamma,
        lr=args.lr,
        batch_size=args.batch_size,
        buffer_size=args.buffer_size,
        target_update_freq=args.target_update_freq,
        ma_window=args.ma_window
    )
    
    # Train iDQN
    print(f"\nTraining iDQN (K={args.K})...")
    idqn_rewards, idqn_ma = train_agent(
        env_name=args.env,
        agent_type='idqn',
        K=args.K,
        D=args.D,
        T=args.T,
        num_episodes=args.episodes,
        max_steps=args.max_steps,
        gamma=args.gamma,
        lr=args.lr,
        batch_size=args.batch_size,
        buffer_size=args.buffer_size,
        target_update_freq=args.target_update_freq,
        ma_window=args.ma_window
    )
    
    # Plot comparison
    plot_comparison(dqn_rewards, idqn_rewards, args.ma_window, args.env, args.K)

if __name__ == "__main__":
    main()