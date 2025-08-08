
from agent import DQNAgent
from env import GomokuEnv
import torch
import argparse

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    parser = argparse.ArgumentParser(description="Gomoku Training Script")

    parser.add_argument("--init_epsilon", type=float, default=0.95, help="Initial exploration rate")
    parser.add_argument("--load_model", type=lambda x: (str(x).lower() == 'true'), default=False, help="Load existing model or not")
    parser.add_argument("--max_episodes", type=int, default=10000, help="Number of episodes to train the agent")
    parser.add_argument("--epsilon_decay_interval", type=int, default=100, help="Number of episodes to decay epsilon")

    args = parser.parse_args()

    UPDATE_MODEL_INTERVAL = 100  # Print every 100 episodes
    EPSILON_DECAY_INTERVAL = args.epsilon_decay_interval  # Decay epsilon every specified interval
    # get arg to check whether to use GUI or not
    print(f"Using device: {device}")
    gomokuEnv = GomokuEnv()
    agent = DQNAgent(device=device, init_epsilon=args.init_epsilon, load_model=args.load_model)

    max_episodes = args.max_episodes  # Set a high number of episodes for training

    recent_losses = []
    player1_wins = 0
    player2_wins = 0
    player1_cum_rewards = 0
    player2_cum_rewards = 0

    for episode in range(max_episodes):
        if episode % UPDATE_MODEL_INTERVAL == 0 and episode > 0:
            avg_loss = sum(recent_losses) / len(recent_losses) if recent_losses else 0
            print(f"Average loss over last {UPDATE_MODEL_INTERVAL} episodes: {avg_loss:.4f}")
            print(f"Player 1 wins: {player1_wins}, Player 2 wins: {player2_wins}")
            print(f"Player 1 cumulative rewards: {player1_cum_rewards}, Player 2 cumulative rewards: {player2_cum_rewards}")
            player1_wins = 0
            player2_wins = 0
            player1_cum_rewards = 0
            player2_cum_rewards = 0
            recent_losses = []
            print(f"Episode {episode}/{max_episodes} completed. Epsilon: {agent.epsilon:.4f}. Saving model...")
            agent.update_target_model()
            agent.save_model()
        gomokuEnv.reset()
        gomokuEnv.first_move()
        total_turns = 1
        completed = False

        if episode % EPSILON_DECAY_INTERVAL == 0 and agent.epsilon > agent.epsilon_min:
            agent.epsilon *= agent.epsilon_decay
            agent.epsilon = max(agent.epsilon, agent.epsilon_min)

        move_count = 0

        # Buffer to hold samples for both players
        # Will be pushed to memory after the game ends
        player1_samples = []
        player2_samples = []
        winner = None

        while move_count < 225:  # 15x15 board, max 225 turns
            # gui.draw_board(gomokuEnv.state[0])
            current_turn = total_turns & 1  

            action = agent.get_action(gomokuEnv.state, current_turn)
            # print(f"Action chosen: {action} by player {current_turn}")

            # Take the action in the environment
            next_state, completed, reward, valid_move = gomokuEnv.step(action, current_turn)
            if current_turn == 0:
                player1_samples.append((gomokuEnv.state[0], action, reward, next_state, completed))
                player1_cum_rewards += reward
            else:
                player2_samples.append((gomokuEnv.state[1], action, reward, next_state, completed))
                player2_cum_rewards += reward

            if not valid_move:
                continue

            move_count += 1
            total_turns += 1
            
            if completed:
                if current_turn == 0:
                    player1_wins += 1
                    winner = 0
                else:
                    player2_wins += 1
                    winner = 1
                break

        if winner is not None:
            if winner == 0:
                num_moves = len(player2_samples)
                bad_move_start = max(0, num_moves - 5)  # Start from the last 5 moves
                denom = max(1, num_moves - bad_move_start)  # Avoid division by zero
                for i in range(bad_move_start, num_moves):
                    # Scale linearly: earlier moves get smaller negative reward
                    relative_i = i - bad_move_start
                    negative_reward = -10 * ((relative_i + 1) / denom)
                    player2_samples[i] = (
                        player2_samples[i][0],
                        player2_samples[i][1],
                        negative_reward,
                        player2_samples[i][3],
                        player2_samples[i][4],
                    )
                num_moves = len(player1_samples)
                good_move_start = max(0, num_moves - 5)  # Start from the last 5 moves
                denom = max(1, num_moves - good_move_start)  # Avoid division by zero
                for i in range(good_move_start, num_moves):
                    # update the reward for the last 5 moves only if the move was valid
                    if player1_samples[i][2] > 0:
                        relative_i = i - good_move_start
                        positive_reward = 10 * ((relative_i + 1) / denom)
                        player1_samples[i] = (
                            player1_samples[i][0],
                            player1_samples[i][1],
                            positive_reward,
                            player1_samples[i][3],
                            player1_samples[i][4],
                        )
            else:
                num_moves = len(player1_samples)
                bad_move_start = max(0, num_moves - 5)  # Start from the last 5 moves
                denom = max(1, num_moves - bad_move_start)  # Avoid division by zero
                for i in range(bad_move_start, num_moves):
                    relative_i = i - bad_move_start
                    negative_reward = -10 * ((relative_i + 1) / denom)
                    player1_samples[i] = (
                        player1_samples[i][0],
                        player1_samples[i][1],
                        negative_reward,
                        player1_samples[i][3],
                        player1_samples[i][4],
                    )

                num_moves = len(player2_samples)
                good_move_start = max(0, num_moves - 5)  # Start from the last 5 moves
                denom = max(1, num_moves - good_move_start)  # Avoid division by zero
                for i in range(good_move_start, num_moves):
                    if player2_samples[i][2] > 0:
                        relative_i = i - good_move_start
                        positive_reward = 10 * ((relative_i + 1) / denom)
                        player2_samples[i] = (
                            player2_samples[i][0],
                            player2_samples[i][1],
                            positive_reward,
                            player2_samples[i][3],
                            player2_samples[i][4],
                        )
        
        # Add samples to the agent's memory
        for sample in player1_samples:
            agent.save_sample(sample, turn=0)
        for sample in player2_samples:
            agent.save_sample(sample, turn=1)

        # Train the models
        if len(agent.memory1) >= agent.batch_size:
            for _ in range(move_count):
                loss = agent.train_model(agent.model1, agent.target_model1, agent.memory1, agent.optimizer1)
                if loss is not None:
                    recent_losses.append(loss)

        if len(agent.memory2) >= agent.batch_size:
            for _ in range(move_count):
                loss = agent.train_model(agent.model2, agent.target_model2, agent.memory2, agent.optimizer2)
                if loss is not None:
                    recent_losses.append(loss)

    agent.save_model()
    print(f"Model saved at episode {episode + 1}")  