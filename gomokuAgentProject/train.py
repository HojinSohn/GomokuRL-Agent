
import sys
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
    parser.add_argument("--load_memory", type=lambda x: (str(x).lower() == 'true'), default=False, help="Load memory from file or not")
    parser.add_argument("--start_training_episode", type=int, default=500, help="Episode to start training")

    args = parser.parse_args()

    UPDATE_MODEL_INTERVAL = 50  # Update target model every 50 episodes
    LOGGING_INTERVAL = 100  # Print every 100 episodes
    START_TRAINING_EPISODE = args.start_training_episode  # Start training after specified episode
    EPSILON_DECAY_INTERVAL = args.epsilon_decay_interval  # Decay epsilon every specified interval
    SAVING_INTERVAL = 100  # Save model every 100 episodes
    

    # get arg to check whether to use GUI or not
    print(f"Using device: {device}")
    gomokuEnv = GomokuEnv()
    agent = DQNAgent(device=device, init_epsilon=args.init_epsilon, load_model=args.load_model)
    load_memory = args.load_memory
    if load_memory:
        print("Loading memory from file...")
        agent.load_memory()

    max_episodes = args.max_episodes  # Set a high number of episodes for training

    recent_losses_p1 = []
    recent_losses_p2 = []
    player1_wins = 0
    player2_wins = 0

    for episode in range(max_episodes):
        if episode < 101:
            print(f"Episode {episode} - Simulating game...")
        if episode > START_TRAINING_EPISODE and episode % SAVING_INTERVAL == 0:
            # checkpoint the model
            print(f"Saving model at episode {episode}")
            agent.save_model()
            # save the memory in the file for later use
            agent.save_memory()
            avg_loss_p1 = sum(recent_losses_p1) / len(recent_losses_p1) if recent_losses_p1 else 0
            avg_loss_p2 = sum(recent_losses_p2) / len(recent_losses_p2) if recent_losses_p2 else 0
            print(f"Average loss over last {20} episodes: Player 1: {avg_loss_p1:.4f}, Player 2: {avg_loss_p2:.4f}")
            recent_losses_p1 = []
            recent_losses_p2 = []

        if episode > START_TRAINING_EPISODE and episode % UPDATE_MODEL_INTERVAL == 0:
            agent.update_target_model(agent.model1, agent.target_model1)
            agent.update_target_model(agent.model2, agent.target_model2)
            
        if episode % LOGGING_INTERVAL == 0 and episode > 0:
            print(f"Player 1 wins: {player1_wins}, Player 2 wins: {player2_wins}")
            player1_wins = 0
            player2_wins = 0
        gomokuEnv.reset()
        gomokuEnv.first_move()
        total_turns = 1
        completed = False

        if episode > START_TRAINING_EPISODE and episode % EPSILON_DECAY_INTERVAL == 0 and agent.epsilon > agent.epsilon_min:
            agent.epsilon *= agent.epsilon_decay
            agent.epsilon = max(agent.epsilon, agent.epsilon_min)

        move_count = 0

        # Buffer to hold samples for both players
        # Will be pushed to memory after the game ends
        player1_samples = []
        player2_samples = []
        winner = None

        while move_count < 150:  # 15x15 board, max 150 turns
            # gui.draw_board(gomokuEnv.state[0])
            current_turn = total_turns & 1  

            action = agent.get_action(gomokuEnv, current_turn)

            # Take the action in the environment
            current_state_record, next_state_record, completed, reward, valid_move = gomokuEnv.step(action, current_turn)
            if current_turn == 0:
                player1_samples.append((current_state_record, action, reward, next_state_record, completed))
            else:
                player2_samples.append((current_state_record, action, reward, next_state_record, completed))

            if not valid_move:
                continue

            gomokuEnv.update_state(current_turn, action)

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
                    negative_reward = -1 * ((relative_i + 1) / denom)
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
                        positive_reward = 1 * ((relative_i + 1) / denom)
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
                    negative_reward = -1 * ((relative_i + 1) / denom)
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
                        positive_reward = 1 * ((relative_i + 1) / denom)
                        player2_samples[i] = (
                            player2_samples[i][0],
                            player2_samples[i][1],
                            positive_reward,
                            player2_samples[i][3],
                            player2_samples[i][4],
                        )
        
        # Add samples to the agent's memory
        for i, sample in enumerate(player1_samples):
            agent.save_sample(sample, turn=0)
            if episode > START_TRAINING_EPISODE:
                # Train the models
                loss = agent.train_model(agent.model1, agent.target_model1, agent.memory1, agent.optimizer1)
                if loss is not None:
                    recent_losses_p1.append(loss)
        for i, sample in enumerate(player2_samples):
            agent.save_sample(sample, turn=1)
            if episode > START_TRAINING_EPISODE:
                # Train the models
                loss = agent.train_model(agent.model2, agent.target_model2, agent.memory2, agent.optimizer2)
                if loss is not None:
                    recent_losses_p2.append(loss)

    agent.save_model()