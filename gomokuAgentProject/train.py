
import sys
from agent import DQNAgent
from env import GomokuEnv
import torch
import argparse

def collect_episodes(gomokuEnv: GomokuEnv, agent: DQNAgent, max_episodes: int):
    print("Collecting data for training...")
    player1_wins = 0
    player2_wins = 0

    for episode in range(max_episodes):
        # Print progress every 100 episodes for checking the progress
        if episode < 100 and episode % 10 == 0:
            print(f"Episode {episode} - Simulating game...")

        # If in data collection mode, save memory every 100 episodes
        if episode >= 99 and (episode + 1) % 500 == 0:
            print(f"Saving memory at episode {episode}")
            print(f"len(agent.memory1): {len(agent.memory1)}")
            print(f"len(agent.memory2): {len(agent.memory2)}")
            agent.save_memory()
            # clear memory to avoid memory overflow in the data collection mode
            agent.memory1.clear() 
            agent.memory2.clear()
            
        # reset the environment for a new episode
        gomokuEnv.reset()
        gomokuEnv.first_move()
        total_turns = 1
        completed = False

        # Buffer to hold samples for both players
        # Will be pushed to memory after the game ends
        player1_samples = []
        player2_samples = []
        winner = None

        # toggle between 0 and 1 to represent which player plays random move (not tactical) each episode
        not_tactical_player = episode & 1

        while total_turns < 70:  # 9x9 board, max 70 turns
            # gui.draw_board(gomokuEnv.state[0])
            current_turn = total_turns & 1  

            # check if the player at the current turn is not tactical
            is_not_tactical = not_tactical_player == current_turn

            action = agent.get_action(gomokuEnv, current_turn, is_not_tactical)

            # Take the action in the environment
            current_state_record, next_state_record, completed, reward, valid_move = gomokuEnv.step(action, current_turn)
            if current_turn == 0:
                player1_samples.append((current_state_record, action, reward, next_state_record, completed))
            else:
                player2_samples.append((current_state_record, action, reward, next_state_record, completed))

            if not valid_move:
                continue

            # Update the environment state with the action taken
            gomokuEnv.update_state(current_turn, action)

            total_turns += 1
            
            # if the game is completed, determine the winner
            if completed:
                if current_turn == 0:
                    player1_wins += 1
                    winner = 0
                else:
                    player2_wins += 1
                    winner = 1
                break

        # Adjust rewards based on the winner
        # Last 5 moves get a higher reward if they were moves from winner, and a negative reward if they were from loser
        # This is to encourage the agent to learn from its mistakes
        if winner is not None:
            if winner == 0:
                num_moves = len(player2_samples)
                bad_move_start = max(0, num_moves - 5)  # Start from the last 5 moves
                denom = max(1, num_moves - bad_move_start)  # Avoid division by zero
                for i in range(bad_move_start, num_moves):
                    # Scale linearly: earlier moves get smaller negative reward
                    relative_i = i - bad_move_start
                    negative_reward = -2 * ((relative_i + 1) / denom)
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
                        positive_reward = 2 * ((relative_i + 1) / denom)
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
                    negative_reward = -2 * ((relative_i + 1) / denom)
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
                        positive_reward = 2 * ((relative_i + 1) / denom)
                        player2_samples[i] = (
                            player2_samples[i][0],
                            player2_samples[i][1],
                            positive_reward,
                            player2_samples[i][3],
                            player2_samples[i][4],
                        )
        
        # Add updated samples to the agent's memory
        for i, sample in enumerate(player1_samples):
            agent.save_sample(sample, turn=0)
        for i, sample in enumerate(player2_samples):
            agent.save_sample(sample, turn=1)
    
    # save the leftover memory after all episodes
    agent.save_memory()

    print(f"Data collection completed. Player 1 wins: {player1_wins}, Player 2 wins: {player2_wins}")

def train_agent(gomokuEnv: GomokuEnv, agent: DQNAgent, num_training: int, update_model_interval: int):
    print("Training DQN Agent: Size of memory1:", len(agent.memory1), "Size of memory2:", len(agent.memory2))
    loss1_sum = 0
    loss2_sum = 0
    for i in range(num_training):
        if i % update_model_interval == 0:
            agent.update_target_model(agent.model1, agent.target_model1)
            agent.update_target_model(agent.model2, agent.target_model2)
            print(f"Updated target models at training iteration {i}")
        loss1 = agent.train_model(agent.model1, agent.target_model1, agent.memory1, agent.optimizer1)
        loss2 = agent.train_model(agent.model2, agent.target_model2, agent.memory2, agent.optimizer2)
        if loss1 is not None:
            loss1_sum += loss1
        if loss2 is not None:
            loss2_sum += loss2
        if i % 100 == 0:
            print(f"Training iteration {i}: Loss1: {loss1_sum / 100 if i > 0 else loss1}, Loss2: {loss2_sum / 100 if i > 0 else loss2}")
            loss1_sum = 0
            loss2_sum = 0

    print("Training completed. Final model saved.")
    agent.save_model()

def run_and_learn(gomokuEnv: GomokuEnv, agent: DQNAgent, max_episodes: int, update_model_interval: int, epsilon_decay_interval: int, start_training_episode, saving_interval= 100):
    recent_losses_p1 = []
    recent_losses_p2 = []
    player1_wins = 0
    player2_wins = 0

    for episode in range(max_episodes):
        # Log the progress every saving_interval episodes after the start training episode
        if episode > start_training_episode and episode % saving_interval == 0:
            # checkpoint the model
            print(f"Saving model at episode {episode}")
            agent.save_model()
            # save the memory in the file for later use
            agent.save_memory()
            avg_loss_p1 = sum(recent_losses_p1) / len(recent_losses_p1) if recent_losses_p1 else 0
            avg_loss_p2 = sum(recent_losses_p2) / len(recent_losses_p2) if recent_losses_p2 else 0
            print(f"Average loss over last {saving_interval} episodes: Player 1: {avg_loss_p1:.4f}, Player 2: {avg_loss_p2:.4f}")
            recent_losses_p1 = []
            recent_losses_p2 = []

        # Update target model every update_model_interval episodes after the start training episode
        if episode > start_training_episode and episode % update_model_interval == 0:
            agent.update_target_model(agent.model1, agent.target_model1)
            agent.update_target_model(agent.model2, agent.target_model2)

        gomokuEnv.reset()
        gomokuEnv.first_move()
        total_turns = 1
        completed = False

        # Decay epsilon after the start training episode
        if episode > start_training_episode and episode % epsilon_decay_interval == 0 and agent.epsilon > agent.epsilon_min:
            agent.epsilon *= agent.epsilon_decay
            agent.epsilon = max(agent.epsilon, agent.epsilon_min)

        move_count = 0

        # Buffer to hold samples for both players
        # Will be pushed to memory after the game ends
        player1_samples = []
        player2_samples = []
        winner = None

        # toggle between 0 and 1 to represent which player plays random move (not tactical) each episode
        not_tactical_player = episode & 1

        while move_count < 70:  # 9x9 board, max 70 turns
            # gui.draw_board(gomokuEnv.state[0])
            current_turn = total_turns & 1  

            action = agent.get_action(gomokuEnv, current_turn, not_tactical_player)

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
                    negative_reward = -2 * ((relative_i + 1) / denom)
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
                        positive_reward = 2 * ((relative_i + 1) / denom)
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
                    negative_reward = -2 * ((relative_i + 1) / denom)
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
                        positive_reward = 2 * ((relative_i + 1) / denom)
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
            if episode > start_training_episode:
                # Train the models
                loss = agent.train_model(agent.model1, agent.target_model1, agent.memory1, agent.optimizer1)
                if loss is not None:
                    recent_losses_p1.append(loss)
        for i, sample in enumerate(player2_samples):
            agent.save_sample(sample, turn=1)
            if episode > start_training_episode:
                # Train the models
                loss = agent.train_model(agent.model2, agent.target_model2, agent.memory2, agent.optimizer2)
                if loss is not None:
                    recent_losses_p2.append(loss)

    agent.save_model()

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # get arg to check whether to use GUI or not
    print(f"Using device: {device}")

    parser = argparse.ArgumentParser(description="Gomoku Training Script")

    parser.add_argument("--init_epsilon", type=float, default=0.95, help="Initial exploration rate")
    parser.add_argument("--load_model", type=lambda x: (str(x).lower() == 'true'), default=False, help="Load existing model or not")
    parser.add_argument("--max_episodes", type=int, default=10000, help="Number of episodes to train the agent")
    parser.add_argument("--epsilon_decay_interval", type=int, default=100, help="Number of episodes to decay epsilon")
    parser.add_argument("--load_memory", type=lambda x: (str(x).lower() == 'true'), default=False, help="Load memory from file or not")
    parser.add_argument("--start_training_episode", type=int, default=500, help="Episode to start training")
    parser.add_argument("--num_training", type=int, default=1000, help="Number of training iterations")
    parser.add_argument("--updating_interval", type=int, default=200, help="Number of episodes to update the target model")
    parser.add_argument("--mode", type=str, choices=["data_collection", "training", "both"], default="training", help="Run in data collection mode or training mode")


    args = parser.parse_args()
    MODE = args.mode
    UPDATE_MODEL_INTERVAL = args.updating_interval
    LOGGING_INTERVAL = 100  # Print every 100 episodes
    START_TRAINING_EPISODE = args.start_training_episode  # Start training after specified episode
    EPSILON_DECAY_INTERVAL = args.epsilon_decay_interval  # Decay epsilon every specified interval
    SAVING_INTERVAL = 100  # Save model every 100 episodes
    INIT_EPSILON = args.init_epsilon  # Initial exploration rate
    LOAD_MODEL = args.load_model  # Load existing model or not
    LOAD_MEMORY = args.load_memory
    MAX_EPISODES = args.max_episodes  # Set a high number of episodes for training
    NUM_TRAINING = args.num_training  # Number of training iterations

    # Setting up the environment and agent
    gomokuEnv = GomokuEnv()
    agent = DQNAgent(device=device, init_epsilon=INIT_EPSILON, load_model=LOAD_MODEL, load_memory=LOAD_MEMORY)

    if MODE == "data_collection":
        # data collection mode
        collect_episodes(gomokuEnv, agent, MAX_EPISODES)
    elif MODE == "training":
        # train mode
        print("Starting Gomoku training...")
        train_agent(gomokuEnv, agent, NUM_TRAINING, UPDATE_MODEL_INTERVAL)
    elif MODE == "both":
        print("Starting Gomoku training...")
        run_and_learn(gomokuEnv, agent, max_episodes=MAX_EPISODES, update_model_interval=UPDATE_MODEL_INTERVAL, epsilon_decay_interval=EPSILON_DECAY_INTERVAL, start_training_episode=START_TRAINING_EPISODE, saving_interval=SAVING_INTERVAL)
