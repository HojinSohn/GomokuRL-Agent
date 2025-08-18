
import sys
from agent import Agent
from game import Game
import torch
import argparse

def run_and_learn(game: Game, agent: Agent, max_episodes: int, starting_episode: int, saving_interval: int):
    losses = []
    entropies = []
    player1_wins = 0
    player2_wins = 0

    for episode in range(max_episodes):
        if (episode + 1) % saving_interval == 0:
            agent.save_memory()
            agent.save_model()
            print(f"Memory and model saved at episode {episode + 1}")

        if episode + 1 < starting_episode:
            # log the episode number
            print(f"Episode {episode + 1}/{max_episodes}")

        game.reset()
        game.first_move()
        total_turns = 1
        completed = False

        move_count = 0

        samples = {0: [], 1: []}

        winner = None
        while move_count < 70:  # 9x9 board, max 70 turns
            # gui.draw_board(game.state[0])
            current_turn = total_turns & 1  

            current_board = game.state[current_turn]

            action, act_probs = agent.get_action_and_probs(game, current_turn)

            value_place = 0

            game.update_state(current_turn, action)

            winner = game.check_winner(current_turn, action)

            if winner:
                completed = True

            samples[current_turn].append((current_board, act_probs, value_place))

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
        
        final_samples = []
        if winner is not None:
            if winner == 0:
                for sample in samples[0]:
                    sample = (sample[0], sample[1], 1)
                    final_samples.append(sample)
                for sample in samples[1]:
                    sample = (sample[0], sample[1], -1)
                    final_samples.append(sample)
            else:
                for sample in samples[0]:
                    sample = (sample[0], sample[1], -1)
                    final_samples.append(sample)
                for sample in samples[1]:
                    sample = (sample[0], sample[1], 1)
                    final_samples.append(sample)
        else:
            for sample in samples[0]:
                sample = (sample[0], sample[1], 0)
                final_samples.append(sample)
            for sample in samples[1]:
                sample = (sample[0], sample[1], 0)
                final_samples.append(sample)

        for sample in final_samples:
            agent.save_sample(sample)
            if episode >= starting_episode:
                loss, entropy = agent.train_step()
                losses.append(loss)
                entropies.append(entropy)
                # Print loss and entropy every 500 training steps
                if len(losses) >= 500:
                    avg_loss = sum(losses) / 500
                    avg_entropy = sum(entropies) / 500
                    print(f"Episode {episode + 1}, Loss: {avg_loss:.4f}, Entropy: {avg_entropy:.4f}")
                    losses = []
                    entropies = []

    agent.save_memory()
    agent.save_model()
    print(f"Training completed. Player 1 wins: {player1_wins}, Player 2 wins: {player2_wins}")

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # get arg to check whether to use GUI or not
    print(f"Using device: {device}")

    parser = argparse.ArgumentParser(description="Gomoku Training Script")

    parser.add_argument("--load_model", type=lambda x: (str(x).lower() == 'true'), default=False, help="Load existing model or not")
    parser.add_argument("--load_memory", type=lambda x: (str(x).lower() == 'true'), default=False, help="Load memory from file or not")
    parser.add_argument("--start_training_episode", type=int, default=500, help="Episode to start training")
    parser.add_argument("--mcts_iterations", type=int, default=500, help="Number of MCTS iterations")
    parser.add_argument("--max_episodes", type=int, default=1000, help="Number of episodes to train")
    parser.add_argument("--saving_interval", type=int, default=100, help="Number of episodes to save the model")


    args = parser.parse_args()
    START_TRAINING_EPISODE = args.start_training_episode  # Start training after specified episode
    SAVING_INTERVAL = args.saving_interval  # Save model every 100 episodes
    LOAD_MODEL = args.load_model  # Load existing model or not
    LOAD_MEMORY = args.load_memory  # Load memory from file or not
    MAX_EPISODES = args.max_episodes  # Set a high number of episodes for training
    MCTS_ITERATIONS = args.mcts_iterations  # Number of MCTS iterations

    # Setting up the environment and agent
    game = Game()
    # device=torch.device("cpu"), learning_rate=0.001, load_model=False, load_memory=False, mcts_iterations=10000
    agent = Agent(device=device, learning_rate=0.001, load_model=LOAD_MODEL, load_memory=LOAD_MEMORY, mcts_iterations=MCTS_ITERATIONS)

    if LOAD_MODEL:
        agent.policy_value_network.load_model()
    if LOAD_MEMORY:
        agent.load_memory()

    print(f"Starting training from episode {START_TRAINING_EPISODE} with max episodes {MAX_EPISODES} and saving interval {SAVING_INTERVAL}")
    run_and_learn(game, agent, MAX_EPISODES, START_TRAINING_EPISODE, SAVING_INTERVAL)


"""

python train.py --load_model False --load_memory False --start_training_episode 50 --mcts_iterations 800 --max_episodes 1000 --saving_interval 200

"""