
import os
import sys
from agent import Agent
from game import Game
import torch
import argparse
import multiprocessing as mp
import pandas as pd

def run_self_play(game: Game, agent: Agent):
    """Run a self-play game and return the collected samples."""
    # Reset the game state and make the first move
    game.reset()
    game.first_move()
    total_turns = 1

    move_count = 1

    # Collect samples for both players
    samples = {0: [], 1: []}

    winner = None
    while move_count < 81:  # 9x9 board, max 80 turns
        # winner = game.check_winner(current_turn, action)
        winner, _ = game.get_winner_indirect()
        if winner is not None:
            # fuck this 
            break

        current_turn = total_turns & 1

        # Get the current board state from the perspective of the current player
        # Copy the current board state to avoid modifying the original state
        current_board = game.state[current_turn]
        current_board = current_board.copy()

        action, act_probs = agent.get_action_and_probs(game, current_turn)

        value_place = 0

        # game.update_state(current_turn, action)
        move = divmod(action, 9)
        game.do_move(move)

        samples[current_turn].append((current_board, act_probs, value_place))

        move_count += 1
        total_turns += 1

    final_samples = []
    # based on the winner, assign values (1 for win, -1 for loss, 0 for draw) to each sample
    if winner is not None:
        if winner == 0:
            for sample in samples[0]:
                sample = (sample[0], sample[1], 1)
                final_samples.append(sample)
            for sample in samples[1]:
                sample = (sample[0], sample[1], -1)
                final_samples.append(sample)
        elif winner == 1:
            for sample in samples[0]:
                sample = (sample[0], sample[1], -1)
                final_samples.append(sample)
            for sample in samples[1]:
                sample = (sample[0], sample[1], 1)
                final_samples.append(sample)
        else:
            # draw
            for sample in samples[0]:
                sample = (sample[0], sample[1], 0)
                final_samples.append(sample)
            for sample in samples[1]:
                sample = (sample[0], sample[1], 0)
                final_samples.append(sample)
    else:
        for sample in samples[0]:
            sample = (sample[0], sample[1], 0)
            final_samples.append(sample)
        for sample in samples[1]:
            sample = (sample[0], sample[1], 0)
            final_samples.append(sample)

    return final_samples

def self_play_worker(game: Game, agent: Agent, episodes, queue, worker_id):
    """Each worker generates self-play data."""
    for _ in range(episodes):
        samples = run_self_play(game, agent)
        queue.put(samples)  # Send samples to the main process
    print(f"Worker {worker_id} finished {episodes} episodes")

def run_and_learn_parallel(game, agent, max_episodes, num_workers, start_training_samples=500, saving_interval=1):
    """Run self-play in parallel using multiple workers and train the agent."""
    ctx = mp.get_context("spawn") 
    queue = ctx.Queue()

    # Divide episodes across workers
    episodes_per_worker = max_episodes // num_workers
    workers = []
    for i in range(num_workers):
        p = ctx.Process(target=self_play_worker, args=(game, agent, episodes_per_worker, queue, i))
        p.start()
        workers.append(p)

    # Training loop runs in the main process
    samples_collected = 0
    episodes_played = 0
    losses = []
    policy_losses = []
    value_losses = []
    entropies = []
    while episodes_played < max_episodes:
        for sample in queue.get():
            agent.save_sample(sample)
            # train the agent when enough samples are collected
            if samples_collected >= start_training_samples:
                loss, policy_loss, value_loss, entropy = agent.train_step(batch_size=512, num_epochs=5)
                losses.append(loss)
                policy_losses.append(policy_loss)
                value_losses.append(value_loss)
                entropies.append(entropy)
            else:
                print(f"Collected sample {samples_collected + 1}, not training yet")
            if samples_collected > 0 and samples_collected % saving_interval == 0:
                agent.save_memory()
                agent.save_model()
                print(f"Memory and model saved at sample {samples_collected} after {episodes_played + 1} episodes")
            samples_collected += 1
        # Calculate average loss and entropy
        avg_loss = sum(losses) / len(losses) if losses else 0
        avg_policy_loss = sum(policy_losses) / len(policy_losses) if policy_losses else 0
        avg_value_loss = sum(value_losses) / len(value_losses) if value_losses else 0
        avg_entropy = sum(entropies) / len(entropies) if entropies else 0
        save_loss_entropy(avg_loss, avg_policy_loss, avg_value_loss, avg_entropy, episodes_played)
        losses = []
        policy_losses = []
        value_losses = []
        entropies = []
        episodes_played += 1

    # Wait for workers
    for p in workers:
        p.join()

    print(f"Training completed. Total samples collected: {samples_collected}")
    agent.save_memory()
    agent.save_model()


def save_loss_entropy(loss, policy_loss, value_loss, entropy, episode):
    df = pd.DataFrame({
        "loss": [loss],
        "policy_loss": [policy_loss],
        "value_loss": [value_loss],
        "entropy": [entropy]
    })
    write_header = not os.path.exists("loss_entropy.csv")
    df.to_csv("loss_entropy.csv", mode="a", header=write_header, index=False)

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    parser = argparse.ArgumentParser(description="Gomoku Training Script")

    parser.add_argument("--load_model", type=lambda x: (str(x).lower() == 'true'), default=False, help="Load existing model or not")
    parser.add_argument("--load_memory", type=lambda x: (str(x).lower() == 'true'), default=False, help="Load memory from file or not")
    parser.add_argument("--start_training_samples", type=int, default=1, help="Samples to start training")
    parser.add_argument("--mcts_iterations", type=int, default=500, help="Number of MCTS iterations")
    parser.add_argument("--max_episodes", type=int, default=1000, help="Number of episodes to train")
    parser.add_argument("--saving_interval", type=int, default=100, help="Number of episodes to save the model")
    parser.add_argument("--num_workers", type=int, default=1, help="Number of parallel workers for self-play")


    args = parser.parse_args()
    START_TRAINING_SAMPLES = args.start_training_samples  # Start training after specified samples
    SAVING_INTERVAL = args.saving_interval  # Save model every 100 episodes
    LOAD_MODEL = args.load_model  # Load existing model or not
    LOAD_MEMORY = args.load_memory  # Load memory from file or not
    MAX_EPISODES = args.max_episodes  # Set a high number of episodes for training
    MCTS_ITERATIONS = args.mcts_iterations  # Number of MCTS iterations
    NUM_WORKERS = args.num_workers  # Number of parallel workers for self-play

    # Setting up the environment and agent
    game = Game()
    agent = Agent(device=device, learning_rate=0.003, mcts_iterations=MCTS_ITERATIONS)

    if LOAD_MODEL:
        agent.load_model()
    if LOAD_MEMORY:
        agent.load_memory()

    

    #game, agent, max_episodes, num_workers=4, start_training_samples=500, saving_interval=1
    print(f"Starting training from {START_TRAINING_SAMPLES} with max episodes {MAX_EPISODES} and saving interval {SAVING_INTERVAL}")
    run_and_learn_parallel(game, agent, MAX_EPISODES, NUM_WORKERS, START_TRAINING_SAMPLES, SAVING_INTERVAL)


"""

python train.py --load_model False --load_memory False --start_training_samples 50 --mcts_iterations 800 --max_episodes 1000 --saving_interval 200

"""