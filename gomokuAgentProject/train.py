
from agent import DQNAgent
from env import GomokuEnv
import pygame
import torch
from gui_play import GUI

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    PRINT_INTERVAL = 20  # Print every 20 episodes
    # get arg to check whether to use GUI or not
    use_gui = False  # Change this to False if you want to run without GUI
    print(f"Using device: {device}")
    gomokuEnv = GomokuEnv()
    agent = DQNAgent(device=device)

    max_episodes = 5000
    save_model_interval = 10

    gui = None
    if use_gui:
        gui = GUI()
    for episode in range(max_episodes):
        if episode % PRINT_INTERVAL == 0:
            print(f"Episode {episode}/{max_episodes} completed. Epsilon: {agent.epsilon:.4f}")
        gomokuEnv.reset()
        gomokuEnv.first_move()
        total_turns = 1
        completed = False

        if agent.epsilon > agent.epsilon_min:
            agent.epsilon *= agent.epsilon_decay
            agent.epsilon = max(agent.epsilon, agent.epsilon_min)

        try_count = 0

        while not completed:
            # gui.draw_board(gomokuEnv.state[0])
            current_turn = total_turns % 2
            action = agent.get_action(gomokuEnv.state, current_turn)
            # print(f"Action chosen: {action} by player {current_turn}")

            # Take the action in the environment
            next_state, completed, reward, valid_move = gomokuEnv.step(action, current_turn)
            agent.save_sample((gomokuEnv.state[current_turn], action, reward, next_state, completed), current_turn)

            if not valid_move:
                try_count += 1
                if try_count > 5:
                    print("Too many invalid moves, resetting the game. Episode:", episode, "Turns:", total_turns, "Reward:", reward)
                    agent.save_sample((gomokuEnv.state[current_turn], action, reward, next_state, True), current_turn)
                    break
                continue
            
            total_turns += 1
            try_count = 0

            # Update the agent's model
            if current_turn == 0 and len(agent.memory1) >= agent.batch_size:
                agent.train_model(agent.model1, agent.target_model1, agent.memory1)
            elif current_turn == 1 and len(agent.memory2) >= agent.batch_size:
                agent.train_model(agent.model2, agent.target_model2, agent.memory2)

            if completed:
                # print(f"Game completed after {total_turns} turns.")
                # gui.draw_board(gomokuEnv.state[0])
                break
        if use_gui:
            gui.draw_board(gomokuEnv.state[0])
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    gui.quit()
    
    agent.save_model()
    print(f"Model saved at episode {episode + 1}")  

    if use_gui:
        gui.quit()