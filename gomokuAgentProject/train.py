
from agent import DQNAgent
from env import GomokuEnv
import pygame
from gui_play import GUI

if __name__ == "__main__":
    gomokuEnv = GomokuEnv()
    agent = DQNAgent()

    max_episodes = 10
    save_model_interval = 10

    gui = GUI()
    for episode in range(max_episodes):
        print(f"Episode {episode + 1}/{max_episodes}")
        gomokuEnv.reset()
        gomokuEnv.first_move()
        total_turns = 1
        completed = False

        if agent.epsilon > agent.epsilon_min:
            agent.epsilon *= agent.epsilon_decay
            print(f"Epsilon decayed to {agent.epsilon}")
            agent.epsilon = max(agent.epsilon, agent.epsilon_min)

        while not completed:
            # gui.draw_board(gomokuEnv.state[0])
            current_turn = total_turns % 2
            action = agent.get_action(gomokuEnv.state, current_turn)
            # print(f"Action chosen: {action} by player {current_turn}")

            # Take the action in the environment
            next_state, completed, reward, valid_move = gomokuEnv.step(action, current_turn)
            agent.save_sample((gomokuEnv.state[current_turn], action, reward, next_state, completed), current_turn)

            if not valid_move:
                # print("Invalid move, trying again.")
                continue
            
            total_turns += 1

            # Update the agent's model
            if current_turn == 0 and len(agent.memory1) >= agent.batch_size:
                agent.train_model(agent.model1, agent.target_model1, agent.memory1, completed)
            elif current_turn == 1 and len(agent.memory2) >= agent.batch_size:
                agent.train_model(agent.model2, agent.target_model2, agent.memory2, completed)

            if completed:
                print(f"Game completed after {total_turns} turns.")
                # gui.draw_board(gomokuEnv.state[0])
                break
        gui.draw_board(gomokuEnv.state[0])
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                gui.quit()
        
        if (episode + 1) % save_model_interval == 0:
            agent.save_model()
            print(f"Model saved at episode {episode + 1}")  

    gui.quit()