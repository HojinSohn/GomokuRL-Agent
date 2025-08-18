


from agent import DQNAgent
from gui_play import GUI
from game import Game
import pygame
import sys
import numpy as np

if __name__ == '__main__':
    gui = GUI()
    game = Game()
    turn = 0  # Start with player 1

    dqnAgent = DQNAgent()
    dqnAgent.load_memory()  # Load the agent's memory if available 

    print(len(dqnAgent.memory1), "samples loaded from memory")

    for i, sample in enumerate(dqnAgent.memory1):
        state, action, reward, next_state, done = sample[0], sample[1], sample[2], sample[3], sample[4]
        board = np.array(state)
        next_board = np.array(next_state)
        print(f"Sample state {i}: {board}")
        print(f"Sample next state {i}: {next_board}")
        print(f"Action: {action}, Reward: {reward}, Done: {done}")

        # Handle pygame events first
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()
        
        pygame.time.wait(1000)
        gui.draw_board(board)  # Draw the board
        
        # Wait 1 second using clock (non-blocking)
        pygame.time.wait(100)
        gui.draw_board(next_board)  # Draw the board