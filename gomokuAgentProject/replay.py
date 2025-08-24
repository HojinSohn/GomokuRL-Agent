


from agent import Agent
from gui_play import GUI
from game import Game
import pygame
import sys
import numpy as np

if __name__ == '__main__':
    gui = GUI()
    game = Game()
    turn = 0  # Start with player 1

    agent = Agent()
    agent.load_memory()  # Load the agent's memory if available

    print(len(agent.memory), "samples loaded from memory")

    for i, sample in enumerate(agent.memory):
        current_board, act_probs, value_place = sample
        board = np.array(current_board)
        print(f"Sample state {i}: {board}")
        print(f"Action probabilities:\n {act_probs}, Value place: {value_place}")

        # Handle pygame events first
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()
        
        pygame.time.wait(100)
        gui.draw_board(board)  # Draw the board
        