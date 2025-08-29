from agent import Agent
from gui_play import GUI
from game import Game
import pygame
import sys
import numpy as np

if __name__ == '__main__':
    gui = GUI()
    game = Game()

    agent = Agent()
    agent.load_memory()  # Load the agent's memory if available

    total_samples = len(agent.memory)
    print(total_samples, "samples loaded from memory")

    # Ask user for a starting index
    try:
        start_index = int(input(f"Enter starting memory index (0 - {total_samples - 1}): "))
    except ValueError:
        start_index = 0
    index = max(0, min(start_index, total_samples - 1))

    # Initialize pygame properly
    pygame.init()

    # Function to show sample
    def show_sample(idx):
        current_board, act_probs, value_place = agent.memory[idx]
        board = np.array(current_board).reshape(9, 9)
        act_probs = np.array(act_probs).reshape(9, 9)
        gui.draw_board_with_probs(board, act_probs)
        print(f"Sample {idx}/{total_samples - 1}")
        print(f"Value place: {value_place}")
        print(f"Action probabilities:\n{act_probs}")
        print(f"Board state:\n{board}")

    # Show first chosen board
    show_sample(index)

    while True:
        event = pygame.event.wait()  # Wait for key press or quit

        if event.type == pygame.QUIT:
            pygame.quit()
            sys.exit()

        elif event.type == pygame.KEYDOWN:
            if event.key == pygame.K_RIGHT:
                index = min(index + 1, total_samples - 1)
                show_sample(index)
            elif event.key == pygame.K_LEFT:
                index = max(index - 1, 0)
                show_sample(index)
