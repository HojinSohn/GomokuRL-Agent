import copy
import sys
from game import Game
from agent import Agent
from mcts import MCTS
from policy_value_network import Model
import pygame
from gui_play import GUI
import torch
import numpy as np

if __name__ == "__main__":
    gui = GUI()
    game = Game()
    turn = 0  # Start with player 1
    agent = Agent(load_model=True) 

    agent.load_model()
    
    done = False
    while not done:
        if turn == 0:
            game.first_move()
            print("Player 1 (Black) starts the game")
            turn += 1
            gui.draw_board(game.state[0])
            continue
        if turn % 2 == 0:
            # Agent's turn
            print("Agent 1's turn")
            print("Current state:", game.state[0])
            _, action_probs = agent.get_action_and_probs(game, 0)
            # pick the most probable action
            action = np.argmax(action_probs)
            game.update_state(0, action)
            row, col = divmod(action, 9)
            action_probs = action_probs.reshape(9, 9)
            print(action_probs)
            print(f"Agent 1 placed stone at ({row}, {col}) with probability {action_probs[row, col]}")
            gui.draw_board_with_probs(game.state[0], action_probs)
            win = game.check_winner(0, action)
            if win:
                print("Agent 1 wins!")
                done = True
            turn += 1
        else:
            print("Waiting for player's move...")

            move_made = False
            while not move_made:
                for event in pygame.event.get():
                    if event.type == pygame.QUIT:
                        gui.quit()
                        pygame.quit()
                        sys.exit()
                    elif event.type == pygame.MOUSEBUTTONDOWN and not move_made:
                        print("Mouse button pressed")
                        if event.button == 1:  # Left click
                            x, y = event.pos
                            row = (y - gui.MARGIN) // gui.CELL_SIZE
                            col = (x - gui.MARGIN) // gui.CELL_SIZE
                            action = row * 9 + col
                            print(f"Player clicked on cell ({row}, {col})")
                            game.update_state(1, action)
                            move_made = True  # <== Exit the wait loop
                            gui.draw_board(game.state[0])
                            win = game.check_winner(1, action)
                            if win:
                                print("Player wins!")
                                done = True
                            turn += 1

                    elif event.type == pygame.KEYDOWN:
                        if event.key == pygame.K_ESCAPE:
                            gui.quit()
                            pygame.quit()
                            sys.exit()
                        elif event.key == pygame.K_r:  # Reset the board
                            state = [[0 for _ in range(9)] for _ in range(9)]
                            gui.draw_board(state)
                            turn = 0
                            move_made = True  # Exit the wait loop so main loop can restart

                pygame.time.delay(100)
                pygame.display.flip()


        pygame.time.delay(100)
        pygame.display.flip()