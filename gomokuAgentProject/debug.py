import numpy as np
import copy
from game import Game
from gui_play import GUI
from policy_value_network import PolicyValueNetwork
from mcts import MCTS  # assuming your Node and MCTS classes are in mcts.py

import sys
import pygame

# Open a file in write mode
log_file = open("output.log", "w")

# Redirect stdout to the file
sys.stdout = log_file

# board_array = np.array([
#  [ 0.,  1.,  1., -1., -1.,  0.,  1.,  0.,  0.],
#  [ 0., -1.,  0.,  0.,  0.,  0., -1., -1.,  1.],
#  [ 0.,  0.,  1.,  0.,  0.,  0., -1., -1.,  0.],
#  [-1.,  0.,  0.,  0.,  1.,  0.,  0.,  0.,  0.],
#  [ 0.,  0.,  0.,  0., -1., -1., -1.,  0.,  0.],
#  [ 0.,  1.,  1., -1.,  0.,  0.,  0.,  0.,  0.],
#  [ 0.,  0.,  0.,  0.,  0.,  0.,  0.,  0., -1.],
#  [ 0.,  1.,  0.,  1.,  0.,  1.,  0.,  0.,  0.],
#  [ 1.,  0., -1.,  0.,  0.,  0.,  1.,  0.,  0.]
# ])

board_array = np.array([
    [ 0.,  1.,  1., -1., -1.,  0.,  0.,  0.,  0.],
    [ 0., -1.,  0.,  0.,  0.,  0., -1., -1.,  1.],
    [ 0.,  0.,  1.,  0.,  0.,  0.,  0., -1.,  0.],
    [-1.,  0.,  0.,  0.,  1.,  0.,  0.,  0.,  0.],
    [ 0.,  0., -1.,  0., -1., -1., -1.,  0.,  0.],
    [ 0.,  1.,  1.,  0.,  0.,  0.,  0.,  0.,  0.],
    [ 0.,  0.,  0.,  0.,  0.,  0.,  0.,  0., -1.],
    [ 0.,  1.,  1.,  0.,  0.,  1.,  0.,  0.,  0.],
    [ 1.,  0., -1.,  0.,  0.,  0.,  1.,  0.,  0.]
])

game = Game()

game.current_player = 1

game.state[0] = -board_array
game.state[1] = board_array

game.move_history = [r * 9 + c for r in range(9) for c in range(9) if board_array[r, c] != 0]

policy_net = PolicyValueNetwork()
# policy_net.load()

mcts = MCTS(policy_net, iterations=7000)  
action_probs = mcts.get_action_probs(game, turn=game.current_player)

# --- 4. Display results ---
print("MCTS action probabilities (flattened 9x9 board):")
print(action_probs.reshape(9, 9))

best_action = np.argmax(action_probs)
best_row, best_col = divmod(best_action, 9)
print(f"Best move according to MCTS: ({best_row}, {best_col})")

sys.stdout = sys.__stdout__
log_file.close()

gui = GUI()
pygame.init()
gui.draw_board_with_probs(board_array, action_probs)
while True:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            pygame.quit()
            sys.exit()
