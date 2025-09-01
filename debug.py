import numpy as np
import copy
from game import Game
from gui_play import GUI
from policy_value_network import PolicyValueNetwork
from mcts import MCTS 

import sys
import pygame


board_array = np.array([
    [0., 0., 0., 0., 0., 0., 0., 0., 0.],
    [0., 0., 0., 0., 0., 0., 0., 0., 0.],
    [0., 0., 0., 0., 0., 0., 0., 0., 0.],
    [0., 0., 0., 1., 0., 0., 0., 0., 0.],
    [0., 0., 0., -1., 1., 1., 0., 0., 0.],
    [0., 0., 0., -1., -1., -1., 0., 0., 0.],
    [0., 0., 0., 0., 0., 0., 1., 0., 0.],
    [0., 0., 0., 0., 0., 0., 0., 0., 0.],
    [0., 0., 0., 0., 0., 0., 0., 0., 0.]
])


gui = GUI()
pygame.init()
gui.draw_board(board_array)
while True:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            pygame.quit()
            sys.exit()



# # Open a file in write mode
# log_file = open("output.log", "w")

# # Redirect stdout to the file
# sys.stdout = log_file

# # board_array = np.array([
# #  [ 0.,  1.,  1., -1., -1.,  0.,  1.,  0.,  0.],
# #  [ 0., -1.,  0.,  0.,  0.,  0., -1., -1.,  1.],
# #  [ 0.,  0.,  1.,  0.,  0.,  0., -1., -1.,  0.],
# #  [-1.,  0.,  0.,  0.,  1.,  0.,  0.,  0.,  0.],
# #  [ 0.,  0.,  0.,  0., -1., -1., -1.,  0.,  0.],
# #  [ 0.,  1.,  1., -1.,  0.,  0.,  0.,  0.,  0.],
# #  [ 0.,  0.,  0.,  0.,  0.,  0.,  0.,  0., -1.],
# #  [ 0.,  1.,  0.,  1.,  0.,  1.,  0.,  0.,  0.],
# #  [ 1.,  0., -1.,  0.,  0.,  0.,  1.,  0.,  0.]
# # ])

# board_array = np.array([
#     [0., 0., 0., 0., 0., 0., 0., 0., 0.],
#     [0., 0., 0., 0., 0., 0., 0., 0., 0.],
#     [0., 0., 0., 0., 0., 0., 0., 0., 0.],
#     [0., 0., 0., 1., 0., 0., 0., 0., 0.],
#     [0., 0., 0., -1., 1., 1., 0., 0., 0.],
#     [0., 0., 0., -1., -1., -1., 0., 0., 0.],
#     [0., 0., 0., 0., 0., 0., 1., 0., 0.],
#     [0., 0., 0., 0., 0., 0., 0., 0., 0.],
#     [0., 0., 0., 0., 0., 0., 0., 0., 0.]
# ])

# game = Game()

# game.current_player = 0

# game.state[0] = board_array
# game.state[1] = -board_array

# game.move_history = [r * 9 + c for r in range(9) for c in range(9) if board_array[r, c] != 0]

# # game.do_move((5, 6))
# # game.do_move((0, 0))
# # game.do_move((5, 2))

# policy_net = PolicyValueNetwork()
# policy_net.load()

# winner = game.get_winner()
# if winner is not None:
#     print(f"Winner: Player {winner + 1}")
# else:
#     print("No winner yet.")

# winner, empoty = game.get_winner_indirect()
# if winner is not None:
#     print(f"Winner: Player {winner + 1}")
# else:
#     print("No winner yet.")

# obvious_move = game.check_obvious_move()
# if obvious_move is not None:
#     print(f"Obvious move found: {obvious_move}")
# else:
#     print("No obvious move found.")

# policy_net.model.eval()  # Set the model to evaluation mode
# action_probs = policy_net.get_action_probs(board_array)
# value = policy_net.get_state_value(board_array)
# for i, prob in action_probs:
#     row, col = divmod(i, 9)
#     print(f"Position ({row}, {col}): Probability {prob:.4f}")

# print(f"State value: {value:.4f}")

# sys.stdout = sys.__stdout__
# log_file.close()
# mcts = MCTS(policy_net, iterations=13000)  
# action_probs = mcts.get_action_probs(game, turn=game.current_player)

# # --- 4. Display results ---
# print("MCTS action probabilities (flattened 9x9 board):")
# print(action_probs.reshape(9, 9))

# best_action = np.argmax(action_probs)
# best_row, best_col = divmod(best_action, 9)
# print(f"Best move according to MCTS: ({best_row}, {best_col})")


# gui = GUI()
# pygame.init()
# gui.draw_board_with_probs(game.state[0], action_probs)
# while True:
#     for event in pygame.event.get():
#         if event.type == pygame.QUIT:
#             pygame.quit()
#             sys.exit()
