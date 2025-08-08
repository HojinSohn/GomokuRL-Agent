import numpy as np
from scipy.signal import convolve2d

rewards = {"win" : 10.0, "lose" : -10.0, "step" : 0.001, "invalid" : -0.5}

class GomokuEnv:
    def __init__(self):
        self.size = 15
        self.state = {0: np.zeros((15, 15)), 1: np.zeros((15, 15))}
        # Directional kernels to detect 5-in-a-row
        self.kernels = [
            np.ones((1, 5)),           # Horizontal
            np.ones((5, 1)),           # Vertical
            np.eye(5),                 # Diagonal ↘
            np.fliplr(np.eye(5))       # Diagonal ↙
        ]

    def reset(self):
        self.state = {0: np.zeros((15, 15)), 1: np.zeros((15, 15))}

    def display(self, turn):
        for row in range(15):
            print(" ".join(str(self.state[turn][row, col]) for col in range(15)))

    def first_move(self):
        center = 7
        self.state[0][center, center] = 1
        self.state[1][center, center] = -1

    def check_winner(self, turn):
        board = self.state[turn]
        masked_board = (board == 1).astype(int)
        for kernel in self.kernels:
            conv = convolve2d(masked_board, kernel, mode='valid')
            if np.any(conv == 5):
                return True
        return False
    
    # Perform an action and return the new state, whether the game is completed, reward, and whether the action was valid
    def step(self, action, turn):
        row, col = divmod(action, 15)
        if self.state[turn][row, col] != 0:
            return self.state[turn], False, rewards["invalid"], False
        # Place the piece
        self.state[turn][row, col] = 1
        self.state[0 if turn == 1 else 1][row, col] = -1
        completed = False
        if self.check_winner(turn):
            completed = True
            reward = rewards["win"]
        else:
            reward = rewards["step"]

        return self.state[turn], completed, reward, True
    


# def test_vertical_win():
#     env = GomokuEnv()
#     env.reset()

#     # Player 1 vertical line at column 7 (rows 7,8,9,10,11)
#     moves_p1 = [7*15 + 7, 8*15 + 7, 9*15 + 7, 10*15 + 7, 11*15 + 7]
#     for move in moves_p1:
#         should_succeed = env.step(move, 0)
#         print(f"Attempting to place at {move} for Player 1: {should_succeed}")
#         should_fail = env.step(move, 0)
#         print(f"Attempting to place at {move} for Player 1: {should_fail}")

#     if env.check_winner(0):
#         print("Player 1 vertical win detected correctly.")
#     else:
#         print("Failed to detect Player 1 vertical win.")

#     if env.check_winner(1):
#         print("False positive for Player 2 vertical win.")
#     else:
#         print("No false positive for Player 2 vertical win.")

# def test_horizontal_win():
#     env = GomokuEnv()
#     env.reset()

#     # Player 1 horizontal line at row 7 (columns 7,8,9,10,11)
#     moves_p1 = [7*15 + 7, 7*15 + 8, 7*15 + 9, 7*15 + 10, 7*15 + 11]
#     for move in moves_p1:
#         env.step(move, 0)

#     if env.check_winner(0):
#         print("Player 1 horizontal win detected correctly.")
#     else:
#         print("Failed to detect Player 1 horizontal win.")

#     if env.check_winner(1):
#         print("False positive for Player 2 horizontal win.")
#     else:
#         print("No false positive for Player 2 horizontal win.")

# def test_diagonal_win():
#     env = GomokuEnv()
#     env.reset()

#     # Player 2 diagonal line (positions: (5,5), (6,6), (7,7), (8,8), (9,9))
#     moves_p2 = [5*15 + 5, 6*15 + 6, 7*15 + 7, 8*15 + 8, 9*15 + 9]
#     for move in moves_p2:
#         env.step(move, 1)

#     if env.check_winner(1):
#         print("Player 2 diagonal win detected correctly.")
#     else:
#         print("Failed to detect Player 2 diagonal win.")

#     if env.check_winner(0):
#         print("False positive for Player 1 diagonal win.")
#     else:
#         print("No false positive for Player 1 diagonal win.")

# def test_diagonal_win2():
#     env = GomokuEnv()
#     env.reset()

#     # Player 2 diagonal line (positions: (7,7), (6,8), (5,9), (4,10), (3,11))
#     moves_p2 = [7*15 + 7, 6*15 + 8, 5*15 + 9, 4*15 + 10, 3*15 + 11]
#     for move in moves_p2:
#         env.step(move, 1)

#     if env.check_winner(1):
#         print("Player 2 diagonal win detected correctly.")
#     else:
#         print("Failed to detect Player 2 diagonal win.")

#     if env.check_winner(0):
#         print("False positive for Player 1 diagonal win.")
#     else:
#         print("No false positive for Player 1 diagonal win.")

# if __name__ == "__main__":
#     test_vertical_win()
#     print("-" * 40)
#     test_horizontal_win()
#     print("-" * 40)
#     test_diagonal_win()
#     print("-" * 40)
#     test_diagonal_win2()