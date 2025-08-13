import copy
import numpy as np

rewards = {"win" : 1.0, "lose" : -1.0, "step" : 0.001, "invalid" : -0.5}

class GomokuEnv:
    def __init__(self):
        self.size = 9
        self.state = {0: np.zeros((9, 9)), 1: np.zeros((9, 9))}
        # Directional kernels to detect 5-in-a-row
        self.kernels = [
            np.ones((1, 5)),           # Horizontal
            np.ones((5, 1)),           # Vertical
            np.eye(5),                 # Diagonal ↘
            np.fliplr(np.eye(5))       # Diagonal ↙
        ]

    def reset(self):
        self.state = {0: np.zeros((9, 9)), 1: np.zeros((9, 9))}

    def display(self, turn):
        for row in range(9):
            print(" ".join(str(self.state[turn][row, col]) for col in range(9)))

    def first_move(self):
        center = self.size // 2
        self.state[0][center, center] = 1
        self.state[1][center, center] = -1

    def check_winner(self, board, prev_move):
        # Assume board will always have the player's perspective as 1, and the opponent's as -1
        player_just_put = 1 
        rows, cols = board.shape
        r, c = prev_move

        # Check if the move is valid and belongs to player_just_put
        if not (0 <= r < rows and 0 <= c < cols and board[r, c] == player_just_put):
            return None

        # Directions: horizontal, vertical, main diagonal, anti-diagonal
        directions = [(0, 1), (1, 0), (1, 1), (1, -1)]

        for dr, dc in directions:
            # Count consecutive stones in both directions (e.g., left and right for horizontal)
            count = 1  # Include the current move
            # Forward direction
            for i in range(1, 5):  # Check up to 4 cells forward
                nr, nc = r + dr * i, c + dc * i
                if not (0 <= nr < rows and 0 <= nc < cols and board[nr, nc] == player_just_put):
                    break
                count += 1
            # Backward direction
            for i in range(1, 5):  # Check up to 4 cells backward
                nr, nc = r - dr * i, c - dc * i
                if not (0 <= nr < rows and 0 <= nc < cols and board[nr, nc] == player_just_put):
                    break
                count += 1
            # Check if we have 5 or more consecutive stones
            if count >= 5:
                return player_just_put

        return None
    
    # Perform an action and return the new state, whether the game is completed, reward, and whether the action was valid
    def step(self, action, turn):
        row, col = divmod(action, 9)
        if self.state[turn][row, col] != 0:
            return self.state[turn].copy(), self.state[turn].copy(), False, rewards["invalid"], False
        
        next_state = self.state[turn].copy()
        # Place the piece
        next_state[row, col] = 1
        completed = False
        if self.check_winner(next_state, (row, col)): # pass prev_move as a tuple
            completed = True
            reward = rewards["win"]
        else:
            reward = rewards["step"]

        return self.state[turn].copy(), next_state, completed, reward, True

    def update_state(self, turn, action):
        row, col = divmod(action, 9)
        self.state[turn][row, col] = 1
        # Update the opponent's state
        opponent_turn = 1 - turn
        self.state[opponent_turn][row, col] = -1


# def test_vertical_win():
#     env = GomokuEnv()
#     env.reset()

#     # Player 1 vertical line at column 7 (rows 7,8,9,10,11)
#     moves_p1 = [7*9 + 7, 8*9 + 7, 9*9 + 7, 10*9 + 7, 11*9 + 7]
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
#     moves_p1 = [7*9 + 7, 7*9 + 8, 7*9 + 9, 7*9 + 10, 7*9 + 11]
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
#     moves_p2 = [5*9 + 5, 6*9 + 6, 7*9 + 7, 8*9 + 8, 9*9 + 9]
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
#     moves_p2 = [7*9 + 7, 6*9 + 8, 5*9 + 9, 4*9 + 10, 3*9 + 11]
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