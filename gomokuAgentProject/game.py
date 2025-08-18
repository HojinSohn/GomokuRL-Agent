import copy
import numpy as np

class Game:
    def __init__(self):
        # board size
        self.size = 9
        # state is a dictionary storing the board state from the perspective of each player
        # 0 for empty, 1 for the current player's stone, -1 for the opponent's stone
        # state[0] is the board state of black's perspective, state[1] is for white's perspective
        self.state = {0: np.zeros((9, 9)), 1: np.zeros((9, 9))}

    def reset(self):
        self.state = {0: np.zeros((9, 9)), 1: np.zeros((9, 9))}

    def display(self, turn):
        for row in range(9):
            print(" ".join(str(self.state[turn][row, col]) for col in range(9)))

    def first_move(self):
        center = self.size // 2
        self.state[0][center, center] = 1
        self.state[1][center, center] = -1

    def check_winner(self, turn, prev_move):
        # Assume board will always have the player's perspective as 1, and the opponent's as -1
        board = self.state[turn]
        player_just_put = 1 # prev_move is made by current turn's player, so player_just_put is 1 (current turn player's perspective)
        rows, cols = board.shape
        r, c = divmod(prev_move, self.size)

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
    
    def update_state(self, turn, action):
        row, col = divmod(action, 9)
        self.state[turn][row, col] = 1
        # Update the opponent's state
        opponent_turn = 1 - turn
        self.state[opponent_turn][row, col] = -1
