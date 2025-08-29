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
        
        self.move_history = []  # Stack to store moves for undoing
        self.current_player = 0 # Start with player 0

    def reset(self):
        self.state = {0: np.zeros((9, 9)), 1: np.zeros((9, 9))}
        self.move_history = []
        self.current_player = 0

    def display(self, turn):
        for row in range(9):
            print(" ".join(str(self.state[turn][row, col]) for col in range(9)))

    def first_move(self):
        center = self.size // 2
        self.state[0][center, center] = 1
        self.state[1][center, center] = -1
        self.move_history.append(center * 9 + center)
        self.current_player = 1 # next player

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
        if self.state[turn][row, col] != 0:
            raise ValueError("Invalid move: Cell is already occupied.")
        self.state[turn][row, col] = 1
        # Update the opponent's state
        opponent_turn = 1 - turn
        self.state[opponent_turn][row, col] = -1

    def get_legal_moves(self):
        legal_moves = []
        for r in range(9):
            for c in range(9):
                if self.state[self.current_player][r, c] == 0:
                    legal_moves.append((r, c))
        return legal_moves

    def do_move(self, move):
        action = move[0] * 9 + move[1]
        self.move_history.append(action)
        self.update_state(self.current_player, action)
        self.turn_swap()

    def undo_move(self):
        if not self.move_history:
            return
        last_move_action = self.move_history.pop()
        row, col = divmod(last_move_action, self.size) # Convert action to (row, col)
        if self.state[self.current_player][row, col] == 0:
            raise ValueError("Invalid undo: Cell is already empty.")
        player_who_made_move = 1 - self.current_player
        self.state[player_who_made_move][row, col] = 0
        self.state[self.current_player][row, col] = 0
        self.turn_swap()

    def turn_swap(self):
        self.current_player = 1 - self.current_player

    def get_winner(self):
        if not self.move_history:
            return None
        
        last_move = self.move_history[-1]
        opponent_player = 1 - self.current_player

        winner = self.check_winner(opponent_player, last_move)
        if winner is not None:
            return opponent_player
        
        # Optional: detect draw
        if len(self.move_history) == self.size * self.size:
            return -1  # draw
        
        return None