import copy
import numpy as np

class Game:
    """
    Game class for managing the state and rules of the Gomoku game. 
    Store the board state from each player's perspective and handle game logic.
    """
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
        """
        Check for a winner in the current board state. Use previous move for efficient checking.
        """
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
        """
        Update the board state for the current player and the opponent based on the action.
        """
        row, col = divmod(action, 9)
        if self.state[turn][row, col] != 0:
            raise ValueError("Invalid move: Cell is already occupied.")
        self.state[turn][row, col] = 1
        # Update the opponent's state
        opponent_turn = 1 - turn
        self.state[opponent_turn][row, col] = -1

    def get_legal_moves(self):
        """
        Get all legal moves for the current player.
        """
        legal_moves = []
        for r in range(9):
            for c in range(9):
                if self.state[self.current_player][r, c] == 0:
                    legal_moves.append((r, c))
        return legal_moves

    def do_move(self, move):
        """
        Apply a move to the game state.
        """
        action = move[0] * 9 + move[1]
        self.move_history.append(action)
        self.update_state(self.current_player, action)
        self.turn_swap()

    def undo_move(self):
        """
        Undo the last move.
        """
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
        """
        Swap the current player.
        """
        self.current_player = 1 - self.current_player

    def check_open_four(self, turn, prev_move):
        """
        Checks for an open 'four' pattern around the last move.
        Used to check for potential winning moves.
        """
        if prev_move is None:
            return False, None

        board = self.state[turn]
        r, c = divmod(prev_move, self.size)
        player_stone = 1  # The current player's stone is always '1' in their perspective

        # Define the four directions to check
        directions = [(0, 1),  # Horizontal
                      (1, 0),  # Vertical
                      (1, 1),  # Main diagonal
                      (1, -1)] # Anti-diagonal

        for dr, dc in directions:
            # For each direction, check the 5 possible 5-cell windows that include the new stone.
            # This is our "sliding window", where we shift the starting point.
            for i in range(5):
                # Define the starting coordinate of the current 5-cell window
                start_r, start_c = r - i * dr, c - i * dc
                
                window_coords = [(start_r + j * dr, start_c + j * dc) for j in range(5)]

                # 1. BOUNDARY CHECK: Ensure the entire window is on the board
                on_board = all(0 <= nr < self.size and 0 <= nc < self.size for nr, nc in window_coords)
                if not on_board:
                    continue

                # 2. CONTENT ANALYSIS: Get the values from the board for the current window
                window_values = [board[nr, nc] for nr, nc in window_coords]

                # 3. PATTERN MATCHING: Check if the window contains 4 of our stones and 1 empty cell
                player_stones_count = window_values.count(player_stone)
                empty_cells_count = window_values.count(0)
                
                if player_stones_count == 4 and empty_cells_count == 1:
                    # get the empty cell position
                    empty_cell_pos = window_coords[window_values.index(0)]
                    return True, empty_cell_pos  # Found a winning pattern

        # If we check all windows in all directions and find nothing, return False
        return False, None

    def check_obvious_move(self):
        """
        Check for obvious moves that can block the opponent's winning chances.
        Returns the position of the moves that can block the opponent's direct winning move.
        """
        obvious_move = None
        opponent_turn = 1 - self.current_player
        # check for open four for opponent
        win, empty_cell_pos = self.check_open_four(opponent_turn, self.move_history[-1] if self.move_history else None)
        if win:
            obvious_move = empty_cell_pos[0] * self.size + empty_cell_pos[1]
        return obvious_move

    def get_winner_indirect(self):
        '''
        Check for a winner. Based on the board state, check whether the game can be finished by one move by current player.
        Only used by MCTS iteration for efficient search and simulation.
        '''
        if not self.move_history:
            return None, None

        last_move = self.move_history[-1]
        opponent_player = 1 - self.current_player

        winner = self.check_winner(opponent_player, last_move)
        if winner is not None:
            return opponent_player, None

        # check open four for current player
        # if the current player has the four in a row with at least one of the sides open, this player will win
        last_move_by_current_player = self.move_history[-2] if len(self.move_history) > 1 else None
        current_player = self.current_player
        win, empty_cell_pos = self.check_open_four(current_player, last_move_by_current_player)
        if win:
            return current_player, empty_cell_pos

        # Optional: detect draw
        if len(self.move_history) == self.size * self.size:
            return -1, None  # draw

        return None, None

    def get_winner(self):
        '''
        Check for a winner by examining the last move. (consecutive row of 5)
        '''
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