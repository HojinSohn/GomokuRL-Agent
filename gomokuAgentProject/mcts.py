# Node class for Tree Search
import random
import numpy as np
import copy
from scipy.signal import convolve2d
class Node:
    def __init__(self, current_player, parent=None, move=None):
        self.parent = parent
        self.children = []
        self.children_moves = set()  # Set to track moves of existing children for quick lookup
        self.visits = 0
        self.wins = 0.0
        self.move = move # represent the last move made to reach this node
        self.moves = (parent.moves + [move] if parent else [move]) if move is not None else []
        self.current_player = current_player
        self.is_terminal = False

    def ucb1(self):
        if self.visits == 0:
            return float('inf')
        return self.wins / self.visits + np.sqrt((2 * np.log(self.parent.visits)) / self.visits)

BOARD_SIZE = 9
class MCTS:
    def __init__(self, iterations=10):
        self.iterations = iterations

    def get_current_player(self, board):
        """Determine whose turn it is based on piece count"""
        count_1 = np.sum(board == 1)
        count_neg1 = np.sum(board == -1)
        return 1 if count_1 <= count_neg1 else -1

    def apply_moves_to_board(self, board, moves, initial_player):
        """Apply a sequence of moves to a board, alternating players"""
        board = board.copy()  # Create a copy to avoid modifying the original
        for i, move in enumerate(moves):
            if move is not None:
                row, col = move
                # Alternate players: initial_player, then -initial_player, etc.
                player = initial_player if i % 2 == 0 else -initial_player
                board[row][col] = player
        return board

    def undo_moves_from_board(self, board, moves, initial_player):
        """Undo a sequence of moves from a board, alternating players"""
        board = board.copy()  # Create a copy to avoid modifying the original
        for move in moves:
            if move is not None:
                row, col = move
                board[row][col] = 0
        return board
    
    def traverse_down_to_leaf_node(self, node):
        '''
            This function traverses the tree from the element node to a leaf node,
            selecting child nodes based on the UCB1 score.

            @param node: Node to traverse
            @return: Leaf node after traversing down the tree
        '''
        while node.children:
            node = max(node.children, key=lambda n: n.ucb1())
        return node
    
    
    
    def get_reasonable_moves(self, board, current_player, last_move=None):
        # return any empty cells adjacent to stones
        rows, cols = board.shape
        empty_cells = set()
        directions = [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]
        for r in range(rows):
            for c in range(cols):
                if board[r, c] != 0:
                    for dr, dc in directions:
                        nr, nc = r + dr, c + dc
                        if 0 <= nr < rows and 0 <= nc < cols and board[nr, nc] == 0:
                            empty_cells.add((nr, nc))
        return list(map(tuple, empty_cells))

    def get_obvious_moves(self, board, current_player):
        '''
            This function generates obvious moves based on the current board state.
            It checks for open four and open three positions, and returns them.

            @param board: Current board state (2d array) 9 x 9. 0 = empty, 1 = player, -1 = opponent
            @param current_player: Current player (1 or -1)
            @return: List of obvious moves, prioritized by importance
        '''
        good_moves = self.detect_open_four(board, current_player)
        if good_moves:
            return good_moves

        good_moves = self.detect_open_four(board, -1 * current_player)
        if good_moves:
            return good_moves
        
        good_moves = self.detect_open_three(board, current_player)
        if good_moves:
            return good_moves
        
        good_moves = self.detect_open_three(board, -1 * current_player)
        if good_moves:
            return good_moves
        return []


    def get_good_moves(self, board, current_player):
        '''
            First check the obvious moves. Then check the good moves.
        '''
        obvious_moves = self.get_obvious_moves(board, current_player)
        if obvious_moves:
            return obvious_moves
        
        return self.get_reasonable_moves(board, current_player)

    def expand(self, node, board):
        '''
            This function expands a leaf node by adding child nodes for legal or promising next moves.
            It will be called when a leaf node is reached second time during the traversal.

            @param node: Leaf node (board state) to expand
            @param board: Current board state (2D array) 9x9. 0 = empty, 1 = player, -1 = opponent
            @return: New child node added to the tree, or None if no promising moves are available
        '''
        # Apply moves to get current board state
        promising_moves = None
        board = self.apply_moves_to_board(board, node.moves, initial_player=self.root.current_player)
        if node.current_player != self.not_tactical_player:
            promising_moves = self.get_obvious_moves(board, node.current_player)
        if not promising_moves:
            promising_moves = self.get_reasonable_moves(board, node.current_player, last_move=node.moves[-1] if node.moves else None)
        random.shuffle(promising_moves)

        if len(promising_moves) == 0:
            # no promising moves available, return
            print("Board state after applying moves:")
            print(board)
            print("No promising moves available for expansion.")
            # undo moves to restore the board state
            # self.undo_moves_from_board(board, node.moves, initial_player=self.root.current_player)
            node.is_terminal = True
            return None

        child_node = None
        # Randomly select one child and add it to the node
        for move in promising_moves:
            if len(node.children_moves) > 0 and move in node.children_moves:
                continue  # Skip already existing child moves
            child_node = Node(current_player=-1 * node.current_player, parent=node, move=move)
            node.children.append(child_node)
            node.children_moves.add(move)  # Add move to the set of existing children moves

        # undo moves to restore the board state
        # self.undo_moves_from_board(board, node.moves, initial_player=self.root.current_player)

        if child_node is None:
            # This node is already fully expanded
            node.is_terminal = True

        # return the last child node added / it will be rolled out
        return child_node

    def check_winner(self, board, player_just_put, prev_move):
        """
        Checks if the player who just moved has won by detecting five consecutive stones
        including the most recent move.

        @param board: Current board state (2D array) 9x9. 0 = empty, 1 = player, -1 = opponent
        @param player_just_put: Player who made the most recent move (1 or -1)
        @param prev_move: Tuple (row, col) of the most recent move
        @return: player_just_put if they won, None otherwise
        """
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

    def get_obvious_moves_light(self, board, current_player, last_move=None):
        """Lightweight version of get_obvious_moves for rollouts."""
        moves = []
        if last_move is None:
            return moves  # Fall back to reasonable moves if no last move
        r, c = last_move
        directions = [(0, 1), (1, 0), (1, 1), (1, -1)]
        for dr, dc in directions:
            for i in [-4, -3, -2, -1, 1, 2, 3, 4]:  # Check nearby cells
                nr, nc = r + dr * i, c + dc * i
                if 0 <= nr < BOARD_SIZE and 0 <= nc < BOARD_SIZE and board[nr, nc] == 0:
                    # Check if placing a stone here creates an open four or blocks one
                    board[nr, nc] = current_player
                    if self.check_winner(board, current_player, (nr, nc)):
                        moves.append((nr, nc)) 
                    elif self.check_winner(board, -current_player, (nr, nc)):
                        moves.append((nr, nc))
                    board[nr, nc] = 0  # Undo
        return list(set(moves))

    def rollout(self, node, board):
        '''
            This function simulates a random game from the current board state.
            It will be called when a leaf node is reached for the first time during the traversal.

            @param node: Leaf node (board state) to simulate
        '''
        # Apply the moves to the board state by applying moves
        applied_moves = node.moves
        board = self.apply_moves_to_board(board, node.moves, initial_player=self.root.current_player)

        # Determine whose turn it is next
        current_player = node.current_player

        # Start the simulation
        result = 0.5
        MAX_SIMULATION_STEPS = 40  # Limit the number of steps to prevent infinite loops
        moves_in_simulation = []
        for _ in range(MAX_SIMULATION_STEPS):
            valid_moves = None
            if current_player != self.not_tactical_player:
                valid_moves = self.get_obvious_moves_light(board, node.current_player, last_move=moves_in_simulation[-1] if moves_in_simulation else None)
            
            if valid_moves is None or len(valid_moves) == 0:
                # Randomly select a move
                valid_moves = self.get_reasonable_moves(board, current_player, last_move=moves_in_simulation[-1] if moves_in_simulation else None)
            if not valid_moves:
                # draw
                result = 0.5
                break
            move = valid_moves[np.random.choice(len(valid_moves))]
            # Place the piece on the board
            board[move[0]][move[1]] = current_player
            moves_in_simulation.append(move)
            # Check if the game is completed
            winner = self.check_winner(board, current_player, move)
            if winner is not None:
                # since the current player is the one who just placed the piece, we return 10 for win
                result = 1 if winner == self.root.current_player else -1
                break

            valid_moves = None
            if current_player != self.not_tactical_player:
                valid_moves = self.get_obvious_moves_light(board, -1 * current_player, last_move=moves_in_simulation[-1] if moves_in_simulation else None)

            if not valid_moves:
                # Randomly select a move for the opponent
                valid_moves = self.get_reasonable_moves(board, -1 * current_player, last_move=moves_in_simulation[-1] if moves_in_simulation else None)
            if not valid_moves:
                # draw
                result = 0.5
                break
            move = valid_moves[np.random.choice(len(valid_moves))]
            board[move[0]][move[1]] = -1 * current_player
            moves_in_simulation.append(move)
            # Check if the game is completed by the opponent's move
            winner = self.check_winner(board, -1 * current_player, move)
            if winner is not None:
                # since the opponent is the one who just placed the piece, we return -10 for loss
                result = 1 if winner == self.root.current_player else -1
                break

        # self.undo_moves_from_board(board, moves_in_simulation, initial_player=self.root.current_player)
        # self.undo_moves_from_board(board, applied_moves, initial_player=self.root.current_player)

        return result

    def backpropagate(self, node, result):
        while node is not None:
            node.visits += 1
            node.wins += result if node.current_player == self.root.current_player else -result
            node = node.parent

    def get_action(self, board, not_tactical=False):
        '''
            This function performs MCTS to find the best action for the current board state.

            @param board: Current board state (2d array) 9 x 9. 0 = empty, 1 = player, -1 = opponent
            @return: Best action based on MCTS
        '''
        # Implement MCTS logic here
        '''
        1. A tree is initialized with a node of the given board state, representing the current game state.
        2. When a leaf node (a node with no children) is reached, expand it by adding child nodes for legal or promising next moves.
        3. In each iteration, traverse the tree from the root by choosing child nodes with the highest UCB1 score, until reaching a leaf node.
        4. From the leaf, simulate a full game to the end (a rollout), then backpropagate the result up the tree to update visit counts and scores.
        5. Repeat this select → expand → simulate → backpropagate process for many iterations.
        6. After a set number of iterations, select the child node with the highest visit count as the best action.
        '''
        # this player will not use tactical moves, so we will not use the get_obvious_moves function
        current_player = self.get_current_player(board)
        promising_moves = None
        if not_tactical:
            self.not_tactical_player = current_player
        else:
            self.not_tactical_player = -1 * current_player  # Opponent player for the random player
            promising_moves = self.get_obvious_moves(board, current_player)

        if promising_moves:
            # If there are obvious moves, select one randomly
            move = promising_moves[np.random.choice(len(promising_moves))]
            return move[0] * 9 + move[1]

        self.root = Node(current_player=current_player, parent=None, move=None)  # Reset the root node with the current board state
        for it in range(self.iterations):
            # traverse down the tree to a leaf node
            leaf_node = self.traverse_down_to_leaf_node(self.root)
            if leaf_node.visits > 0:
                # If the leaf node has been visited before, expand it
                added_child_node = self.expand(leaf_node, board)
                if added_child_node is None:
                    # if the leaf node is in terminal state, we need to simulate a random game from this state
                    print("Leaf node is in terminal state...@@.@@@@@@@@@@.@@@@@@@@@@.@@@@@@@@@@.@@@@@@@@@@.@@@@@@@@@@.@@@@@@@@@@@@@@@@@@@@")
                else:
                    # if the child node is added, we need to simulate a random game from this state
                    result = self.rollout(added_child_node, board)
                    self.backpropagate(added_child_node, result)
            else:
                # If the leaf node has not been visited before, we need to simulate a random game from this state
                result = self.rollout(leaf_node, board)
                self.backpropagate(leaf_node, result)

        if len(self.root.children) == 0:
            print("Root's child node empty....")
            print("Current board state:")
            print(board)
            valid_moves = self.get_reasonable_moves(board, current_player)
            return valid_moves[0][0] * 9 + valid_moves[0][1]

        best_child = max(self.root.children, key=lambda n: n.visits)
        # Convert the best child's element (board state) to an action
        best_action = best_child.move
        if best_action is None:
            return None
        # Convert the best action (row, col) to a single action index
        return best_action[0] * 9 + best_action[1] 



    def add_open_three(self, board, player, r, c, good_moves):
        """
        Adds an open three pattern to the board for the given player.
        Open three: _XXX_ (three consecutive stones with empty ends).

        @param board: 9x9 board (2D list/array). 0 = empty, 1 = player, -1 = opponent
        @param player: Player to check (1 or -1)
        @param row: Row index of the open three
        @param col: Column index of the open three
        """
        rows, cols = len(board), len(board[0])
        directions = [(0, 1), (1, 0), (1, 1), (1, -1)]  # Horizontal, vertical, main diag, anti-diag
        # Helper function to check if a cell is empty
        def is_empty(r, c):
            return 0 <= r < rows and 0 <= c < cols and board[r][c] == 0

        # Helper function to check if a cell matches the player
        def is_player(r, c):
            return 0 <= r < rows and 0 <= c < cols and board[r][c] == player
        
        for dr, dc in directions:
            # Check open three: _XXX_
            if is_player(r, c) and is_player(r + dr, c + dc) and is_player(r + 2*dr, c + 2*dc):
                # Check both ends: (r-dr, c-dc) and (r+3*dr, c+3*dc)
                left_end = (r - dr, c - dc)
                right_end = (r + 3*dr, c + 3*dc)
                if is_empty(*left_end) and is_empty(*right_end):
                    good_moves.add(left_end)
                    good_moves.add(right_end)

            # Check detached three pattern 1: _X_XX_ (gap at position 2)
            if (is_empty(r, c) and is_player(r + dr, c + dc) and is_empty(r + 2*dr, c + 2*dc) and
                is_player(r + 3*dr, c + 3*dc) and is_player(r + 4*dr, c + 4*dc) and
                is_empty(r + 5*dr, c + 5*dc)):
                good_moves.add((r, c))  # First empty
                good_moves.add((r + 2*dr, c + 2*dc))  # Middle empty
                good_moves.add((r + 5*dr, c + 5*dc))  # Last empty

            # Check detached three pattern 2: _XX_X_ (gap at position 3)
            if (is_empty(r, c) and is_player(r + dr, c + dc) and is_player(r + 2*dr, c + 2*dc) and
                is_empty(r + 3*dr, c + 3*dc) and is_player(r + 4*dr, c + 4*dc) and
                is_empty(r + 5*dr, c + 5*dc)):
                good_moves.add((r, c))  # First empty
                good_moves.add((r + 3*dr, c + 3*dc))  # Middle empty
                good_moves.add((r + 5*dr, c + 5*dc))  # Last empty

    
    def add_open_four(self, board, player, r, c, good_moves):
        rows, cols = len(board), len(board[0])
        directions = [(0, 1), (1, 0), (1, 1), (1, -1)]  # Horizontal, vertical, main diag, anti-diag
        # Helper function to check if a cell is empty
        def is_empty(r, c):
            return 0 <= r < rows and 0 <= c < cols and board[r][c] == 0

        # Helper function to check if a cell matches the player
        def is_player(r, c):
            return 0 <= r < rows and 0 <= c < cols and board[r][c] == player
        
        for dr, dc in directions:
            # Check open four: _XXXX_ or _XXXX, XXXX_
            if (is_player(r, c) and is_player(r + dr, c + dc) and 
                is_player(r + 2*dr, c + 2*dc) and is_player(r + 3*dr, c + 3*dc)):
                # Check left end: (r-dr, c-dc)
                left_end = (r - dr, c - dc)
                if is_empty(*left_end):
                    good_moves.add(left_end)
                # Check right end: (r+4*dr, c+4*dc)
                right_end = (r + 4*dr, c + 4*dc)
                if is_empty(*right_end):
                    good_moves.add(right_end)
            # check open four: XX_XX or XXX_X or X_XXX
            elif (is_player(r, c) and is_player(r + dr, c + dc) and 
                is_empty(r + 2*dr, c + 2*dc) and is_player(r + 3*dr, c + 3*dc) and is_player(r + 4*dr, c + 4*dc)):
                good_moves.add((r + 2*dr, c + 2*dc))  # Add the empty cell in the middle
            elif (is_player(r, c) and is_empty(r + dr, c + dc) and 
                is_player(r + 2*dr, c + 2*dc) and is_player(r + 3*dr, c + 3*dc) and is_player(r + 4*dr, c + 4*dc)):
                good_moves.add((r + dr, c + dc))
            elif (is_player(r, c) and is_player(r + dr, c + dc) and is_player(r + 2*dr, c + 2*dc) and 
                is_empty(r + 3*dr, c + 3*dc) and is_player(r + 4*dr, c + 4*dc)):
                good_moves.add((r + 3*dr, c + 3*dc))
    
    def detect_open_three(self, board, player):
        """
        Detects open three and detached three patterns for the given player.
        Open three: _XXX_ (three consecutive stones with empty ends).
        Detached three: _X_XX_ or _XX_X_ (three stones with one gap and empty ends).

        @param board: 9x9 board (2D list/array). 0 = empty, 1 = player, -1 = opponent
        @param player: Player to check (1 or -1)
        @return: List of (row, col) tuples for empty cells that complete open/detached three patterns
        """
        rows, cols = len(board), len(board[0])
        good_moves = set()

        for r in range(rows):
            for c in range(cols):
                self.add_open_three(board, player, r, c, good_moves)

        return list(good_moves)

    def detect_open_four(self, board, player):
        """
        Detects open four patterns for the given player.
        Open four: _XXXX_ or _XXXX, XXXX_ (four consecutive stones with at least one empty end).

        @param board: 9x9 board (2D list/array). 0 = empty, 1 = player, -1 = opponent
        @param player: Player to check (1 or -1)
        @return: List of (row, col) tuples for empty cells that complete open four patterns
        """
        rows, cols = len(board), len(board[0])
        good_moves = set()

        for r in range(rows):
            for c in range(cols):
                self.add_open_four(board, player, r, c, good_moves)

        return list(good_moves)

# # Example usage and test function
# def test_open_three_detection():
#     """Test the open three detection with sample boards"""

#     mcts = MCTS(iterations=1)  # Initialize MCTS with 1 iteration

#     def print_test_case(name, board, player):
#         print(f"\n{'='*50}")
#         print(f"TEST CASE: {name}")
#         print("Board:")
#         print(board)
#         result = mcts.detect_open_three(board, player)
#         print(f"Open threes for player {player}: {sorted(result)}")
#         return result
    
#     # Test Case 1: Horizontal Continuous Open Three
#     board1 = np.array([
#         [0, 0, 0, 0, 0, 0, 0, 0, 0],
#         [0, 0, 0, 0, 0, 0, 0, 0, 0],
#         [0, 1, 1, 1, 0, 0, 0, 0, 0],  # _XXX_ at positions 0,4
#         [0, 0, 0, 0, 0, 0, 0, 0, 0],
#         [0, 0, 0, 0, 0, 0, 0, 0, 0],
#     ])
#     print_test_case("Horizontal Continuous Open Three", board1, 1)
    
#     # Test Case 2: Vertical Continuous Open Three  
#     board2 = np.array([
#         [0, 0, 0, 0, 0, 0, 0],
#         [0, 1, 0, 0, 0, 0, 0],
#         [0, 1, 0, 0, 0, 0, 0],  # Vertical XXX
#         [0, 1, 0, 0, 0, 0, 0],
#         [0, 0, 0, 0, 0, 0, 0],
#         [0, 0, 0, 0, 0, 0, 0],
#     ])
#     print_test_case("Vertical Continuous Open Three", board2, 1)
    
#     # Test Case 3: Diagonal (SE) Continuous Open Three
#     board3 = np.array([
#         [0, 0, 0, 0, 0, 0, 0],
#         [0, 0, 1, 0, 0, 0, 0],
#         [0, 0, 0, 1, 0, 0, 0],  # Diagonal XXX ↘
#         [0, 0, 0, 0, 1, 0, 0],
#         [0, 0, 0, 0, 0, 0, 0],
#         [0, 0, 0, 0, 0, 0, 0],
#     ])
#     print_test_case("Diagonal (SE) Continuous Open Three", board3, 1)
    
#     # Test Case 4: Anti-Diagonal (SW) Continuous Open Three
#     board4 = np.array([
#         [0, 0, 0, 0, 0, 0, 0],
#         [0, 0, 0, 0, 1, 0, 0],
#         [0, 0, 0, 1, 0, 0, 0],  # Anti-diagonal XXX ↙
#         [0, 0, 1, 0, 0, 0, 0],
#         [0, 0, 0, 0, 0, 0, 0],
#         [0, 0, 0, 0, 0, 0, 0],
#     ])
#     print_test_case("Anti-Diagonal (SW) Continuous Open Three", board4, 1)
    
#     # Test Case 5: Horizontal Detached Pattern _X_XX_
#     board5 = np.array([
#         [0, 0, 0, 0, 0, 0, 0, 0, 0],
#         [0, 0, 0, 0, 0, 0, 0, 0, 0],
#         [0, 1, 0, 1, 1, 0, 0, 0, 0],  # _X_XX_ pattern
#         [0, 0, 0, 0, 0, 0, 0, 0, 0],
#         [0, 0, 0, 0, 0, 0, 0, 0, 0],
#     ])
#     print_test_case("Horizontal Detached Pattern _X_XX_", board5, 1)
    
#     # Test Case 6: Horizontal Detached Pattern _XX_X_
#     board6 = np.array([
#         [0, 0, 0, 0, 0, 0, 0, 0, 0],
#         [0, 0, 0, 0, 0, 0, 0, 0, 0],
#         [0, 1, 1, 0, 1, 0, 0, 0, 0],  # _XX_X_ pattern
#         [0, 0, 0, 0, 0, 0, 0, 0, 0],
#         [0, 0, 0, 0, 0, 0, 0, 0, 0],
#     ])
#     print_test_case("Horizontal Detached Pattern _XX_X_", board6, 1)
    
#     # Test Case 7: Vertical Detached Pattern _X_XX_
#     board7 = np.array([
#         [0, 0, 0, 0, 0],
#         [0, 1, 0, 0, 0],  # _
#         [0, 0, 0, 0, 0],  # X
#         [0, 1, 0, 0, 0],  # _  
#         [0, 1, 0, 0, 0],  # X
#         [0, 0, 0, 0, 0],  # X
#         [0, 0, 0, 0, 0],  # _
#     ])
#     print_test_case("Vertical Detached Pattern _X_XX_", board7, 1)
    
#     # Test Case 8: Vertical Detached Pattern _XX_X_
#     board8 = np.array([
#         [0, 0, 0, 0, 0],
#         [0, 1, 0, 0, 0],  # _
#         [0, 1, 0, 0, 0],  # X
#         [0, 0, 0, 0, 0],  # X
#         [0, 1, 0, 0, 0],  # _
#         [0, 0, 0, 0, 0],  # X
#         [0, 0, 0, 0, 0],  # _
#     ])
#     print_test_case("Vertical Detached Pattern _XX_X_", board8, 1)
    
#     # Test Case 9: Diagonal (SE) Detached Pattern _X_XX_
#     board9 = np.array([
#         [0, 0, 0, 0, 0, 0, 0, 0],
#         [0, 1, 0, 0, 0, 0, 0, 0],  # _
#         [0, 0, 0, 0, 0, 0, 0, 0],  #  X
#         [0, 0, 0, 1, 0, 0, 0, 0],  #   _
#         [0, 0, 0, 0, 1, 0, 0, 0],  #    X
#         [0, 0, 0, 0, 0, 0, 0, 0],  #     X
#         [0, 0, 0, 0, 0, 0, 0, 0],  #      _
#         [0, 0, 0, 0, 0, 0, 0, 0],
#     ])
#     print_test_case("Diagonal (SE) Detached Pattern _X_XX_", board9, 1)
    
#     # Test Case 10: Anti-Diagonal (SW) Detached Pattern _XX_X_
#     board10 = np.array([
#         [0, 0, 0, 0, 0, 0, 0, 0],
#         [0, 0, 0, 0, 0, 1, 0, 0],  #      _
#         [0, 0, 0, 0, 1, 0, 0, 0],  #     X
#         [0, 0, 0, 0, 0, 0, 0, 0],  #    X
#         [0, 0, 1, 0, 0, 0, 0, 0],  #   _
#         [0, 0, 0, 0, 0, 0, 0, 0],  #  X
#         [0, 0, 0, 0, 0, 0, 0, 0],  # _
#     ])
#     print_test_case("Anti-Diagonal (SW) Detached Pattern _XX_X_", board10, 1)
    
#     # Test Case 11: Multiple Patterns (Both Players)
#     board11 = np.array([
#         [0, 0, 0, 0, 0, 0, 0, 0, 0],
#         [0, 1, 1, 1, 0, 0, 0, 0, 0],  # Horizontal open three for player 1
#         [0, 0, 0, 0, 0, 0, 0, 0, 0],
#         [0, 0, -1, 0, -1, -1, 0, 0, 0],  # Detached pattern for player -1
#         [0, 0, 0, 0, 0, 0, 0, 0, 0],
#         [0, -1, 0, 0, 0, 0, 0, 0, 0],  # Start of vertical pattern
#         [0, -1, 0, 0, 0, 0, 0, 0, 0],  # for player -1
#         [0, -1, 0, 0, 0, 0, 0, 0, 0],
#         [0, 0, 0, 0, 0, 0, 0, 0, 0],
#     ])
#     print_test_case("Multiple Patterns - Player 1", board11, 1)
#     print_test_case("Multiple Patterns - Player -1", board11, -1)
    
#     # Test Case 12: Edge Cases - Near Board Boundaries
#     board12 = np.array([
#         [1, 1, 1, 0, 0, 0, 0],  # Open three at left edge (only right side open)
#         [0, 0, 0, 0, 0, 0, 0],
#         [0, 0, 0, 0, 1, 1, 1],  # Open three at right edge (only left side open)
#         [0, 0, 0, 0, 0, 0, 0],
#         [0, 0, 0, 0, 0, 0, 0],
#         [0, 0, 0, 0, 0, 0, 0],
#     ])
#     print_test_case("Edge Cases - Near Boundaries", board12, 1)
    
#     # Test Case 13: No Open Threes (Blocked)
#     board13 = np.array([
#         [0, 0, 0, 0, 0, 0, 0],
#         [-1, 1, 1, 1, -1, 0, 0],  # Blocked three (no open ends)
#         [0, 0, 0, 0, 0, 0, 0],
#         [0, 1, 0, 1, 1, 0, 0],  # Almost detached pattern but missing end empty
#         [0, 0, 0, 0, 0, 0, 0],
#     ])
#     print_test_case("No Open Threes (All Blocked)", board13, 1)

# if __name__ == "__main__":
#     test_open_three_detection()