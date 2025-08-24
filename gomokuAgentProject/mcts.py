# Node class for Tree Search
import random
import numpy as np
import copy

from game import Game
from policy_value_network import PolicyValueNetwork

"""
Stone: 1 (player's stone) and -1 (opponent's stone)
Turn: 0 and 1
"""
class Node:
    def __init__(self, current_turn, probability, parent=None, move=None):
        self.parent = parent
        self.children = []
        self.children_moves = set()  # Set to track moves of existing children for quick lookup
        self.visits = 0
        self.Q_value = 0.0
        self.move = move # represent the last move made to reach this node
        self.moves = (parent.moves + [move] if parent else [move]) if move is not None else []
        self.current_turn = current_turn
        self.is_terminal = False
        self.probability = probability

        self.punc_val = 5

    def update(self, value_from_rollout):
        """ Update the node with the value from the rollout. Negative value indicates a loss for the current player.
        """
        self.visits += 1
        self.Q_value += (value_from_rollout - self.Q_value) / self.visits

    def get_value(self):
        """ Calculate the value of the node based on Q-value and UCB1 formula.
        """
        return self.Q_value + self.punc_val * self.probability * np.sqrt(self.parent.visits) / (1 + self.visits)

BOARD_SIZE = 9
class MCTS:
    def __init__(self, policy_value_network: PolicyValueNetwork, iterations=10000):
        self.policy_value_network = policy_value_network
        self.iterations = iterations

    def apply_moves_to_board(self, board, moves, first_move_stone):
        """Apply a sequence of moves to a board, alternating players"""
        board = board.copy()  # Create a copy to avoid modifying the original
        for i, move in enumerate(moves):
            if move is not None:
                row, col = move
                # Alternate players: first_move_stone, then -first_move_stone, etc.
                player = first_move_stone if i % 2 == 0 else -first_move_stone
                board[row][col] = player
        return board

    def undo_moves_from_board(self, board, moves, initial_stone):
        """Undo a sequence of moves from a board, alternating players"""
        board = board.copy()  # Create a copy to avoid modifying the original
        for move in moves:
            if move is not None:
                row, col = move
                board[row][col] = 0
        return board
    
    def traverse_down_to_leaf_node(self, node):
        """ This function traverses the tree from the element node to a leaf node,
            selecting child nodes based on the UCB1 score.
        """
        while node.children:
            node = max(node.children, key=lambda n: n.get_value())
        return node
    
    def expand(self, node: Node, game: Game):
        '''
            This function expands a leaf node by adding child nodes for legal or promising next moves.
            It will be called when a leaf node is reached second time during the traversal.

            @param node: Leaf node (board state) to expand
            @param board: Current board state (2D array) 9x9. 0 = empty, 1 = player, -1 = opponent
            @return: New child node added to the tree, or None if no promising moves are available
        '''
        # need to get the board state from the current turn's perspective
        # get the root's board state from the current turn's perspective
        root_board = game.state[node.current_turn]
        # first move turn is the root's current turn
        # but since the board state is from the current turn's perspective, we need to adjust
        if self.root.current_turn == node.current_turn:
            # in this case, first move is 1
            first_move_stone = 1
        else:
            # in this case, first move is -1
            first_move_stone = -1
        current_board = self.apply_moves_to_board(root_board, node.moves, first_move_stone)

        # action probabilities from the current state
        act_probs = self.policy_value_network.get_action_probs(current_board)
        if not act_probs:
            print("Expand: No action is available")
            return None
        next_turn = 1 - node.current_turn
        for action, prob in act_probs:
            if action in node.children_moves:
                continue
            row, col = divmod(action, BOARD_SIZE)
            # Create a new child node for this action
            child_node = Node(next_turn, prob, parent=node, move=(row, col))
            # Add the child node to the current node's children
            node.children.append(child_node)
            node.children_moves.add(action)
            child_node.parent = node

    def check_end(self, node: Node, game: Game):
        """
        Checks if the player who just moved has won by detecting five consecutive stones
        including the most recent move.
        """
        board = game.state[node.current_turn]
        board = self.apply_moves_to_board(board, node.moves, 1)
        r, c = node.move
        if not (0 <= r < BOARD_SIZE and 0 <= c < BOARD_SIZE and board[r, c] == 1):
            return None

        # Directions: horizontal, vertical, main diagonal, anti-diagonal
        directions = [(0, 1), (1, 0), (1, 1), (1, -1)]

        for dr, dc in directions:
            # Count consecutive stones in both directions (e.g., left and right for horizontal)
            count = 1  # Include the current move
            # Forward direction
            for i in range(1, 5):  # Check up to 4 cells forward
                nr, nc = r + dr * i, c + dc * i
                if not (0 <= nr < BOARD_SIZE and 0 <= nc < BOARD_SIZE and board[nr, nc] == 1):
                    break
                count += 1
            # Backward direction
            for i in range(1, 5):  # Check up to 4 cells backward
                nr, nc = r - dr * i, c - dc * i
                if not (0 <= nr < BOARD_SIZE and 0 <= nc < BOARD_SIZE and board[nr, nc] == 1):
                    break
                count += 1
            # Check if we have 5 or more consecutive stones
            if count >= 5:
                return node.current_turn  # Return the winning player (0 or 1)

        return None

    def rollout(self, node: Node, game: Game):
        """ Perform a rollout from the given node to a terminal state. It won't use random simulation, 
            but will use the value network to get the value of the state.
        """
        # need to get the board state from the current turn's perspective
        # get the root's board state from the current turn's perspective
        root_board = game.state[node.current_turn]
        # first move turn is the root's current turn
        # but since the board state is from the current turn's perspective, we need to adjust
        if self.root.current_turn == node.current_turn:
            # in this case, first move is 1
            first_move_stone = 1
        else:
            # in this case, first move is -1
            first_move_stone = -1
        current_board = self.apply_moves_to_board(root_board, node.moves, first_move_stone)

        # Get the value from the policy value network by passing the current board state from the current turn's perspective
        value = self.policy_value_network.get_state_value(current_board)

        # if value < 0:
        #     # If the value is negative, current player lost
        #     value = -1
        # elif value > 0:
        #     # If the value is positive, current player won
        #     value = 1
        # else:
        #     # If the value is zero, draw
        #     value = 0

        # This value is the result for the current turn
        return value

    def backpropagate(self, node, result):
        """ Backpropagate the result up the tree to update the nodes.
        """
        while node is not None:
            node.update(result)
            result = -result  # Flip the result for the parent node
            node = node.parent

    def get_action_probs(self, game: Game, turn: int):
        """
        Get action probabilities from the model for the current board state.
        Perform Monte Carlo Tree Search (MCTS) to find the best action.
        At expansion, it will use the model to get probability distribution of actions to explore.
        At rollout, it will use the model to get the value of the state, instead of random simulation.
        This function will return the action probabilities based on the MCTS results.

        Game should have not been ended yet.
        """
        self.root = Node(turn, 1.0, parent=None, move=None) 
        for it in range(self.iterations):
            # traverse down the tree to a leaf node
            leaf_node = self.traverse_down_to_leaf_node(self.root)
            winner = self.check_end(leaf_node, game) if leaf_node != self.root else None
            if winner is not None:
                if winner == leaf_node.current_turn:
                    leaf_node_result = 1
                else:
                    leaf_node_result = -1
            else:
                # Expand leaf node
                self.expand(leaf_node, game)
                if not leaf_node.children:
                    # If no children were added, it means no legal moves are available
                    winner = self.check_end(leaf_node, game) if leaf_node != self.root else None
                    if winner is not None:
                        if winner == leaf_node.current_turn:
                            leaf_node_result = 1
                        else:
                            leaf_node_result = -1
                    else:
                        # If no winner can be determined, it's a draw
                        leaf_node_result = 0
                else:
                    leaf_node_result = self.rollout(leaf_node, game)
            self.backpropagate(leaf_node, leaf_node_result)

        # Calculate action probabilities based on visits
        action_probs = np.zeros(BOARD_SIZE * BOARD_SIZE)
        if not self.root.children:
            print(f"Root has no children.")
            print("Board state:")
            print(game.state[self.root.current_turn])
        for child in self.root.children:
            if child.visits > 0:
                action = child.move[0] * BOARD_SIZE + child.move[1]
                action_probs[action] = child.visits / (self.root.visits - 1)
        
        return action_probs

    # def get_reasonable_moves(self, board, current_turn, last_move=None):
    #     # return any empty cells in the center 5x5 area
    #     rows, cols = board.shape
    #     center_rows = range(max(0, rows // 2 - 2), min(rows, rows // 2 + 3))
    #     center_cols = range(max(0, cols // 2 - 2), min(cols, cols // 2 + 3))
    #     empty_cells = [(r, c) for r in center_rows for c in center_cols if board[r, c] == 0]

    #     if not empty_cells:
    #         # If no empty cells in the center, return all empty cells
    #         empty_cells = [(r, c) for r in range(rows) for c in range(cols) if board[r, c] == 0]
    #     return empty_cells

    # def get_obvious_moves(self, board, current_turn):
    #     '''
    #         This function generates obvious moves based on the current board state.
    #         It checks for open four and open three positions, and returns them.

    #         @param board: Current board state (2d array) 9 x 9. 0 = empty, 1 = player, -1 = opponent
    #         @param current_turn: Current player (1 or -1)
    #         @return: List of obvious moves, prioritized by importance
    #     '''
    #     good_moves = self.detect_open_four(board, current_turn)
    #     if good_moves:
    #         return good_moves

    #     good_moves = self.detect_open_four(board, -1 * current_turn)
    #     if good_moves:
    #         return good_moves
        
    #     good_moves = self.detect_open_three(board, current_turn)
    #     if good_moves:
    #         return good_moves
        
    #     good_moves = self.detect_open_three(board, -1 * current_turn)
    #     if good_moves:
    #         return good_moves
    #     return []

    # def get_obvious_moves_light(self, board, current_turn, last_move=None):
    #     """Lightweight version of get_obvious_moves for rollouts."""
    #     moves = []
    #     if last_move is None:
    #         return moves  # Fall back to reasonable moves if no last move
    #     r, c = last_move
    #     directions = [(0, 1), (1, 0), (1, 1), (1, -1)]
    #     for dr, dc in directions:
    #         for i in [-4, -3, -2, -1, 1, 2, 3, 4]:  # Check nearby cells
    #             nr, nc = r + dr * i, c + dc * i
    #             if 0 <= nr < BOARD_SIZE and 0 <= nc < BOARD_SIZE and board[nr, nc] == 0:
    #                 # Check if placing a stone here creates an open four or blocks one
    #                 board[nr, nc] = current_turn
    #                 if self.check_winner(board, current_turn, (nr, nc)):
    #                     moves.append((nr, nc)) 
    #                 elif self.check_winner(board, -current_turn, (nr, nc)):
    #                     moves.append((nr, nc))
    #                 board[nr, nc] = 0  # Undo
    #     return list(set(moves))

    # def add_open_three(self, board, player, r, c, good_moves):
    #     """
    #     Adds an open three pattern to the board for the given player.
    #     Open three: _XXX_ (three consecutive stones with empty ends).

    #     @param board: 9x9 board (2D list/array). 0 = empty, 1 = player, -1 = opponent
    #     @param player: Player to check (1 or -1)
    #     @param row: Row index of the open three
    #     @param col: Column index of the open three
    #     """
    #     rows, cols = len(board), len(board[0])
    #     directions = [(0, 1), (1, 0), (1, 1), (1, -1)]  # Horizontal, vertical, main diag, anti-diag
    #     # Helper function to check if a cell is empty
    #     def is_empty(r, c):
    #         return 0 <= r < rows and 0 <= c < cols and board[r][c] == 0

    #     # Helper function to check if a cell matches the player
    #     def is_player(r, c):
    #         return 0 <= r < rows and 0 <= c < cols and board[r][c] == player
        
    #     for dr, dc in directions:
    #         # Check open three: _XXX_
    #         if is_player(r, c) and is_player(r + dr, c + dc) and is_player(r + 2*dr, c + 2*dc):
    #             # Check both ends: (r-dr, c-dc) and (r+3*dr, c+3*dc)
    #             left_end = (r - dr, c - dc)
    #             right_end = (r + 3*dr, c + 3*dc)
    #             if is_empty(*left_end) and is_empty(*right_end):
    #                 good_moves.add(left_end)
    #                 good_moves.add(right_end)

    #         # Check detached three pattern 1: _X_XX_ (gap at position 2)
    #         if (is_empty(r, c) and is_player(r + dr, c + dc) and is_empty(r + 2*dr, c + 2*dc) and
    #             is_player(r + 3*dr, c + 3*dc) and is_player(r + 4*dr, c + 4*dc) and
    #             is_empty(r + 5*dr, c + 5*dc)):
    #             good_moves.add((r, c))  # First empty
    #             good_moves.add((r + 2*dr, c + 2*dc))  # Middle empty
    #             good_moves.add((r + 5*dr, c + 5*dc))  # Last empty

    #         # Check detached three pattern 2: _XX_X_ (gap at position 3)
    #         if (is_empty(r, c) and is_player(r + dr, c + dc) and is_player(r + 2*dr, c + 2*dc) and
    #             is_empty(r + 3*dr, c + 3*dc) and is_player(r + 4*dr, c + 4*dc) and
    #             is_empty(r + 5*dr, c + 5*dc)):
    #             good_moves.add((r, c))  # First empty
    #             good_moves.add((r + 3*dr, c + 3*dc))  # Middle empty
    #             good_moves.add((r + 5*dr, c + 5*dc))  # Last empty

    
    # def add_open_four(self, board, player, r, c, good_moves):
    #     rows, cols = len(board), len(board[0])
    #     directions = [(0, 1), (1, 0), (1, 1), (1, -1)]  # Horizontal, vertical, main diag, anti-diag
    #     # Helper function to check if a cell is empty
    #     def is_empty(r, c):
    #         return 0 <= r < rows and 0 <= c < cols and board[r][c] == 0

    #     # Helper function to check if a cell matches the player
    #     def is_player(r, c):
    #         return 0 <= r < rows and 0 <= c < cols and board[r][c] == player
        
    #     for dr, dc in directions:
    #         # Check open four: _XXXX_ or _XXXX, XXXX_
    #         if (is_player(r, c) and is_player(r + dr, c + dc) and 
    #             is_player(r + 2*dr, c + 2*dc) and is_player(r + 3*dr, c + 3*dc)):
    #             # Check left end: (r-dr, c-dc)
    #             left_end = (r - dr, c - dc)
    #             if is_empty(*left_end):
    #                 good_moves.add(left_end)
    #             # Check right end: (r+4*dr, c+4*dc)
    #             right_end = (r + 4*dr, c + 4*dc)
    #             if is_empty(*right_end):
    #                 good_moves.add(right_end)
    #         # check open four: XX_XX or XXX_X or X_XXX
    #         elif (is_player(r, c) and is_player(r + dr, c + dc) and 
    #             is_empty(r + 2*dr, c + 2*dc) and is_player(r + 3*dr, c + 3*dc) and is_player(r + 4*dr, c + 4*dc)):
    #             good_moves.add((r + 2*dr, c + 2*dc))  # Add the empty cell in the middle
    #         elif (is_player(r, c) and is_empty(r + dr, c + dc) and 
    #             is_player(r + 2*dr, c + 2*dc) and is_player(r + 3*dr, c + 3*dc) and is_player(r + 4*dr, c + 4*dc)):
    #             good_moves.add((r + dr, c + dc))
    #         elif (is_player(r, c) and is_player(r + dr, c + dc) and is_player(r + 2*dr, c + 2*dc) and 
    #             is_empty(r + 3*dr, c + 3*dc) and is_player(r + 4*dr, c + 4*dc)):
    #             good_moves.add((r + 3*dr, c + 3*dc))
    
    # def detect_open_three(self, board, player):
    #     """
    #     Detects open three and detached three patterns for the given player.
    #     Open three: _XXX_ (three consecutive stones with empty ends).
    #     Detached three: _X_XX_ or _XX_X_ (three stones with one gap and empty ends).

    #     @param board: 9x9 board (2D list/array). 0 = empty, 1 = player, -1 = opponent
    #     @param player: Player to check (1 or -1)
    #     @return: List of (row, col) tuples for empty cells that complete open/detached three patterns
    #     """
    #     rows, cols = len(board), len(board[0])
    #     good_moves = set()

    #     for r in range(rows):
    #         for c in range(cols):
    #             self.add_open_three(board, player, r, c, good_moves)

    #     return list(good_moves)

    # def detect_open_four(self, board, player):
    #     """
    #     Detects open four patterns for the given player.
    #     Open four: _XXXX_ or _XXXX, XXXX_ (four consecutive stones with at least one empty end).

    #     @param board: 9x9 board (2D list/array). 0 = empty, 1 = player, -1 = opponent
    #     @param player: Player to check (1 or -1)
    #     @return: List of (row, col) tuples for empty cells that complete open four patterns
    #     """
    #     rows, cols = len(board), len(board[0])
    #     good_moves = set()

    #     for r in range(rows):
    #         for c in range(cols):
    #             self.add_open_four(board, player, r, c, good_moves)

    #     return list(good_moves)

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
