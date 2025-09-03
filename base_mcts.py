
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
    def __init__(self, current_turn, parent=None, move=None):
        self.parent = parent
        self.children = []
        self.children_moves = set()  # Set to track moves of existing children for quick lookup
        self.visits = 0
        self.Q_value = 0.0
        self.move = move # represent the last move made to reach this node
        self.moves = (parent.moves + [move] if parent else [move]) if move is not None else []
        self.current_turn = current_turn
        self.is_terminal = False

    def update(self, value_from_rollout):
        """ 
        Update the node with the value from the rollout. Positive value represents good state for the node's current player.
        """
        self.visits += 1
        self.Q_value += (value_from_rollout - self.Q_value) / self.visits

    def get_value(self):
        """ 
        Calculate the value of the node based on UCB1.
        """
        return -self.Q_value + np.sqrt(2 * self.parent.visits) / (1 + self.visits)

BOARD_SIZE = 9
class BaseMCTS:
    def __init__(self, iterations=10000):
        """
        iterations: number of playouts (simulation) for one move
        """
        self.iterations = iterations

    def rollout(self, game: Game, turn: int):
        """
        Perform a random rollout from the given board state.
        """
        current_turn = turn
        game = copy.deepcopy(game)
        while True:
            winner, empty_cell_pos = game.get_winner_indirect()
            if winner is not None:
                return 1 if winner == current_turn else -1

            # Get a list of all possible actions
            possible_moves = game.get_legal_moves()
            if not possible_moves:
                return 0  # Draw

            # Randomly select an action
            move = random.choice(possible_moves)
            game.do_move(move)
            current_turn = 1 - current_turn

    def expand_node(self, node: Node, game: Game):
        """
        Expand the node by adding all possible child nodes.
        """
        # Get action probabilities from the model for the current board state
        board = game.state[node.current_turn]
        possible_moves = game.get_legal_moves()
        # print(f"[EXPAND_NODE] Expanding {node.move}, turn={node.current_turn}, got {len(act_probs)} moves")
        if not possible_moves:
            print("Expand: No action is available")
            return None
        
        good_moves = []

        for possible_move in possible_moves:
            row, col = possible_move
            if 1 <= row <= 7 and 1 <= col <= 7:
                good_moves.append(possible_move)

        if not good_moves:
            good_moves = possible_moves
        
        next_turn = 1 - node.current_turn
        for move in good_moves:
            action = move[0] * BOARD_SIZE + move[1]
            if action in node.children_moves:
                continue
            row, col = move
            # print(f"   -> Creating child at {(row,col)} with prob={prob:.3f}")
            # Create a new child node for this action
            child_node = Node(next_turn, parent=node, move=(row, col))
            # Add the child node to the current node's children
            node.children.append(child_node)
            node.children_moves.add(action)
            child_node.parent = node

    def search(self, node: Node, game: Game):
        """
        A recursive function to traverse, expand, and backpropagate.
        Returns the value of the state from the parent's perspective.
        The node value will be updated based on the value from current node's perspective, 
        but the value from the parent node's perspective will be returned
        """
        # 1. BASE CASE: Check for a terminal state
        winner, empty_cell_pos = game.get_winner_indirect()
        if winner is not None:
            leaf_value = 1 if winner == node.current_turn else -1
            # print(f"[WINNER] Node {node.move}, turn={node.current_turn}, winner={winner}, visits={node.visits}, leaf_value={leaf_value}")
            node.update(leaf_value)
            return -leaf_value

        # 2. EXPANSION
        if not node.children:
            # print(f"[EXPAND] Expanding node {node.move}, visits={node.visits}")
            value = self.rollout(game, node.current_turn)
            self.expand_node(node, game)
            node.update(value)
            return -value # Return the negated value for the parent

        # 3. SELECTION: If not a leaf, select the best child and recurse
        best_child = max(node.children, key=lambda n: n.get_value())
        # print(f"[SELECT] Node {node.move} at turn {node.current_turn}, choosing child {best_child.move}, visits={best_child.visits}, val={best_child.get_value():.3f}, Q_val={best_child.Q_value:.3f}, prob={best_child.probability:.3f}")
        # 4. RECURSIVE CALL & UNDO
        game.do_move(best_child.move)           # Apply move
        value = self.search(best_child, game)   # Recurse
        game.undo_move()                        # Undo move

        # 5. BACKPROPAGATION: Update the node's stats
        node.update(value)
        # if abs(value) == 1:
        #     print(f"[TERMINAL] Node {best_child.move}, turn={node.current_turn}, value={value}, visits={node.visits}")
        return -value # Return the negated value for the parent
    

    def find_three_threats(self, game: Game, turn: int):
        """
        Finds all "three-threats" for a given player. This includes:
        1. Open Threes:  _XXX_   (e.g., .OOO.)
        2. Broken Threes: _X_XX_  (e.g., .O.OO.)
        3. Broken Threes: _XX_X_  (e.g., .OO.O.)

        Args:
            board (np.ndarray): The game board, where -1 represents an empty spot.
            player (int): The player to check for (e.g., 0 or 1).

        Returns:
            set: A set of (row, col) tuples for all empty spots that create an
                "open four" when played.
        """
        BOARD_SIZE = 9
        EMPTY = 0

        board = game.state[turn]
        player = 1
        
        # Using a set to automatically handle duplicate empty spots
        threat_spots = set()

        # Directions: horizontal, vertical, diagonal down-right, diagonal down-left
        directions = [(0, 1), (1, 0), (1, 1), (1, -1)]
        
        # --- Define all three threat patterns ---
        # Pattern 1: Open Three (_XXX_)
        p_open_three = [EMPTY, player, player, player, EMPTY]
        
        # Pattern 2: Broken Three (_X_XX_)
        p_broken_three_A = [EMPTY, player, EMPTY, player, player, EMPTY]
        
        # Pattern 3: Broken Three (_XX_X_)
        p_broken_three_B = [EMPTY, player, player, EMPTY, player, EMPTY]

        for r in range(BOARD_SIZE):
            for c in range(BOARD_SIZE):
                for dr, dc in directions:
                    # --- Check for Open Three (length 5 pattern) ---
                    line_coords_5 = [(r + i * dr, c + i * dc) for i in range(5)]
                    if all(0 <= nr < BOARD_SIZE and 0 <= nc < BOARD_SIZE for nr, nc in line_coords_5):
                        line_values = [board[nr][nc] for nr, nc in line_coords_5]
                        if line_values == p_open_three:
                            # For _XXX_, both empty ends are threat spots
                            threat_spots.add(line_coords_5[0])
                            threat_spots.add(line_coords_5[4])

                    # --- Check for Broken Threes (length 6 patterns) ---
                    line_coords_6 = [(r + i * dr, c + i * dc) for i in range(6)]
                    if all(0 <= nr < BOARD_SIZE and 0 <= nc < BOARD_SIZE for nr, nc in line_coords_6):
                        line_values = [board[nr][nc] for nr, nc in line_coords_6]
                        
                        # Check for _X_XX_
                        if line_values == p_broken_three_A:
                            # The key empty spot is in the middle
                            threat_spots.add(line_coords_6[2])
                        
                        # Check for _XX_X_
                        elif line_values == p_broken_three_B:
                            # The key empty spot is in the middle
                            threat_spots.add(line_coords_6[3])

        return threat_spots


    def get_action_probs(self, game: Game, turn: int):
        """
        Get action probabilities from the model for the current board state.
        Perform Monte Carlo Tree Search (MCTS) to find the best action.
        At expansion, it will use the model to get probability distribution of actions to explore.
        At rollout, it will use the model to get the value of the state, instead of random simulation.
        This function will return the action probabilities based on the MCTS results.

        Game should have not been ended yet.
        """

        print("Getting action probabilities from base MCTS")
        action_probs = np.zeros(BOARD_SIZE * BOARD_SIZE)

        # # the game should not be done, but it is possible to have indirect win here
        # # check the indirect win first
        # winner, empty_cell_pos = game.get_winner_indirect()
        # if winner == turn:
        #     action_probs[empty_cell_pos[0] * BOARD_SIZE + empty_cell_pos[1]] = 1.0
        #     return action_probs
        
        # if the current turn cannot finish the game, check the obvious moves, which is the move required to block immediate row of 5
        # when opponent has row of 4 with one open end, it should be blocked no matter what, given that current turn cannot win this turn
        moves = game.check_obvious_moves()
        if moves and len(moves) > 0:
            for move in moves:
                action_probs[move] = 1.0
            # normalize
            action_probs /= action_probs.sum()
            return action_probs
        
        # check open three of current
        open_three_moves = self.find_three_threats(game, turn)
        if len(open_three_moves) > 0:
            for move in open_three_moves:
                action = move[0] * BOARD_SIZE + move[1]
                action_probs[action] = 1.0
            # normalize
            action_probs /= action_probs.sum()
            return action_probs
        

        # check open three of opponent
        open_three_moves = self.find_three_threats(game, 1 - turn)
        if len(open_three_moves) > 0:
            for move in open_three_moves:
                action = move[0] * BOARD_SIZE + move[1]
                action_probs[action] = 1.0
            # normalize
            action_probs /= action_probs.sum()
            return action_probs

        # print(f"[INFO] Starting MCTS for turn {turn}")
        self.root = Node(turn, parent=None, move=None)

        for _ in range(self.iterations):
            self.search(self.root, game)

        # Calculate action probabilities based on visits
        total_child_visits = sum(child.visits for child in self.root.children)

        if total_child_visits == 0:
            print('This should not happen: No visits recorded for any child nodes')
            print('Parent node visits:', self.root.visits)
            print('Board:', game.state[turn])
            print('len of root.children:', len(self.root.children))
            for child in self.root.children:
                print(f'Child move: {child.move}, visits: {child.visits}, its children length: {len(child.children)}')
            return action_probs

        for child in self.root.children:
            if child.visits > 0:
                action = child.move[0] * BOARD_SIZE + child.move[1]
                action_probs[action] = child.visits / total_child_visits
        
        return action_probs
