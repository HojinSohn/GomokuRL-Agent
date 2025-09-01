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

        self.punc_val = 3.0  # Exploration value

    def update(self, value_from_rollout):
        """ 
        Update the node with the value from the rollout. Positive value represents good state for the node's current player.
        """
        self.visits += 1
        self.Q_value += (value_from_rollout - self.Q_value) / self.visits

    def get_value(self):
        """ 
        Calculate the value of the node based on PUCT formula.
        """
        return -self.Q_value + self.punc_val * self.probability * np.sqrt(self.parent.visits) / (1 + self.visits)

BOARD_SIZE = 9
class MCTS:
    def __init__(self, policy_value_network: PolicyValueNetwork, iterations=10000):
        """
        iterations: number of playouts (simulation) for one move
        """
        self.policy_value_network = policy_value_network
        self.iterations = iterations

    def expand_node(self, node: Node, game: Game):
        """
        Expand the node by adding all possible child nodes.
        """
        # Get action probabilities from the model for the current board state
        board = game.state[node.current_turn]
        act_probs = self.policy_value_network.get_action_probs(board)
        # print(f"[EXPAND_NODE] Expanding {node.move}, turn={node.current_turn}, got {len(act_probs)} moves")
        if not act_probs:
            print("Expand: No action is available")
            return None
        next_turn = 1 - node.current_turn
        for action, prob in act_probs:
            if action in node.children_moves:
                continue
            row, col = divmod(action, BOARD_SIZE)
            # print(f"   -> Creating child at {(row,col)} with prob={prob:.3f}")
            # Create a new child node for this action
            child_node = Node(next_turn, prob, parent=node, move=(row, col))
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
            board = game.state[node.current_turn]
            value = self.policy_value_network.get_state_value(board)
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

    def get_action_probs(self, game: Game, turn: int):
        """
        Get action probabilities from the model for the current board state.
        Perform Monte Carlo Tree Search (MCTS) to find the best action.
        At expansion, it will use the model to get probability distribution of actions to explore.
        At rollout, it will use the model to get the value of the state, instead of random simulation.
        This function will return the action probabilities based on the MCTS results.

        Game should have not been ended yet.
        """
        action_probs = np.zeros(BOARD_SIZE * BOARD_SIZE)
        # the game should not be done, but it is possible to have indirect win here
        # check the indirect win first
        winner, empty_cell_pos = game.get_winner_indirect()
        if winner == turn:
            action_probs[empty_cell_pos[0] * BOARD_SIZE + empty_cell_pos[1]] = 1.0
            return action_probs
        
        # if the current turn cannot finish the game, check the obvious moves, which is the move required to block immediate row of 5
        # when opponent has row of 4 with one open end, it should be blocked no matter what, given that current turn cannot win this turn
        move = game.check_obvious_move()
        if move is not None:
            action_probs[move] = 1.0
            return action_probs

        # print(f"[INFO] Starting MCTS for turn {turn}")
        self.root = Node(turn, 1.0, parent=None, move=None)

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
