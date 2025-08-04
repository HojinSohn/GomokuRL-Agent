# Node class for Tree Search
import numpy as np


class Node:
    def __init__(self, element, parent=None, move=None):
        self.element = element
        self.parent = parent
        self.children = []
        self.visits = 0
        self.wins = 0.0
        self.move = move

    def ucb1(self):
        if self.visits == 0:
            return float('inf')
        return self.wins / self.visits + (2 * np.log(self.parent.visits) / self.visits) ** 0.5


class MCTS:
    def __init__(self, iterations=5000):
        self.iterations = iterations

    def traverse(self, node):
        '''
            This function traverses the tree from the element node to a leaf node,
            selecting child nodes based on the UCB1 score.

            @param node: Node to traverse
            @return: Leaf node after traversing down the tree
        '''
        while node.children:
            node = max(node.children, key=lambda n: self.get_ucb1(n))
        return node
    
    def get_promising_moves(self, board):
        '''
            This function generates promising moves based on the current board state.
            It should return a list of promising moves.

            @param board: Current board state (2d array) 15 x 15. 0 = empty, 1 = player, -1 = opponent
            @return: List of promising moves
        '''
        # Implement logic to find promising moves based on the current board state
        '''
        1. For any open four consecutive pieces of the opponent.
        2. For any open three consecutive pieces of the opponent.
        3. For any open three consecutive pieces of the player.
        4. For any open four consecutive pieces of the player.
        5. Any other cells that are adjacent to existing pieces. (at most 4 cells away)
        '''
        promising_moves = []
        
        direction_vectors = [
                (1, 0),   # vertical
                (0, 1),   # horizontal
                (1, 1),   # diagonal down-right
                (-1, 1),  # diagonal up-right
            ]
        sequence_length = 4

        for i in range(15):
            for j in range(15):
                for dx, dy in direction_vectors:
                    for k in [1, -1]:
                        if all(0 <= i + dx * n < 15 and 0 <= j + dy * n < 15 and board[i + dx * n][j + dy * n] == k for n in range(sequence_length)):
                            if i + dx * sequence_length < 15 and j + dy * sequence_length < 15 and board[i + dx * sequence_length][j + dy * sequence_length] == 0:
                                promising_moves.append((i + dx * sequence_length, j + dy * sequence_length))
                            if i - dx >= 0 and j - dy >= 0 and board[i - dx][j - dy] == 0:
                                promising_moves.append((i - dx * sequence_length, j - dy * sequence_length))
                        elif all(0 <= i + dx * n < 15 and 0 <= j + dy * n < 15 and board[i + dx * n][j + dy * n] == k for n in range(sequence_length - 1)):
                            if i + dx * (sequence_length - 1) < 15 and j + dy * (sequence_length - 1) < 15 and board[i + dx * (sequence_length - 1)][j + dy * (sequence_length - 1)] == 0:
                                promising_moves.append((i + dx * (sequence_length - 1), j + dy * (sequence_length - 1)))
                            if i - dx >= 0 and j - dy >= 0 and board[i - dx][j - dy] == 0:
                                promising_moves.append((i - dx * (sequence_length - 1), j - dy * (sequence_length - 1)))

        # add any other cells that are adjacent to existing pieces
        # This is a heuristic to find promising moves
        # It will add any cell that is adjacent to an existing piece
        visited = [[False] * 15 for _ in range(15)]
        for i in range(15):
            for j in range(15):
                # If the cell is occupied, check its neighbors
                if board[i][j] != 0:
                    for dx in [-2, -1, 0, 1, 2]:
                        for dy in [-2, -1, 0, 1, 2]:
                            if dx == 0 and dy == 0:
                                continue
                            ni, nj = i + dx, j + dy
                            if not visited[ni][nj] and 0 <= ni < 15 and 0 <= nj < 15 and board[ni][nj] == 0:
                                visited[ni][nj] = True
                                promising_moves.append((ni, nj)) 
        return promising_moves

    def expand(self, node):
        '''
            This function expands a leaf node by adding child nodes for legal or promising next moves.
            It will be called when a leaf node is reached second time during the traversal.

            @param node: Leaf node (board state) to expand
        '''
        # Implement logic to generate 10 promising moves based on the current board state
        for _ in range(10):
            # Here apply a random legal move to new_board
            # Heuristically, find 10 promising moves
            promising_moves = self.get_promising_moves(new_board)
            for move in promising_moves:
                new_board = np.copy(node.element)
                # Apply the move to the new board state
                new_board[move[0]][move[1]] = 1
                # Update the child node's element to the new board state
                child_node.element = new_board
                child_node = Node(element=new_board, parent=node, move=move)
                node.children.append(child_node)

    def check_winner(self, board):
        '''
            This function checks if there is a winner in the current board state.
            It will be called during the rollout to determine if the game is completed.

            @param board: Current board state (2d array) 15 x 15. 0 = empty, 1 = player, -1 = opponent
            @return: True if there is a winner, False otherwise
        '''
        # Implement logic to check if there is a winner in the current board state
        direction_vectors = [
                (1, 0),   # vertical
                (0, 1),   # horizontal
                (1, 1),   # diagonal down-right
                (-1, 1),  # diagonal up-right
            ]
        sequence_length = 5

        for i in range(15):
            for j in range(15):
                for dx, dy in direction_vectors:
                    for k in [1, -1]:
                        try:
                            if all(0 <= i + dx * n < 15 and 0 <= j + dy * n < 15 and board[i + dx * n][j + dy * n] == k for n in range(sequence_length)):
                                return True
                        except IndexError:
                            continue
        return False
        
    def rollout(self, node):
        '''
            This function simulates a random game from the current board state.
            It will be called when a leaf node is reached for the first time during the traversal.

            @param node: Leaf node (board state) to simulate
        '''
        # Implement logic to simulate a random game from the current board state
        # For simplicity, we can just randomly select moves until the game ends
        # deep copy the board state
        current_board = np.copy(node.element)
        while True:
            # Randomly select a move
            valid_moves = [(i, j) for i in range(15) for j in range(15) if current_board[i][j] == 0]
            if not valid_moves:
                break
            move = np.random.choice(len(valid_moves))
            current_board[valid_moves[move][0]][valid_moves[move][1]] = 1  # Assume player 1 is making the move
            # Check if the game is completed
            if self.check_winner(current_board):
                node.wins += 1
                node.visits += 1
            # Switch to the other player
            current_board[valid_moves[move][0]][valid_moves[move][1]] = -1  # Assume player -1 is making the move
            # Check if the game is completed
            if self.check_winner(current_board):
                node.wins -= 1
                node.visits += 1
        # traverse back up the tree and update the visit counts and wins
        while node.parent is not None:
            node.parent.visits += 1
            node.parent.wins += node.wins
            node = node.parent
        

    def get_action(self, board):
        '''
            This function performs MCTS to find the best action for the current board state.

            @param board: Current board state (2d array) 15 x 15. 0 = empty, 1 = player, -1 = opponent
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
        root = Node(element=board)

        for _ in range(self.iterations):
            # traverse down the tree to a leaf node
            leaf_node = self.traverse(root)
            if leaf_node.visits > 0:
                # If the leaf node has been visited before, expand it
                self.expand(leaf_node)
                children_nodes = leaf_node.children
                if children_nodes:
                    # Since childrens are just expanded, select one of them randomly
                    selected_node = np.random.choice(children_nodes)
                    self.rollout(selected_node)
            else:
                # If the leaf node has not been visited before, simulate a random game from this state
                self.rollout(leaf_node)
        
        # After all iterations, select the child node with the highest visit count
        best_child = max(root.children, key=lambda n: n.visits)
        # Convert the best child's element (board state) to an action
        best_action = best_child.move
        if best_action is None:
            return None
        # Convert the best action (row, col) to a single action index
        return best_action[0] * 15 + best_action[1] 