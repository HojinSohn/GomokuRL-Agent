# Node class for Tree Search
import numpy as np
import copy
from scipy.signal import convolve2d
class Node:
    def __init__(self, current_player, parent=None, move=None):
        self.parent = parent
        self.children = []
        self.visits = 0
        self.wins = 0.0
        self.move = move
        self.moves = (parent.moves + [move] if parent else [move]) if move is not None else []
        self.current_player = current_player

    def ucb1(self):
        if self.visits == 0:
            return float('inf')
        return self.wins / self.visits + np.sqrt((2 * np.log(self.parent.visits)) / self.visits)


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
        current_board = copy.deepcopy(board)
        
        for i, move in enumerate(moves):
            if move is not None:
                row, col = move
                # Alternate players: initial_player, then -initial_player, etc.
                player = initial_player if i % 2 == 0 else -initial_player
                current_board[row][col] = player
        return current_board
    
    def traverse(self, node):
        '''
            This function traverses the tree from the element node to a leaf node,
            selecting child nodes based on the UCB1 score.

            @param node: Node to traverse
            @return: Leaf node after traversing down the tree
        '''
        while node.children:
            node = max(node.children, key=lambda n: n.ucb1())
        return node

    def detect_open_three(self, board, player):
        kernel_h = np.array([[1, 1, 1]])
        kernel_v = kernel_h.T
        kernel_d1 = np.eye(3, dtype=int)
        kernel_d2 = np.fliplr(kernel_d1)

        player_mask = (board == player).astype(int)
        empty_mask = (board == 0).astype(int)

        good_moves = set()
        # Horizontal
        conv_h = convolve2d(player_mask, kernel_h, mode='valid')
        for r in range(conv_h.shape[0]):
            for c in range(conv_h.shape[1]):
                if conv_h[r, c] == 3:
                    # Check ends at (r, c-1) and (r, c+3)
                    left_end = c - 1
                    right_end = c + 3
                    if left_end >= 0 and empty_mask[r, left_end] and right_end < board.shape[1] and empty_mask[r, right_end]:
                        good_moves.add((r, left_end))
                        good_moves.add((r, right_end))

        # Vertical
        conv_v = convolve2d(player_mask, kernel_v, mode='valid')
        for r in range(conv_v.shape[0]):
            for c in range(conv_v.shape[1]):
                if conv_v[r, c] == 3:
                    top_end = r - 1
                    bottom_end = r + 3
                    if top_end >= 0 and empty_mask[top_end, c] and bottom_end < board.shape[0] and empty_mask[bottom_end, c]:
                        good_moves.add((top_end, c))
                        good_moves.add((bottom_end, c))

        # Diagonal ↘
        conv_d1 = convolve2d(player_mask, kernel_d1, mode='valid')
        for r in range(conv_d1.shape[0]):
            for c in range(conv_d1.shape[1]):
                if conv_d1[r, c] == 3:
                    # Check ends diagonally: (r-1, c-1) and (r+3, c+3)
                    top_left = (r - 1, c - 1)
                    bottom_right = (r + 3, c + 3)
                    if 0 <= top_left[0] < board.shape[0] and 0 <= top_left[1] < board.shape[1] and empty_mask[top_left] and \
                        0 <= bottom_right[0] < board.shape[0] and 0 <= bottom_right[1] < board.shape[1] and empty_mask[bottom_right]:
                        good_moves.add(top_left)
                        good_moves.add(bottom_right)

        # Anti-diagonal ↙
        conv_d2 = convolve2d(player_mask, kernel_d2, mode='valid')
        for r in range(conv_d2.shape[0]):
            for c in range(conv_d2.shape[1]):
                if conv_d2[r, c] == 3:
                    # Check ends anti-diagonally: (r-1, c+3) and (r+3, c-1)
                    top_right = (r - 1, c + 3)
                    bottom_left = (r + 3, c - 1)
                    if 0 <= top_right[0] < board.shape[0] and 0 <= top_right[1] < board.shape[1] and empty_mask[top_right] and \
                        0 <= bottom_left[0] < board.shape[0] and 0 <= bottom_left[1] < board.shape[1] and empty_mask[bottom_left]:
                        good_moves.add(top_right)
                        good_moves.add(bottom_left)

        def check_detached_patterns():
            """Check for detached patterns using vectorized numpy operations for speed"""
            
            # Define the two patterns we're looking for
            pattern1 = np.array([0, player, 0, player, player, 0])  # _X_XX_
            pattern2 = np.array([0, player, player, 0, player, 0])  # _XX_X_
            
            def find_pattern_matches(segments, pattern):
                """Vectorized pattern matching"""
                return np.all(segments == pattern, axis=-1)
            
            def check_horizontal_detached():
                if board.shape[1] < 6:
                    return
                    
                # Extract all horizontal 6-segments
                segments = []
                positions = []
                for r in range(board.shape[0]):
                    for c in range(board.shape[1] - 5):
                        segment = board[r, c:c+6]
                        segments.append(segment)
                        positions.append((r, c))
                
                if not segments:
                    return
                    
                segments = np.array(segments)
                
                # Find matches for both patterns
                matches1 = find_pattern_matches(segments, pattern1)
                matches2 = find_pattern_matches(segments, pattern2)
                
                # Add good moves for pattern 1: _X_XX_
                for i, match in enumerate(matches1):
                    if match:
                        r, c = positions[i]
                        good_moves.add((r, c))      # First empty
                        good_moves.add((r, c+2))    # Middle empty  
                        good_moves.add((r, c+5))    # Last empty
                
                # Add good moves for pattern 2: _XX_X_
                for i, match in enumerate(matches2):
                    if match:
                        r, c = positions[i]
                        good_moves.add((r, c))      # First empty
                        good_moves.add((r, c+3))    # Middle empty
                        good_moves.add((r, c+5))    # Last empty
            
            def check_vertical_detached():
                if board.shape[0] < 6:
                    return
                    
                # Extract all vertical 6-segments
                segments = []
                positions = []
                for r in range(board.shape[0] - 5):
                    for c in range(board.shape[1]):
                        segment = board[r:r+6, c]
                        segments.append(segment)
                        positions.append((r, c))
                
                if not segments:
                    return
                    
                segments = np.array(segments)
                
                # Find matches for both patterns
                matches1 = find_pattern_matches(segments, pattern1)
                matches2 = find_pattern_matches(segments, pattern2)
                
                # Add good moves for pattern 1: _X_XX_
                for i, match in enumerate(matches1):
                    if match:
                        r, c = positions[i]
                        good_moves.add((r, c))      # First empty
                        good_moves.add((r+2, c))    # Middle empty  
                        good_moves.add((r+5, c))    # Last empty
                
                # Add good moves for pattern 2: _XX_X_
                for i, match in enumerate(matches2):
                    if match:
                        r, c = positions[i]
                        good_moves.add((r, c))      # First empty
                        good_moves.add((r+3, c))    # Middle empty
                        good_moves.add((r+5, c))    # Last empty
            
            def check_diagonal_detached():
                if board.shape[0] < 6 or board.shape[1] < 6:
                    return
                    
                # Extract all main diagonal 6-segments
                segments = []
                positions = []
                for r in range(board.shape[0] - 5):
                    for c in range(board.shape[1] - 5):
                        segment = np.array([board[r+i, c+i] for i in range(6)])
                        segments.append(segment)
                        positions.append((r, c))
                
                if not segments:
                    return
                    
                segments = np.array(segments)
                matches1 = find_pattern_matches(segments, pattern1)
                matches2 = find_pattern_matches(segments, pattern2)
                
                # Add good moves for pattern 1: _X_XX_
                for i, match in enumerate(matches1):
                    if match:
                        r, c = positions[i]
                        good_moves.add((r, c))          # First empty
                        good_moves.add((r+2, c+2))      # Middle empty  
                        good_moves.add((r+5, c+5))      # Last empty
                
                # Add good moves for pattern 2: _XX_X_
                for i, match in enumerate(matches2):
                    if match:
                        r, c = positions[i]
                        good_moves.add((r, c))          # First empty
                        good_moves.add((r+3, c+3))      # Middle empty
                        good_moves.add((r+5, c+5))      # Last empty
            
            def check_anti_diagonal_detached():
                if board.shape[0] < 6 or board.shape[1] < 6:
                    return
                    
                # Extract all anti-diagonal 6-segments
                segments = []
                positions = []
                for r in range(board.shape[0] - 5):
                    for c in range(5, board.shape[1]):
                        segment = np.array([board[r+i, c-i] for i in range(6)])
                        segments.append(segment)
                        positions.append((r, c))
                
                if not segments:
                    return
                    
                segments = np.array(segments)
                matches1 = find_pattern_matches(segments, pattern1)
                matches2 = find_pattern_matches(segments, pattern2)
                
                # Add good moves for pattern 1: _X_XX_
                for i, match in enumerate(matches1):
                    if match:
                        r, c = positions[i]
                        good_moves.add((r, c))          # First empty
                        good_moves.add((r+2, c-2))      # Middle empty  
                        good_moves.add((r+5, c-5))      # Last empty
                
                # Add good moves for pattern 2: _XX_X_
                for i, match in enumerate(matches2):
                    if match:
                        r, c = positions[i]
                        good_moves.add((r, c))          # First empty
                        good_moves.add((r+3, c-3))      # Middle empty
                        good_moves.add((r+5, c-5))      # Last empty
            
            check_horizontal_detached()
            check_vertical_detached() 
            check_diagonal_detached()
            check_anti_diagonal_detached()
        
        check_detached_patterns()
        
        return list(good_moves)

    
    def detect_open_four(self, board, player):
        kernel_h = np.array([[1, 1, 1, 1]])
        kernel_v = kernel_h.T
        kernel_d1 = np.eye(4, dtype=int)
        kernel_d2 = np.fliplr(kernel_d1)

        player_mask = (board == player).astype(int)
        empty_mask = (board == 0).astype(int)

        good_moves = set()
        # Horizontal
        conv_h = convolve2d(player_mask, kernel_h, mode='valid')
        for r in range(conv_h.shape[0]):
            for c in range(conv_h.shape[1]):
                if conv_h[r, c] == 4:
                    # Check ends at (r, c-1) and (r, c+4)
                    left_end = c - 1
                    right_end = c + 4
                    if left_end >= 0 and empty_mask[r, left_end]:
                        good_moves.add((r, left_end))
                    if right_end < board.shape[1] and empty_mask[r, right_end]:
                        good_moves.add((r, right_end))

        # Vertical
        conv_v = convolve2d(player_mask, kernel_v, mode='valid')
        for r in range(conv_v.shape[0]):
            for c in range(conv_v.shape[1]):
                if conv_v[r, c] == 4:
                    top_end = r - 1
                    bottom_end = r + 4
                    if top_end >= 0 and empty_mask[top_end, c]:
                        good_moves.add((top_end, c))
                    if bottom_end < board.shape[0] and empty_mask[bottom_end, c]:
                        good_moves.add((bottom_end, c))

        # Diagonal ↘
        conv_d1 = convolve2d(player_mask, kernel_d1, mode='valid')
        for r in range(conv_d1.shape[0]):
            for c in range(conv_d1.shape[1]):
                if conv_d1[r, c] == 4:
                    # Check ends diagonally: (r-1, c-1) and (r+4, c+4)
                    top_left = (r - 1, c - 1)
                    bottom_right = (r + 4, c + 4)
                    if 0 <= top_left[0] < board.shape[0] and 0 <= top_left[1] < board.shape[1]:
                        if empty_mask[top_left]:
                            good_moves.add(top_left)
                    if 0 <= bottom_right[0] < board.shape[0] and 0 <= bottom_right[1] < board.shape[1]:
                        if empty_mask[bottom_right]:
                            good_moves.add(bottom_right)

        # Anti-diagonal ↙
        conv_d2 = convolve2d(player_mask, kernel_d2, mode='valid')
        for r in range(conv_d2.shape[0]):
            for c in range(conv_d2.shape[1]):
                if conv_d2[r, c] == 4:
                    # Check ends anti-diagonally: (r-1, c+4) and (r+4, c-1)
                    top_right = (r - 1, c + 4)
                    bottom_left = (r + 4, c - 1)
                    if 0 <= top_right[0] < board.shape[0] and 0 <= top_right[1] < board.shape[1]:
                        if empty_mask[top_right]:
                            good_moves.add(top_right)
                    if 0 <= bottom_left[0] < board.shape[0] and 0 <= bottom_left[1] < board.shape[1]:
                        if empty_mask[bottom_left]:
                            good_moves.add(bottom_left)

        # detached patterns
        kernel_h = np.array([[1, 1, 1, 1, 1]])
        kernel_v = kernel_h.T
        kernel_d1 = np.eye(5, dtype=int)
        kernel_d2 = np.fliplr(kernel_d1)

        # Horizontal
        conv_h = convolve2d(board, kernel_h, mode='valid')
        for r in range(conv_h.shape[0]):
            for c in range(conv_h.shape[1]):
                if conv_h[r, c] == 4 * player:
                    for offset in range(5):
                        end_col = c + offset
                        if 0 <= end_col < board.shape[1] and board[r, end_col] == 0:
                            good_moves.add((r, end_col))
        # Vertical
        conv_v = convolve2d(board, kernel_v, mode='valid')
        for r in range(conv_v.shape[0]):
            for c in range(conv_v.shape[1]):
                if conv_v[r, c] == 4 * player:
                    for offset in range(5):
                        end_row = r + offset
                        if 0 <= end_row < board.shape[0] and board[end_row, c] == 0:
                            good_moves.add((end_row, c))

        # Diagonal ↘
        conv_d1 = convolve2d(board, kernel_d1, mode='valid')
        for r in range(conv_d1.shape[0]):
            for c in range(conv_d1.shape[1]):
                if conv_d1[r, c] == 4 * player:
                    for offset in range(5):
                        end_pos = (r + offset, c + offset)
                        if 0 <= end_pos[0] < board.shape[0] and 0 <= end_pos[1] < board.shape[1]:
                            if board[end_pos] == 0:
                                good_moves.add(end_pos)

        # Anti-diagonal ↙
        conv_d2 = convolve2d(board, kernel_d2, mode='valid')
        for r in range(conv_d2.shape[0]):
            for c in range(conv_d2.shape[1]):
                if conv_d2[r, c] == 4 * player:
                    for offset in range(5):
                        end_pos = (r + offset, c - offset + 4)
                        if 0 <= end_pos[0] < board.shape[0] and 0 <= end_pos[1] < board.shape[1]:
                            if board[end_pos] == 0:
                                good_moves.add(end_pos)

        return list(good_moves)

    def get_good_moves2(self, board, current_player):
        # If good moves are not found, return random empty positions
        stone_mask = (board != 0).astype(int)
        # Step 2: Convolve to get neighbor counts
        kernel = np.array([
            [1, 1, 1],
            [1, 0, 1],
            [1, 1, 1]
        ])
        neighbor_counts = convolve2d(stone_mask, kernel, mode='same', boundary='fill', fillvalue=0)

        # Step 3: Mask out non-empty cells
        empty_mask = (board == 0)
        valid_scores = neighbor_counts * empty_mask

        # Step 4: Get index with highest value among empty cells
        max_value = np.max(valid_scores)
        indices = np.argwhere(valid_scores == max_value)
        return list(indices)
    
    def get_obvious_moves(self, board, current_player):
        '''
            This function generates obvious moves based on the current board state.
            It checks for open four and open three positions, and returns them.

            @param board: Current board state (2d array) 15 x 15. 0 = empty, 1 = player, -1 = opponent
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
        
        return self.get_good_moves2(board, current_player)

    def expand(self, node):
        '''
            This function expands a leaf node by adding child nodes for legal or promising next moves.
            It will be called when a leaf node is reached second time during the traversal.

            @param node: Leaf node (board state) to expand
        '''
        # Apply moves to get current board state
        board = self.apply_moves_to_board(self.board, node.moves, initial_player=self.root.current_player)
        promising_moves = self.get_good_moves2(board, current_player=node.current_player)
        promising_moves = np.random.permutation(promising_moves)  # Shuffle the promising moves to add some randomness

        for move in promising_moves[:5]:  # Take first 5
            child_node = Node(current_player=-node.current_player, parent=node, move=move)
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
                                return k
                        except IndexError:
                            continue
        return None  # No winner found
        
    def rollout(self, node):
        '''
            This function simulates a random game from the current board state.
            It will be called when a leaf node is reached for the first time during the traversal.

            @param node: Leaf node (board state) to simulate
        '''
        # Get current board state by applying moves
        current_board = self.apply_moves_to_board(self.board, node.moves, initial_player=self.root.current_player)

        # Determine whose turn it is next
        current_player = node.current_player

        # Start the simulation
        result = 0.0
        MAX_SIMULATION_STEPS = 10  # Limit the number of steps to prevent infinite loops
        for _ in range(MAX_SIMULATION_STEPS):
            # Randomly select a move
            valid_moves = self.get_good_moves(current_board, current_player)
            if not valid_moves:
                # draw
                result = 0.5
                break
            random_index = np.random.randint(len(valid_moves))
            row, col = valid_moves[random_index]
            current_board[row][col] = current_player
            # Check if the game is completed
            winner = self.check_winner(current_board)
            if winner is not None:
                result = 1 if winner == current_player else -1
                break
            # Randomly select a move for the opponent
            valid_moves = self.get_good_moves(current_board, -1 * current_player)
            if not valid_moves:
                # draw
                result = 0.5
                break
            random_index = np.random.randint(len(valid_moves))
            row, col = valid_moves[random_index]
            current_board[row][col] = -1 * current_player
            # Check if the game is completed
            winner = self.check_winner(current_board)
            if winner is not None:
                result = 1 if winner == current_player else -1
                break
        node.visits += 1
        node.wins += result
        # traverse back up the tree and update the visit counts and wins
        while node.parent is not None:
            node.parent.visits += 1
            result = -1 * result  # Invert the result for the parent node
            node.parent.wins += result
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
        self.board = board  # Update the board state
        current_player = self.get_current_player(self.board)
        obvious_moves = self.get_obvious_moves(board, current_player)
        if obvious_moves:
            obvious_moves = np.random.permutation(obvious_moves)  # Shuffle the obvious moves to add some randomness
            # If there are obvious moves, return the first one
            # This is a heuristic to prioritize obvious moves over MCTS
            return obvious_moves[0][0] * 15 + obvious_moves[0][1]
        self.root = Node(current_player=current_player, parent=None, move=None)  # Reset the root node with the current board state
        for it in range(self.iterations):
            # traverse down the tree to a leaf node
            leaf_node = self.traverse(self.root)
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
        best_child = max(self.root.children, key=lambda n: n.visits)
        # Convert the best child's element (board state) to an action
        best_action = best_child.move
        if best_action is None:
            return None
        # Convert the best action (row, col) to a single action index
        return best_action[0] * 15 + best_action[1] 


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