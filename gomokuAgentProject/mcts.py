# Node class for Tree Search
import numpy as np
import copy
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

    def get_good_moves(self, board, current_player):
        '''
            This function generates promising moves based on the current board state.
            Prioritizes critical moves and returns up to 10 most promising positions.

            @param board: Current board state (2d array) 15 x 15. 0 = empty, 1 = player, -1 = opponent
            @param current_player: Current player (1 or -1)
            @return: List of up to 10 promising moves, prioritized by importance
        '''
        # Priority sets for different move types
        direct_win_moves = set()  # Direct winning moves = open four for player
        direct_block_moves = set()  # Add stone to block opponent's open four / half open four
        indirect_win_moves = set()  # Add one stone to make open four in a row ==> win
        defend_moves = set()  # open three in a row for opponent / need to block to prevent opponent's win

        direction_vectors = [
            (1, 0),   # vertical
            (0, 1),   # horizontal
            (1, 1),   # diagonal down-right
            (1, -1),  # diagonal down-left
        ]

        # Check all positions and directions for patterns
        for i in range(15):
            for j in range(15):
                if board[i][j] == 0:  # Only check empty positions
                    continue
                    
                player = board[i][j]
                
                for dx, dy in direction_vectors:
                    block_before = False
                    bx, by = i - dx, j - dy
                    if 0 <= bx < 15 and 0 <= by < 15 and board[bx][by] == player:
                        continue  # Since the previous position is occupied by the same player, no need to check further

                    if 0 > bx or bx >= 15 or 0 > by or by >= 15 or board[bx][by] == -player:
                        block_before = True

                    # Count consecutive pieces in this direction
                    three_open = False
                    count = 1  # Count the current piece
                    # check four in a row / or four in a row with one empty space
                    for k in range(1, 5):
                        nx, ny = i + dx * k, j + dy * k
                        if 0 <= nx < 15 and 0 <= ny < 15 and board[nx][ny] == player:
                            count += 1
                            if count == 4:
                                break
                        elif 0 > nx or nx >= 15 or 0 > ny or ny >= 15 or board[nx][ny] == -player:
                            count = -1  # Blocked by opponent
                            break
                        else:
                            if count == 3 and not block_before:
                                # If we have three in a row and the next is empty, it's a good move
                                three_open = True

                    if count == 4:
                        # If we have four in a row, but need to check the end of the row

                        # check if the prev position is empty
                        if not block_before:
                            # check if the stone should be placed in the middle of the four
                            put_in_the_middle = False
                            for n in range(1, 4):  # Check four steps away
                                nx, ny = i + dx * n, j + dy * n
                                if 0 <= nx < 15 and 0 <= ny < 15 and board[nx][ny] == 0:
                                    if player == current_player:
                                        direct_win_moves.add((nx, ny))
                                    else:
                                        direct_block_moves.add((nx, ny))
                                    put_in_the_middle = True
                        
                            if not put_in_the_middle:
                                # Add the position before the four
                                if player == current_player:
                                    direct_win_moves.add((i - dx, j - dy))
                                else:
                                    direct_block_moves.add((i - dx, j - dy))
                                
                                # Add the position after the four
                                nx, ny = i + dx * 4, j + dy * 4
                                if 0 <= nx < 15 and 0 <= ny < 15 and board[nx][ny] == 0:
                                    if player == current_player:
                                        direct_win_moves.add((nx, ny))
                                    else:
                                        direct_block_moves.add((nx, ny))
                        else:
                            # If we have four in a row and the next is blocked, 
                            # check if the stone should be placed in the middle of the four
                            put_in_the_middle = False
                            for n in range(1, 4):  # Check four steps away
                                nx, ny = i + dx * n, j + dy * n
                                if 0 <= nx < 15 and 0 <= ny < 15 and board[nx][ny] == 0:
                                    if player == current_player:
                                        direct_win_moves.add((nx, ny))
                                    else:
                                        direct_block_moves.add((nx, ny))
                                    put_in_the_middle = True
                            if not put_in_the_middle:
                                # Cannot add the position before the four since its blocked
                                # Add the position after the four
                                nx, ny = i + dx * 4, j + dy * 4
                                if 0 <= nx < 15 and 0 <= ny < 15 and board[nx][ny] == 0:
                                    if player == current_player:
                                        direct_win_moves.add((nx, ny))
                                    else:
                                        direct_block_moves.add((nx, ny))

                    if three_open:
                        put_in_the_middle = False
                        for n in range(1, 4):  # Check three steps away
                            nx, ny = i + dx * n, j + dy * n
                            if 0 <= nx < 15 and 0 <= ny < 15 and board[nx][ny] == 0:
                                # If we have three in a row, it's a good move
                                if player == current_player:
                                    indirect_win_moves.add((nx, ny)) 
                                else:
                                    defend_moves.add((nx, ny))
                                put_in_the_middle = True
                        if not put_in_the_middle:
                            # If we have three in a row, it's a good move
                            if player == current_player:
                                indirect_win_moves.add((i - dx, j - dy)) # Before the three
                            else:
                                defend_moves.add((i - dx, j - dy))
        print("Turn: ", current_player, " Direct win moves:", direct_win_moves)
        print("Turn: ", current_player, " Direct block moves:", direct_block_moves)
        if direct_win_moves:
            # If we have open four positions, return them
            return list(direct_win_moves)
        if direct_block_moves:
            # If we have open three positions, return them
            return list(direct_block_moves)
        
        if indirect_win_moves:
            # If we have indirect win positions, return them
            return list(indirect_win_moves)

        if defend_moves:
            # If we have defend positions, return them
            return list(defend_moves)
        
        good_moves = set()

        # If good moves are not found, return random empty positions
        for i in range(15):
            for j in range(15):
                if board[i][j] != 0:
                    # add adjacent empty positions
                    for dx, dy in direction_vectors:
                        for n in range(1, 3):  # Check two steps away
                            nx, ny = i + dx * n, j + dy * n
                            if 0 <= nx < 15 and 0 <= ny < 15 and board[nx][ny] == 0:
                                good_moves.add((nx, ny))
        return list(good_moves)

    def expand(self, node):
        '''
            This function expands a leaf node by adding child nodes for legal or promising next moves.
            It will be called when a leaf node is reached second time during the traversal.

            @param node: Leaf node (board state) to expand
        '''
        # Apply moves to get current board state
        board = self.apply_moves_to_board(self.board, node.moves, initial_player=self.root.current_player)
        promising_moves = self.get_good_moves(board, current_player=node.current_player)
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
        MAX_SIMULATION_STEPS = 1  # Limit the number of steps to prevent infinite loops
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
        self.root = Node(current_player=current_player, parent=None, move=None)  # Reset the root node with the current board state
        for it in range(self.iterations):
            if it % 100 == 0:
                print("Iteration:", it)
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