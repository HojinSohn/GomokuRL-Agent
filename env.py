import numpy as np

rewards = {"win" : 10.0, "lose" : -10.0, "step" : 0.0001, "invalid" : -0.1}

class GomokuEnv:
    def __init__(self):
        self.size = 15
        self.state = {0: np.zeros((15, 15)), 1: np.zeros((15, 15))}

    def reset(self):
        self.state = {0: np.zeros((15, 15)), 1: np.zeros((15, 15))}

    def display(self, turn):
        for row in range(15):
            print(" ".join(str(self.state[turn][row, col]) for col in range(15)))

    def first_move(self):
        center = 7
        self.state[0][center, center] = 1
        self.state[1][center, center] = -1

    def check_winner(self):
        # check rows, columns, and diagonals for a winner
        # if 5 in a row, return the piece number
        direction_vectors = [
                (1, 0),   # vertical
                (0, 1),   # horizontal
                (1, 1),   # diagonal down-right
                (-1, 1),  # diagonal up-right
            ]
        sequence_length = 5

        for i in range(self.size):
            for j in range(self.size):
                for dx, dy in direction_vectors:
                    for k in [1, -1]:
                        try:
                            if all(0 <= i + dx * n < self.size and 0 <= j + dy * n < self.size and self.state[0][i + dx * n][j + dy * n] == k for n in range(sequence_length)):
                                print(f"Player {k} wins!")
                                print(f"Winning sequence starts at ({i}, {j}) in direction ({dx}, {dy})")
                                return True
                        except IndexError:
                            continue
        return False
    
    # Perform an action and return the new state, whether the game is completed, reward, and whether the action was valid
    def step(self, action, turn):
        row, col = divmod(action, 15)
        if self.state[turn][row, col] != 0:
            return self.state[turn], False, rewards["invalid"], False
        # Place the piece
        self.state[turn][row, col] = 1
        self.state[0 if turn == 1 else 1][row, col] = -1
        completed = False
        if self.check_winner():
            completed = True
            reward = rewards["win"]
        else:
            reward = rewards["step"]

        return self.state[turn], completed, reward, True