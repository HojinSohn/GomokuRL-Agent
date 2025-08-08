import copy
from env import GomokuEnv
from mcts import MCTS
from model import Model
import pygame
from gui_play import GUI
import torch
import numpy as np

class Agent(object):
    def __init__(self, turn):
        # turn 1 = first
        # turn -1 = second
        self.turn = turn

        self.mcts = MCTS()
        
        # load the model
        self.model = Model() 
        if turn == 0:
            self.model.load_state_dict(torch.load('models/model1.pth', map_location=torch.device('cpu')))
        else:
            self.model.load_state_dict(torch.load('models/model2.pth', map_location=torch.device('cpu')))
        self.model.eval()

    def make_move_mcts(self, state):
        print("Agent is making a move using MCTS")

        # moves = self.mcts.get_obvious_moves(np.array(state), self.turn)
        # if not moves:
        #     print("No obvious moves found, using MCTS to find the best move")
        #     moves = self.mcts.get_good_moves2(np.array(state), self.turn)
        # row, col = moves[0]
        # print(f"Agent selected move: ({row}, {col})")
        # return row * 15 + col

        # Run MCTS to get the best move
        board = np.array(copy.deepcopy(state))
        best_move = self.mcts.get_action(board)
        print(f"Best move found: {best_move}")
        return best_move
    
    def convert_states_to_3channel(self, states, device=torch.device("cpu")):
        batch_size = len(states)
        input_tensor = np.zeros((batch_size, 3, 15, 15), dtype=np.float32)

        for i, state in enumerate(states):
            state_array = np.array(state)
            input_tensor[i, 0] = (state_array == -1).astype(np.float32)   # Player 1 stones
            input_tensor[i, 1] = (state_array == 1).astype(np.float32)   # Player 2 stones
            input_tensor[i, 2] = (state_array != 0).astype(np.float32)    # Non Empty cells

        return torch.from_numpy(input_tensor).to(device)

    def make_move(self, state):
        print("Agent is making a move")
        input_tensor = self.convert_states_to_3channel([state])
        with torch.no_grad():
            q_values = self.model(input_tensor).squeeze(0).squeeze(0)

        invalid_moves = set()
        max_attempts = 100
        attempts = 0

        while attempts < max_attempts:
            q_values_clone = q_values.clone()
            if invalid_moves:
                q_values_clone[list(invalid_moves)] = float('-inf')
            action = torch.argmax(q_values_clone).item()
            row, col = divmod(action, 15)
            if state[row][col] == 0:
                print(f"Agent placed stone at ({row}, {col})")
                return action
            else:
                print(f"Cell ({row}, {col}) is already occupied, trying another action")
                invalid_moves.add(action)
                attempts += 1

        raise RuntimeError("No valid moves available for the agent!")

if __name__ == "__main__":
    gui = GUI()
    env = GomokuEnv()
    turn = 0  # Start with player 1
    agent = Agent(-1) # Agent plays second (white stones)
    done = False
    while not done:
        if turn == 0:
            env.first_move()
            print("Player 1 (Black) starts the game")
            turn += 1
            gui.draw_board(env.state[0])
            continue
        if turn % 2 == 1:
            # Agent's turn
            print("Agent's turn")
            action = agent.make_move_mcts(env.state[0])
            env.step(action, 1)
            row, col = divmod(action, 15)
            print(f"Agent placed stone at ({row}, {col})")
            gui.draw_board(env.state[0])
            win = env.check_winner(1)
            if win:
                print("Agent wins!")
                done = True
            turn += 1
        else:
            print("Waiting for player's move...")
            move_made = False
            while not move_made:
                for event in pygame.event.get():
                    if event.type == pygame.QUIT:
                        gui.quit()
                        pygame.quit()
                        sys.exit()
                    elif event.type == pygame.MOUSEBUTTONDOWN and not move_made:
                        print("Mouse button pressed")
                        if event.button == 1:  # Left click
                            x, y = event.pos
                            row = (y - gui.MARGIN) // gui.CELL_SIZE
                            col = (x - gui.MARGIN) // gui.CELL_SIZE
                            action = row * 15 + col
                            print(f"Player clicked on cell ({row}, {col})")
                            env.step(action, 0)
                            move_made = True  # <== Exit the wait loop
                            gui.draw_board(env.state[0])
                            win = env.check_winner(0)
                            if win:
                                print("Player wins!")
                                done = True
                            turn += 1

                    elif event.type == pygame.KEYDOWN:
                        if event.key == pygame.K_ESCAPE:
                            gui.quit()
                            pygame.quit()
                            sys.exit()
                        elif event.key == pygame.K_r:  # Reset the board
                            state = [[0 for _ in range(15)] for _ in range(15)]
                            gui.draw_board(state)
                            turn = 0
                            move_made = True  # Exit the wait loop so main loop can restart

                pygame.time.delay(100)
                pygame.display.flip()


        pygame.time.delay(100)
        pygame.display.flip()