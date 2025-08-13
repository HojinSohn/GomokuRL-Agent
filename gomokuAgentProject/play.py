import copy
import sys
from env import GomokuEnv
from agent import DQNAgent
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

        self.mcts = MCTS(iterations=200)
        
        # # load the model
        self.model = Model() 
        # if turn == -1:
        #     self.model.load_state_dict(torch.load('models/model1.pth', map_location=torch.device('cpu')))
        # else:
        #     self.model.load_state_dict(torch.load('models/model2.pth', map_location=torch.device('cpu')))
        # self.model.eval()

    def make_move_mcts(self, state, random):
        print("Agent is making a move using MCTS")

        # moves = self.mcts.get_obvious_moves(np.array(state), self.turn)
        # if not moves:
        #     print("No obvious moves found, using MCTS to find the best move")
        #     moves = self.mcts.get_good_moves2(np.array(state), self.turn)
        # row, col = moves[0]
        # print(f"Agent selected move: ({row}, {col})")
        # return row * 9 + col

        # Run MCTS to get the best move
        board = np.array(copy.deepcopy(state))
        best_move = self.mcts.get_action(board, random)
        print(f"Best move found: {best_move}")
        return best_move
    
    def convert_states_to_3channel(self, states, device=torch.device("cpu")):
        batch_size = len(states)
        input_tensor = np.zeros((batch_size, 3, 9, 9), dtype=np.float32)

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
            row, col = divmod(action, 9)
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

    # dqnAgent = DQNAgent()
    # dqnAgent.load_memory()  # Load the agent's memory if available 
    # clock = pygame.time.Clock()

    # print(len(dqnAgent.memory1), "samples loaded from memory")

    # for i, sample in enumerate(dqnAgent.memory1):
    #     state, action, reward, next_state, done = sample[0], sample[1], sample[2], sample[3], sample[4]
    #     board = np.array(state)
    #     next_board = np.array(next_state)
    #     print(f"Sample state {i}: {board}")
    #     print(f"Sample next state {i}: {next_board}")
    #     print(f"Action: {action}, Reward: {reward}, Done: {done}")

    #     # Handle pygame events first
    #     for event in pygame.event.get():
    #         if event.type == pygame.QUIT:
    #             pygame.quit()
    #             sys.exit()
        
    #     pygame.time.wait(4000)
    #     gui.draw_board(board)  # Draw the board
        
    #     # Wait 1 second using clock (non-blocking)
    #     pygame.time.wait(1000)
    #     gui.draw_board(next_board)  # Draw the board

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
            print("Agent 1's turn")
            action = agent.make_move_mcts(env.state[0], False)
            env.step(action, 1)
            row, col = divmod(action, 9)
            env.update_state(1, action)
            print(f"Agent 1 placed stone at ({row}, {col})")
            gui.draw_board(env.state[0])
            win = env.check_winner(env.state[1], (row, col))
            if win:
                print("Agent 1 wins!")
                done = True
            turn += 1
        else:
            print("Agent 2's turn")
            action = agent.make_move_mcts(env.state[0], True)
            env.step(action, 0)
            row, col = divmod(action, 9)
            env.update_state(0, action)
            print(f"Agent 2 placed stone at ({row}, {col})")
            gui.draw_board(env.state[0])
            win = env.check_winner(env.state[0], (row, col))
            if win:
                print("Agent 2 wins!")
                done = True
            turn += 1
            print("Waiting for player's move...")

            move_made = True
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
                            action = row * 9 + col
                            print(f"Player clicked on cell ({row}, {col})")
                            env.step(action, 0)
                            env.update_state(0, action)
                            move_made = True  # <== Exit the wait loop
                            gui.draw_board(env.state[0])
                            win = env.check_winner(env.state[0], (row, col))
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
                            state = [[0 for _ in range(9)] for _ in range(9)]
                            gui.draw_board(state)
                            turn = 0
                            move_made = True  # Exit the wait loop so main loop can restart

                pygame.time.delay(100)
                pygame.display.flip()


        pygame.time.delay(100)
        pygame.display.flip()