import copy
from mcts import MCTS
from model import Model
import pygame
from gui_play import GUI
import torch

class Agent(object):
    def __init__(self, turn):
        # turn 0 = first
        # turn 1 = second
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
        # Run MCTS to get the best move
        temp = copy.deepcopy(state)
        best_move = self.mcts.get_action(state)
        print(f"Best move found: {best_move}")
        state[:] = copy.deepcopy(temp)
        return best_move

    def make_move(self, state):
        print("Agent is making a move")
        state_tensor = torch.tensor(state, dtype=torch.float).unsqueeze(0).unsqueeze(0)
        with torch.no_grad():
            q_values = self.model(state_tensor).squeeze(0).squeeze(0)

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
    state = [[0 for _ in range(15)] for _ in range(15)]
    turn = 0  # Start with player 1
    agent = Agent(1) # Agent plays second (white stones)
    while True:
        gui.draw_board(state)
        if turn % 2 == 1:
            # Agent's turn
            print("Agent's turn")
            action = agent.make_move_mcts(state)
            row, col = divmod(action, 15)
            if state[row][col] == 0:
                print(f"Agent placed stone at ({row}, {col})")
                state[row][col] = -1
                gui.draw_board(state)
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
                    elif event.type == pygame.MOUSEBUTTONDOWN:
                        print("Mouse button pressed")
                        if event.button == 1:  # Left click
                            x, y = event.pos
                            row = (y - gui.MARGIN) // gui.CELL_SIZE
                            col = (x - gui.MARGIN) // gui.CELL_SIZE
                            if 0 <= row < 15 and 0 <= col < 15 and state[row][col] == 0:
                                state[row][col] = 1
                                gui.draw_board(state)
                                turn += 1
                                move_made = True  # <== Exit the wait loop

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