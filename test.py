import numpy as np
from game import Game
from agent import Agent
from mcts import MCTS
from policy_value_network import PolicyValueNetwork
from gui_play import GUI
import pygame
import sys

def play_game_agent_first(agent1: Agent, agent2: Agent, board_size=9, gui: GUI=None):
    game = Game()
    game.first_move()
    if gui:
        gui.draw_board(game.state[0])
    game_turn = 1
    while True:
        print(game_turn)
        # --- SOLUTION: Add Pygame event loop to handle window events ---
        if gui:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                    sys.exit()

        if game_turn % 2 == 0:  
            _, action_probs = agent1.get_action_and_probs(game, 0)
        else:  
            _, action_probs = agent2.get_action_and_probs(game, 1)

        # choose the max
        action = np.argmax(action_probs)
        row, col = divmod(action, board_size)
        game.do_move((row, col))

        print(f"Turn {game_turn}: Player {game.current_player} placed at ({row}, {col})")

        if gui:
            gui.draw_board_with_probs(game.state[0], action_probs)
            pygame.time.wait(1000) # Wait 1 second to show final board

        winner, winning_moves = game.get_winner_indirect()
        if winner is not None:
            if winning_moves and len(winning_moves) > 0:
                game.do_move(winning_moves[0])
            if gui:
                pygame.time.wait(2000) # Wait 2 seconds to show final board
                gui.draw_board(game.state[0])
                pygame.time.wait(2000) # Wait 4 seconds to show final board
            return winner
        game_turn += 1

def evaluate(agent_with_nn, agent_base, n_games=100, board_size=9, gui=None):
    results = {"agent_wins": 0, "random_wins": 0, "draws": 0}
    for i in range(n_games):
        if i % 2 == 0:
            winner = play_game_agent_first(agent_with_nn, agent_base, board_size=board_size, gui=gui)
            if winner == 0:
                results["agent_wins"] += 1
            elif winner == 1:
                results["random_wins"] += 1
            else:
                results["draws"] += 1
        else:
            winner = play_game_agent_first(agent_base, agent_with_nn, board_size=board_size, gui=gui)
            if winner == 1:
                results["agent_wins"] += 1
            elif winner == 0:
                results["random_wins"] += 1
            else:
                results["draws"] += 1

    return results

if __name__ == "__main__":
    agent_with_nn = Agent(mcts_iterations=4000)
    agent_with_nn.load_model()

    agent_base = Agent(mcts_iterations=4000, base_mcts=True)

    gui = GUI()
    pygame.init()

    stats = evaluate(agent_with_nn, agent_base, n_games=50, board_size=9, gui=gui)
    print(f"Agent MCTS Wins: {stats['agent_wins']}")
    print(f"Random MCTS Wins: {stats['random_wins']}")
    print(f"Draws: {stats['draws']}")

    win_rate = stats['agent_wins'] / sum(stats.values())
    print(f"Agent Win Rate: {win_rate * 100:.2f}%")
