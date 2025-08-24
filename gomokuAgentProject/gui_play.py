import pygame
import sys

# Settings
BOARD_SIZE = 9
CELL_SIZE = 40
MARGIN = 20
WINDOW_SIZE = BOARD_SIZE * CELL_SIZE + MARGIN * 2

# Colors
BLACK = (0, 0, 0)
WHITE = (255, 255, 255)
STONE_BLACK = (0, 0, 0)
STONE_WHITE = (255, 255, 255)
WOOD = (205, 170, 125)

class GUI:
    def __init__(self):
        pygame.init()
        self.screen = pygame.display.set_mode((WINDOW_SIZE, WINDOW_SIZE))
        pygame.display.set_caption("Gomoku")
        self.CELL_SIZE = CELL_SIZE
        self.BOARD_SIZE = BOARD_SIZE
        self.MARGIN = MARGIN

    def draw_board(self, state):
        self.screen.fill(WOOD)
        
        # Draw grid lines
        for i in range(BOARD_SIZE + 1):
            # Horizontal lines
            pygame.draw.line(self.screen, BLACK,
                            (MARGIN, MARGIN + i * CELL_SIZE),
                            (WINDOW_SIZE - MARGIN, MARGIN + i * CELL_SIZE))
            # Vertical lines
            pygame.draw.line(self.screen, BLACK,
                            (MARGIN + i * CELL_SIZE, MARGIN),
                            (MARGIN + i * CELL_SIZE, WINDOW_SIZE - MARGIN))

        # Draw pieces
        for i in range(BOARD_SIZE):
            for j in range(BOARD_SIZE):
                piece = state[i][j]
                if piece != 0:
                    color = STONE_BLACK if piece == 1 else STONE_WHITE
                    pygame.draw.circle(
                        self.screen,
                        color,
                        (MARGIN + j * CELL_SIZE, MARGIN + i * CELL_SIZE),
                        CELL_SIZE // 2 - 2
                    )

        pygame.display.flip()

    def draw_board_with_probs(self, state, action_probs):
        self.draw_board(state)  # draw the board + stones first

        font = pygame.font.SysFont(None, 18)  # small font
        probs_grid = action_probs.reshape(self.BOARD_SIZE, self.BOARD_SIZE)

        for i in range(self.BOARD_SIZE):
            for j in range(self.BOARD_SIZE):
                prob = probs_grid[i][j]

                if prob > 0.001:  # skip near-zero values
                    # pick color based on probability value
                    if prob > 0.1:
                        color = (255, 0, 0)       # red (high prob)
                    elif prob > 0.05:
                        color = (255, 165, 0)     # orange (medium)
                    elif prob > 0.03:
                        color = (0, 128, 0)       # green (low-mid)
                    else:
                        color = (0, 0, 255)       # blue (very low)

                    text = font.render(f"{prob:.2f}", True, color)
                    text_rect = text.get_rect(center=(
                        self.MARGIN + j * self.CELL_SIZE,
                        self.MARGIN + i * self.CELL_SIZE
                    ))
                    self.screen.blit(text, text_rect)

        pygame.display.flip()

    def quit(self):
        pygame.quit()
        sys.exit()