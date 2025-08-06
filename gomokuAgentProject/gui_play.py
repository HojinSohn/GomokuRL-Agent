import pygame
import sys

# Settings
BOARD_SIZE = 15
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

    def quit(self):
        pygame.quit()
        sys.exit()