BOARD_SIZE = 8
BINARY_SHAPE = (2, BOARD_SIZE, BOARD_SIZE)
BLACK_PLAYER = 1
WHITE_PLAYER = -1
WALL = 2
DIRECTIONS = [
    (1, 1), (1, 0), (1, -1),
    (0, -1), (0, 1),
    (-1, -1), (-1, 0), (-1, 1)
]
ALL_ACTIONS = [
    (x, y)
    for x in range(BOARD_SIZE)
    for y in range(BOARD_SIZE)
]
