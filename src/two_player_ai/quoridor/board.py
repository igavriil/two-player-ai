import numpy as np
import numba as nb

board_size = 9
board = np.zeros(
    (3, board_size, board_size),
    dtype=np.int8
)
center = int(9 / 2)
pawn_board = board[0]
pawn_board[0][center] = 1
pawn_board[board_size - 1][center] = -1

horizontal_fences = board[1]
vertical_fences = board[2]

horizontal_actions = []
vertical_actions = []


@nb.jit(nopython=True, nogil=True, cache=True)
def horizontal_actions(horizontal_fences, vertical_fences):
    aux = np.zeros((9, 9), dtype=np.int)
    for row, col in zip(*np.where(horizontal_fences != 0)):
        if vertical_fences[row, col] == 0:
            aux[row, col] = horizontal_fences[row, col]
            aux[row, col + 1] = horizontal_fences[row, col]

    for row, col in zip(*np.where(vertical_fences != 0)):
        aux[row, col] = vertical_fences[row, col]

    return np.where(aux[0:8, 0:8] == 0)


@nb.jit(nopython=True, nogil=True, cache=True)
def vertical_actions(horizontal_fences, vertical_fences, aux):
    for row, col in zip(*np.where(vertical_fences != 0)):
        if horizontal_fences[row, col] == 0:
            aux[row, col] = vertical_fences[row, col]
            aux[row + 1, col] = vertical_fences[row, col]

    for row, col in zip(*np.where(horizontal_fences != 0)):
        aux[row, col] = horizontal_fences[row, col]

    return np.where(aux[0:8, 0:8] == 0)


@nb.jit(nopython=True, nogil=True, cache=True)
def flatten(board, flat):
    pieces = board[0]
    height, width = pieces.shape
    for row, col in zip(*np.where(pieces != 0)):
        flat[2 * row, 2 * col] = pieces[row, col]

    horizontal_fences = board[1]
    for row, col in zip(*np.where(horizontal_fences != 0)):
        fences_row = 2 * row + 1

        fence_a_col = 2 * col
        fence_b_col = 2 * (col + 1)
        if (0 <= fences_row < 17
            and 0 <= fence_a_col < 17
                and 0 <= fence_b_col < 17):
            flat[fences_row, fence_a_col] = horizontal_fences[row, col] * 2
            flat[fences_row, fence_b_col] = horizontal_fences[row, col] * 2

    vertical_fences = board[2]
    for row, col in zip(*np.where(vertical_fences != 0)):
        fences_col = 2 * col + 1

        fence_a_row = 2 * row
        fence_b_row = 2 * (row + 1)
        if (0 <= fences_col < 17
            and 0 <= fence_a_row < 17
                and 0 <= fence_b_row < 17):
            flat[fence_a_row, fences_col] = vertical_fences[row, col] * 2
            flat[fence_b_row, fences_col] = vertical_fences[row, col] * 2

    return flat


def fence_actions(board):
    horizontal_fences = board[1]
    vertical_fences = board[2]

    v_actions = vertical_actions(
        horizontal_fences, vertical_fences, np.zeros((9, 9), dtype=np.int8)
    )
    h_actions = horizontal_actions(
        horizontal_fences, vertical_fences, np.zeros((9, 9), dtype=np.int8)
    )
    total_fences = (np.count_nonzero(horizontal_fences) +
                    np.count_nonzero(vertical_fences))
    actions = set()




    for v_action in list(zip(*v_actions)):
        temp = np.copy(board)
        temp[2][v_action] = 1
        flat_board = flatten(temp, np.zeros((17, 17), dtype=np.int8))

        if total_fences < 6:
            actions.add((1, v_action))
        else:
            x1, y1 = np.where(board[0] == 1)
            x2, y2 = np.where(board[0] == -11)

            if flood_fill(
                flat_board,
                np.zeros((17, 17), dtype=np.int8),
                (x1[0], y1[0]),
                16) and flood_fill(
                    flat_board,
                    np.zeros((17, 17), dtype=np.int8),
                    (x1[0], y1[0]),
                    16) and flo

            flood_fill(
            flat_board,
            np.zeros((17, 17), dtype=np.int8), (x[0], y[0]), target):
            actions.add((2, v_action))

    for h_action in list(zip(*h_actions)):
        temp = np.copy(board)
        temp[1][v_action] = 1
        flat_board = flatten(temp, np.zeros((17, 17), dtype=np.int8))

        if total_fences < 6:
            actions.add((1, h_action))
        elif flood_fill(
            flat_board,
            np.zeros((17, 17), dtype=np.int8), (x[0], y[0]), target):
            actions.add((1, h_action))

    return actions


@nb.jit(nopython=True, nogil=True, cache=True)
def flood_fill(board, aux, start, goal_row):
    frontier = [start]

    while len(frontier):
        position = frontier.pop()
        if aux[position] != 1:
            row, col = position
            for row_dir, col_dir in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
                neighbor = (row + 2 * row_dir, col + 2 * col_dir)
                if aux[neighbor] != 1:
                    if (0 <= row + row_dir < 17 and 0 <= col + col_dir < 17):
                        if board[row + row_dir, col + col_dir] == 0:

                            frontier.append(neighbor)

                            if row + 2 * row_dir == goal_row:
                                return True

        aux[position] = 1

    return False
