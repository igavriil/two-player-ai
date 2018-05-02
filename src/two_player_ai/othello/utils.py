import numba as nb
import numpy as np
import two_player_ai.othello.constants as othello_constants


@nb.jit(nopython=True, nogil=True, cache=True)
def cannonical_board(board, player):
    return board * player


@nb.jit(nopython=True, nogil=True, cache=True)
def boards_equal(board, other_board):
    return np.all(np.equal(board, other_board))


@nb.jit(nopython=True, nogil=True, cache=True)
def binary_board(board):
    binary = np.zeros(
        (2, othello_constants.BOARD_SIZE, othello_constants.BOARD_SIZE),
        dtype=np.int8
    )
    for row, col in zip(*np.where(board == othello_constants.BLACK_PLAYER)):
        binary[0, row, col] = 1

    for row, col in zip(*np.where(board == othello_constants.WHITE_PLAYER)):
        binary[1, row, col] = 1

    return binary
