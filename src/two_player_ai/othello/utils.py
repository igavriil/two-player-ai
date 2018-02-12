import numba as nb
import numpy as np


@nb.jit(nopython=True, cache=True)
def cannonical_board(board, player):
    return board * player


@nb.jit(nopython=True, cache=True)
def boards_equal(board, other_board):
    return np.all(np.equal(board, other_board))
