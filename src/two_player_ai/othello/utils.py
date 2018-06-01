import numba as nb
import numpy as np
import two_player_ai.othello.constants as othello_constants


@nb.jit(nopython=True, nogil=True, cache=True)
def cannonical_board(board, player):
    return board * player


@nb.jit(nopython=True, nogil=True, cache=True)
def boards_equal(board, other_board):
    return np.all(np.equal(board, other_board))
