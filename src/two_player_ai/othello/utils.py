import numba as nb


@nb.jit(nopython=True, cache=True)
def cannonical_board(board, player):
    return board * player
