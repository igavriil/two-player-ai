import numba as nb
import numpy as np


class Zobrist(object):
    MAX_RAND = pow(10, 16)
    BLACK_TABLE = np.random.seed(3) or np.random.randint(MAX_RAND, size=(8, 8))
    WHITE_TABLE = np.random.seed(7) or np.random.randint(MAX_RAND, size=(8, 8))

    @staticmethod
    def from_state(state):
        return Zobrist.hash(state.board,
                            Zobrist.BLACK_TABLE,
                            Zobrist.WHITE_TABLE)

    @staticmethod
    def update_action(previous, action, player):
        return Zobrist.update(previous, action,
                              Zobrist.BLACK_TABLE,
                              Zobrist.WHITE_TABLE,
                              [player])

    @staticmethod
    def update_flip(previous, flip):
        return Zobrist.update(previous, flip,
                              Zobrist.BLACK_TABLE,
                              Zobrist.WHITE_TABLE,
                              [1, -1])

    @staticmethod
    @nb.jit(nopython=True, cache=True)
    def hash(board, black_table, white_table):
        result = 0
        for row, col in zip(*np.where(board == 1)):
            result ^= black_table[row, col]

        for row, col in zip(*np.where(board == -1)):
            result ^= white_table[row, col]

        return result

    @staticmethod
    @nb.jit(nopython=True, cache=True)
    def update(previous, square, black_table, white_table, players):
        result = previous
        row, col = square
        for player in players:
            if player == 1:
                result ^= black_table[row, col]
            elif player == -1:
                result ^= white_table[row, col]
        return result
