import numba as nb
import numpy as np

from two_player_ai.othello.game import Othello


class Heuristic(object):
    @staticmethod
    def evaluate(state):
        raise NotImplementedError

    @staticmethod
    @nb.jit(nopython=True, cache=True)
    def evaluate_weighted_sum(board, weights):
        return np.sum(board * weights)


class Weighted(Heuristic):
    WEIGHTS = np.array([
        [100, -25, 10, 5, 5, 10, -25, 100],
        [-25, -25,  2, 2, 2,  2, -25, -25],
        [10,    2,  5, 1, 1,  5,   2,  10],
        [5,     2,  1, 2, 2,  1,   2,   5],
        [5,     2,  1, 2, 2,  1,   2,   5],
        [10,    2,  5, 1, 1,  5,   2,  10],
        [-25, -25,  2, 2, 2,  2, -25, -25],
        [100, -25, 10, 5, 5, 10, -25, 100]
    ], dtype=np.int8)

    @staticmethod
    def evaluate(state):
        return Weighted.evaluate_weighted_sum(state.board, Weighted.WEIGHTS)


class PiecesCount(Heuristic):
    @staticmethod
    def evaluate(state):
        return PiecesCount._evaluate(state.board)

    def _evaluate(board):
        return np.sum(board)


class CornerPieces(Heuristic):
    CORNERS = np.array([
        [1, 0, 0, 0, 0, 0, 0, 1],
        [0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0],
        [1, 0, 0, 0, 0, 0, 0, 1]
    ], dtype=np.int8)

    @staticmethod
    def evaluate(state):
        return CornerPieces.evaluate_weighted_sum(state.board,
                                                  CornerPieces.CORNERS)


class EdgePieces(Heuristic):
    EDGES = np.array([
        [0, 0, 1, 1, 1, 1, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0],
        [1, 0, 0, 0, 0, 0, 0, 1],
        [1, 0, 0, 0, 0, 0, 0, 1],
        [1, 0, 0, 0, 0, 0, 0, 1],
        [1, 0, 0, 0, 0, 0, 0, 1],
        [0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 1, 1, 1, 1, 0, 0]
    ], dtype=np.int8)

    @staticmethod
    def evaluate(state):
        return EdgePieces.evaluate_weighted_sum(state.board, EdgePieces.EDGES)


class CornerEdgePieces(Heuristic):
    TOP_LEFT_CORNER_EDGES = np.array([
        [0, 2, 0, 0, 0, 0, 0, 0],
        [2, 1, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0]
    ], dtype=np.int8)

    TOP_RIGHT_CORNER_EDGES = np.array([
        [0, 0, 0, 0, 0, 0, 2, 0],
        [0, 0, 0, 0, 0, 0, 1, 2],
        [0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0]
    ], dtype=np.int8)

    BOTTON_LEFT_CORNER_EDGES = np.array([
        [0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0],
        [2, 1, 0, 0, 0, 0, 0, 0],
        [0, 2, 0, 0, 0, 0, 0, 0]
    ], dtype=np.int8)

    BOTTOM_RIGHT_CORNER_EDGES = np.array([
        [0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 1, 2],
        [0, 0, 0, 0, 0, 0, 2, 0]
    ], dtype=np.int8)

    @staticmethod
    def evaluate(state):
        weights = np.zeros((8, 8), dtype=np.int8)
        if state.board[0, 0] == 0:
            weights -= CornerEdgePieces.TOP_LEFT_CORNER_EDGES
        if state.board[0, 7] == 0:
            weights -= CornerEdgePieces.TOP_RIGHT_CORNER_EDGES
        if state.board[7, 0] == 0:
            weights -= CornerEdgePieces.BOTTON_LEFT_CORNER_EDGES
        if state.board[7, 7] == 0:
            weights += CornerEdgePieces.BOTTOM_RIGHT_CORNER_EDGES

        return CornerEdgePieces.evaluate_weighted_sum(state.board,
                                                      EdgePieces.EDGES)


class Mobility(Heuristic):
    @staticmethod
    def evaluate(state):
        return len(Othello.actions(state, 1)) - len(Othello.actions(state, -1))


class Stability(Heuristic):
    EDGE_FILTER = np.array([
        [1, 0, 0, 0, 0, 0, 0, 1],
        [0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0],
        [1, 0, 0, 0, 0, 0, 0, 1]
    ], dtype=np.int8)

    def evaluate(state):
        return Stability.stability_squares(state.board, Stability.EDGE_FILTER,
                                           +1, Othello.DIRECTIONS) - \
               Stability.stability_squares(state.board, Stability.EDGE_FILTER,
                                           -1, Othello.DIRECTIONS)

    @staticmethod
    @nb.jit(nopython=True, cache=True)
    def stability_squares(board, edge_filter, player, directions):
        size = board[0].size
        for row, col in zip(*np.where(board * edge_filter == player)):
            for hdir, vdir in directions:
                x, y = row + hdir, col + vdir
                step = 1
                while (0 <= x < size and 0 <= y < size and
                       board[x, y] == player):
                    step += 1
                    x, y = row + step * hdir, col + step * vdir

        return step
