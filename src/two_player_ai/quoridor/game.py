import numba as nb
import numpy as np

import two_player_ai.quoridor.constants as quoridor_constants

from two_player_ai.game import Game
from two_player_ai.quoridor.utils import (
    boards_equal, cannonical_board
)
from two_player_ai.quoridor.zobrist import Zobrist


class QuoridorBoard(object):
    def __init__(self, board=None, uid=None, compute_uid=True):
        self.board = self.initial_board() if board is None else board
        #self.uid = Zobrist.from_state(self) if not uid and compute_uid else uid

    def initial_board(self):
        """
        The initial board

        Returns:
            A np.array(3 x size x size) representing the board
        """
        board_size = 9
        board = np.zeros(
            (3, board_size, board_size),
            dtype=np.int8
        )
        center = int(9 / 2)
        pieces_board = board[0]
        pieces_board[0][center] = 1
        pieces_board[board_size - 1][center] = -1

        return board

    def position_id(self, position):
        i, j = position
        return i * 9 + j

    def clone(self):
        """
        Deep clone the Othello state.

        Returns:
            A clone the current state
        """
        return QuoridorBoard(board=np.copy(self.board), uid=self.uid)

    def flat_board(self):
        aux = np.zeros((17, 17), dtype=np.int8)
        return QuoridorBoard.flatten(self.board, aux)

    @staticmethod
    @nb.jit(nopython=True, nogil=True, cache=True)
    def flatten(board, flat):
        pieces = board[0]
        height, width = pieces.shape
        for row, col in zip(*np.where(pieces != 0)):
            flat[2 * row, 2 * col] = pieces[row, col]

        h_fences = board[1]
        for row, col in zip(*np.where(h_fences != 0)):
            fences_row = 2 * row + 1

            fence_a_col = 2 * col
            fence_b_col = 2 * (col + 1)
            if (0 <= fences_row < 17
                and 0 <= fence_a_col < 17
                    and 0 <= fence_b_col < 17):
                flat[fences_row, fence_a_col] = h_fences[row, col] * 2
                flat[fences_row, fence_b_col] = h_fences[row, col] * 2

        v_fences = board[2]
        for row, col in zip(*np.where(v_fences != 0)):
            fences_col = 2 * col + 1

            fence_a_row = 2 * row
            fence_b_row = 2 * (row + 1)
            if (0 <= fences_col < 17
                and 0 <= fence_a_row < 17
                    and 0 <= fence_b_row < 17):
                flat[fence_a_row, fences_col] = h_fences[row, col] * 2
                flat[fence_b_row, fences_col] = h_fences[row, col] * 2

        return flat

    def binary_form(self, wrap=True):
        if wrap:
            return np.array([np.array([self.board])])
        else:
            return np.array([self.board])

    def __eq__(self, other):
        return boards_equal(self.board, other.board)

    def __hash__(self):
        return self.uid if self.uid else Zobrist.from_state(self)


class Quoridor(Game):
    @staticmethod
    def board_size():
        return quoridor_constants.BOARD_SIZE, quoridor_constants.BOARD_SIZE

    @staticmethod
    def action_size():
        return quoridor_constants.BOARD_SIZE * quoridor_constants.BOARD_SIZE

    @staticmethod
    def all_actions():
        return quoridor_constants.ALL_ACTIONS

    @staticmethod
    def symmetries(board, policy):
        symmetries = []

        for rotation in range(1, 5):
            for flip in [True, False]:
                symmetric_board = np.rot90(board, rotation)
                symmetric_policy = np.rot90(policy, rotation)
                if flip:
                    symmetric_board = np.fliplr(symmetric_board)
                    symmetric_policy = np.fliplr(symmetric_policy)

                symmetries.append((symmetric_board, symmetric_policy))
        return symmetries

    @staticmethod
    def initial_state():
        """
        The initial state of the game.

        Returns: The initial board of the Othello game and
                 player playing

        """
        return Quoridor(), quoridor_constants.WHITE_PLAYER

    @staticmethod
    def cannonical_state(state, player):
        cannonical = state.clone()
        cannonical.board = cannonical_board(cannonical.board, player)
        return cannonical

    @staticmethod
    def actions(state, player):
        """
        All available actions for the Othello game state.

        Args:
            state: An Othello game state
            player: The player playing
        Returns:
        """
        board = state.board

    @staticmethod
    def pawn_actions(board, player):
        size = board[0].size

        row, col = np.where(board == player)
        row, col = row[0], col[0]

        actions = []

        for hdir, vdir in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
            if 0 <= row + hdir < size and 0 <= col + vdir < size:
                if board[row + hdir, col + vdir] != 2:
                    if board[row + 2 * hdir, col + 2 * vdir] == 0:
                        actions.append((row + 2 * hdir, col + 2 * vdir))
                    elif board[row + 2 * hdir, col + 2 * vdir] == -player:
                        target_fence = board[row + 3 * hdir, col + 3 * vdir]
                        if target_fence == 2:
                            if hdir == 0:
                                h_neighbor_dirs = (1, -1)
                                for h_neighbor_dir in h_neighbor_dirs:
                                    if board[row + h_neighbor_dir, col + 2 * vdir] != 2:
                                        actions.append((row + 2 * h_neighbor_dir, col + 4 * vdir))
                            elif vdir == 0:
                                v_neighbor_dirs = (1, -1)
                                for v_neighbor_dir in v_neighbor_dirs:
                                    if board[row + 2 * hdir, col + v_neighbor_dir] != 2:
                                        actions.append((row + 2 * hdir, col + 2 * v_neighbor_dir))
                        else:
                            actions.append((row + 4 * hdir, col + 4 * vdir))

        return actions

    def foo(state, position):
        if board[x + hdir, y + hdir] != WALL:



    @staticmethod
    def result(state, player, action):
        """
        The resulting state that is produced when the
        player applies the action to the game state.
        In turn-based games the resulting player
        is the opponent (-player). Assumes that the action
        is valid for the given state.

        If no action is specified current player loses it's
        turn without affecting the board.

        Args:
            state: An Othello game state
            action: action taken by current player
        Returns:
            result_state: the resulting Othello game state
        """

    @staticmethod
    def terminal_test(state, player):
        """
        Check if the game has ended.

        Args:
            state: An Othello game state
        Returns:
            True if game has not ended else False
        """

    @staticmethod
    def winner(state):
        """
        Check winning player of the game.

        Args:
            state: An Othello game state
        Returns:
            WHITE_PLAYER if white player wins
            BLACK_PLAYER if black player wins
            0.5 if tie
        """
        white = state.count(quoridor_constants.WHITE_PLAYER)
        black = state.count(quoridor_constants.BLACK_PLAYER)
        if white > black:
            return quoridor_constants.WHITE_PLAYER
        elif black > white:
            return quoridor_constants.BLACK_PLAYER
        else:
            return 0.5
