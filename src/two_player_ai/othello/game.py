import numba as nb
import numpy as np

import two_player_ai.othello.constants as othello_constants

from two_player_ai.game import Game
from two_player_ai.othello.utils import (
    boards_equal, cannonical_board
)
from two_player_ai.othello.zobrist import Zobrist


class OthelloBoard(object):
    def __init__(self, board=None, uid=None, compute_uid=True):
        self.board = self.initial_board() if board is None else board
        self.uid = Zobrist.from_state(self) if not uid and compute_uid else uid

    def initial_board(self):
        """
        The initial board

        Returns:
            A np.array(size x size) representing the board
        """
        board = np.zeros(
            (othello_constants.BOARD_SIZE, othello_constants.BOARD_SIZE),
            dtype=np.int8
        )
        center = int(othello_constants.BOARD_SIZE / 2)
        board[center - 1][center - 1] = othello_constants.BLACK_PLAYER
        board[center - 1][center] = othello_constants.WHITE_PLAYER
        board[center][center] = othello_constants.BLACK_PLAYER
        board[center][center - 1] = othello_constants.WHITE_PLAYER

        return board

    def count(self, player):
        return np.count_nonzero(self.board == player)

    def clone(self):
        """
        Deep clone the Othello state.

        Returns:
            A clone the current state
        """
        return OthelloBoard(board=np.copy(self.board), uid=self.uid)

    def binary_form(self, wrap=True):
        if wrap:
            return np.array([np.array([self.board])])
        else:
            return np.array([self.board])

    def __eq__(self, other):
        return boards_equal(self.board, other.board)

    def __hash__(self):
        return self.uid if self.uid else Zobrist.from_state(self)


class Othello(Game):
    @staticmethod
    def board_size():
        return othello_constants.BOARD_SIZE, othello_constants.BOARD_SIZE

    @staticmethod
    def action_size():
        return othello_constants.BOARD_SIZE * othello_constants.BOARD_SIZE

    @staticmethod
    def all_actions():
        return othello_constants.ALL_ACTIONS

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
        return OthelloBoard(), othello_constants.BLACK_PLAYER

    @staticmethod
    def cannonical_state(state, player):
        cannonical = state.clone()
        cannonical.board = cannonical_board(cannonical.board, player)
        return cannonical

    @staticmethod
    def actions(state, player):
        """
        All available actions for the Othello game state.
        Actions are blank squares where the user can place his chips.

        Args:
            state: An Othello game state
            player: The player playing
        Returns:
            A list of (x, y) tuples denoting all squares the user
            can play
        """
        board = state.board
        aux = np.zeros(board.shape, dtype=np.bool)
        directions = othello_constants.DIRECTIONS
        if state.count(0) > state.count(player):
            actions = Othello.forward_actions(board, player, aux, directions)
        else:
            actions = Othello.reverse_actions(board, player, aux, directions)
        return list(zip(*actions))

    @staticmethod
    def actions_mask(state, player):
        """
        All available actions for the Othello game state.
        Actions are blank squares where the user can place his chips.

        Args:
            state: An Othello game state
            player: The player playing
        Returns:
            A mask of the board with True filled in valid squares.
        """
        board = state.board
        aux = np.zeros(board.shape[0] * board.shape[1], dtype=np.int8)
        directions = othello_constants.DIRECTIONS
        if state.count(0) > state.count(player):
            actions = Othello.forward_actions_mask(
                board, player, aux, directions
            )
        else:
            actions = Othello.reverse_actions_mask(
                board, player, aux, directions
            )
        return actions

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
        result_state = state.clone()
        board = result_state.board

        uid = result_state.uid

        if action:
            aux = np.zeros(result_state.board.shape, dtype=np.bool)

            for direction in othello_constants.DIRECTIONS:
                flips = Othello.flips(board, player, aux, action, direction)
                if flips:
                    for flip in zip(*flips):
                        result_state.board[flip] = player
                        uid = Zobrist.update_flip(uid, flip)

            uid = Zobrist.update_action(uid, action, player)
            result_state.board[action] = player
            result_state.uid = uid

        return result_state, -player

    @staticmethod
    def terminal_test(state, player):
        """
        Check if the game has ended.

        Args:
            state: An Othello game state
        Returns:
            True if game has not ended else False
        """
        dirs = othello_constants.DIRECTIONS

        if state.count(0) >= state.count(player):
            if not Othello.has_forward_actions(state.board, player, dirs):
                if not Othello.has_forward_actions(state.board, -player, dirs):
                    return True
        else:
            if not Othello.has_reverse_actions(state.board, player, dirs):
                if not Othello.has_reverse_actions(state.board, -player, dirs):
                    return True

        return False

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
        white = state.count(othello_constants.WHITE_PLAYER)
        black = state.count(othello_constants.BLACK_PLAYER)
        if white > black:
            return othello_constants.WHITE_PLAYER
        elif black > white:
            return othello_constants.BLACK_PLAYER
        else:
            return 0.5

    @staticmethod
    @nb.jit(nopython=True, nogil=True, cache=True)
    def forward_actions(board, player, actions, directions):
        size = board[0].size
        for row, col in zip(*np.where(board == player)):
            for hdir, vdir in directions:
                x, y = row + hdir, col + vdir
                step = 1
                if 0 <= x < size and 0 <= y < size and board[x, y] == -player:
                    step += 1
                    x, y = row + step * hdir, col + step * vdir
                    while (0 <= x < size and 0 <= y < size and
                           board[x, y] != player):
                        if board[x, y] == 0:
                            actions[x, y] = True
                            break

                        step += 1
                        x, y = row + step * hdir, col + step * vdir
        return np.where(actions)

    @staticmethod
    @nb.jit(nopython=True, nogil=True, cache=True)
    def forward_actions_mask(board, player, actions, directions):
        size = board[0].size
        for row, col in zip(*np.where(board == player)):
            for hdir, vdir in directions:
                x, y = row + hdir, col + vdir
                step = 1
                if 0 <= x < size and 0 <= y < size and board[x, y] == -player:
                    step += 1
                    x, y = row + step * hdir, col + step * vdir
                    while (0 <= x < size and 0 <= y < size and
                           board[x, y] != player):
                        if board[x, y] == 0:
                            actions[x * size + y] = 1
                            break

                        step += 1
                        x, y = row + step * hdir, col + step * vdir
        return actions

    @staticmethod
    @nb.jit(nopython=True, nogil=True, cache=True)
    def reverse_actions(board, player, actions, directions):
        size = board[0].size
        for row, col in zip(*np.where(board == 0)):
            for hdir, vdir in directions:
                x, y = row + hdir, col + vdir
                step = 1
                if 0 <= x < size and 0 <= y < size and board[x, y] == -player:
                    step += 1
                    x, y = row + step * hdir, col + step * vdir
                    while 0 <= x < size and 0 <= y < size and board[x, y] != 0:
                        if board[x, y] == player:
                            actions[row, col] = True
                            break

                        step += 1
                        x, y = row + step * hdir, col + step * vdir
        return np.where(actions)

    @staticmethod
    @nb.jit(nopython=True, nogil=True, cache=True)
    def reverse_actions_mask(board, player, actions, directions):
        size = board[0].size
        for row, col in zip(*np.where(board == 0)):
            for hdir, vdir in directions:
                x, y = row + hdir, col + vdir
                step = 1
                if 0 <= x < size and 0 <= y < size and board[x, y] == -player:
                    step += 1
                    x, y = row + step * hdir, col + step * vdir
                    while 0 <= x < size and 0 <= y < size and board[x, y] != 0:
                        if board[x, y] == player:
                            actions[row * size + col] = 1
                            break

                        step += 1
                        x, y = row + step * hdir, col + step * vdir
        return actions

    @staticmethod
    @nb.jit(nopython=True, nogil=True, cache=True)
    def has_forward_actions(board, player, directions):
        size = board[0].size
        for row, col in zip(*np.where(board == player)):
            for hdir, vdir in directions:
                x, y = row + hdir, col + vdir
                step = 1
                if 0 <= x < size and 0 <= y < size and board[x, y] == -player:
                    step += 1
                    x, y = row + step * hdir, col + step * vdir
                    while (0 <= x < size and 0 <= y < size and
                           board[x, y] != player):
                        if board[x, y] == 0:
                            return True

                        step += 1
                        x, y = row + step * hdir, col + step * vdir

        return False

    @staticmethod
    @nb.jit(nopython=True, nogil=True, cache=True)
    def has_reverse_actions(board, player, directions):
        size = board[0].size
        for row, col in zip(*np.where(board == 0)):
            for hdir, vdir in directions:
                x, y = row + hdir, col + vdir
                step = 1
                if 0 <= x < size and 0 <= y < size and board[x, y] == -player:
                    step += 1
                    x, y = row + step * hdir, col + step * vdir
                    while 0 <= x < size and 0 <= y < size and board[x, y] != 0:
                        if board[x, y] == player:
                            return True

                        step += 1
                        x, y = row + step * hdir, col + step * vdir

        return False

    @staticmethod
    @nb.jit(nopython=True, nogil=True, cache=True)
    def flips(board, player, aux, square, direction):
        size = board[0].size
        row, col = square
        hdir, vdir = direction

        x, y = row + hdir, col + vdir
        step = 1
        if 0 <= x < size and 0 <= y < size and board[x, y] == -player:
            aux[x, y] = True
            step += 1
            x, y = row + step * hdir, col + step * vdir
            while 0 <= x < size and 0 <= y < size and board[x, y] != 0:
                if board[x, y] == player:
                    return np.where(aux)

                step += 1
                x, y = row + step * hdir, col + step * vdir
