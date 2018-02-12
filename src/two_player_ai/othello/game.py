import numba as nb
import numpy as np

from two_player_ai.game import Game
from two_player_ai.othello.utils import boards_equal
from two_player_ai.othello.zobrist import Zobrist


class OthelloBoard(object):
    BLACK_PLAYER = 1
    WHITE_PLAYER = -1

    def __init__(self, board=None, uid=None):
        self.size = 8
        self.board = self.initial_board() if board is None else board
        self.uid = Zobrist.from_state(self) if uid is None else uid

    def initial_board(self):
        """
        The initial board

        Returns:
            A np.array(size x size) representing the board
        """
        board = np.zeros((self.size, self.size), dtype=np.int8)
        center = int(self.size / 2)
        board[center - 1][center - 1] = OthelloBoard.BLACK_PLAYER
        board[center - 1][center] = OthelloBoard.WHITE_PLAYER
        board[center][center] = OthelloBoard.BLACK_PLAYER
        board[center][center - 1] = OthelloBoard.WHITE_PLAYER

        return board

    def count(self, player):
        return np.count_nonzero(self.board == player)

    def clone(self):
        """
        Deep clone the Othello state.

        Returns:
            A clone of the current state
        """
        return OthelloBoard(board=np.copy(self.board), uid=self.uid)

    def __eq__(self, other):
        return boards_equal(self.board, other.board)

    def __hash__(self):
        return self.uid


class Othello(Game):
    """
    The game of Othello.initial_state
    """

    """A list of all directions, denoted as (x,y) vector offsets"""
    DIRECTIONS = [
        (1, 1), (1, 0), (1, -1),
        (0, -1), (0, 1),
        (-1, -1), (-1, 0), (-1, 1)
    ]

    @staticmethod
    def initial_state():
        """
        The initial state of the game.

        Returns: The initial board of the Othello game and
                 player playing

        """
        return OthelloBoard(), OthelloBoard.BLACK_PLAYER

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
        directions = Othello.DIRECTIONS
        if state.count(0) > state.count(player):
            actions = Othello.forward_actions(board, player, aux, directions)
        else:
            actions = Othello.reverse_actions(board, player, aux, directions)
        return list(zip(*actions))

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

            for direction in Othello.DIRECTIONS:
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
        Check if the game has ended and if yes determine
        the outcome of game.

        Args:
            state: An Othello game state
        Returns:
            True if game has not ended else False
        """
        dirs = Othello.DIRECTIONS

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
        white = state.count(OthelloBoard.WHITE_PLAYER)
        black = state.count(OthelloBoard.BLACK_PLAYER)
        if white > black:
            return OthelloBoard.WHITE_PLAYER
        elif black > white:
            return OthelloBoard.BLACK_PLAYER
        else:
            return 0.5

    @staticmethod
    @nb.jit(nopython=True, cache=True)
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
    @nb.jit(nopython=True, cache=True)
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
    @nb.jit(nopython=True, cache=True)
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
    @nb.jit(nopython=True, cache=True)
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
    @nb.jit(nopython=True, cache=True)
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
