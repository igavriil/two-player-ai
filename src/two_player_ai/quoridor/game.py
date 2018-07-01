import numba as nb
import numpy as np

import two_player_ai.quoridor.constants as quoridor_constants

from two_player_ai.game import Game
from two_player_ai.quoridor.utils import (
    boards_equal, cannonical_board
)
from two_player_ai.quoridor.zobrist import Zobrist
from two_player_ai.quoridor import board_utils


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
        center = int(board_size / 2)
        self.pawn_board = board[0]
        self.horizontal_fences = board[1]
        self.vertical_fences = board[2]

        self.adjacency_matrix = board_utils.create_adjacency_matrix(
            board_size, board_size
        )

        self.pawn_board[0][center] = 1
        self.pawn_board[board_size - 1][center] = -1

        # player 1  targets pawn_board[board_size - 1]
        # player -1 targets pawn_board[0]

    def clone(self):
        """
        Deep clone the Othello state.

        Returns:
            A clone the current state
        """
        return QuoridorBoard(board=np.copy(self.board), uid=self.uid)

    def __eq__(self, other):
        return boards_equal(self.board, other.board)

    def __hash__(self):
        return self.uid if self.uid else Zobrist.from_state(self)


class Quoridor(Game):

    @staticmethod
    def initial_state():
        """
        The initial state of the game.

        Returns: The initial board of the Othello game and
                 player playing

        """
        return QuoridorBoard(), quoridor_constants.WHITE_PLAYER

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
