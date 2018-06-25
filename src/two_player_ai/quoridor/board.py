import numpy as np
from two_player_ai.quoridor import board_utils


class Board(object):
    def __init__(self):
        board_size = 9
        board = np.zeros(
            (3, board_size, board_size),
            dtype=np.int8
        )
        center = int(board_size / 2)
        self.pawn_board = board[0]
        self.horizontal_fences = board[1]
        self.vertical_fences = board[2]

        self.adjacency_matrix = Board.create_adjacency_matrix(
            board_size, board_size
        )

        self.pawn_board[0][center] = 1
        self.pawn_board[board_size - 1][center] = -1

        # player 1  targets pawn_board[board_size - 1]
        # player -1 targets pawn_board[0]
