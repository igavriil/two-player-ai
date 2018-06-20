import numpy as np
import numba as nb


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

        self.adjacency_matrix = Board.create_adjacency_matrix()

        self.pawn_board[0][center] = 1
        self.pawn_board[board_size - 1][center] = -1

    @staticmethod
    @nb.jit(nopython=True, nogil=True, cache=True, fastmath=False)
    def create_adjacency_matrix():
        m = np.zeros((81, 81), dtype=np.int8)
        for row in range(9):
            for col in range(9):
                from_point = 9 * row + col
                for row_offset, col_offset in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
                    to_point = 9 * (row + row_offset) + (col + col_offset)
                    if 0 <= to_point < 81:
                        m[from_point][to_point] = 1
        return m


def remove_edge(from_point, to_point):
    from_point = one_d_point(from_point[0], from_point[1])
    to_point = one_d_point(to_point[0], to_point[1])

    m[from_point][to_point] = 0


remove_edge((0, 1), (1, 1))
remove_edge((0, 2), (1, 2))
remove_edge((0, 3), (1, 3))
remove_edge((0, 0), (1, 0))

@nb.jit(nopython=True, nogil=True, cache=True, fastmath=True)
def valid_fence_positions(horizontal_fences, vertical_fences):
    invalid_horizontal = np.zeros((8, 8), dtype=np.uint8)
    unsafe_horizontal = np.zeros((8, 8), dtype=np.uint8)

    invalid_vertical = np.zeros((8, 8), dtype=np.uint8)
    unsafe_vertical = np.zeros((8, 8), dtype=np.uint8)

    for row in range(8):
        for col in range(8):
            if col == 0 or col == 7:
                horizontal_fences[row, col] += 1
            if row == 0 or row == 7:
                vertical_fences[row, col] += 1

            if vertical_fences[row, col]:
                invalid_vertical[row, col] += 1
                invalid_horizontal[row, col] += 1
                if 0 <= row + 1 < 8:
                    invalid_vertical[row + 1, col] += 1
                if 0 <= row - 1 < 8:
                    invalid_vertical[row - 1, col] += 1
                for row_offset in [-2, 2]:
                    neighbor_row = row + row_offset
                    if (0 <= neighbor_row < 8 and 0 <= col < 8):
                        unsafe_horizontal[neighbor_row][col] += 1
                for row_offset in [-1, 0, 1]:
                    for col_offset in [-1, 0, 1]:
                        neighbor_row = row + row_offset
                        neighbor_col = col + col_offset
                        if (0 <= neighbor_row < 8 and 0 <= neighbor_col < 8):
                            unsafe_horizontal[neighbor_row][neighbor_col] += 1

            if horizontal_fences[row, col]:
                invalid_vertical[row, col] += 1
                invalid_horizontal[row, col] += 1
                if 0 <= col + 1 < 8:
                    invalid_horizontal[row, col + 1] += 1
                if 0 <= col - 1 < 8:
                    invalid_horizontal[row, col - 1] += 1
                for col_offset in [-2, 2]:
                    neighbor_col = col + col_offset
                    if (0 <= row < 8 and 0 <= neighbor_col < 8):
                        unsafe_horizontal[row][neighbor_col] += 1
                for row_offset in [-1, 0, 1]:
                    for col_offset in [-1, 0, 1]:
                        neighbor_row = row + row_offset
                        neighbor_col = col + col_offset
                        if (0 <= neighbor_row < 8 and 0 <= neighbor_col < 8):
                            unsafe_vertical[neighbor_row][neighbor_col] += 1

    return invalid_vertical, invalid_horizontal, unsafe_vertical, unsafe_horizontal


@nb.jit(nopython=True, nogil=True, cache=True, fastmath=False)
def connected_components(matrix, source):
    vertices = matrix[0].size

    queue = np.zeros(vertices, dtype=np.int8)
    queue_index = 0
    queue[queue_index] = source
    queue_items = 1

    components = np.zeros(vertices, dtype=np.int8)
    components[source] = 1

    while queue_items:
        current = queue[queue_index]
        queue_index += 1
        queue_items -= 1

        for i in range(vertices):
            if matrix[current][i] and components[i] != 1:
                components[i] = 1
                queue[queue_index] = i
                queue_items += 1

    return components

@benchmark
def benchmark_cc():
    for i in range(10000):
        connected_components(m)



+---+---+---+---+---+---+
|   |   |   |   |   |   |
+---+---+---+---+---+---+
|   |   |   |   |   |   |
+---+---+---+---+---+---+
|   |   |   |   |   |   |
+---+---+---+---+---+---+
|   |   |   |   |   |   |
+---+---+---+---+---+---+
|   |   |   |   |   |   |
+---+---+---+---+---+---+
|   |   |   |   |   |   |
+---+---+---+---+---+---+
