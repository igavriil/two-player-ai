import numpy as np
import numba as nb


@nb.jit(nopython=True, nogil=True, cache=True, fastmath=True)
def create_adjacency_matrix(rows, cols):
    """
    Create an adjacency matrix representing connections between
    poard positions. Each position is represented as a single integer
    using the following rule for conversion:
        i = rows * row + col

    Args:
        rows: The number of rows in the board
        cols: The number of cols in the board
    Returns:
        A 2d array with (rows * cols x rows * cols) dimension
        where non-zero values m[row][col] represent connections
        between row and col position.
    """
    m = np.zeros((rows * cols, rows * cols), dtype=np.int8)
    for row in range(rows):
        for col in range(cols):
            from_point = rows * row + col
            for row_offset, col_offset in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
                to_point = rows * (row + row_offset) + (col + col_offset)
                to_point_row = (row + row_offset)
                to_point_col = (col + col_offset)
                if 0 <= to_point_row < rows and 0 <= to_point_col < cols:
                    m[from_point][to_point] = 1
    return m


@nb.jit(nopython=True, nogil=True, cache=True, fastmath=True)
def fence_actions(board_size, horizontal_fences, vertical_fences):
    """
    Args:
        horizontal_fences: A 2d with (board_size x board_size) dimension
            where no-zero values represent a horizontal fence that remove
            the following connections:
            horizontail_fence [r,c] -> removes:
                * (r, c) <-> (r + 1, c)
                * (r, c + 1) <-> (r + 1, c + 1)

        vertical_fences: A 2d with (board_size x board_size) dimension
            where no-zero values represent a vertical fence that remove
            the following connections:
            horizontail_fence [r,c] -> removes:
                * (r, c) <-> (r, c + 1)
                * (r + 1, c) <-> (r + 1, c + 1)
    Returns:
        4 (board_size - 1 x board_size -1) arrays:
        * invalid_horizontal: invalid horizontal positions
        * invalid_vertical: invalid vertical postions
        * unsafe_horizontal: horizontal positions that are directly connected
          with one or more fences or boundary positions. On each cell the
          number of connected fences or boundary positions will be recorded.
        * unsafe_vertical: vertical positions that are directly connected
          with one or more fences or boundary positions. On each cell the
          number of connected fences or boundary positions will be recorded.
    """
    fences_size = board_size - 1
    invalid_horizontal = np.zeros((fences_size, fences_size), dtype=np.uint8)
    unsafe_horizontal = np.zeros((fences_size, fences_size), dtype=np.uint8)

    invalid_vertical = np.zeros((fences_size, fences_size), dtype=np.uint8)
    unsafe_vertical = np.zeros((fences_size, fences_size), dtype=np.uint8)

    for row in range(fences_size):
        for col in range(fences_size):
            if col == 0 or col == fences_size - 1:
                unsafe_horizontal[row, col] += 1
            if row == 0 or row == fences_size - 1:
                unsafe_vertical[row, col] += 1

            if vertical_fences[row, col]:
                invalid_vertical[row, col] += 1
                invalid_horizontal[row, col] += 1
                if 0 <= row + 1 < fences_size:
                    invalid_vertical[row + 1, col] += 1
                if 0 <= row - 1 < fences_size:
                    invalid_vertical[row - 1, col] += 1
                for row_offset in [-2, 2]:
                    neighbor_row = row + row_offset
                    if (0 <= neighbor_row < fences_size and
                            0 <= col < fences_size):
                        unsafe_horizontal[neighbor_row][col] += 1
                for row_offset in [-1, 0, 1]:
                    for col_offset in [-1, 0, 1]:
                        neighbor_row = row + row_offset
                        neighbor_col = col + col_offset
                        if (0 <= neighbor_row < fences_size and
                                0 <= neighbor_col < fences_size):
                            unsafe_horizontal[neighbor_row][neighbor_col] += 1

            if horizontal_fences[row, col]:
                invalid_vertical[row, col] += 1
                invalid_horizontal[row, col] += 1
                if 0 <= col + 1 < fences_size:
                    invalid_horizontal[row, col + 1] += 1
                if 0 <= col - 1 < fences_size:
                    invalid_horizontal[row, col - 1] += 1
                for col_offset in [-2, 2]:
                    neighbor_col = col + col_offset
                    if (0 <= row < fences_size and
                            0 <= neighbor_col < fences_size):
                        unsafe_horizontal[row][neighbor_col] += 1
                for row_offset in [-1, 0, 1]:
                    for col_offset in [-1, 0, 1]:
                        neighbor_row = row + row_offset
                        neighbor_col = col + col_offset
                        if (0 <= neighbor_row < fences_size and
                                0 <= neighbor_col < fences_size):
                            unsafe_vertical[neighbor_row][neighbor_col] += 1

    return (invalid_horizontal,
            invalid_vertical,
            unsafe_horizontal,
            unsafe_vertical)


@nb.jit(nopython=True, nogil=True, cache=True, fastmath=True)
def valid_pawn_actions(board, adj_matrix, player):
    player_position = np.where(board == player)
    player_row, player_col = player_position
    player_row, player_col = player_row[0], player_col[0]

    opponent_position = np.where(board == -player)
    opponent_row, opponent_col = opponent_position
    opponent_row, opponent_col = opponent_row[0], opponent_col[0]

    size = board[0].size
    actions = []

    for row_offset, col_offset in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
        target_row = row + row_offset
        target_col = col + col_offset

        if 0 <= target_row < size and 0 <= target_col < size:
            if adj_matrix[size * row + col, size * target_col + target_row]:
                if board[target_row, target_col] == -player:
                    if row_offset == 0:
                        hop_row = row + 2 * row_offset
                        hop_col = col + 2 * col_offset
                        if 0 <= hop_row < size and 0 <= hop_col < size:
                            if adj_matrix[
                                size * target_row + target_col,
                                size * hop_col + hop_row
                            ]:
                                actions.append((target_row, target_col))
                        else:
                            for side_row_offset in [1, -1]:
                                side_hop_row = target_row + side_row_offset
                                if 0 <= side_hop_row < size:
                                    if adj_matrix[
                                        size * target_row + target_col,
                                        size * side_hop_row + target_col
                                    ]:
                                        actions.append(
                                            (side_hop_row, target_col)
                                        )
                    else:  # col_offset == 0
                        hop_row = row + 2 * row_offset
                        hop_col = col + 2 * col_offset
                        if 0 <= hop_row < size and 0 <= hop_col < size:
                            if adj_matrix[
                                size * target_row + target_col,
                                size * hop_col + hop_row
                            ]:
                                actions.append((target_row, target_col))
                        else:
                            for side_col_offset in [1, -1]:
                                side_hop_col = target_col + side_col_offset
                                if 0 <= side_hop_col < size:
                                    if adj_matrix[
                                        size * target_row + target_col,
                                        size * target_row + side_hop_col
                                    ]:
                                        actions.append(
                                            (target_row, side_hop_col)
                                        )
                else:
                    actions.append((target_row, target_col))

    return actions


@nb.jit(nopython=True, nogil=True, cache=True, fastmath=True)
def valid_fence_actions(board, horizontal_fences, vertical_fences, adj_matrix):
    (invalid_horizontal,
     invalid_vertical,
     unsafe_horizontal,
     unsafe_vertical) = fence_actions(horizontal_fences, vertical_fences)

    board_size = board[0].size
    fences_size = board_size - 1
    player_1_position = np.where(board == 1)
    player_1_row, player_1_col = player_1_position
    player_1_row, player_1_col = player_1_row[0], player_1_col[0]

    player_2_position = np.where(board == -1)
    player_2_row, player_2_col = player_2_position
    player_2_row, player_2_col = player_2_row[0], player_2_col[0]

    total = (horizontal_fences == 1).sum() + (vertical_fences == 1).sum()
    valid_vertical = np.zeros((fences_size, fences_size), dtype=np.uint8)
    valid_horizontal = np.zeros((fences_size, fences_size), dtype=np.uint8)

    for row in range(fences_size):
        for col in range(fences_size):
            if invalid_vertical[row, col] != 0:
                continue

            if unsafe_vertical[row, col] <= 1 or total <= 4:
                valid_vertical[row, col] = 1
            else:
                adj_matrix[
                    board_size * row + col,
                    board_size * (row + 1) + col
                ] = 0
                adj_matrix[
                    board_size * (row + 1) + col,
                    board_size * row + col
                ] = 0
                adj_matrix[
                    board_size * row + col + 1,
                    board_size * (row + 1) + col + 1
                ] = 0
                adj_matrix[
                    board_size * (row + 1) + col + 1,
                    board_size * row + col + 1
                ] = 0

                connected = connected_components(adj_matrix, 0)
                if connected.all():
                    valid_vertical[row, col] = 1

                adj_matrix[
                    board_size * row + col,
                    board_size * (row + 1) + col
                ] = 1
                adj_matrix[
                    board_size * (row + 1) + col,
                    board_size * row + col
                ] = 1
                adj_matrix[
                    board_size * row + col + 1,
                    board_size * (row + 1) + col + 1
                ] = 1
                adj_matrix[
                    board_size * (row + 1) + col + 1,
                    board_size * row + col + 1
                ] = 1

    for row in range(fences_size):
        for col in range(fences_size):
            if invalid_horizontal[row, col] != 0:
                continue

            if unsafe_horizontal[row, col] <= 1 or total <= 5:
                valid_horizontal[row, col] = 1
            else:
                adj_matrix[
                    board_size * row + col,
                    board_size * row + col + 1
                ] = 0
                adj_matrix[
                    board_size * row + col + 1,
                    board_size * row + col
                ] = 0
                adj_matrix[
                    board_size * (row + 1) + col,
                    board_size * (row + 1) + col + 1
                ] = 0
                adj_matrix[
                    board_size * (row + 1) + col + 1,
                    board_size * (row + 1) + col
                ] = 0

                connected = connected_components(adj_matrix, 0)
                if connected.all():
                    valid_horizontal[row, col] = 1

                adj_matrix[
                    board_size * row + col,
                    board_size * row + col + 1
                ] = 1
                adj_matrix[
                    board_size * row + col + 1,
                    board_size * row + col
                ] = 1
                adj_matrix[
                    board_size * (row + 1) + col,
                    board_size * (row + 1) + col + 1
                ] = 1
                adj_matrix[
                    board_size * (row + 1) + col + 1,
                    board_size * (row + 1) + col
                ] = 1

    return valid_vertical, valid_horizontal


@nb.jit(nopython=True, nogil=True, cache=True, fastmath=True)
def dfs_bridge(matrix, u, v, cnt, low, pre, bridges):
    """

    Args:
        matrix: An adjacency matrix
        u: The source vertex
        v: The source vertex
        cnt: The depth of dfs search
        low: A 1-d array with size equal to the number of vertices that holds
             the smallest pre-order number of any vertex reachable.
        pre: Identifies each vertex by its depth in the dfs tree. Holds the
             pre-order traversal numbering.
    """
    vertices = matrix[0].size

    cnt += 1
    pre[v] = cnt
    low[v] = pre[v]

    for w in range(vertices):
        if matrix[v][w]:
            if pre[w] == -1:
                dfs_bridge(matrix, v, w, cnt, low, pre, bridges)

                if low[v] > low[w]:
                    low[v] = low[w]

                if low[w] == pre[w]:
                    bridges[v, w] += 1
                    bridges[w, v] += 1
            elif w != u:
                if low[v] > pre[w]:
                    low[v] = pre[w]


@nb.jit(nopython=True, nogil=True, cache=True, fastmath=True)
def bridges(matrix):
    """
    Args:
        matrix: An adjacency matrix
    Returns:
        A 2-d matrix with non-zero elements representing bridges on the graph.
        Bridges are edges that if removed will result in disconnecting previous
        connected components.
    """
    vertices = matrix[0].size

    b = np.zeros(matrix.shape, dtype=np.int8)
    pre = -1 * np.ones(vertices, dtype=np.int8)
    low = -1 * np.ones(vertices, dtype=np.int8)
    cnt = 0

    for v in range(vertices):
        dfs_bridge(matrix, v, v, cnt, low, pre, b)

    return b, pre, low


@nb.jit(nopython=True, nogil=True, cache=True, fastmath=False)
def connected_components(matrix, source):
    """
    Args:
        matrix: An adjacency matrix
        source: The source vertex
    Returns:
        A 1-d array of size equal number of vertices where non-zero
        elements denote connectivity with the source vertex at depth
        equal to the value stored in the array.
    """
    vertices = matrix[0].size

    queue = np.zeros(vertices + 1, dtype=np.int32)
    queue_start = 0
    queue[queue_start] = source
    queue_items = 1
    queue_end = 1

    components = np.zeros(vertices, dtype=np.int32)

    while queue_items:
        current = queue[queue_start]
        queue_start += 1
        queue_items -= 1

        depth = components[current]

        for i in range(vertices):
            if matrix[current][i] and components[i] == 0:
                components[i] = depth + 1
                queue[queue_end] = i
                queue_end += 1
                queue_items += 1

    rows = cols = np.int_(np.sqrt(vertices))
    return np.reshape(components, (rows, cols))


@nb.jit(nopython=True, nogil=True, cache=True, fastmath=False)
def astar(matrix, source, target):
    vertices = matrix[0].size
    dim = np.int_(np.sqrt(vertices))

    closed_set = np.zeros(vertices, dtype=np.int8)
    open_set = np.inf * np.ones(vertices)
    g_score = np.inf * np.ones(vertices)
    f_score = np.inf * np.ones(vertices)
    came_from = np.zeros(vertices, dtype=np.int8)

    source_row, source_col = np.floor_divide(source, dim), np.mod(source, dim)
    target_row, target_col = np.floor_divide(target, dim), np.mod(target, dim)

    open_set[source] = 1
    g_score[source] = 0
    f_score[source] = (np.abs(source_row - target_row) +
                       np.abs(source_col - target_col))

    while np.any(open_set == 1):
        current = np.argmin(f_score * open_set)
        if current == target:
            path = [current]
            while came_from[current] != source:
                current = came_from[current]
                path.append(current)
            return path

        open_set[current] = np.inf
        closed_set[current] = 1

        for i in range(vertices):
            if matrix[current][i]:
                if closed_set[i]:
                    continue

                if open_set[i] == np.inf:
                    open_set[i] = 1

                t_score = g_score[current] + 1
                if t_score >= g_score[i]:
                    continue

                source_row = np.floor_divide(i, dim)
                source_col = np.mod(i, dim)

                came_from[i] = current
                g_score[i] = t_score
                f_score[i] = g_score[i] + (np.abs(source_row - target_row) +
                                           np.abs(source_col - target_col))
