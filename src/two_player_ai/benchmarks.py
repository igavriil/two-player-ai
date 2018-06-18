import time
import numpy as np
from two_player_ai.othello.game import Othello
from two_player_ai.mcts import Mcts, MctsTreeNode
from two_player_ai.othello.zobrist import Zobrist
import multiprocessing
from two_player_ai.utils import benchmark, cached_property
from functools import partial
from two_player_ai.alpha_beta import AlphaBeta
from two_player_ai.othello.heuristics import Weighted, PiecesCount, CornerPieces, EdgePieces, CornerEdgePieces, Mobility, Stability
from two_player_ai.othello.utils import cannonic

import numba as nb

@nb.jit(nopython=True, nogil=True, cache=True)
def foo(array):
    array.reshape((8,8))
def choose_next_action(node):
    available_actions = game.actions(node.state, node.player)
    action_probabilities = [
        Mcts.get_policy(Othello, r)[a] for a in available_actions
    ]
    action_index = np.random.choice(
        len(available_actions), p=action_probabilities
    )
    action = available_actions[action_index]


def heuristic(state, player):
    s, p = Othello.cannonical_state(state, player)
    h1 = Weighted.evaluate(s)
    h2 = PiecesCount.evaluate(s)
    h3 = CornerPieces.evaluate(s)
    h4 = EdgePieces.evaluate(s)
    h5 = CornerEdgePieces.evaluate(s)
    h6 = Mobility.evaluate(s)
    h7 = Stability.evaluate(s)
    return h1 + h2 + h3 + h4 + h5 + h6 + h7

    state, player = Othello.initial_state()
mcts = Mcts(Othello)
Mcts.uct(state, player)
ab = AlphaBeta(Othello, heuristic)
ab.run(state, player, False)

actions = Othello.actions(state, player)
state, player = Othello.result(state, player, actions[1])
game = Othello

root_node = MctsTreeNode(game, state, player)
selected_node = Mcts.tree_policy(root_node, 1)

@benchmark
def parallel(num):
    with multiprocessing.Pool(processes=num) as pool:
        simulation_results = pool.starmap(
            Mcts.simulate,
            ((game, selected_node) for _ in range(num))
            )
        print(simulation_results)

@benchmark
def p3(num):
    d = [(game, selected_node) for _ in range(num)]
    pool = multiprocessing.Pool(processes=4)
    result_list = pool.map(helper, d)
    print(result_list)

def helper(args):
    return Mcts.simulate(*args)


@benchmark
def paralle_2(num):
    jobs = []
    for i in range(num):
        process = multiprocessing.Process(
            target=Mcts.simulate,
            args=(game, selected_node)
        )
        jobs.append(process)

    for j in jobs:
        j.start()

    for j in jobs:
        j.join()


@benchmark
def sequential(num):
    simulation_results = []
    for _ in range(num):
        simulation_results.append(
            Mcts.simulate(game, selected_node)
        )

    print(simulation_results)

import json
from two_player_ai.alpha_zero.models.alpha_zero_model import AlphaZeroModel
from two_player_ai.alpha_zero.data_loaders.alpha_zero_data_loader import AlphaZeroDataLoader

from two_player_ai.othello.game import Othello


state, player = Othello.initial_state()
config_file = "./src/two_player_ai/alpha_zero/configs/alpha_zero_config.json"
with open(config_file, 'r') as config:
    config_dict = json.load(config)

model = AlphaZeroModel(Othello, config_dict).model

dl = AlphaZeroDataLoader(None, Othello, model)
dl.execute_episode()




r = Mcts.uct(Othello, state, player, model, c_puct=0.8, iterations=10)


from collections import defaultdict
h = defaultdict(int)

def traverse(node):
    h[node.state] += 1
    for n in node.child_nodes:
        traverse(n)
    if len(node.child_nodes) == 0:
        print(node.state.board)
    return h

def board():
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

def flatten(board):
    flat = np.zeros(
        (17, 17),
        dtype=np.int8
    )
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



@nb.jit(nopython=True, nogil=True, cache=True)
def nb_flatten(board, flat):
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

@benchmark
def foo():
    h = Heap(11000)

    for i in range(10000):
        h.push_item(10000 - i)

    for i in range(10000):
        h.pop_item()


0,1,2,3,  4,  5,6,7,8

0,1,2,3,4,5,6,7,8,   9,   10,11,12,13,14,15,16,17,18


import json
from two_player_ai.alpha_zero.models.alpha_zero_model import AlphaZeroModel
from two_player_ai.alpha_zero.data_loaders.alpha_zero_data_loader import AlphaZeroDataLoader
from two_player_ai.alpha_zero.trainers.alpha_zero_trainer import AlphaZeroModelTrainer

from two_player_ai.othello.game import Othello


state, player = Othello.initial_state()
config_file = "./src/two_player_ai/alpha_zero/configs/alpha_zero_config.json"
with open(config_file, 'r') as config:
    config_dict = json.load(config)

model = AlphaZeroModel(Othello, config_dict).model

dl = AlphaZeroDataLoader(Othello, model, 1)

t = AlphaZeroModelTrainer(model, dl.get_train_data(), dl.get_test_data(), 1)
