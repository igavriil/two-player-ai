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
from two_player_ai.othello.utils import cannonical_board


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
from two_player_ai.alpha_zero.mcts import Mcts
from two_player_ai.othello.game import Othello


state, player = Othello.initial_state()
config_file = "./src/two_player_ai/alpha_zero/configs/alpha_zero_config.json"
with open(config_file, 'r') as config:
    config_dict = json.load(config)

model = AlphaZeroModel(Othello, config_dict).model
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
