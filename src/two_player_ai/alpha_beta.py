import numpy as np
from two_player_ai.utils import benchmark


class AlphaBeta(object):
    def __init__(self, game=None, heuristic=None):
        self.game = game
        self.heuristic = heuristic

    @benchmark
    def run(self, state, player, maximize, alpha=-np.inf, beta=np.inf, depth=10):
        if depth == 0 or self.game.terminal_test(state, player):
            return state, self.heuristic(state, player)

        actions = self.game.actions(state, player)
        best_action = None
        if not actions:
            return best_action, 0

        if maximize:
            value = -np.inf
            for action in actions:
                next_state, next_player = self.game.result(state, player,
                                                           action)

                _, result = self.run(next_state, next_player, False, alpha,
                                     beta, depth - 1)
                if result > value:
                    value = result
                    best_action = action

                alpha = np.max([alpha, value])
                if beta <= alpha:
                    break
            return best_action, value
        else:
            value = +np.inf
            for action in actions:
                next_state, next_player = self.game.result(state, player,
                                                           action)
                _, result = self.run(next_state, next_player, True, alpha,
                                     beta, depth - 1)
                if result < value:
                    value = result
                    best_action = action
                beta = np.min([beta, value])
                if beta <= alpha:
                    break
            return best_action, value
