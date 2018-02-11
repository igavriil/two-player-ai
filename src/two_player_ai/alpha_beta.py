import numpy as np
from two_player_ai.utils import benchmark


class AlphaBeta(object):
    @staticmethod
    @benchmark
    def run(game, state, player, maximize, alpha=-np.inf, beta=np.inf,
            depth=10):
        if depth == 0 or game.terminal_test(state, player):
            return game.evaluate(state, player)

        actions = game.actions(state, player)
        best_action = None
        if not actions:
            return None, 0

        if maximize:
            value = -np.inf
            for action in actions:
                next_state, next_player = game.result(state, player, action)
                (_, result) = AlphaBeta.run(game, next_state, next_player,
                                            False, alpha, beta, depth - 1)
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
                next_state, next_player = game.result(state, player, action)
                _, result = AlphaBeta.run(game, next_state, next_player,
                                          True, alpha, beta, depth - 1)
                if result < value:
                    value = result
                    best_action = action
                beta = np.min([beta, value])
                if beta <= alpha:
                    break
            return best_action, value
