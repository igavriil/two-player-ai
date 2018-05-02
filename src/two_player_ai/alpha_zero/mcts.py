import math
import multiprocessing
import numpy as np
from random import sample
from two_player_ai.utils import benchmark, cached_property


class MctsTreeNode(object):
    def __init__(self, state, player, parent=None, action=None, previous_probabily=0):
        self.state = state
        self.player = player
        self.parent = parent
        self.action = action
        self.visit_count = 0
        self.total_reward = 0.0
        self.previous_probabily = previous_probabily
        self.child_nodes = []

    @cached_property
    def performer(self):
        return self.parent.player if self.parent else None

    @cached_property
    def is_root(self):
        return bool(self.parent)

    def add_child(self, child):
        self.child_nodes.append(child)
        return child

    def get_domain_theoretic_value(self):
        return self.total_reward / self.visit_count

    def update_domain_theoretic_value(self, reward):
        self.visit_count += 1
        self.total_reward += reward

    def __eq__(self, other):
        return self.state == other.state and self.action == other.action

    def __hash__(self):
        return self.state.__hash__() ^ self.action.__hash__()


class Mcts(object):
    @staticmethod
    @benchmark
    def uct(game, state, player, exploration=0.8, iterations=10):
        root_node = MctsTreeNode(state, player, root=True)

        return Mcts.uct_node(game, root_node, player, exploration, iterations)

    @staticmethod
    def uct_node(game, root_node, player, exploration, iterations):
        for i in range(iterations):
            node = Mcts.tree_policy(game, root_node, exploration)
            terminal_state, last_player = Mcts.simulate(game, node)
            Mcts.backpropagate(game, node, terminal_state)
        best_child = Mcts.uct_select_child(root_node, exploration)

        return best_child

    @staticmethod
    def tree_policy(game, node, exploration, dirilecht_alpha=0.8, espilon=0.2):
        while not game.terminal_test(node.state, node.player):
            available_actions = set(game.actions(node.state, node.player))
            if node.is_root:
                noise = np.random.dirichlet(
                    [dirilecht_alpha] * len(available_actions)
                )
            else:
                noise = [0] * len(available_actions)
                epsilon = 0

            if not available_actions:
                node = Mcts.expand_without_action(node)
                return node
            elif not len(node.child_nodes) == len(available_actions):
                node = Mcts.expand_with_action(game, node)
                return node
            else:
                node = Mcts.uct_select_child(node, noise, exploration, epsilon)
        return node

    @staticmethod
    def expand_with_action(game, node):
        available_actions = set(game.actions(node.state, node.player))
        explored_actions = set([child.action for child in node.child_nodes])

        action = sample(available_actions - explored_actions, 1)[0]
        state, player = game.result(node.state, node.player, action)
        child = MctsTreeNode(state, player, parent=node, action=action)
        return node.add_child(child)

    @staticmethod
    def expand_without_action(node):
        node.add_child(None)
        return node

    @staticmethod
    def uct_select_child(node, noise, exploration, epsilon):
        return Mcts.best_child(node.child_nodes, noise, exploration, epsilon)

    @staticmethod
    def best_child(chidlren, noise, exploration, epsilon):
        result = None
        max_uct = -np.inf
        for child, child_noise in zip(chidlren, noise):
            child_uct = Mcts.calculate_uct_value(
                child, child_noise, exploration, epsilon
            )
            if child_uct > max_uct:
                max_uct = child_uct
                result = child

        return result

    @staticmethod
    def simulate(game, node):
        state, player = node.state, node.player
        while not game.terminal_test(state, player):
            available_actions = game.actions(state, player)
            if available_actions:
                action = sample(available_actions, 1)[0]
            else:
                action = None
            state, player = game.result(state, player, action)
        return state, player

    @staticmethod
    def backpropagate(game, node, terminal_state):
        winner = game.winner(terminal_state)
        while node:
            if winner == node.performer:
                reward = 1
            else:
                reward = 0
            node.update_domain_theoretic_value(reward)
            node = node.parent

    @staticmethod
    def calculate_uct_value(node, noise, exploration, epsilon):
        Q = node.get_domain_theoretic_value()
        U = exploration * (
            (1 - epsilon) * node.previous_probabily +
            epsilon * noise
        ) * (
            np.sqrt(node.parent.visit_count) / (1 + node.visit_count)
        )
        return Q + U
