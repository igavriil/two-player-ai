import math
import multiprocessing
import numpy as np
from random import sample
from two_player_ai.utils import benchmark, cached_property


class MctsTreeNode(object):
    def __init__(self, state, player, parent=None, action=None, prob=0.0):
        self.state = state
        self.player = player
        self.parent = parent
        self.action = action
        self.visit_count = 0
        self.total_reward = 0.0
        self.child_nodes = []
        self.prob = prob

    @cached_property
    def performer(self):
        return self.parent.player if self.parent else None

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
    def uct(game, state, player, c_puct=0.8, iterations=10):
        root_node = MctsTreeNode(state, player)

        return Mcts.uct_node(game, root_node, player, c_puct, iterations)

    @staticmethod
    @benchmark
    def uct_root(game, state, player, c_puct=0.8, iterations=10):
        processes = multiprocessing.cpu_count()
        root_node = MctsTreeNode(state, player)
        iterations = math.ceil(iterations / processes)
        with multiprocessing.Pool(processes) as pool:
            results = pool.starmap(
                Mcts.uct_node,
                [
                    (game, root_node, player, c_puct, iterations)
                    for _ in range(processes)
                ]
            )

        return Mcts.best_child(results, 0)

    @staticmethod
    @benchmark
    def uct_leaf(game, state, player, c_puct=0.8, iterations=10):
        processes = multiprocessing.cpu_count()
        root_node = MctsTreeNode(state, player)
        iterations = math.ceil(iterations / processes)
        with multiprocessing.Pool(processes) as pool:
            for i in range(processes):
                node = Mcts.tree_policy(game, root_node, c_puct)
                results = pool.starmap(
                    Mcts.simulate,
                    [
                        (game, node)
                        for _ in range(iterations)
                    ]
                )
                for reward in results:
                    Mcts.backpropagate(game, node, reward)

        best_child = Mcts.uct_select_child(root_node, 0)

        return best_child

    @staticmethod
    def uct_node(game, root_node, player, c_puct, iterations):
        for i in range(iterations):
            node = Mcts.tree_pohttps://github.com/explorelicy(game, root_node, c_puct)
            reward = Mcts.simulate(game, node)
            Mcts.backpropagate(game, node, reward)
        best_child = Mcts.uct_select_child(root_node, 0)

        return best_child

    @staticmethod
    def tree_policy(game, node, c_puct):
        while not game.terminal_test(node.state, node.player):
            available_actions = set(game.actions(node.state, node.player))
            if not available_actions:
                node = Mcts.expand_without_action(node)
                return node
            elif not len(node.child_nodes) == len(available_actions):
                node = Mcts.expand_with_action(game, node)
                return node
            else:
                node = Mcts.uct_select_child(node, c_puct)
        return node

    @staticmethod
    def expand_with_action(game, node):
        available_actions = set(game.actions(node.state, node.player))
        explored_actions = set([child.action for child in node.child_nodes])

        action = sample(available_actions - explored_actions, 1)[0]
        state, player = game.result(node.state, node.player, action)
        child = MctsTreeNode(
            state, player, parent=node, action=action,
            prob=(1/len(available_actions))
        )
        return node.add_child(child)

    @staticmethod
    def expand_without_action(node):
        node.add_child(None)
        return node

    @staticmethod
    def uct_select_child(node, c_puct):
        return Mcts.best_child(node.child_nodes, c_puct)

    @staticmethod
    def best_child(chidlren, c_puct):
        result = None
        max_uct = -np.inf
        for child in chidlren:
            child_uct = Mcts.calculate_uct_value(child, c_puct)
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
                action_probs = np.random.rand(len(available_actions))
                rollout_policy = zip(available_actions, action_probs)
                action = max(rollout_policy, key=lambda p: p[1])[0]
            else:
                action = None
            state, player = game.result(state, player, action)

        winning_player = game.winner(state)

        return Mcts.reward(winning_player)

    @staticmethod
    def backpropagate(game, node, reward):
        while node:
            node.update_domain_theoretic_value(reward)
            node = node.parent
            reward = -reward

    @staticmethod
    def reward(node, winning_player):
        if winning_player == node.performer:
            reward = 1
        elif winning_player == node.player:
            reward = -1
        else:
            reward = 0

        return reward

    @staticmethod
    def calculate_uct_value(node, c_puct):
        return node.get_domain_theoretic_value() + \
               c_puct * np.sqrt(
               np.log(node.parent.visit_count) / node.visit_count
            )
