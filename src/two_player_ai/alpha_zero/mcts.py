import numpy as np
from random import sample
from two_player_ai.utils import cached_property
from two_player_ai.utils import normalize, reshape


class MctsTreeNode(object):
    def __init__(self, state, player, parent=None, action=None, prob=0.0):
        self.state = state
        self.player = player
        self.parent = parent
        self.action = action
        self.visit_count = 0
        self.total_reward = 0.0
        self.prob = prob
        self.child_probs = {}
        self.child_nodes = {}
        self.value = None

    @cached_property
    def performer(self):
        return self.parent.player if self.parent else None

    @cached_property
    def is_root(self):
        return not bool(self.parent)

    def add_child(self, child):
        if child:
            self.child_nodes[child.action] = child
        else:
            child = MctsTreeNode(
                self.state, -self.player, parent=self, prob=1
            )
            self.child_nodes[None] = child
        return child

    def get_domain_theoretic_value(self):
        return self.total_reward

    def update_domain_theoretic_value(self, reward):
        self.total_reward = (
            (self.visit_count * self.total_reward + reward) /
            (self.visit_count + 1)
        )
        self.visit_count += 1

    def __eq__(self, other):
        return self.state == other.state and self.action == other.action

    def __hash__(self):
        return self.state.__hash__() ^ self.action.__hash__()


class Mcts(object):
    @staticmethod
    def uct_node(game, root_node, model, c_puct=None, iterations=None):
        for i in range(iterations):
            node = Mcts.tree_policy(game, root_node, model, c_puct)
            value = Mcts.simulate(game, node, model)
            Mcts.backpropagate(game, node, value)

        c_puct = 0
        noise = [0] * len(root_node.child_nodes)
        epsilon = 0

        best_child = Mcts.uct_select_child(
            root_node, c_puct, noise, epsilon
        )

        return root_node, best_child

    @staticmethod
    def tree_policy(game, node, model, c_puct, dirichlet_alpha=0.03, epsilon=0.25):
        while not game.terminal_test(node.state, node.player):
            available_actions = set(game.actions(node.state, node.player))

            if not available_actions:
                node = Mcts.expand_without_action(node)
                return node
            elif not len(node.child_nodes) == len(available_actions):
                node.child_probs = node.child_probs or Mcts.children_probs(
                    game, node, model
                )

                node = Mcts.expand_with_action(game, node)
                return node
            else:
                if node.is_root:
                    noise = np.random.dirichlet(
                        [dirichlet_alpha] * len(node.child_nodes)
                    )
                else:
                    noise = [0] * len(node.child_nodes)
                    epsilon = 0

                node = Mcts.uct_select_child(
                    node, c_puct, noise, epsilon
                )
        return node

    @staticmethod
    def children_probs(game, node, model):
        available_actions = game.actions(node.state, node.player)
        policy, value = model.predict(node.state.binary_form())

        policy = reshape(policy[0], game.board_size())
        value = value[0]

        action_probabilities = np.zeros(game.board_size())
        for row, col in available_actions:
            action_probabilities[row, col] = policy[row, col]

        action_probabilities = normalize(action_probabilities)

        child_probs = {}
        for action in available_actions:
            row, col = action
            child_probs[action] = action_probabilities[row, col]

        return child_probs

    @staticmethod
    def expand_with_action(game, node):
        available_actions = set(game.actions(node.state, node.player))
        explored_actions = set(node.child_nodes.keys())

        action = sample(available_actions - explored_actions, 1)[0]
        state, player = game.result(node.state, node.player, action)
        child = MctsTreeNode(
            state, player, parent=node, action=action,
            prob=node.child_probs[action]
        )
        return node.add_child(child)

    @staticmethod
    def expand_without_action(node):
        node.add_child(None)
        return node

    @staticmethod
    def uct_select_child(node, c_puct, noise, epsilon):
        result = None
        max_uct = -np.inf
        for child, child_noise in zip(node.child_nodes.values(), noise):
            child_uct = Mcts.calculate_uct_value(
                child, c_puct, child_noise, epsilon
            )
            if child_uct > max_uct:
                max_uct = child_uct
                result = child

        return result

    @staticmethod
    def simulate(game, node, model):
        state, player = node.state, node.player

        if node.value is not None:
            value = node.value
        elif game.terminal_test(state, player):
            value = game.winner(state) * player
        else:
            _, value = model.predict(state.binary_form())
            value = value[0][0]

        node.value = value
        return value

    @staticmethod
    def backpropagate(game, node, value):
        while node:
            reward = value * node.player
            node.update_domain_theoretic_value(reward)
            node = node.parent

    @staticmethod
    def get_policy(game, node):
        available_actions = game.actions(node.state, node.player)

        if available_actions:
            policy = np.zeros(game.board_size())
            for row, col in available_actions:
                child_node = node.child_nodes.get((row, col))
                if child_node:
                    policy[row, col] = child_node.visit_count

            return normalize(policy)
        else:
            return np.zeros(game.board_size())

    @staticmethod
    def calculate_uct_value(node, c_puct, noise, epsilon=0.25):
        Q = node.get_domain_theoretic_value()
        U = c_puct * (
            (1 - epsilon) * node.prob +
            epsilon * noise
        ) * (
            np.sqrt(node.parent.visit_count) / (1 + node.visit_count)
        )
        return Q + U
