import numpy as np
from random import sample
from two_player_ai.utils import benchmark, cached_property
from two_player_ai.alpha_zero.utils.utils import normalize, reshape


class MctsAggregation(object):
    def __init__(self):
        self.Ns = {}
        self.Ps = {}


class MctsTreeNode(object):
    def __init__(self, state, player, parent=None, action=None, prob=None):
        self.state = state
        self.player = player
        self.parent = parent
        self.action = action
        self.visit_count = 0  # N(s, a)
        self.total_reward = 0.0  # W(s, a) total_action_value
        self.prob = prob  # P(s, a) prior probability
        self.child_probs = {}
        self.child_nodes = []

    @cached_property
    def performer(self):
        return self.parent.player if self.parent else None

    @cached_property
    def is_root(self):
        return not bool(self.parent)

    def add_child(self, child):
        self.child_nodes.append(child)
        return child

    def get_domain_theoretic_value(self):  # Q(s,a)
        return self.total_reward

    def update_domain_theoretic_value(self, reward):
        self.total_reward = (
            self.visit_count * self.total_reward /
            self.visit_count + 1
        )
        self.visit_count += 1

    def __eq__(self, other):
        return self.state == other.state and self.action == other.action

    def __hash__(self):
        return self.state.__hash__() ^ self.action.__hash__()

class Mcts(object):
    @staticmethod
    def search(game, state, player, model):
        cannonical_state = game.cannonical_state(state, player)
        player *= player

        root_node = MctsTreeNode(cannonical_state, player)

        if game.terminal_test(cannonical_state, player):
            return -game.winner(cannonical_state)


    @staticmethod
    @benchmark
    def uct(memory, game, state, player, model, c_puct=0.8, iterations=10):
        root_node = MctsTreeNode(state, player)

        return Mcts.uct_node(
            game, root_node, player, model, c_puct, iterations
        )

    @staticmethod
    def uct_node(game, root_node, model, c_puct, iterations):
        for i in range(iterations):
            node = Mcts.tree_policy(game, root_node, c_puct)
            value = Mcts.simulate(game, node, model)
            Mcts.backpropagate(game, node, value)
        best_child = Mcts.uct_select_child(root_node, None, c_puct, 0)

        return best_child

    @staticmethod
    def tree_policy(game, node, model, c_puct, dirichlet_alpha=0.03, epsilon=0.25):
        while not game.terminal_test(node.state, node.player):
            available_actions = set(game.actions(node.state, node.player))
            # if node.is_root:
            #     noise = np.random.dirichlet(
            #         [dirichlet_alpha] * len(available_actions)
            #     )
            # else:
            #     noise = [0] * len(available_actions)
            #     epsilon = 0

            if not available_actions:
                node = Mcts.expand_without_action(node)
                return node
            elif not len(node.child_nodes) == len(available_actions):
                node.children_probabilities = (
                    node.children_probabilities or
                    Mcts.children_probabilities(game, node, model)
                )
                node = Mcts.expand_with_action(game, node)
                return node
            else:
                node = Mcts.uct_select_child(
                    node, c_puct, dirichlet_alpha, epsilon
                )
        return node

    @staticmethod
    def children_probabilities(game, node, model):
        available_actions = game.actions(node.state, node.player)
        policy, value = model.predict(node.state.binary_form)

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
    def expand_with_action(game, node, model):
        available_actions = set(game.actions(node.state, node.player))
        explored_actions = set([child.action for child in node.child_nodes])

        if not node.policy:
            policy, value = model.predict(node.state.binary_form)
            action_probs = [
                ((x, y), policy[0][x * 8 + y])
                for (x, y) in game.actions(node.state, node.player)
            ]
            node.action_probs = action_probs

        action = sample(available_actions - explored_actions, 1)[0]
        state, player = game.result(node.state, node.player, action)
        child = MctsTreeNode(
            state, player, parent=node, action=action, prob=node.action_probs[action]
        )
        return node.add_child(child)

    @staticmethod
    def expand_without_action(node):
        node.add_child(None)
        return node

    @staticmethod
    def uct_select_child(node, c_puct, dirichlet_alpha, epsilon):
        result = None
        max_uct = -np.inf
        for child, child_noise in zip(node.child_nodes, noise):
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

        if game.terminal_test(state, player):
            return game.winner(state) * player
        else:
            _, value = model.predict(state.binary_form)
            return value

    @staticmethod
    def backpropagate(game, node, value):
        while node:
            reward = value * node.player
            node.update_domain_theoretic_value(reward)
            node = node.parent

    @staticmethod
    def calculate_uct_value(node, c_puct, noise, epsilon=0.25):
        Q = node.get_domain_theoretic_value()
        U = c_puct * (
            (1 - epsilon) * node.prior_prob +
            epsilon * noise
        ) * (
            np.sqrt(node.parent.visit_count) / (1 + node.visit_count)
        )
        return Q + U
