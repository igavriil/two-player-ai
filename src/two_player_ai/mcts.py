import numpy as np
from random import sample
from two_player_ai.utils import benchmark, cached_property


class MctsTreeNode(object):
    def __init__(self, game, state, player, parent=None, action=None):
        self.game = game
        self.state = state
        self.player = player
        self.parent = parent
        self.action = action
        self.visit_count = 0
        self.total_reward = 0.0
        self.child_nodes = []

    @cached_property
    def performer(self):
        return self.parent.player if self.parent else None

    @cached_property
    def terminal_state(self):
        return self.game.terminal_test(self.state, self.player)

    def has_child_nodes(self):
        return len(self.child_nodes) > 0

    def is_fully_expanded(self):
        return len(self.get_all_actions) == len(self.child_nodes)

    @cached_property
    def has_available_actions(self):
        return len(self.get_all_actions) > 0

    def has_unvisited_child(self):
        return any([child.visit_count == 0 for child in self.child_nodes])

    @cached_property
    def get_all_actions(self):
        return set(self.game.actions(self.state, self.player))

    def get_tried_actions(self):
        return set([child_node.action for child_node in self.child_nodes])

    def get_untried_actions(self):
        return self.get_all_actions - self.get_tried_actions()

    def add_child(self, action=None):
        child_state, player = self.game.result(self.state, self.player, action)
        child_node = MctsTreeNode(self.game, child_state, player, parent=self,
                                  action=action)
        self.child_nodes.append(child_node)
        return child_node

    def get_domain_theoretic_value(self):
        return self.total_reward / self.visit_count

    def update_domain_theoretic_value(self, reward):
        self.visit_count += 1
        self.total_reward += reward


class Mcts(object):
    @staticmethod
    @benchmark
    def uct(game, state, player, exploration=0.8, iterations=10):
        root_node = MctsTreeNode(game, state, player)

        for i in range(iterations):
            selected_node = Mcts.tree_policy(root_node, exploration)
            terminal_state, last_player = Mcts.simulate(game, selected_node)
            Mcts.backpropagate(selected_node, game, terminal_state)

        best_child = Mcts.uct_select_child(root_node, 0.0)
        return best_child

    @staticmethod
    def tree_policy(node, exploration):
        while not node.terminal_state:
            if not node.has_available_actions:
                node = Mcts.expand_without_action(node)
                return node
            elif not node.is_fully_expanded():
                node = Mcts.expand_with_action(node)
                return node
            else:
                node = Mcts.uct_select_child(node, exploration)
        return node

    @staticmethod
    def expand_with_action(node):
        action = sample(node.get_untried_actions(), 1)[0]
        return node.add_child(action=action)

    @staticmethod
    def expand_without_action(node):
        return node.add_child(action=None)

    @staticmethod
    def uct_select_child(node, exploration):
        result = None
        max_uct = -1
        for child in node.child_nodes:
            child_uct = Mcts.calculate_uct_value(child, exploration)
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
    def backpropagate(node, game, terminal_state):
        winner = game.winner(terminal_state)
        while node:
            # node.state player is the player to play next
            # not the player that played the move
            if winner == node.performer:
                reward = 1
            elif winner == node.player:
                reward = -1
            else:
                reward = 0
            node.update_domain_theoretic_value(reward)
            node = node.parent

    @staticmethod
    def calculate_uct_value(node, exploration):
        return node.get_domain_theoretic_value() + exploration * np.sqrt(np.log(node.parent.visit_count) / node.visit_count)
