import os
import numpy as np

from tqdm import tqdm
from two_player_ai.alpha_zero.base.base_data_loader import BaseDataLoader
from two_player_ai.alpha_zero.mcts import Mcts, MctsTreeNode
from two_player_ai.utils import flatten
from random import shuffle


class AlphaZeroDataLoader(BaseDataLoader):
    def __init__(self, game, model, config):
        self.game = game
        self.model = model
        self.execute_episode(game, model, config)
        super(AlphaZeroDataLoader, self).__init__(config)

    def get_train_data(self):
        return self.X_train, self.y_train

    def get_test_data(self):
        return [], []

    def execute_episode(self, game, model, config):
        non_symmetrical_pairs = []

        state, player = game.initial_state()
        root_node = MctsTreeNode(state, player)
        while not game.terminal_test(state, player):
            available_actions = game.actions(
                root_node.state, root_node.player
            )
            if available_actions:
                root_node, child_node = Mcts.uct_node(
                    game, root_node, model,
                    c_puct=0.8,
                    iterations=20
                )
                policy = Mcts.get_policy(game, root_node)
                cannonical_state = game.cannonical_state(
                    root_node.state, root_node.player
                )
                non_symmetrical_pairs.append(
                    (cannonical_state.board, player, policy)
                )
                action_probabilities = [
                    policy[action] for action in available_actions
                ]
                action_index = np.random.choice(
                    len(available_actions), p=action_probabilities
                )
                action = available_actions[action_index]
            else:
                action = None

            state, player = game.result(state, player, action)
            root_node = root_node.child_nodes[action]
            root_node.parent = None

        winning_player = game.winner(state)

        episode_results = []

        for state, player, policy in non_symmetrical_pairs:
            for sym_state, sym_policy in game.symmetries(state, policy):
                episode_results.append(
                    (
                        np.array([sym_state]),
                        flatten(sym_policy),
                        player * winning_player
                        if winning_player != 0.5
                        else 0.5
                    )
                )
        return episode_results

    def execute_episodes(self):
        pbar = tqdm(total=config[])
        results = []

        for i in range(1):
            episode_results.extend(
                self.execute_episode(self.game, self.model, self.config)
            )
            pbar.update(1)

        pbar.close()
        shuffle(episode_results)
        input_boards, target_pis, target_vs = list(zip(*episode_results))

        self.X_train = np.asarray(input_boards)
        self.y_train = [np.asarray(target_pis), np.asarray(target_vs)]

    def save(self, id, folder='checkpoint', filename='{}.pth.tar.examples'):
        filepath = os.path.join(folder, filename)
        if not os.path.exists(folder):
            os.mkdir(folder)
        self.model.save_weights(filepath)

    def load(self, folder='checkpoint', filename='weights.pth.tar'):
        filepath = os.path.join(folder, filename)
        if not os.path.exists(folder):
            os.mkdir(folder)
        self.model.load_weights(filepath)
