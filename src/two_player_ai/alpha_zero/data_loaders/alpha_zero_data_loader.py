import numpy as np

from tqdm import tqdm
from two_player_ai.alpha_zero.base.base_data_loader import BaseDataLoader
from two_player_ai.alpha_zero.mcts import Mcts
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
        pbar = tqdm(total=60)
        train_examples = []

        state, player = game.initial_state()
        while not game.terminal_test(state, player):
            root_node, child_node = Mcts.uct(
                game,
                state,
                player,
                model,
                c_puct=0.8,
                iterations=600
            )

            policy = Mcts.get_policy(game, root_node)
            cannonical_state, _ = game.cannonical_state(
                root_node.state, root_node.player
            )
            symmetries = game.symmetries(cannonical_state, policy)

            for symmetry_state, symmetry_policy in symmetries:
                train_examples.append(
                    (symmetry_state, player, symmetry_policy.flatten())
                )

            available_actions = game.actions(
                root_node.state, root_node.player
            )
            if available_actions:
                action_probabilities = [policy[a] for a in available_actions]
                action_index = np.random.choice(
                    len(available_actions), p=action_probabilities
                )
                action = available_actions[action_index]
            else:
                action = None

            state, player = game.result(state, player, action)
            pbar.update(1)

        pbar.close()
        winning_player = game.winner(state)

        shuffle(train_examples)

        self.X_train, self.y_train = [], []

        examples = []

        for state, player, policy in train_examples:
            examples.extend(
                [(
                    state.binary_form(wrap=False),
                    policy,
                    winning_player * player
                )]
            )

        input_boards, target_pis, target_vs = list(zip(*examples))

        self.X_train = np.asarray(input_boards)
        self.y_train = [
            np.asarray(np.asarray(target_pis)), np.asarray(target_vs)
        ]
