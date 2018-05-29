import numpy as np

from tqdm import tqdm
from two_player_ai.alpha_zero.base.base_data_loader import BaseDataLoader
from two_player_ai.alpha_zero.mcts import Mcts, MctsTreeNode
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
        pbar = tqdm(total=5)
        train_examples = []

        for i in range(6):
            state, player = game.initial_state()
            root_node = MctsTreeNode(state, player)
            episode_results = []
            while not game.terminal_test(state, player):
                root_node, child_node = Mcts.uct_node(
                    game,
                    root_node,
                    model,
                    c_puct=0.8,
                    iterations=20
                )

                policy = Mcts.get_policy(game, root_node)
                if policy is not None:
                    cannonical_state, _ = game.cannonical_state(
                        root_node.state, root_node.player
                    )
                    symmetries = game.symmetries(cannonical_state, policy)

                    for symmetry_state, symmetry_policy in symmetries:
                        episode_results.append(
                            (symmetry_state, player, symmetry_policy.flatten())
                        )
                else:
                    print("None policy. Actions {}".format(
                        game.actions(
                            root_node.state, root_node.player
                        )
                    ))

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
                root_node = root_node.child_nodes[action]
                root_node.parent = None

            pbar.update(1)
            winning_player = game.winner(state)

            episode_results = [
                (state, policy, player * winning_player) for
                state, player, policy in episode_results
            ]
            train_examples.extend(episode_results)
        pbar.close()

        #import ipdb; ipdb.set_trace()
        shuffle(train_examples)

        self.X_train, self.y_train = [], []

        examples = []

        for state, policy, value in train_examples:
            examples.extend(
                [(
                    state.binary_form(wrap=False),
                    policy,
                    value
                )]
            )

        input_boards, target_pis, target_vs = list(zip(*examples))

        self.X_train = np.asarray(input_boards)
        self.y_train = [
            np.asarray(np.asarray(target_pis)), np.asarray(target_vs)
        ]
