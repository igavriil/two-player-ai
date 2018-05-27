import time
import os
import json

from two_player_ai.othello.game import Othello
from two_player_ai.alpha_zero.data_loaders.alpha_zero_data_loader import (
    AlphaZeroDataLoader
)
from two_player_ai.alpha_zero.models.alpha_zero_model import AlphaZeroModel
from two_player_ai.alpha_zero.trainers.mnist_trainer import MnistModelTrainer
from two_player_ai.alpha_zero.utils.config import process_config
from two_player_ai.alpha_zero.utils.dirs import create_dirs
from two_player_ai.alpha_zero.utils.args import get_args


def main():
    config_file = "./configs/alpha_zero_config.json"
    with open(config_file, 'r') as config:
        config_dict = json.load(config)

    checkpoint_dir = os.path.join(
        "experiments",
        time.strftime("%Y-%m-%d/", time.localtime()),
        "alpha_zero",
        "checkpoints/"
    )

    tensorboard_log_dir = os.path.join(
        "experiments",
        time.strftime("%Y-%m-%d/", time.localtime()),
        "alpha_zero",
        "logs/"
    )
    # create the experiments dirs
    create_dirs([checkpoint_dir, tensorboard_log_dir])

    print('Create the model.')
    model = AlphaZeroModel(Othello, config_dict).model

    print('Create the data generator.')
    data_loader = AlphaZeroDataLoader(config)

    print('Create the trainer')
    trainer = MnistModelTrainer(
        model.model,
        data_loader.get_train_data(),
        data_loader.get_test_data(),
        config
    )

    print('Start training the model.')
    trainer.train()


if __name__ == '__main__':
    main()
