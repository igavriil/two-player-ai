from two_player_ai.alpha_zero.data_loaders.mnist_data_loader import (
    MnistDataLoader
)
from two_player_ai.alpha_zero.models.mnist_model import MnistModel
from two_player_ai.alpha_zero.trainers.mnist_trainer import MnistModelTrainer
from two_player_ai.alpha_zero.utils.config import process_config
from two_player_ai.alpha_zero.utils.dirs import create_dirs
from two_player_ai.alpha_zero.utils.args import get_args


def main():
    # capture the config path from the run arguments
    # then process the json configuration file
    try:
        args = get_args()
        config = process_config(args.config)
    except Exception as e:
        print("missing or invalid arguments")
        exit(0)

    # create the experiments dirs
    create_dirs([config["checkpoint_dir"], config["tensorboard_log_dir"]])

    print('Create the data generator.')
    data_loader = MnistDataLoader(config)

    print('Create the model.')
    model = MnistModel(config)

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
