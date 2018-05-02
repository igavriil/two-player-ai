import json
import os
import time


def get_config(config_file):
    with open(config_file, 'r') as config:
        config_dict = json.load(config)

    return config_dict


def process_config(config_file):
    config = get_config(config_file)
    config["tensorboard_log_dir"] = os.path.join(
        "experiments",
        time.strftime("%Y-%m-%d/", time.localtime()),
        config["exp_name"],
        "logs"
    )
    config["checkpoint_dir"] = os.path.join(
        "experiments",
        time.strftime("%Y-%m-%d/", time.localtime()),
        config["exp_name"],
        "checkpoints"
    )
    return config
