import os
import time

from two_player_ai.alpha_zero.base.base_trainer import BaseTrainer
from keras.callbacks import ModelCheckpoint, TensorBoard


class AlphaZeroModelTrainer(BaseTrainer):
    def __init__(self, model, data, test_data, config):
        super(AlphaZeroModelTrainer, self).__init__(model, data, test_data, config)
        self.callbacks = []
        self.loss = []
        self.acc = []
        self.val_loss = []
        self.val_acc = []
        self.init_callbacks()

    def init_callbacks(self):

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

        self.callbacks.append(
            ModelCheckpoint(
                filepath=os.path.join(
                    checkpoint_dir,
                    '%s-{epoch:02d}-{val_loss:.2f}.hdf5' %
                    "alpha_zero"
                ),
                monitor="val_loss",
                mode="min",
                save_best_only=True,
                save_weights_only=True,
                verbose=True,
            )
        )

        self.callbacks.append(
                TensorBoard(
                    log_dir=tensorboard_log_dir,
                    write_graph=True,
                )
            )

    def train(self):
        history = self.model.fit(
            x=self.data[0],
            y=self.data[1],
            epochs=25,
            verbose=True,
            batch_size=256,
            validation_split=0.15,
            callbacks=self.callbacks,
        )
        self.loss.extend(history.history['loss'])
        self.val_loss.extend(history.history['val_loss'])
