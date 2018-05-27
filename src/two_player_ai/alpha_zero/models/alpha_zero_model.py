from two_player_ai.alpha_zero.base.base_model import BaseModel
from keras.models import Model
from keras.layers import (
    Activation, BatchNormalization, Conv2D, Dense, Flatten, Input, Dropout
)
from keras.optimizers import Adam
from keras.regularizers import l2
from keras.layers.merge import Add


class AlphaZeroModel(BaseModel):
    def __init__(self, game, config):
        super(AlphaZeroModel, self).__init__(config)
        self.game = game
        self.build_model()

    def build_residual_layer(self, x):
        in_x = x

        x = Conv2D(
            filters=self.config["cnn_filter_num"],
            kernel_size=self.config["cnn_filter_size"],
            padding="same",
            data_format="channels_first",
            kernel_regularizer=l2(self.config["l2_reg"])
        )(x)

        x = BatchNormalization(axis=1)(x)
        x = Activation("relu")(x)

        x = Conv2D(
            filters=self.config["cnn_filter_num"],
            kernel_size=self.config["cnn_filter_size"],
            padding="same",
            data_format="channels_first",
            kernel_regularizer=l2(self.config["l2_reg"])
        )(x)
        x = BatchNormalization(axis=1)(x)

        x = Add()([in_x, x])
        x = Activation("relu")(x)
        return x

    def build_model(self):
        in_x = x = Input((1, 8, 8))

        x = Conv2D(
            filters=self.config["cnn_filter_num"],
            kernel_size=self.config["cnn_filter_size"],
            data_format="channels_first",
            padding="same",
            kernel_regularizer=l2(self.config["l2_reg"]),
        )(x)

        x = BatchNormalization(axis=1)(x)
        x = Activation("relu")(x)

        for _ in range(self.config["res_layer_num"]):
            x = self.build_residual_layer(x)

        res_out = x

        x = Conv2D(
            filters=2,
            kernel_size=1,
            data_format="channels_first",
            padding="valid",
            kernel_regularizer=l2(self.config["l2_reg"]),
        )(res_out)

        x = BatchNormalization(axis=1)(x)
        x = Activation("relu")(x)
        x = Dropout(self.config["dropout"])(x)

        x = Flatten()(x)

        policy_network = Dense(
            8 * 8,
            kernel_regularizer=l2(self.config["l2_reg"]),
            activation="softmax",
            name="policy_out"
        )(x)

        x = Conv2D(
            filters=1,
            kernel_size=1,
            padding="valid",
            data_format="channels_first",
            kernel_regularizer=l2(self.config["l2_reg"]),
        )(res_out)

        x = BatchNormalization(axis=1)(x)
        x = Activation("relu")(x)
        x = Dropout(self.config["dropout"])(x)

        x = Flatten()(x)
        x = Dense(
            self.config["value_fc_size"],
            kernel_regularizer=l2(self.config["l2_reg"]),
            activation="relu"
        )(x)

        value_network = Dense(
            1,
            kernel_regularizer=l2(self.config["l2_reg"]),
            activation="tanh",
            name="value_out"
        )(x)

        self.model = Model(
            inputs=in_x,
            outputs=[policy_network, value_network],
            name="reversi_model"
        )

        self.model.compile(
            loss=['categorical_crossentropy', 'mean_squared_error'],
            optimizer=Adam(
                lr=self.config['adam_learning_rate'],
                beta_1=self.config['adam_beta_1'],
                beta_2=self.config['adam_beta_2'],
            ),
        )
