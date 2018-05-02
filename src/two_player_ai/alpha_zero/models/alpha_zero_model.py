import two_player_ai.othello.constants as othello_constants

from two_player_ai.alpha_zero.base.base_model import BaseModel
from keras.models import Model
from keras.layers import (
    Activation, BatchNormalization, Conv2D, Dense, Flatten, Input,
)
from keras.optimizers import Adam
from keras.regularizers import l2
from keras.layers.merge import Add


class AlphaZeroModel(BaseModel):
    def __init__(self, game, config):
        super(AlphaZeroModel, self).__init__(config)
        self.game = game
        self.build_model()

    def build_residual_layer(self, x, index):
        in_x = x

        x = Conv2D(
            filters=self.config["cnn_filter_num"],
            kernel_size=self.config["cnn_filter_size"],
            padding="same",
            data_format="channels_first",
            use_bias=False,
            kernel_regularizer=l2(self.config["l2_reg"]),
            name="residual_conv_1-{index}-{filter_size}-{kernel_size}".format(
                index=index,
                filter_size=self.config["cnn_first_filter_size"],
                kernel_size=self.config["cnn_filter_num"]
            )
        )(x)

        x = BatchNormalization(
            axis=1,
            name="residual_batch_normalization_1-{index}".format(
                index=index
            )
        )(x)

        x = Activation(
            "relu",
            name="residual_activation_1-{index}".format(
                index=index
            )
        )(x)

        x = Conv2D(
            filters=self.config["cnn_filter_num"],
            kernel_size=self.config["cnn_filter_size"],
            padding="same",
            data_format="channels_first",
            use_bias=False,
            kernel_regularizer=l2(self.config["l2_reg"]),
            name="residual_conv_2-{index}-{filter_size}-{kernel_size}".format(
                index=index,
                filter_size=self.config["cnn_first_filter_size"],
                kernel_size=self.config["cnn_filter_num"]
            )
        )(x)

        x = BatchNormalization(
            axis=1,
            name="residual_batch_normalization_2-{index}".format(
                index=index
            )
        )(x)

        x = Add(
            name="residual_add-{index}".format(
                index=index
            )
        )([in_x, x])

        x = Activation(
            "relu",
            name="residual_activation_2-{index}".format(
                index=index
            )
        )(x)
        return x

    def build_model(self):
        input_layer = x = Input(
            shape=othello_constants.BINARY_SHAPE,
            name='input_layer'
        )

        x = Conv2D(
            filters=self.config["cnn_filter_num"],
            kernel_size=self.config["cnn_first_filter_size"],
            data_format="channels_first",
            use_bias=False,
            kernel_regularizer=l2(self.config["l2_reg"]),
            name="input_conv-{filter_size}-{kernel_size}".format(
                filter_size=self.config["cnn_first_filter_size"],
                kernel_size=self.config["cnn_filter_num"]
            )
        )(x)

        x = BatchNormalization(axis=1, name="input_batch_normalization")(x)
        x = Activation("relu", name="input_relu")(x)

        for index in range(self.config["residual_layer_num"]):
            x = self.build_residual_layer(x, index)

        out_layer = x

        p = Conv2D(
            filters=2,
            kernel_size=1,
            padding="same",
            data_format="channels_first",
            use_bias=False,
            kernel_regularizer=l2(self.config["l2_reg"]),
            name="policy_network-1-2"
        )(out_layer)

        p = BatchNormalization(axis=1, name="policy_batch_normalization")(p)
        p = Activation("relu", name="policy_relu")(p)
        p = Flatten(name="policy_flatten")(p)

        policy_network = Dense(
            self.game.action_size(),
            kernel_regularizer=l2(self.config["l2_reg"]),
            activation="softmax",
            name="policy_out"
        )(p)

        v = Conv2D(
            filters=4,
            kernel_size=1,
            padding="same",
            data_format="channels_first",
            use_bias=False,
            kernel_regularizer=l2(self.config["l2_reg"]),
            name="value_network-1-4"
        )(out_layer)

        v = BatchNormalization(axis=1, name="value_batch_normalization")(v)
        v = Activation("relu", name="value_relu")(v)
        v = Flatten(name="value_flatten")(v)
        v = Dense(
            self.game.action_size(),
            kernel_regularizer=l2(self.config["l2_reg"]),
            activation="relu",
            name="value_dense"
        )(v)

        value_network = Dense(
            1,
            kernel_regularizer=l2(self.config["l2_reg"]),
            activation="tanh",
            name="value_out"
        )(v)

        self.model = Model(
            inputs=input_layer,
            outputs=[policy_network, value_network]
        )

        self.model.compile(
            loss=['categorical_crossentropy', 'mean_squared_error'],
            optimizer=Adam(
                lr=self.config['adam_learning_rate'],
                beta_1=self.config['adam_beta_1'],
                beta_2=self.config['adam_beta_2'],
            ),
        )
