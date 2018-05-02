from two_player_ai.alpha_zero.base.base_model import BaseModel
from keras.models import Sequential
from keras.layers import Dense


class MnistModel(BaseModel):
    def __init__(self, config):
        super(MnistModel, self).__init__(config)
        self.build_model()

    def build_model(self):
        model = Sequential()
        model.add(Dense(32, activation='relu', input_shape=(28 * 28,)))
        model.add(Dense(16, activation='relu'))
        model.add(Dense(10, activation='softmax'))

        self.model = model
        self.model.compile(
            loss='sparse_categorical_crossentropy',
            optimizer=self.config["optimizer"],
            metrics=['acc'],
        )
