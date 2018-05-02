from two_player_ai.alpha_zero.base.base_data_loader import BaseDataLoader
from keras.datasets import mnist


class MnistDataLoader(BaseDataLoader):
    def __init__(self, config):
        super(MnistDataLoader, self).__init__(config)
        (X_train, self.y_train), (X_test, self.y_test) = mnist.load_data()
        self.X_train = X_train.reshape((-1, 28 * 28))
        self.X_test = X_test.reshape((-1, 28 * 28))

    def get_train_data(self):
        return self.X_train, self.y_train

    def get_test_data(self):
        return self.X_test, self.y_test
