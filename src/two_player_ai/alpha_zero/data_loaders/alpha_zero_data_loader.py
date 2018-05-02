from two_player_ai.alpha_zero.base.base_data_loader import BaseDataLoader


class AlphaZeroDataLoader(BaseDataLoader):
    def __init__(self, config):
        super(AlphaZeroDataLoader, self).__init__(config)

    def get_train_data(self):
        return self.X_train, self.y_train

    def get_test_data(self):
        return self.X_test, self.y_test
