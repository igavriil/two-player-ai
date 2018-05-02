class BaseTrainer(object):
    def __init__(self, model, data, test_data, config):
        self.model = model
        self.data = data
        self.test_data = test_data
        self.config = config

    def train(self):
        raise NotImplementedError
