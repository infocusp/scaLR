from _scalr.nn.trainer import build_trainer


class ModelTrainer:

    def __init__(self, model_config, train_config, data_config):
        """Wrapper class for training of a model"""
        self.trainer = build_trainer(model_config, train_config, data_config)

    def train(self):
        """Trains the model"""
        self.trainer.train()

