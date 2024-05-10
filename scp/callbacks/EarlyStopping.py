class EarlyStopping:
    """
    Implementing Early stopping for model in case of learning stalled based upon validation loss
    """
    def __init__(self, stop_patience=3, stop_min_delta=1e-4):
        """
        Args:
            stop_patience: number of epochs with no improvement after which training will be stopped
            stop_min_delta: Minimum change in the monitored quantity to qualify as an improvement,
                            i.e. an absolute change of less than min_delta, will count as no improvement.
        """
        self.patience = int(stop_patience)
        self.min_delta = float(stop_min_delta)
        self.epoch = 0
        self.min_validation_loss = float('inf')

    def __call__(self, validation_loss):
        if validation_loss < self.min_validation_loss:
            self.min_validation_loss = validation_loss
            self.epoch = 0
        elif validation_loss >= (self.min_validation_loss + self.min_delta):
            self.epoch += 1
            if self.epoch >= self.patience:
                return True
        return False