class EarlyStopping:
    """
    Implementing Early stopping for model in case of learning stalled based upon validation loss
    """
    def __init__(self, patience, min_delta):
        """
        Args:
            patience: number of epochs for which model does not improve
            min_delta: minimum improvement in val_loss considered as model improved
        """
        self.patience = int(patience)
        self.min_delta = float(min_delta)
        self.counter = 0
        self.min_validation_loss = float('inf')

    def __call__(self, validation_loss):
        if validation_loss < self.min_validation_loss:
            self.min_validation_loss = validation_loss
            self.counter = 0
        elif validation_loss >= (self.min_validation_loss + self.min_delta):
            self.counter += 1
            if self.counter >= self.patience:
                return True
        return False