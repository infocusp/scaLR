class LinearModelTrainer(TrainerBase):
    def __init__():
        super().__init__()
        pass
    
    def train_one_epoch(self, dl: DataLoader) -> (float, float):
        """Trains one epoch

        Args:
            dl: training dataloader

        Returns:
            Train Loss, Train Accuracy
        """
        return total_loss, accuracy


    def validation(self, dl: DataLoader) -> (float, float):
        """ Validates after training one epoch

        Args:
            dl: validation dataloader

        Returns:
            Validation Loss, Validation Accuracy
        """
        
        return total_loss, accuracy