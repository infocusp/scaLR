from _scalr.trainer import build_trainer
from _scalr.model import build_model
from _scalr.model.callbacks import build_callbacks
from _scalr.model.dataloader import build_dataloader

class ModelTrainer:
    
    def __init__(self, model_config, train_config, data_config, trainer_name='linear'):
        # Load params here
        # Load paths
        
        # Initialize the model, optimizer, loss and callbacks class objects here
        # Initialize the weights (in case of resume from checkpointing) for model and opt 
        # Finally create a trainer object for model training
        
        self.opt = None
        self.loss_fn = None
        self.model = build_model
        self.callbacks = build_callbacks
        self.trainer = build_trainer(trainer_name, model, opt, loss, callbacks)
        
        pass

    def build_loss_fn():
        return loss_fn
    
    def build_optimizer():
        return optimizer
    
    def train(self):
        """Trains the model.

        Args:
            epochs: max number of epochs to train model on
            train_dl: training dataloader
            val_dl: validation dataloader
        """

        callback_executor = CallbackExecutor(
            dirpath=self.dirpath, callback_paramaters=self.callback_params)

        for epoch in range(epochs):
            ep_start = time()
            print(f'Epoch {epoch+1}:')
            train_loss, train_acc = self.train_one_epoch(train_dl)
            print(
                f'Training Loss: {train_loss} || Training Accuracy: {train_acc}'
            )
            val_loss, val_acc = self.validation(val_dl)
            print(
                f'Validation Loss: {val_loss} || Validation Accuracy: {val_acc}'
            )
            ep_end = time()
            print(f'Time: {ep_end-ep_start}\n', flush=True)

            if callback_executor.execute(self.model.state_dict(),
                                         self.opt.state_dict(), train_loss,
                                         train_acc, val_loss, val_acc):
                break

        self.model.load_state_dict(
            torch.load(path.join(self.dirpath, 'best_model',
                                 'model.pt'))['model_state_dict'])
        torch.save(self.model,
                   path.join(self.dirpath, 'best_model', 'model.bin'))