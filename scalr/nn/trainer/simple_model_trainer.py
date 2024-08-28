from scalr.nn.trainer import TrainerBase


class SimpleModelTrainer(TrainerBase):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
