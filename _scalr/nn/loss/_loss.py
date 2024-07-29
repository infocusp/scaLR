class Loss:
    pass

def build_loss(name):
    return getattr(torch.nn, name)