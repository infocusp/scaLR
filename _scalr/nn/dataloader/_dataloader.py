from anndata.experimental import AnnLoader

class DataLoaderBase:

    def collate_fn(self, batch, **kwargs):
        """Given an input anndata of batch_size,
        the collate function creates inputs and outputs
        """
        pass
    
    # This method returns a torch DataLoader object
    def __call__(self, adata, batch_size, **kwargs):
        return AnnLoader(adata,
                         batch_size=batch_size,
                         collate_fn=lambda batch: self.collate_fn(
                         batch, **kwargs))
        

def build_dataloader(name, adata, batch_size, **kwargs):
    
    dataloader = getattr(_scalr.nn.dataloader, name)(adata, batch_size, **kwargs)
    return dataloader