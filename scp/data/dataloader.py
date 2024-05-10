import torch
import anndata as ad
from anndata.experimental import AnnLoader

from .preprocess import binning
from ..tokenizer import tokenize_and_pad_batch

def simpleDataLoader(adata, target, batch_size=1, label_mappings=None):
    """
    A simple data loader to prepare inputs to be fed into linear model and corresponding labels

    Args:
        adata: anndata object containing the data
        target: corresponding metadata name to be treated as training objective in classification.
                must be present as a column_name in adata.obs
        batch_size: size of batches returned
        label_mappings: mapping the target name to respective ids

    Return:
        PyTorch DataLoader object with (X: Tensor [batch_size, features], y: Tensor [batch_size, ])
    """
    label_mappings = label_mappings[target]['label2id']
    
    def collate_fn(batch, target, label_mappings):
        x = batch.X.float()
        y = torch.as_tensor(batch.obs[target].astype('category').cat.rename_categories(label_mappings).astype('int64').values)
        return x,y
    
    return AnnLoader(adata, batch_size=batch_size, collate_fn=lambda batch: collate_fn(batch, target, label_mappings))
        
def transformerDataLoader(adata,
                          target,
                          batch_size=16,
                          label_mappings={},
                          value_bin=True,
                          n_bins=51,
                          gene_ids=None,
                          max_len=3001,
                          vocab=None,
                          pad_token='<pad>',
                          pad_value=-2,
                          append_cls=True,
                          include_zero_gene=False
                         ):
    """
    A data loader to prepare inputs to be fed into transformer model and corresponding labels

    Args:
        adata: anndata object containing the data
        target: corresponding metadata name to be treated as training objective in classification.
                must be present as a column_name in adata.obs
        batch_size: size of batches returned
        label_mappings: mapping the target name to respective ids
        value_bin: boolean to specify value binning preprocessing
        n_bins: number of bins
        gene_ids: all gene_id values in vocab
        max_len: max sequence length for input
        vocab: torch vocab for bidirectional lookup (gene_ids <-> gene_names)
        pad_token: padding token <pad>
        pad_value: padding value expressed
        append_cls: appending cls token at beginning of sequence
        include_zero_gene: boolean to include zero expressed genes in sequence

    Return:
        PyTorch DataLoader object with (input_gene_ids: Tensor [batch_size, seq_len],
                                        input_values: Tensor [batch_size, seq_len]
                                        src_key_padding_mask: [batch_size, seq_len]
                                        y: Tensor [batch_size, ]
                                        )
    """
    label_mappings = label_mappings[target]['label2id']

    def collate_fn( batch, 
                    target,
                    label_mappings,
                    value_bin,
                    n_bins,
                    gene_ids,
                    max_len,
                    vocab,
                    pad_token,
                    pad_value,
                    append_cls,
                    include_zero_gene):
        
        y = torch.as_tensor(batch.obs[target].cat.rename_categories(label_mappings).astype('int64').values)
        
        if value_bin:
            batch = binning(batch, n_bins)
        data = batch.X
            
        tokenized_data = tokenize_and_pad_batch(
                data,
                gene_ids,
                max_len=3001,
                vocab=vocab,
                pad_token=pad_token,
                pad_value=pad_value,
                append_cls=True,
                include_zero_gene=False,
            )
        
        input_gene_ids = tokenized_data["genes"]
        input_values = tokenized_data["values"]
        src_key_padding_mask = input_gene_ids.eq(vocab[pad_token])
        return input_gene_ids, input_values, src_key_padding_mask, y

    collate_inp = [target, label_mappings, value_bin, n_bins, gene_ids, max_len, vocab, pad_token, pad_value, append_cls, include_zero_gene]
    return AnnLoader(adata, batch_size=batch_size, collate_fn=lambda batch: collate_fn(batch, *collate_inp))
    
    