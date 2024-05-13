import torch
from torch import nn
from torch.nn import TransformerEncoder, TransformerEncoderLayer

class GeneEncoder(nn.Module):
    """
    Encode gene_id values to a vector using embedding layer.
    """
    def __init__(
        self,
        num_embeddings: int,
        embedding_dim: int,
        padding_idx = None,
    ):
        super().__init__()
        self.embedding = nn.Embedding(
            num_embeddings, embedding_dim, padding_idx=padding_idx
        )
        self.enc_norm = nn.LayerNorm(embedding_dim)

    def forward(self, x):
        """
        Args:
            x: Tensor, shape [batch_size, seq_len]
        """
        x = self.embedding(x)  # (batch, seq_len, embsize)
        x = self.enc_norm(x)
        return x

class ContinuousValueEncoder(nn.Module):
    """
    Encode real number values to a vector using neural nets projection.
    """

    def __init__(self, d_model: int, dropout: float = 0.1, max_value: int = 512):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        self.linear1 = nn.Linear(1, d_model)
        self.activation = nn.ReLU()
        self.linear2 = nn.Linear(d_model, d_model)
        self.norm = nn.LayerNorm(d_model)
        self.max_value = max_value

    def forward(self, x):
        """
        Args:
            x: Tensor, shape [batch_size, seq_len]
        """
        # TODO: test using actual embedding layer if input is categorical
        # expand last dimension
        x = x.unsqueeze(-1)
        # clip x to [-inf, max_value]
        x = torch.clamp(x, max=self.max_value)
        x = self.activation(self.linear1(x))
        x = self.linear2(x)
        x = self.norm(x)
        return self.dropout(x)

class ClsDecoder(nn.Module):
    """
    Decoder for classification task.
    """

    def __init__(
        self,
        *args
    ):
        super().__init__()
        # module list
        activation = nn.ReLU
        self._decoder = nn.ModuleList()
        n = len(args)
        for i in range(n):
            if i==n-2:break
            self._decoder.append(nn.Linear(args[i], args[i+1]))
            self._decoder.append(activation())

        self.out_layer = nn.Linear(args[n-2], args[n-1])

    def forward(self, x):
        """
        Args:
            x: Tensor, shape [batch_size, embsize]
        """
        for layer in self._decoder:
            x = layer(x)
        return self.out_layer(x)

class TransformerModel(nn.Module):
    """
    Transformer Model for single-cell data training on classification task.
    """    
    def __init__(self,
                 ntokens,
                 vocab,
                 d_model,
                 nheads,
                 nlayers,
                 dropout,
                 decoder_layers,
                 pad_token='<pad>'
                 ):
        """
        Args:
            ntokens: total number of distinct tokens in vocab
            vocab: torch vocab for bidirectional lookup (gene_ids <-> gene_names)
            d_model: dimension of model hidden layers
            nheads: number of attention heads
            nlayers: number of encoder layers
            dropout: dropout value between layers
            decoder_layers: List of feature_dimensions going from d_model -> number of classes for classification.
            pad_token: padding token, deafult: '<pad>'
        
        """
        
        super().__init__()

        self.gene_encoder = GeneEncoder(ntokens, d_model, padding_idx=vocab[pad_token])
        self.value_encoder = ContinuousValueEncoder(d_model, dropout)
        
        encoder_layers = TransformerEncoderLayer(
                d_model, nheads, d_model, dropout, batch_first=True
            )
        self.transformer_encoder = TransformerEncoder(encoder_layers, nlayers)

        self.cls_decoder = ClsDecoder(*decoder_layers)

        self.bn = nn.BatchNorm1d(d_model, eps=6.1e-5)

    def _encode(
        self,
        src,
        values,
        src_key_padding_mask
    ):
        """
        encodes various inputs to a single d_model dimensional vector and finally passes to the network

        Args:
            src: gene_ids from vocab
            values: gene expression values of corresponding ids
            src_key_padding_mask: mask to avoid attention with pad_token

        Return:
            Tensor of shape [batch_size, <cls> + seq_len, d_model]
        """
        src = self.gene_encoder(src)  # (batch, seq_len, embsize)

        values = self.value_encoder(values)  # (batch, seq_len, embsize)
        total_embs = src + values

        total_embs = self.bn(total_embs.permute(0, 2, 1)).permute(0, 2, 1)

        output = self.transformer_encoder(
            total_embs, src_key_padding_mask=src_key_padding_mask
        )
        return output
    
    def forward(self, src, values, src_key_padding_mask):
        transformer_output = self._encode(
            src, values, src_key_padding_mask
        )
        """
        Args:
            src: gene_ids from vocab
            values: gene expression values of corresponding ids
            src_key_padding_mask: mask to avoid attention with pad_token
            
        Return:
            output dict contains cell_embedding[d_model] and cls_decoder output[n_cls]
        """

        output = {}
        
        cell_emb = transformer_output[:, 0, :]
        output['cell_emb'] = cell_emb

        output['cls_output'] = self.cls_decoder(cell_emb)

        return output
