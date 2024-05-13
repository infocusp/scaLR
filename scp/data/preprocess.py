import anndata as ad
import numpy as np
import torch


def binning(adata, n_bins):
    """
    Value binning for pre-processing of data in transformer model

    Args:
        adata: anndata object containing the data
        n_bins: number fo bins to distribute data into

    Return:
        anndata object containing layer 'X_binned' with binned data values
    """

    data = adata.X
    binned_rows = []

    for row in data:
        if row.max() == 0:
            binned_rows.append(np.zeros_like(row, dtype=np.int64))
            continue
        non_zero_ids = row.nonzero()
        non_zero_row = row[non_zero_ids]
        if isinstance(non_zero_row, torch.Tensor):
            non_zero_ids = non_zero_ids.flatten()
            non_zero_row = non_zero_row.flatten()
        bins = np.quantile(non_zero_row, np.linspace(0, 1, n_bins - 1))
        # bins = np.sort(np.unique(bins))
        # NOTE: comment this line for now, since this will make the each category
        # has different relative meaning across datasets
        non_zero_digits = _digitize(non_zero_row, bins)
        assert non_zero_digits.min() >= 1
        assert non_zero_digits.max() <= n_bins - 1
        binned_row = torch.zeros_like(row, dtype=torch.int64)
        binned_row[non_zero_ids] = non_zero_digits
        # print(binned_row.shape)
        binned_rows.append(binned_row)
    adata = adata.to_adata()
    adata.X = torch.stack(binned_rows)
    return adata


def _digitize(x, bins, side="both") -> np.ndarray:
    """
    Digitize the data into bins. This method spreads data uniformly when bins
    have same values.

    Args:

    x (:class:`np.ndarray`):
        The data to digitize.
    bins (:class:`np.ndarray`):
        The bins to use for digitization, in increasing order.
    side (:class:`str`, optional):
        The side to use for digitization. If "one", the left side is used. If
        "both", the left and right side are used. Default to "one".

    Returns:

    :class:`np.ndarray`:
        The digitized data.
    """
    assert x.ndim == 1 and bins.ndim == 1

    left_digits = np.digitize(x, bins)
    if side == "one":
        return left_digits

    right_difits = np.digitize(x, bins, right=True)

    rands = np.random.rand(len(x))  # uniform random numbers

    digits = torch.as_tensor(rands * (right_difits - left_digits) +
                             left_digits)
    digits = torch.ceil(digits).flatten().type(torch.int64)
    return digits
