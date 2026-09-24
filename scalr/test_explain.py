"""This is a test file for the per-cell/per-class explainability API (explain.py)."""

import pytest

import scalr
from scalr.utils import generate_dummy_dge_anndata


def _toy_adata():
    return generate_dummy_dge_anndata(n_donors=8,
                                      cell_type_list=['B_cell', 'T_cell', 'DC'],
                                      cell_replicate=10,
                                      n_vars=20)


def _toy_model():
    adata = _toy_adata()
    model = scalr.train(adata,
                        labels_key='cell_type',
                        group_key='donor_id',
                        hidden_layers=(16,),
                        epochs=2,
                        batch_size=32,
                        device='cpu',
                        verbose=False)
    return adata, model


def test_explain_returns_one_explanation_per_index():
    """model.explain should return one CellExplanation per requested index, in order."""
    adata, model = _toy_model()
    indices = [0, 5, 10]

    explanations = model.explain(adata, indices=indices, top_k=5, device='cpu')

    assert len(explanations) == len(indices)
    for i, explanation in zip(indices, explanations):
        assert isinstance(explanation, scalr.CellExplanation)
        assert explanation.obs_name == str(adata.obs_names[i])
        assert explanation.prediction in model.class_names
        assert 0.0 <= explanation.confidence <= 1.0


def test_explain_gene_scores_use_model_features():
    """Supporting/contradictory gene names should come from the model's feature list."""
    adata, model = _toy_model()

    explanations = model.explain(adata, indices=[0, 1], top_k=3, device='cpu')

    for explanation in explanations:
        for gene, score in explanation.supporting_genes:
            assert gene in model.features
            assert score > 0
        for gene, score in explanation.contradictory_genes:
            assert gene in model.features
            assert score < 0


def test_explain_respects_top_k():
    """No more than top_k supporting/contradictory genes should be returned per cell."""
    adata, model = _toy_model()

    explanations = model.explain(adata, indices=[0], top_k=2, device='cpu')

    assert len(explanations[0].supporting_genes) <= 2
    assert len(explanations[0].contradictory_genes) <= 2


def test_explain_class_returns_supporting_and_contradictory_genes():
    """model.explain_class should explain a class independent of any specific cell."""
    _, model = _toy_model()
    class_name = model.class_names[0]

    supporting, contradictory = model.explain_class(class_name,
                                                    top_k=5,
                                                    device='cpu')

    assert isinstance(supporting, list)
    assert isinstance(contradictory, list)
    for gene, score in supporting:
        assert gene in model.features
        assert score > 0
    for gene, score in contradictory:
        assert gene in model.features
        assert score < 0


def test_explain_class_rejects_unknown_class():
    """explain_class should raise a clear error for a class the model doesn't have."""
    _, model = _toy_model()

    with pytest.raises(ValueError):
        model.explain_class('not_a_real_class', device='cpu')
