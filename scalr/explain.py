"""This file implements gradient-based gene attributions for individual cell
predictions and per-class signatures.

This is a first-order sensitivity explanation (Grad x Input saliency), not a
causal or mechanistic one: it answers "which genes, in this cell's
expression profile, pushed the model's output most strongly toward (or away
from) the predicted class" via the gradient of the class logit with respect
to each input gene, scaled by that gene's expression. Like the doublet/
refinement heuristics elsewhere in this layer, treat it as a screening/
interpretation aid, not ground truth biology.
"""

from dataclasses import dataclass

import numpy as np
import torch
from torch import nn


@dataclass
class CellExplanation:
    """Per-cell gene attribution for the model's top prediction."""

    obs_name: str
    prediction: str
    confidence: float
    supporting_genes: list[tuple[str, float]]
    contradictory_genes: list[tuple[str, float]]

    def __str__(self) -> str:
        lines = [
            self.obs_name,
            f'Prediction: {self.prediction}',
            f'Confidence: {self.confidence:.2f}',
            '',
            'Supporting genes',
        ]
        lines += [f'{g}  {s:+.2f}' for g, s in self.supporting_genes]
        lines.append('')
        lines.append('Contradictory signals')
        lines += [f'{g}  {s:+.2f}' for g, s in self.contradictory_genes]
        return '\n'.join(lines)


def _grad_x_input(model: nn.Module, x: torch.Tensor,
                  class_idx: torch.Tensor) -> np.ndarray:
    """Gradient-of-logit x input attribution for a batch of cells."""
    x = x.clone().detach().requires_grad_(True)
    logits = model(x)['cls_output']
    selected = logits[torch.arange(logits.shape[0]), class_idx]
    model.zero_grad(set_to_none=True)
    selected.sum().backward()
    attributions = x.grad.detach() * x.detach()
    return attributions.cpu().numpy()


def attribute_cells(
    model: nn.Module,
    x: torch.Tensor,
    features: list[str],
    class_names: list[str],
    predicted_ids: np.ndarray,
    confidence: np.ndarray,
    obs_names: list[str],
    top_k: int = 20,
) -> list[CellExplanation]:
    """Explain each row of `x` in terms of its aligned gene attributions.

    Args:
        model: The underlying network, in eval mode, on the same device as `x`.
        x: Aligned, dense `[n_cells, n_features]` input tensor.
        features: Gene names for each column of `x`, in the model's feature order.
        class_names: Model's class label list.
        predicted_ids: Per-cell predicted class index to explain.
        confidence: Per-cell top-1 probability, carried through for display.
        obs_names: Per-cell identifiers, aligned to `x`'s rows.
        top_k: Number of supporting and contradictory genes to keep per cell.

    Returns:
        One `CellExplanation` per row of `x`, in the same order.
    """
    class_idx = torch.as_tensor(predicted_ids,
                                dtype=torch.long,
                                device=x.device)
    attributions = _grad_x_input(model, x, class_idx)

    explanations = []
    for i in range(x.shape[0]):
        scores = attributions[i]
        descending = np.argsort(-scores)
        supporting = [(features[j], float(scores[j]))
                      for j in descending[:top_k]
                      if scores[j] > 0]
        ascending = descending[::-1]
        contradictory = [(features[j], float(scores[j]))
                         for j in ascending[:top_k]
                         if scores[j] < 0]
        explanations.append(
            CellExplanation(
                obs_name=obs_names[i],
                prediction=class_names[predicted_ids[i]],
                confidence=float(confidence[i]),
                supporting_genes=supporting,
                contradictory_genes=contradictory,
            ))
    return explanations


def attribute_class(
    model: nn.Module,
    features: list[str],
    class_names: list[str],
    class_name: str,
    top_k: int = 50,
    device: str = 'cpu',
) -> tuple[list[tuple[str, float]], list[tuple[str, float]]]:
    """Explain a class in general, independent of any specific cell.

    Computes the gradient of the class logit with respect to a zero-
    expression baseline input. This approximates the genes the model relies
    on most for this class near a neutral expression profile; unlike
    `attribute_cells`, it is not derived from any observed cell.

    Returns:
        `(supporting_genes, contradictory_genes)`, each a list of
        `(gene, score)` sorted by descending/ascending attribution.
    """
    if class_name not in class_names:
        raise ValueError(f'Unknown class {class_name!r}; expected one of '
                         f'{class_names}.')
    class_id = class_names.index(class_name)

    x = torch.zeros((1, len(features)),
                    dtype=torch.float32,
                    device=device,
                    requires_grad=True)
    logits = model(x)['cls_output']
    model.zero_grad(set_to_none=True)
    logits[0, class_id].backward()
    scores = x.grad.detach().cpu().numpy()[0]

    descending = np.argsort(-scores)
    supporting = [(features[j], float(scores[j]))
                  for j in descending[:top_k]
                  if scores[j] > 0]
    ascending = descending[::-1]
    contradictory = [(features[j], float(scores[j]))
                     for j in ascending[:top_k]
                     if scores[j] < 0]
    return supporting, contradictory
