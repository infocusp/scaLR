"""This file implements confidence calibration and open-set (unknown-cell) detection.

A raw softmax maximum is not a calibrated probability. This module fits a
temperature-scaling calibrator on held-out logits/labels, and derives
confidence, entropy, margin and an "unknown" abstention flag from calibrated
probabilities.
"""

from dataclasses import dataclass

import numpy as np
import torch
from torch import nn
from torch import Tensor


class TemperatureScaler:
    """Single-parameter temperature scaling for multi-class calibration.

    Reference: Guo et al., 2017, "On Calibration of Modern Neural Networks".
    """

    def __init__(self, temperature: float = 1.0):
        self.temperature = float(temperature)

    def fit(self,
            logits: Tensor,
            labels: Tensor,
            lr: float = 0.01,
            max_iter: int = 100) -> 'TemperatureScaler':
        """Fit the scalar temperature on held-out (logits, labels) via NLL minimization."""
        logits = logits.detach().float()
        labels = labels.detach().long()

        log_temperature = torch.zeros(1, requires_grad=True)
        optimizer = torch.optim.LBFGS([log_temperature],
                                      lr=lr,
                                      max_iter=max_iter)
        nll = nn.CrossEntropyLoss()

        def closure():
            optimizer.zero_grad()
            temperature = torch.exp(log_temperature)
            loss = nll(logits / temperature, labels)
            loss.backward()
            return loss

        optimizer.step(closure)
        self.temperature = float(torch.exp(log_temperature).item())
        return self

    def calibrate_probabilities(self, logits: np.ndarray) -> np.ndarray:
        """Apply the fitted temperature and return softmax probabilities."""
        logits_t = torch.as_tensor(logits, dtype=torch.float32)
        probs = torch.softmax(logits_t / self.temperature, dim=-1)
        return probs.numpy()

    def to_dict(self) -> dict:
        return {
            'method': 'temperature_scaling',
            'temperature': self.temperature
        }

    @classmethod
    def from_dict(cls, d: dict) -> 'TemperatureScaler':
        return cls(temperature=d.get('temperature', 1.0))


def predictive_entropy(probabilities: np.ndarray) -> np.ndarray:
    """Normalized (0-1) Shannon entropy of each row's class distribution."""
    n_classes = probabilities.shape[1]
    eps = 1e-12
    raw_entropy = -(probabilities * np.log(probabilities + eps)).sum(axis=1)
    max_entropy = np.log(n_classes) if n_classes > 1 else 1.0
    return raw_entropy / max_entropy if max_entropy > 0 else raw_entropy


def top1_top2_margin(probabilities: np.ndarray) -> np.ndarray:
    """Difference between the top-1 and top-2 class probabilities per row."""
    sorted_probs = np.sort(probabilities, axis=1)
    if probabilities.shape[1] < 2:
        return sorted_probs[:, -1]
    return sorted_probs[:, -1] - sorted_probs[:, -2]


def expected_calibration_error(probabilities: np.ndarray,
                               labels: np.ndarray,
                               n_bins: int = 15) -> float:
    """Expected Calibration Error (ECE) using the top-1 confidence/correctness pairs."""
    confidences = probabilities.max(axis=1)
    predictions = probabilities.argmax(axis=1)
    accuracies = (predictions == labels).astype(np.float64)

    bin_edges = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    n = len(labels)
    for lo, hi in zip(bin_edges[:-1], bin_edges[1:]):
        in_bin = (confidences > lo) & (confidences <= hi)
        if not in_bin.any():
            continue
        bin_conf = confidences[in_bin].mean()
        bin_acc = accuracies[in_bin].mean()
        ece += (in_bin.sum() / n) * abs(bin_conf - bin_acc)
    return float(ece)


def brier_score(probabilities: np.ndarray, labels: np.ndarray) -> float:
    """Multi-class Brier score: mean squared error between one-hot labels and probabilities."""
    n_classes = probabilities.shape[1]
    one_hot = np.eye(n_classes)[labels]
    return float(np.mean(np.sum((probabilities - one_hot)**2, axis=1)))


@dataclass
class OpenSetThresholds:
    """Abstention thresholds used to flag predictions as `UNKNOWN`."""

    min_confidence: float = 0.5
    max_entropy: float = 0.7
    min_margin: float = 0.0

    def flag_unknown(self, confidence: np.ndarray, entropy: np.ndarray,
                     margin: np.ndarray) -> np.ndarray:
        """Return a boolean array: True where a cell should be abstained on."""
        return ((confidence < self.min_confidence) |
                (entropy > self.max_entropy) | (margin < self.min_margin))

    def to_dict(self) -> dict:
        return {
            'min_confidence': self.min_confidence,
            'max_entropy': self.max_entropy,
            'min_margin': self.min_margin,
        }

    @classmethod
    def from_dict(cls, d: dict) -> 'OpenSetThresholds':
        return cls(**d)
