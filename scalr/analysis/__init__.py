"""`scalr.analysis`'s heavier submodules (scanpy/statsmodels/matplotlib-backed)
are imported lazily on first attribute access (PEP 562), so importing `scalr`
for something unrelated to analysis (e.g. the `scalr` CLI's `models`/`train`/
`annotate` commands) doesn't pay their import cost. `getattr(module, name)`
(the `name`/`params` builder pattern used by `build_analyser`) transparently
triggers this, same as an eager import would.
"""

import importlib

from ._analyser import AnalysisBase
from ._analyser import build_analyser

_LAZY_SUBMODULES = {
    'DgeLMEM': '.dge_lmem',
    'DgePseudoBulk': '.dge_pseudobulk',
    'generate_and_save_classification_report': '.evaluation',
    'get_accuracy': '.evaluation',
    'GeneRecallCurve': '.gene_recall_curve',
    'Heatmap': '.heatmap',
    'RocAucCurve': '.roc_auc',
}


def __getattr__(name):
    module_name = _LAZY_SUBMODULES.get(name)
    if module_name is None:
        raise AttributeError(f'module {__name__!r} has no attribute {name!r}')
    value = getattr(importlib.import_module(module_name, __name__), name)
    globals()[name] = value
    return value


def __dir__():
    return sorted(set(globals()) | set(_LAZY_SUBMODULES))
