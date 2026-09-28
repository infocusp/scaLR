"""`ShapScorer` pulls in the `shap` package, which is only needed when SHAP-
based scoring is actually configured; it's imported lazily on first attribute
access (PEP 562) so importing `scalr` doesn't pay that cost for unrelated
commands. `getattr(module, name)` (the `name`/`params` builder pattern used
by `build_scorer`) transparently triggers this, same as an eager import would.
"""

import importlib

from ._scoring import build_scorer
from ._scoring import ScoringBase
from .linear_scorer import LinearScorer

_LAZY_SUBMODULES = {
    'ShapScorer': '.shap_scorer',
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
