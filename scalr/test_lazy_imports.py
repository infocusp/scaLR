"""This is a test file verifying scalr.analysis/scalr.feature.scoring's heavy
(scanpy/shap-backed) submodules are lazily imported, not paid for by every
`import scalr` (e.g. the `scalr` CLI's `models`/`train`/`annotate` commands).

Runs in a subprocess: other tests in the same pytest session may already have
imported scanpy/shap, which would make an in-process check order-dependent.
"""

import subprocess
import sys


def _run(code: str) -> str:
    result = subprocess.run([sys.executable, '-c', code],
                            capture_output=True,
                            text=True,
                            check=True)
    return result.stdout.strip()


def test_import_scalr_does_not_eagerly_load_scanpy_or_shap():
    """`import scalr` alone should not pull in scanpy/shap."""
    output = _run("import sys\n"
                  "import scalr\n"
                  "print('scanpy' in sys.modules)\n"
                  "print('shap' in sys.modules)\n")

    assert output == 'False\nFalse'


def test_lazy_analysis_attribute_resolves_and_loads_scanpy_on_demand():
    """Accessing a lazy scalr.analysis attribute should still work, loading scanpy on demand."""
    output = _run("import sys\n"
                  "import scalr\n"
                  "scalr.analysis.DgeLMEM\n"
                  "print('scanpy' in sys.modules)\n")

    assert output == 'True'


def test_lazy_scoring_attribute_resolves_and_loads_shap_on_demand():
    """Accessing scalr.feature.scoring.ShapScorer should still work, loading shap on demand."""
    output = _run("import sys\n"
                  "from scalr.feature.scoring import ShapScorer\n"
                  "print('shap' in sys.modules)\n")

    assert output == 'True'


def test_build_analyser_resolves_lazy_class_by_name():
    """The name/params builder pattern (getattr-based) should still find lazy analysis classes."""
    output = _run(
        "from scalr.analysis import build_analyser\n"
        "analyser, _ = build_analyser({'name': 'Heatmap', 'params': {}})\n"
        "print(type(analyser).__name__)\n")

    assert output == 'Heatmap'
