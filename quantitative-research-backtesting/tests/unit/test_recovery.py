"""Recovery regressions for source-checkout reproducibility."""
from importlib.metadata import PackageNotFoundError

from quantlab import __version__
from quantlab.backtesting import engine


def test_source_checkout_records_its_version_without_package_metadata(monkeypatch):
    original = engine.importlib_metadata.version

    def version(name):
        if name == "quantlab":
            raise PackageNotFoundError(name)
        return original(name)

    engine._dependency_versions.cache_clear()
    monkeypatch.setattr(engine.importlib_metadata, "version", version)
    try:
        assert engine._dependency_versions()["quantlab"] == __version__
    finally:
        engine._dependency_versions.cache_clear()
