"""Pricing, Greeks and brick-by-brick decomposition of option strategies and structured products."""

from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("structured-products-pricer")
except PackageNotFoundError:  # pragma: no cover - source tree used without installation
    __version__ = "0.0.0"

__all__ = ["__version__"]
