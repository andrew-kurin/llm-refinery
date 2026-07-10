from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("llm-refinery")
except PackageNotFoundError:  # pragma: no cover - source tree without installation metadata
    __version__ = "0.0.0"
